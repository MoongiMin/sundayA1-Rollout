"""Accepted R25-300 stationary-base bench rollout on three Feetech boards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
from rollout import BNO055, LegBus  # existing Sunday_v13_Jetson_Rollout installation

ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / "r25_contract.json").read_text())
BOARDS = ("lower_body", "left_arm", "right_arm")
RAD_PER_TICK = 2 * math.pi / 4096


def check_finite(values, label):
    if not np.all(np.isfinite(values)):
        raise ValueError(label + " contains nonfinite values")


def check_imu(sample, allow_uncalibrated_accel, initial=False):
    if sample["gyro_calibration"] < 3:
        raise ValueError("BNO055 gyro calibration below 3")
    if sample["accel_calibration"] < 3 and not allow_uncalibrated_accel:
        raise ValueError("BNO055 accel calibration below 3")
    gravity = np.asarray(sample["gravity"], dtype=float)
    check_finite(gravity, "BNO055 gravity")
    if gravity.shape != (3,) or not .95 <= np.linalg.norm(gravity) <= 1.05:
        raise ValueError("BNO055 gravity norm invalid")
    min_upright = .95 if initial and allow_uncalibrated_accel else .8
    if -gravity[2] < min_upright:
        raise ValueError("Torso tilt exceeds supported bench limit")


class Actor:
    def __init__(self):
        if hashlib.sha256((ROOT / "r25_actor.npz").read_bytes()).hexdigest() != CONTRACT["actor_npz_sha256"]:
            raise ValueError("R25 actor file SHA256 mismatch")
        with np.load(ROOT / "r25_actor.npz") as data:
            self.params = {key: data[key] for key in data.files}
        assert self.params["mlp_0_weight"].shape == (256, 460)
        assert self.params["mlp_2_weight"].shape == (128, 256)
        assert self.params["mlp_4_weight"].shape == (128, 128)
        assert self.params["mlp_6_weight"].shape == (22, 128)

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(1, 460)
        x = (x - self.params["obs_normalizer__mean"]) / (self.params["obs_normalizer__std"] + .01)
        for i in (0, 2, 4, 6):
            x = x @ self.params[f"mlp_{i}_weight"].T + self.params[f"mlp_{i}_bias"]
            if i != 6:
                x = np.where(x > 0, x, np.expm1(x))
        check_finite(x, "actor output")
        return x[0].tolist()


class R25History:
    """R25 term-major five-frame actor observation and command oscillator."""

    def __init__(self):
        self.command = np.zeros(3, dtype=np.float32)
        self.gait_phase = np.float32(.075)
        self.gate = 0.0
        self.heading_target = 0.0
        self.frames = None
        self.step_count = 0

    def period(self):
        c = self.command
        forward = np.clip((c[0] - np.float32(.06)) / np.float32(.09), 0, 1)
        reverse = np.clip((-c[0] - np.float32(.005)) / np.float32(.025), 0, 1)
        yaw = np.clip((abs(c[2]) - np.float32(.12)) / np.float32(.28), 0, 1)
        return np.float32(np.float32(1.4) - np.float32(.4) * max(forward, reverse, yaw))

    def push(self, q, dq, imu, previous_action, yaw_relative, body_velocity_x=0.0):
        if self.step_count > 0:
            self.gait_phase = np.float32(
                (self.gait_phase + np.float32(.02) / self.period()) % np.float32(1.)
            )
            desired = np.clip((np.linalg.norm(self.command[:2]) + .375 * abs(self.command[2])) / .03, 0, 1)
            self.gate += (1 - math.exp(-.02 / .18)) * (desired - self.gate)
        phase = 2 * math.pi * float(self.gait_phase)
        names = CONTRACT["observation_joint_names"]
        defaults = CONTRACT["default_joint_position_rad"]
        terms = [
            np.asarray(imu["angular_velocity"]) * .25,
            imu["gravity"],
            self.command,
            [q[n] - default for n, default in zip(names, defaults)],
            [.05 * dq[n] for n in names],
            previous_action,
            [math.sin(phase) * self.gate, math.cos(phase) * self.gate],
            [math.sin(self.heading_target - yaw_relative), math.cos(self.heading_target - yaw_relative)],
            [body_velocity_x, 0., 0.],  # Experimental fixed body-frame velocity; real IMU remains in use.
        ]
        terms = [np.asarray(term, dtype=np.float32) for term in terms]
        assert [len(term) for term in terms] == [3, 3, 3, 27, 27, 22, 2, 2, 3]
        if self.frames is None:
            self.frames = [np.tile(term, (5, 1)) for term in terms]
        else:
            self.frames = [np.concatenate((old[1:], term[None])) for old, term in zip(self.frames, terms)]
        self.step_count += 1
        result = np.concatenate([frame.ravel() for frame in self.frames])
        check_finite(result, "R25 observation")
        assert result.shape == (460,)
        return result

    def advance_command(self, requested):
        rate = np.asarray((.5, .5, .8), dtype=np.float32) * np.float32(.02)
        self.command += np.clip(np.asarray(requested, dtype=np.float32) - self.command, -rate, rate)
        self.heading_target += float(self.command[2]) * .02


def joint_radians(ticks):
    q = {}
    for name, j in CONTRACT["joints"].items():
        tick = ticks[j["board"]][j["id"]]
        if not 0 <= tick <= 4095:
            raise ValueError(f"{name}: encoder tick outside one turn")
        q[name] = j["reference_angle_rad"] + j["sign"]*(tick-j["home_tick"])*RAD_PER_TICK
    q.update(CONTRACT["stationary_observation_defaults"])
    return q


def targets(raw, previous):
    names = CONTRACT["action_joint_names"]
    clipped = np.clip(np.asarray(raw, dtype=np.float64), -1., 1.)
    out = []
    for k, name in enumerate(names):
        j = CONTRACT["joints"][name]
        lo, hi = CONTRACT["soft_joint_limits_rad"][name]
        requested = j["reference_angle_rad"] + clipped[k] * CONTRACT["action_scales_rad"][name]
        requested = previous[k] + .3 * (requested - previous[k])
        speed_step = CONTRACT["action_speeds_rad_s"][name] * .02
        requested = min(previous[k] + speed_step, max(previous[k] - speed_step, requested))
        out.append(min(hi, max(lo, requested)))
    return clipped.tolist(), out


def goal_ticks(targets_rad):
    result = {board: {} for board in BOARDS}
    for name, angle in zip(CONTRACT["action_joint_names"], targets_rad):
        j = CONTRACT["joints"][name]
        tick = round(j["home_tick"] + (angle-j["reference_angle_rad"])/(j["sign"]*RAD_PER_TICK))
        if not 0 <= tick <= 4095:
            raise ValueError(f"{name}: policy goal outside encoder's 0..4095 format")
        result[j["board"]][j["id"]] = tick
    return result


class Buses:
    def __init__(self, ports):
        self.buses = {}
        self.ids = {board: sorted(j["id"] for j in CONTRACT["joints"].values() if j["board"] == board)
                    for board in BOARDS}
        try:
            for board in BOARDS:
                self.buses[board] = LegBus(ports[board], 1000000, self.ids[board])
            self.readers = {}
            for board in BOARDS:
                bus = self.buses[board]
                reader = bus.sdk.GroupSyncRead(bus.port, bus.packet, 56, 2)
                for mid in self.ids[board]:
                    if not reader.addParam(mid):
                        raise OSError(f"Cannot add {board} ID {mid} to group read")
                self.readers[board] = reader
            self.pool = ThreadPoolExecutor(max_workers=3)
        except BaseException:
            self.close()
            raise

    def read(self):
        tasks = {board: self.pool.submit(self._read_one, board) for board in BOARDS}
        return {board: task.result() for board, task in tasks.items()}

    def _read_one(self, board):
        bus, reader = self.buses[board], self.readers[board]
        for attempt in range(3):
            code = reader.txRxPacket()
            if code == bus.sdk.COMM_SUCCESS:
                if attempt:
                    print(f"READ_RECOVERED: {board} after {attempt} retry(s)", flush=True)
                break
            if code not in (bus.sdk.COMM_RX_TIMEOUT, bus.sdk.COMM_RX_CORRUPT) or attempt == 2:
                raise OSError(f"{board}: group read failed after {attempt + 1} attempt(s), "
                              f"communication={code}")
            # Discard the rest of a partial group reply before a new request.
            bus.port.ser.reset_input_buffer()
            time.sleep(.005)
        result = {}
        for mid in self.ids[board]:
            if not reader.isAvailable(mid, 56, 2):
                raise OSError(f"{board} ID {mid}: group read unavailable")
            result[mid] = reader.getData(mid, 56, 2)
        return result

    def verify(self):
        for board in BOARDS:
            bus = self.buses[board]
            for mid in self.ids[board]:
                if bus.read(mid, 3) != 777 or bus.read(mid, 33, 1) != 0:
                    raise ValueError(f"{board} ID {mid}: model or position mode mismatch")

    def write(self, values):
        tasks = {board: self.pool.submit(self.buses[board].goals, values[board]) for board in BOARDS}
        for task in tasks.values():
            task.result()

    def torque(self, enabled):
        tasks = {board: self.pool.submit(self.buses[board].torque, enabled) for board in BOARDS}
        for task in tasks.values():
            task.result()

    def close(self):
        if hasattr(self, "pool"):
            self.pool.shutdown(wait=True)
        for bus in self.buses.values():
            bus.close()


def run(args):
    actor = Actor()
    actor(np.zeros(460, dtype=np.float32))
    fixed_body_vx = getattr(args, "experimental_fixed_body_vx", None)
    if fixed_body_vx is not None and (not math.isfinite(fixed_body_vx) or
                                   not 0.0 <= fixed_body_vx <= .12):
        raise ValueError("experimental fixed body vx must be between 0 and 0.12 m/s")
    body_vx_input = 0.0 if fixed_body_vx is None else float(fixed_body_vx)
    if args.mode == "status":
        print(json.dumps({"checkpoint_sha256": CONTRACT["source_checkpoint_sha256"],
                          "actor_shape": [460, 22], "joints": len(CONTRACT["joints"]),
                          "home_alignment": CONTRACT.get("home_alignment"),
                          "mode": "supported stationary-base bench only",
                          "experimental_fixed_body_vx_m_s": fixed_body_vx,
                          "motor_output": False}, indent=2))
        return
    if args.mode in ("run", "home-preview") and not (args.enable_motors and args.supported_test):
        raise ValueError(f"{args.mode} requires --enable-motors --supported-test")
    if (args.mode in ("run", "home-preview") and
            CONTRACT.get("home_alignment", {}).get("status") == "v5_home_preserved_left_mismatch"):
        raise ValueError("Direction-only release preserves the v5 Home ticks, including a "
                         "visually confirmed left-arm mismatch; motor-output modes are disabled "
                         "until Home is calibrated separately.")
    if (args.mode in ("run", "home-preview") and
            CONTRACT.get("home_alignment", {}).get("status", "").startswith("calculated_from_v73_") and
            not args.accept_calculated_home):
        raise ValueError("R25 Home ticks were calculated from V73 and have not been "
                         "verified on the robot; inspect the candidate pose first. "
                         "--accept-calculated-home explicitly enables motor output.")
    source = json.loads(Path(args.imu_config).read_text())
    icfg = source["imu"]
    if icfg["bus"] != 7 or icfg["address"] != 40 or not icfg["axes_and_quaternion_verified"]:
        raise ValueError(f"{args.imu_config}: expected verified BNO055 bus=7 address=40 (0x28); "
                         f"got bus={icfg['bus']}, address={icfg['address']}, "
                         f"axes_verified={icfg['axes_and_quaternion_verified']}")
    if not all(Path(args.ports[board]).exists() for board in BOARDS):
        raise FileNotFoundError("One or more stable board ports are disconnected")
    buses = imu = None
    output = args.mode in ("run", "home-preview")
    armed = False
    start = time.monotonic()
    log = ROOT / f"r25_{args.mode}_{time.time_ns()}.jsonl"
    try:
        buses = Buses(args.ports)
        buses.verify()
        imu = BNO055(icfg)
        initial_imu = imu.sample()
        check_imu(initial_imu, args.allow_uncalibrated_accel, initial=True)
        if output:
            for board in BOARDS:
                for mid in buses.ids[board]:
                    if buses.buses[board].read(mid, 40, 1) != 0:
                        raise ValueError(f"{board} ID {mid}: torque already ON")
        first = buses.read()
        joint_radians(first)
        # The hand-observed ranges are metadata, not abort limits. Any valid
        # encoder tick can enter the saved-Home ramp.
        # Measure the serial group-read time for diagnostics. A slow read does
        # not prevent motor output; the loop simply runs at its actual rate.
        read_times = []
        for _ in range(5):
            begin = time.monotonic()
            buses.read()
            read_times.append(time.monotonic()-begin)
        print("Group-read seconds:", [round(t, 4) for t in read_times], flush=True)
        prev_q, prev_time = None, None
        prev_action = [0.]*22
        filtered = [CONTRACT["joints"][n]["reference_angle_rad"] for n in CONTRACT["action_joint_names"]]
        history = R25History()
        yaw_origin = None
        with log.open("x", encoding="utf-8") as f:
            f.write(json.dumps({"mode": args.mode, "requested_body_command": args.command,
                                "contract_sha256": hashlib.sha256((ROOT / "r25_contract.json").read_bytes()).hexdigest(),
                                "supported_stationary_base_assumption": True,
                                "body_linear_velocity_source": ("zero stationary-bench assumption" if fixed_body_vx is None
                                                                 else "experimental fixed body-frame x input"),
                                "body_linear_velocity_input_m_s": [body_vx_input, 0., 0.],
                                "allow_uncalibrated_accel": args.allow_uncalibrated_accel,
                                "checkpoint_sha256": CONTRACT["source_checkpoint_sha256"]})+"\n")
            for s in range(args.delay, 0, -1):
                print(f"Starts in {s}s; Ctrl+C cancels", flush=True)
                time.sleep(1)
            if output:
                initial = buses.read()
                joint_radians(initial)
                home = {board: {} for board in BOARDS}
                for j in CONTRACT["joints"].values():
                    home[j["board"]][j["id"]] = j["home_tick"]
                max_home_delta = max(abs(home[board][mid]-initial[board][mid])
                                     for board in BOARDS for mid in home[board])
                home_steps = max(25, math.ceil(max_home_delta/600 * 50))
                buses.write(initial)
                armed = True
                buses.torque(True)
                print(f"HOME_MOVING: {home_steps/50:.2f}s target ramp; "
                      f"max start difference {max_home_delta} ticks", flush=True)
                for step in range(1, home_steps+1):
                    begun = time.monotonic()
                    goal = {board: {mid: round(initial[board][mid] +
                                               (home[board][mid]-initial[board][mid])*step/home_steps)
                                    for mid in home[board]} for board in BOARDS}
                    buses.write(goal)
                    time.sleep(max(0., .02-(time.monotonic()-begun)))
                next_home_report = 0.
                while True:
                    settled = buses.read()
                    pending = [(name, j["board"], j["id"],
                                settled[j["board"]][j["id"]]-j["home_tick"])
                               for name, j in CONTRACT["joints"].items()
                               if abs(settled[j["board"]][j["id"]]-j["home_tick"]) > 65]
                    if not pending:
                        break
                    now_home = time.monotonic()
                    if now_home >= next_home_report:
                        top = sorted(pending, key=lambda row: abs(row[3]), reverse=True)[:5]
                        print("HOME_WAITING:", len(pending), "outside 65 ticks;",
                              [(name, board, mid, delta) for name, board, mid, delta in top],
                              "Ctrl+C releases torque", flush=True)
                        f.write(json.dumps({"phase": "home_waiting", "pending": pending,
                                            "measured_ticks": settled, "motor_output": True})+"\n")
                        f.flush()
                        next_home_report = now_home+1.
                    check_imu(imu.sample(), args.allow_uncalibrated_accel)
                    time.sleep(.05)
                at_home_imu = imu.sample()
                check_imu(at_home_imu, args.allow_uncalibrated_accel)
                print("HOME_READY: all 22 near saved Home", flush=True)
                if args.mode == "home-preview":
                    print(f"HOME_PREVIEW: holding candidate R25 pose for {args.seconds}s; "
                          "no policy inference or policy motor commands", flush=True)
                    f.write(json.dumps({"phase": "home_preview", "motor_output": True,
                                        "measured_ticks": settled, "imu": at_home_imu}) + "\n")
                    f.flush()
                    time.sleep(args.seconds)
                    print("HOME_PREVIEW_COMPLETE: torque will turn OFF", flush=True)
                    return
            start = time.monotonic()
            while time.monotonic() - start < args.seconds:
                cycle = time.monotonic()
                ticks = buses.read()
                now = time.monotonic()
                q = joint_radians(ticks)
                sample = imu.sample()
                check_imu(sample, args.allow_uncalibrated_accel)
                if yaw_origin is None:
                    yaw_origin = sample["yaw"]
                sample["yaw_relative"] = sample["yaw"]-yaw_origin
                if prev_q is None:
                    prev_q, prev_time = q, now
                    continue
                dt = now-prev_time
                if dt <= 0:
                    prev_q, prev_time = q, now
                    continue
                dq = {n: (q[n]-prev_q[n])/dt for n in q}
                t = now-start
                obs = history.push(q, dq, sample, prev_action, sample["yaw_relative"],
                                   body_vx_input)
                raw = actor(obs)
                clipped, desired = targets(raw, filtered)
                # R25 uses clip, 0.30 EMA, actuator speed/step slew, and CAD soft limits.
                tracking_lag = {
                    name: abs(old-q[name])
                    for name, old in zip(CONTRACT["action_joint_names"], filtered)
                }
                candidate_error = None
                try:
                    goals = goal_ticks(desired)
                except ValueError as exc:
                    if output:
                        raise
                    goals = None
                    candidate_error = str(exc)
                if output:
                    if t < .05:
                        print("POLICY_RUNNING: R25 actor commands 22 joints", flush=True)
                    buses.write(goals)
                f.write(json.dumps({"t": t, "motor_output": output, "raw_action": raw,
                                    "measured_ticks": ticks, "virtual_target_rad": desired,
                                    "imu": sample,
                                    "body_linear_velocity_input_m_s": [body_vx_input, 0., 0.],
                                    "candidate_goal_ticks": goals,
                                    "candidate_error": candidate_error,
                                    "tracking_lag_rad": tracking_lag,
                                    "body_command": history.command.tolist(),
                                    "gait_phase": float(history.gait_phase),
                                    "motion_gate": float(history.gate),
                                    "cycle_s": time.monotonic()-cycle})+"\n")
                f.flush()
                prev_q, prev_time, filtered, prev_action = q, now, desired, clipped
                history.advance_command(args.command)
                elapsed = time.monotonic()-cycle
                time.sleep(max(0., .02-elapsed))
            print(f"{args.mode.upper()}_COMPLETE: {args.seconds}s; motor_output={output}", flush=True)
    finally:
        if buses is not None:
            if armed:
                try:
                    buses.torque(False)
                    still_on = [(board, mid) for board in BOARDS for mid in buses.ids[board]
                                if buses.buses[board].read(mid, 40, 1) != 0]
                    if still_on:
                        print("TORQUE OFF READBACK FAILED:", still_on, flush=True)
                    else:
                        print("ALL 22 torque OFF verified", flush=True)
                except Exception as exc:
                    print("TORQUE OFF ERROR:", exc, flush=True)
            buses.close()
        if imu is not None:
            imu.close()
        print("Log:", log, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("status", "shadow", "home-preview", "run"))
    p.add_argument("--imu-config", default="config.confirmed.json")
    p.add_argument("--ports-json", default=str(Path.home()/"board_ports.json"))
    p.add_argument("--command", choices=("forward", "reverse", "turn-left", "turn-right", "stop"), default="forward")
    p.add_argument("--experimental-fixed-body-vx", type=float, default=None, metavar="M_S",
                   help="Replace only body-frame linear-velocity X observation with a fixed 0..0.12 m/s value; actual BNO055 orientation/gyro stay active")
    p.add_argument("--seconds", type=float, default=2.)
    p.add_argument("--delay", type=int, default=10)
    p.add_argument("--enable-motors", action="store_true")
    p.add_argument("--supported-test", action="store_true")
    p.add_argument("--accept-calculated-home", action="store_true",
                   help="Explicitly use the unverified V73-to-R25 calculated Home ticks for motor output")
    p.add_argument("--allow-uncalibrated-accel", action="store_true",
                   help="Explicit suspended bench exception; gyro must be 3")
    args = p.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 0 <= args.delay <= 60:
        p.error("seconds must be a positive finite number, delay 0..60")
    args.command = {"forward": (.06, 0., 0.), "reverse": (-.04, 0., 0.),
                    "turn-left": (0., 0., .15), "turn-right": (0., 0., -.15),
                    "stop": (0., 0., 0.)}[args.command]
    if args.mode != "status":
        port_info = json.loads(Path(args.ports_json).read_text())
        serials = {"lower_body": "5B3D042991", "left_arm": "5B3D040665", "right_arm": "5AE6082486"}
        for board, serial in serials.items():
            item = port_info[board]
            if item["serial"] != serial or serial not in item["stable_port"]:
                raise ValueError(f"{board}: board serial mismatch")
        args.ports = {board: port_info[board]["stable_port"] for board in BOARDS}
    else:
        args.ports = {}
    run(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")

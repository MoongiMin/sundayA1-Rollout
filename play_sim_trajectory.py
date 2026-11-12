"""Replay archived R25 simulation joint targets on the three Sunday motor boards.

This is an open-loop, supported-robot experiment. No PPO inference or IMU
feedback is used during playback. The source trace and motor conversion are
validated before opening a port. The default 'inspect' mode is read-only.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
TRAJECTORY = HERE / "r25_sim_forward_10s.trajectory.json"
CONTRACT = HERE / "r25_contract.json"
SERIALS = {"lower_body": "5B3D042991", "left_arm": "5B3D040665", "right_arm": "5AE6082486"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    trajectory = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
    if trajectory["schema"] != "sunday.r25.sim_goal_trajectory.v1":
        raise ValueError("Unknown trajectory schema")
    if trajectory["contract_sha256"] != sha(CONTRACT):
        raise ValueError("Trajectory does not match installed R25 mapping/Home")
    if trajectory["checkpoint_sha256"] != contract["source_checkpoint_sha256"]:
        raise ValueError("R25 checkpoint mismatch")
    names = contract["action_joint_names"]
    if len(names) != 22 or names != trajectory["joint_names"] or set(names) != set(contract["joints"]):
        raise ValueError("Trajectory joint order or count differs from mapping")
    frames = trajectory["frames"]
    if not frames or len(frames) != round(trajectory["duration_s"] * 50) or trajectory["sample_period_s"] != .02:
        raise ValueError("Expected an uninterrupted 50 Hz trajectory")
    board_addresses = set()
    for name in names:
        j = contract["joints"][name]
        address = (j["board"], j["id"])
        if address in board_addresses or j["board"] not in SERIALS:
            raise ValueError("Duplicate or unknown motor address")
        board_addresses.add(address)
    for index, frame in enumerate(frames):
        if frame["step"] != index or abs(frame["t_s"] - index * .02) > 1e-5:
            raise ValueError(f"Invalid frame time at {index}")
        if set(frame["goal_rad"]) != set(names) or set(frame["goal_tick"]) != set(names):
            raise ValueError(f"Incomplete joint target at frame {index}")
        for name in names:
            angle = frame["goal_rad"][name]
            tick = frame["goal_tick"][name]
            j = contract["joints"][name]
            expected = round(j["home_tick"] + (angle - j["reference_angle_rad"]) /
                             (j["sign"] * 2 * math.pi / 4096))
            if not math.isfinite(angle) or not isinstance(tick, int) or not 0 <= tick <= 4095 or tick != expected:
                raise ValueError(f"Invalid target for {name} at frame {index}")
    return contract, trajectory


def board_goals(contract, frame):
    goals = {board: {} for board in SERIALS}
    for name, tick in frame["goal_tick"].items():
        j = contract["joints"][name]
        goals[j["board"]][j["id"]] = tick
    return goals


def ports_from(path):
    metadata = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for board, serial in SERIALS.items():
        item = metadata[board]
        port = item["stable_port"]
        if item["serial"] != serial or serial not in port or not Path(port).exists():
            raise ValueError(f"{board}: missing or mismatched stable serial port")
        result[board] = port
    return result


def motor_playback(contract, frames, args):
    from r25_rollout import Buses, BOARDS

    ports = ports_from(args.ports_json)
    buses = None
    armed = False
    log = HERE / f"r25_sim_playback_{time.time_ns()}.jsonl"
    try:
        buses = Buses(ports)
        buses.verify()
        for board in BOARDS:
            for mid in buses.ids[board]:
                if buses.buses[board].read(mid, 40, 1) != 0:
                    raise ValueError(f"{board} ID {mid}: torque already ON")
        with log.open("x", encoding="utf-8") as out:
            out.write(json.dumps({"mode": "simulation_target_playback", "trajectory_sha256": sha(TRAJECTORY),
                                  "contract_sha256": sha(CONTRACT), "seconds_requested": args.seconds,
                                  "motor_output": True, "feedback_policy": False}) + "\n")
            for delay in range(args.delay, 0, -1):
                print(f"Starts in {delay}s; Ctrl+C cancels", flush=True)
                time.sleep(1)
            initial = buses.read()
            home = {board: {} for board in BOARDS}
            for j in contract["joints"].values():
                home[j["board"]][j["id"]] = j["home_tick"]
            maximum = max(abs(home[b][mid] - initial[b][mid]) for b in BOARDS for mid in home[b])
            home_steps = max(25, math.ceil(maximum / 600 * 50))
            buses.write(initial)
            armed = True
            buses.torque(True)
            print(f"HOME_MOVING: {home_steps / 50:.2f}s ramp", flush=True)
            for step in range(1, home_steps + 1):
                cycle = time.monotonic()
                goals = {board: {mid: round(initial[board][mid] +
                                            (home[board][mid] - initial[board][mid]) * step / home_steps)
                                 for mid in home[board]} for board in BOARDS}
                buses.write(goals)
                time.sleep(max(0., .02 - (time.monotonic() - cycle)))
            next_report = 0.
            while True:
                measured = buses.read()
                pending = [(n, j["board"], j["id"],
                            measured[j["board"]][j["id"]] - j["home_tick"])
                           for n, j in contract["joints"].items()
                           if abs(measured[j["board"]][j["id"]] - j["home_tick"]) > 65]
                if not pending:
                    break
                if time.monotonic() >= next_report:
                    print("HOME_WAITING:", len(pending), "outside 65 ticks; Ctrl+C releases torque", flush=True)
                    out.write(json.dumps({"phase": "home_waiting", "pending": pending,
                                          "measured_ticks": measured}) + "\n")
                    out.flush()
                    next_report = time.monotonic() + 1.
                time.sleep(.05)
            print("HOME_READY: playback begins", flush=True)
            start = time.monotonic()
            last_index = -1
            skipped = 0
            while True:
                elapsed = time.monotonic() - start
                index = int(elapsed / .02)
                if index >= len(frames):
                    break
                if index == last_index:
                    time.sleep(min(.002, max(0., (index + 1) * .02 - elapsed)))
                    continue
                skipped += max(0, index - last_index - 1)
                frame = frames[index]
                goals = board_goals(contract, frame)
                buses.write(goals)
                measured = buses.read()
                out.write(json.dumps({"phase": "playback", "index": index, "t_s": elapsed,
                                      "goal_ticks": goals, "measured_ticks": measured,
                                      "motor_output": True}) + "\n")
                last_index = index
            out.flush()
            print(f"PLAYBACK_COMPLETE: {args.seconds:.2f}s; skipped_frames={skipped}", flush=True)
    finally:
        if buses is not None:
            if armed:
                try:
                    buses.torque(False)
                    still_on = [(board, mid) for board in BOARDS for mid in buses.ids[board]
                                if buses.buses[board].read(mid, 40, 1) != 0]
                    print("ALL 22 torque OFF verified" if not still_on else f"TORQUE OFF READBACK FAILED: {still_on}",
                          flush=True)
                except Exception as exc:
                    print(f"TORQUE OFF ERROR: {exc}", flush=True)
            buses.close()
        print("Log:", log, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "run"))
    parser.add_argument("--seconds", type=float, default=2., help="Prefix length; maximum 10 s")
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--ports-json", type=Path, default=Path.home() / "board_ports.json")
    parser.add_argument("--enable-motors", action="store_true")
    parser.add_argument("--supported-test", action="store_true")
    args = parser.parse_args()
    contract, trajectory = load()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= trajectory["duration_s"] or args.delay < 0:
        parser.error("seconds must be inside the trajectory, delay must be nonnegative")
    count = round(args.seconds * 50)
    if abs(args.seconds * 50 - count) > 1e-6:
        parser.error("seconds must be a multiple of 20 ms")
    frames = trajectory["frames"][:count]
    if args.mode == "inspect":
        print(json.dumps({"trajectory_sha256": sha(TRAJECTORY), "checkpoint_sha256": trajectory["checkpoint_sha256"],
                          "frames": len(frames), "seconds": args.seconds, "joint_count": 22,
                          "motor_output": False, "sim_command_m_s": [min(f["sim_command_m_s"] for f in frames),
                                                                max(f["sim_command_m_s"] for f in frames)]}, indent=2))
        return
    if not (args.enable_motors and args.supported_test):
        parser.error("run requires --enable-motors --supported-test")
    motor_playback(contract, frames, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", flush=True)

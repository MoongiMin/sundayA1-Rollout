"""Read all R25 motor encoders and report the strongest recent manual motion.

No goal, torque, mode, PID, or calibration write is made by this program.
The reported anatomical joint name and CAD-angle sign come from the installed
R25 mapping; the physical motion direction must be observed by a person.
"""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from r25_rollout import Buses, BOARDS

HERE = Path(__file__).resolve().parent
CONTRACT_FILE = HERE / "r25_contract.json"
EXPECTED_CONTRACT_SHA256 = "300cf9bf8f25b0d18dcb7d35a090198f9c77a05ec7514ec2290ccc5011b3a1d1"
SERIALS = {"lower_body": "5B3D042991", "left_arm": "5B3D040665", "right_arm": "5AE6082486"}
RAD_PER_TICK = 2 * math.pi / 4096

# Human-readable *simulation predictions* for positive/negative CAD angle.
# These are not observations of the real robot; encoder ticks alone cannot
# determine physical forward/backward or inward/outward motion.
EXPECTED_MOTION_KO = {
    "r_shoulder_roll": ("오른팔이 앞쪽으로 들리는 방향", "오른팔이 뒤쪽으로 내려가는 방향"),
    "r_shoulder_pitch": ("오른팔이 몸 바깥쪽으로 벌어지는 방향", "오른팔이 몸 안쪽으로 모이는 방향"),
    "r_elbow_yaw": ("오른쪽 팔꿈치 아래가 뒤쪽·안쪽으로 도는 방향", "오른쪽 팔꿈치 아래가 앞쪽·바깥쪽으로 도는 방향"),
    "r_elbow_pitch": ("오른쪽 팔꿈치가 굽혀지는 방향", "오른쪽 팔꿈치가 펴지는 방향"),
    "r_wrist_yaw": ("오른손목이 손끝을 뒤쪽·바깥쪽으로 보내는 방향", "오른손목이 손끝을 앞쪽·안쪽으로 보내는 방향"),
    "l_shoulder_roll": ("왼팔이 뒤쪽으로 내려가는 방향", "왼팔이 앞쪽으로 들리는 방향"),
    "l_shoulder_pitch": ("왼팔이 몸 바깥쪽으로 벌어지는 방향", "왼팔이 몸 안쪽으로 모이는 방향"),
    "l_elbow_yaw": ("왼쪽 팔꿈치 아래가 뒤쪽·바깥쪽으로 도는 방향", "왼쪽 팔꿈치 아래가 앞쪽·안쪽으로 도는 방향"),
    "l_elbow_pitch": ("왼쪽 팔꿈치가 굽혀지는 방향", "왼쪽 팔꿈치가 펴지는 방향"),
    "l_wrist_yaw": ("왼손목이 손끝을 뒤쪽·바깥쪽으로 보내는 방향", "왼손목이 손끝을 앞쪽·안쪽으로 보내는 방향"),
    "r_hip_roll": ("오른다리가 앞쪽으로 들리는 방향", "오른다리가 뒤쪽으로 가는 방향"),
    "r_hip_pitch": ("오른다리가 몸 안쪽으로 모이는 방향", "오른다리가 몸 바깥쪽으로 벌어지는 방향"),
    "r_thigh_yaw": ("오른쪽 허벅지가 발끝을 로봇 왼쪽으로 돌리는 방향", "오른쪽 허벅지가 발끝을 로봇 오른쪽으로 돌리는 방향"),
    "r_knee_roll": ("오른쪽 무릎이 뒤로 굽혀지는 방향", "오른쪽 무릎이 펴지는 방향"),
    "r_ankle_roll": ("오른쪽 발끝이 위로 들리는 방향", "오른쪽 발끝이 아래로 내려가는 방향"),
    "r_ankle_pitch": ("오른발 바깥쪽 모서리가 아래로 기우는 방향", "오른발 바깥쪽 모서리가 위로 기우는 방향"),
    "l_hip_roll": ("왼다리가 뒤쪽으로 가는 방향", "왼다리가 앞쪽으로 들리는 방향"),
    "l_hip_pitch": ("왼다리가 몸 바깥쪽으로 벌어지는 방향", "왼다리가 몸 안쪽으로 모이는 방향"),
    "l_thigh_yaw": ("왼쪽 허벅지가 발끝을 로봇 왼쪽으로 돌리는 방향", "왼쪽 허벅지가 발끝을 로봇 오른쪽으로 돌리는 방향"),
    "l_knee_roll": ("왼쪽 무릎이 펴지는 방향", "왼쪽 무릎이 굽혀지는 방향"),
    "l_ankle_roll": ("왼쪽 발끝이 위로 들리는 방향", "왼쪽 발끝이 아래로 내려가는 방향"),
    "l_ankle_pitch": ("왼발 바깥쪽 모서리가 위로 기우는 방향", "왼발 바깥쪽 모서리가 아래로 기우는 방향"),
}


def load_mapping():
    actual_hash = hashlib.sha256(CONTRACT_FILE.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_CONTRACT_SHA256:
        raise ValueError("Install the corrected R25 hip-axis contract from this ZIP")
    cfg = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    if len(cfg["joints"]) != 22:
        raise ValueError("Expected 22 R25 joints")
    if set(cfg["joints"]) != set(EXPECTED_MOTION_KO):
        raise ValueError("Missing a Korean motion description for a joint")
    lookup = {(j["board"], j["id"]): (name, j) for name, j in cfg["joints"].items()}
    if len(lookup) != 22 or set(lookup) != {(board, mid) for board in BOARDS
                                            for mid in ([7,8,9,10,11,12,19,20,21,22,23,24]
                                                        if board == "lower_body" else [1,2,3,4,5])}:
        raise ValueError("Unexpected or duplicate board/ID mapping")
    return lookup


def ports(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for board, serial in SERIALS.items():
        record = data[board]
        port = record["stable_port"]
        if record["serial"] != serial or serial not in port or not Path(port).exists():
            raise ValueError(f"{board}: stable serial-numbered board is unavailable")
        result[board] = port
    return result


def flatten(sample):
    return {(board, mid): tick for board, row in sample.items() for mid, tick in row.items()}


def describe(key, current, baseline, earlier, lookup, elapsed_s):
    name, joint = lookup[key]
    tick_change = current[key] - earlier[key]
    since_start = current[key] - baseline[key]
    cad_change = tick_change * joint["sign"] * RAD_PER_TICK
    direction = "TICK_INCREASE" if tick_change > 0 else "TICK_DECREASE" if tick_change < 0 else "STILL"
    cad_direction = "CAD_ANGLE_INCREASE" if cad_change > 0 else "CAD_ANGLE_DECREASE" if cad_change < 0 else "STILL"
    predicted_motion = (EXPECTED_MOTION_KO[name][0] if cad_change > 0 else
                        EXPECTED_MOTION_KO[name][1] if cad_change < 0 else "움직임이 거의 없음")
    return {"board": key[0], "motor_id": key[1], "mapped_joint": name,
            "current_tick": current[key], "recent_delta_tick": tick_change,
            "from_start_delta_tick": since_start, "tick_direction": direction,
            "inferred_cad_direction": cad_direction,
            "predicted_motion_ko": predicted_motion,
            "recent_cad_delta_rad": round(cad_change, 4),
            "recent_tick_rate_per_s": round(tick_change / elapsed_s, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument("--rate-hz", type=float, default=5.)
    parser.add_argument("--window-s", type=float, default=1.)
    parser.add_argument("--threshold-ticks", type=int, default=5)
    parser.add_argument("--ports-json", type=Path, default=Path.home()/"board_ports.json")
    args = parser.parse_args()
    if (not math.isfinite(args.seconds) or not 1 <= args.seconds <= 3600 or
            not 0 <= args.delay <= 60 or not 1 <= args.rate_hz <= 20 or
            not .2 <= args.window_s <= 3 or not 1 <= args.threshold_ticks <= 100):
        parser.error("Invalid duration, delay, rate, window, or threshold")
    lookup = load_mapping()
    port_names = ports(args.ports_json)
    buses = Buses(port_names)
    log_path = HERE / f"r25_passive_motion_{time.time_ns()}.jsonl"
    try:
        buses.verify()
        enabled = [(board, mid) for board in BOARDS for mid in buses.ids[board]
                   if buses.buses[board].read(mid, 40, 1) != 0]
        if enabled:
            print("TORQUE ENABLED (read-only warning):", enabled, flush=True)
        print("READ ONLY: 22 encoders, no motor commands. Ctrl+C stops.", flush=True)
        print("Move joints only after MONITOR_READY; keep the other joints as still as possible.", flush=True)
        print("Anatomical movement (forward/outward/bending) cannot be sensed from ticks alone.", flush=True)
        print("Korean motion descriptions are SIMULATION PREDICTIONS; compare with the real movement.", flush=True)
        for second in range(args.delay, 0, -1):
            print(f"Baseline in {second}s", flush=True)
            time.sleep(1)
        samples = []
        for _ in range(5):
            samples.append(flatten(buses.read()))
            time.sleep(.05)
        baseline = {key: round(statistics.median(row[key] for row in samples)) for key in lookup}
        history = deque()
        started = time.monotonic()
        history.append((started, baseline))
        print("MONITOR_READY: manually move a joint now", flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            log.write(json.dumps({"type": "header", "read_only": True, "contract_sha256": EXPECTED_CONTRACT_SHA256,
                                  "baseline_ticks": {f"{b}:{i}": v for (b, i), v in baseline.items()},
                                  "window_s": args.window_s, "threshold_ticks": args.threshold_ticks}) + "\n")
            while time.monotonic()-started < args.seconds:
                cycle = time.monotonic()
                current = flatten(buses.read())
                now = time.monotonic()
                history.append((now, current))
                while len(history) > 2 and history[1][0] < now-args.window_s:
                    history.popleft()
                earlier_time, earlier = history[0]
                elapsed = max(.001, now-earlier_time)
                ranking = sorted(lookup, key=lambda key: abs(current[key]-earlier[key]), reverse=True)
                strongest = describe(ranking[0], current, baseline, earlier, lookup, elapsed)
                second = describe(ranking[1], current, baseline, earlier, lookup, elapsed)
                if abs(strongest["recent_delta_tick"]) >= args.threshold_ticks:
                    print(f"TOP {strongest['board']} ID {strongest['motor_id']:2d} {strongest['mapped_joint']:18s} "
                          f"{strongest['tick_direction']:13s} {strongest['recent_delta_tick']:+5d} ticks/{elapsed:.1f}s "
                          f"({strongest['recent_tick_rate_per_s']:+.0f} tick/s), "
                          f"START {strongest['from_start_delta_tick']:+5d}, "
                          f"{strongest['inferred_cad_direction']}", flush=True)
                    print(f"  >>> 시뮬레이션 예상: {strongest['predicted_motion_ko']}", flush=True)
                    if abs(second["recent_delta_tick"]) >= max(args.threshold_ticks, .6*abs(strongest["recent_delta_tick"])):
                        print(f"  ALSO {second['board']} ID {second['motor_id']:2d} "
                              f"{second['mapped_joint']} {second['recent_delta_tick']:+d} ticks", flush=True)
                else:
                    print("STILL: no joint exceeds recent tick threshold", flush=True)
                log.write(json.dumps({"type": "sample", "elapsed_s": round(now-started, 3),
                                      "ticks": {f"{b}:{i}": v for (b, i), v in current.items()},
                                      "top": strongest, "runner_up": second}) + "\n")
                log.flush()
                time.sleep(max(0., 1/args.rate_hz-(time.monotonic()-cycle)))
    finally:
        buses.close()
        print("READ_ONLY_MONITOR_STOPPED; no torque or target commands sent", flush=True)
        print("Log:", log_path, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", flush=True)

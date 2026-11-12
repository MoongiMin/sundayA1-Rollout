"""Record both arm motor ranges without writing motor settings or goals."""
import argparse
import json
import math
from pathlib import Path
import select
import signal
import sys
import time

from rollout import LegBus

IDS = tuple(range(1, 7))
SERIALS = {"left_arm": "5B3D040665", "right_arm": "5AE6082486"}
_JOINTS = json.loads(Path(__file__).with_name('r25_contract.json').read_text(encoding='utf-8'))['joints']
JOINT_LABELS = {arm: {**{j['id']:name for name,j in _JOINTS.items() if j['board']==arm}, 6:side+'_hand'} for arm,side in (('left_arm','l'),('right_arm','r'))}


def read_checked(bus, arm, mid, register, size=2, attempts=3):
    for attempt in range(attempts):
        try:
            return bus.read(mid, register, size)
        except OSError as exc:
            if attempt == attempts - 1:
                raise OSError(f"{arm} ID {mid} register {register}: {exc}") from exc
            time.sleep(.1)


def positions_checked(bus, arm):
    return {mid: read_checked(bus, arm, mid, 56) for mid in IDS}


def summarize(rows, settings, ports, error):
    result = {"completed": error is None, "error": error,
              "sample_count": len(rows), "motor_commands_sent": False,
              "ports": ports, "motor_settings": settings, "arms": {}}
    for arm in SERIALS:
        joints = {}
        for mid in IDS:
            values = [r["ticks"][arm][str(mid)] for r in rows]
            if values:
                joints[str(mid)] = {"min_tick": min(values), "max_tick": max(values),
                                    "span_ticks": max(values) - min(values),
                                    "first_tick": values[0], "last_tick": values[-1],
                                    "reported_joint": JOINT_LABELS[arm][mid],
                                    "joint_mapping_physically_verified": False}
        result["arms"][arm] = joints
    result["note"] = "Observed hand-moved range only. No joint-name mapping, Home, safe limit, or CAD direction inferred."
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ports-json", type=Path, default=Path.home() / "board_ports.json")
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--delay", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("manual_calibration"))
    a = p.parse_args()
    if not math.isfinite(a.seconds) or not 5 <= a.seconds <= 600 or not 0 <= a.delay <= 60:
        p.error("seconds must be 5..600 and delay 0..60")
    metadata = json.loads(a.ports_json.read_text(encoding="utf-8"))
    ports = {}
    for arm, serial in SERIALS.items():
        board = metadata[arm]
        if board["serial"] != serial:
            raise ValueError(f"{arm}: unexpected USB serial")
        port = Path(board["stable_port"])
        if serial not in port.name or not port.exists():
            raise ValueError(f"{arm}: stable port missing or mismatched: {port}")
        ports[arm] = str(port)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    stem = a.output_dir / f"both_arms_{time.time_ns()}"
    buses, rows, settings, error = {}, [], {}, None

    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for arm, port in ports.items():
            bus = LegBus(port, 1000000, list(IDS))
            buses[arm] = bus
            settings[arm] = {}
            for mid in IDS:
                model = read_checked(bus, arm, mid, 3)
                mode = read_checked(bus, arm, mid, 33, 1)
                torque = read_checked(bus, arm, mid, 40, 1)
                raw = read_checked(bus, arm, mid, 31)
                offset = -(raw & 2047) if raw & 2048 else raw
                settings[arm][str(mid)] = {"model": model, "mode": mode,
                                            "torque_enabled": torque, "homing_offset": offset}
                if model != 777 or mode != 0 or torque != 0:
                    raise ValueError(f"{arm} ID {mid}: model/mode/torque unexpected; no recording")
        print("READ ONLY: both arms, IDs 1..6 on each separate board.", flush=True)
        for remaining in range(a.delay, 0, -1):
            print(f"Recording starts in {remaining}s", flush=True)
            time.sleep(1)
        print("RECORDING: move arm joints by hand; Enter finishes.", flush=True)
        started, last_print = time.monotonic(), -1
        with stem.with_suffix(".raw.jsonl").open("x", encoding="utf-8") as log:
            while time.monotonic() - started < a.seconds:
                if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                    break
                begun = time.monotonic()
                ticks = {}
                for arm, bus in buses.items():
                    actual = positions_checked(bus, arm)
                    if set(actual) != set(IDS) or any(type(v) is not int or not 0 <= v <= 4095 for v in actual.values()):
                        raise ValueError(f"{arm}: invalid encoder sample")
                    for mid in IDS:
                        if read_checked(bus, arm, mid, 40, 1) != 0:
                            raise RuntimeError(f"{arm} ID {mid}: torque became enabled")
                    ticks[arm] = {str(k): v for k, v in actual.items()}
                if rows and any(abs(ticks[arm][str(mid)] - rows[-1]["ticks"][arm][str(mid)]) > 2048
                                for arm in SERIALS for mid in IDS):
                    raise ValueError("Possible encoder wrap; review raw data")
                row = {"elapsed_s": time.monotonic() - started,
                       "monotonic_ns": time.monotonic_ns(), "ticks": ticks,
                       "motor_commands_sent": False}
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                second = int(row["elapsed_s"])
                if second != last_print:
                    print(" ARM       ID  REPORTED JOINT        MIN   NOW   MAX  SPAN", flush=True)
                    for arm in SERIALS:
                        for mid in IDS:
                            values = [r["ticks"][arm][str(mid)] for r in rows]
                            print(f" {arm:10s} {mid:2d} {JOINT_LABELS[arm][mid]:19s} {min(values):5d} {values[-1]:5d} "
                                  f"{max(values):5d} {max(values)-min(values):5d}", flush=True)
                    last_print = second
                if time.monotonic() - begun > 1.5:
                    raise TimeoutError("Arm read cycle exceeded 1.5s")
                time.sleep(max(0, .1 - (time.monotonic() - begun)))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for bus in buses.values():
            try:
                bus.close()
            except Exception as exc:
                error = (error or "") + f"; close failed: {exc}"
        result = summarize(rows, settings, ports, error)
        path = stem.with_suffix(".summary.json")
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("Summary file:", path, flush=True)
        print("No calibration or motor settings were changed.", flush=True)
        if error:
            print("Capture ended:", error, flush=True)
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Save a user-posed whole-body Home from three boards; read only."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

from rollout import LegBus

IDS = {
    "lower_body": (7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24),
    "left_arm": (1, 2, 3, 4, 5, 6),
    "right_arm": (1, 2, 3, 4, 5, 6),
}
SERIALS = {"lower_body": "5B3D042991", "left_arm": "5B3D040665",
           "right_arm": "5AE6082486"}


def read_position(bus, label, mid):
    for attempt in range(3):
        try:
            value = bus.read(mid, 56)
            if type(value) is not int or not 0 <= value <= 4095:
                raise ValueError(f"{label} ID {mid}: invalid tick {value}")
            return value
        except OSError as exc:
            if attempt == 2:
                raise OSError(f"{label} ID {mid}: {exc}") from exc
            time.sleep(.1)


def make_home(samples, spread_limit):
    result = {}
    for label, ids in IDS.items():
        result[label] = {}
        for mid in ids:
            values = [sample[label][str(mid)] for sample in samples]
            spread = max(values) - min(values)
            if spread > spread_limit:
                raise ValueError(f"{label} ID {mid}: pose moved {spread} ticks during capture")
            result[label][str(mid)] = {
                "home_tick": round(statistics.median(values)),
                "sample_min_tick": min(values), "sample_max_tick": max(values),
                "sample_spread_tick": spread}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ports-json", type=Path, default=Path.home() / "board_ports.json")
    p.add_argument("--delay", type=int, default=10)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--spread-limit", type=int, default=8)
    p.add_argument("--output-dir", type=Path, default=Path("manual_calibration"))
    a = p.parse_args()
    if not 0 <= a.delay <= 60 or not 3 <= a.samples <= 50 or not 0 <= a.spread_limit <= 50:
        p.error("delay 0..60, samples 3..50, spread-limit 0..50")
    board_metadata = json.loads(a.ports_json.read_text(encoding="utf-8"))
    ports = {}
    for label, serial in SERIALS.items():
        record = board_metadata[label]
        port = Path(record["stable_port"])
        if record["serial"] != serial or serial not in port.name or not port.exists():
            raise ValueError(f"{label}: USB serial/port mismatch or disconnected")
        ports[label] = str(port)
    buses = {}
    try:
        for label, ids in IDS.items():
            bus = LegBus(ports[label], 1000000, list(ids))
            buses[label] = bus
            for mid in ids:
                model = bus.read(mid, 3)
                mode = bus.read(mid, 33, 1)
                torque = bus.read(mid, 40, 1)
                if model != 777 or mode != 0 or torque != 0:
                    raise ValueError(f"{label} ID {mid}: model={model}, mode={mode}, torque={torque}; expected 777/0/0")
                read_position(bus, label, mid)
        print("READ ONLY: all 24 motors responded with torque OFF.", flush=True)
        print("Hold the robot in your chosen Home pose. Do not move during sampling.", flush=True)
        for remaining in range(a.delay, 0, -1):
            print(f"Capture starts in {remaining}s", flush=True)
            time.sleep(1)
        samples = []
        for _ in range(a.samples):
            sample = {label: {str(mid): read_position(buses[label], label, mid)
                              for mid in IDS[label]} for label in IDS}
            samples.append(sample)
            time.sleep(.1)
        home = make_home(samples, a.spread_limit)
    finally:
        for bus in buses.values():
            bus.close()
    result = {"source": "User-posed physical Home captured from live motor positions",
              "complete": True, "motor_commands_sent": False,
              "captured_at_unix_ns": time.time_ns(), "board_ports": ports,
              "samples_per_motor": a.samples, "max_allowed_spread_tick": a.spread_limit,
              "home": home,
              "note": "Motor Home ticks only; no CAD angle, sign, or motor homing offset was changed."}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    path = a.output_dir / f"whole_body_home_{result['captured_at_unix_ns']}.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for label in IDS:
        print(label + ": " + " ".join(f"{mid}={home[label][str(mid)]['home_tick']}"
                                  for mid in IDS[label]), flush=True)
    print("Saved:", path, flush=True)
    print("No motor or rollout configuration was changed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Simulate recorded R25 motor goals in the Week4 MuJoCo model for Viser replay.

This answers what the *model* would do with the goals sent during a real run.
It cannot reconstruct the real robot's root translation or prove real dynamics.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
WEEK = HERE / "assets" / "Week4"
MODEL = WEEK / "model" / "sunday_a1_locomotion.xml"
HOME = WEEK / "model" / "arm_clearance_home.json"
DEFAULT_CONTRACT = HERE / "r25_contract.json"
FRAME_DT = 0.02
RAD_PER_TICK = 2 * math.pi / 4096


def load_model_with_ground():
    """R25's robot XML omits the plane added by the mjlab scene."""
    root = ET.parse(MODEL).getroot()
    root.find("compiler").set("meshdir", str(MODEL.parent / "meshes"))
    root.find("worldbody").insert(0, ET.Element("geom", {
        "name": "r25_replay_ground", "type": "plane", "size": "10 10 0.1",
        "pos": "0 0 0", "friction": "1 0.005 0.0001",
        "rgba": "0.75 0.75 0.75 1",
    }))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations = 20
    model.opt.ls_iterations = 10
    return model


def policy_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if rows and rows[0].get("mode") == "simulation_target_playback":
        sent = [{"t": row["t_s"], "candidate_goal_ticks": row["goal_ticks"]}
                for row in rows[1:] if row.get("phase") == "playback"
                and row.get("motor_output") is True
                and isinstance(row.get("goal_ticks"), dict)]
    else:
        sent = [row for row in rows if row.get("motor_output") is True
                and isinstance(row.get("candidate_goal_ticks"), dict) and "t" in row]
    if len(sent) < 2:
        raise ValueError("Log contains fewer than two completed policy commands")
    times = np.asarray([row["t"] for row in sent], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid or unordered policy timestamps")
    return sent, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT,
                        help="Exact mapping/Home contract used to write the motor goals")
    parser.add_argument("--seconds", type=float, help="Replay this many recorded seconds; default: full log")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows, times = policy_rows(args.log)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    first_row = json.loads(args.log.open(encoding="utf-8").readline())
    if first_row.get("mode") == "simulation_target_playback":
        if first_row.get("contract_sha256") != hashlib.sha256(args.contract.read_bytes()).hexdigest():
            raise ValueError("Playback log contract does not match selected mapping")
    else:
        if first_row.get("checkpoint_sha256") != contract["source_checkpoint_sha256"]:
            raise ValueError("Log checkpoint does not match selected mapping contract")
        if (first_row.get("contract_sha256") is not None and
                first_row["contract_sha256"] != hashlib.sha256(args.contract.read_bytes()).hexdigest()):
            raise ValueError("Log contract does not match selected mapping and Home")
    if len(contract["action_joint_names"]) != 22 or len(contract["joints"]) != 22:
        raise ValueError("Selected mapping does not contain 22 policy joints")
    home = json.loads(HOME.read_text(encoding="utf-8"))
    model = load_model_with_ground()
    data = mujoco.MjData(model)
    substeps = round(FRAME_DT / model.opt.timestep)
    if not math.isclose(substeps * model.opt.timestep, FRAME_DT, abs_tol=1e-9):
        raise ValueError("MuJoCo timestep does not divide 20 ms")
    duration = float(times[-1] + FRAME_DT)
    if args.seconds is not None:
        if not math.isfinite(args.seconds) or args.seconds <= 0:
            parser.error("--seconds must be positive and finite")
        if duration + FRAME_DT < args.seconds:
            raise ValueError(f"Only {duration:.2f}s of policy commands in log")
        duration = min(duration, args.seconds)
    count = math.ceil(duration / FRAME_DT)

    # The simulation starts from the R25 training Home. Real root pose is unknown.
    data.qpos[:3] = (0, 0, home["root_height"])
    data.qpos[3:7] = (1, 0, 0, 0)
    for name, value in home["joint_pos"].items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
    actuator_ids = {}
    for name in contract["action_joint_names"]:
        actuator_ids[name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    for name, value in home["joint_pos"].items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        data.ctrl[actuator_id] = value
    goals = []
    for row in rows:
        angles = {}
        for name, joint in contract["joints"].items():
            tick = int(row["candidate_goal_ticks"][joint["board"]][str(joint["id"])])
            angles[name] = (joint["reference_angle_rad"] +
                            joint["sign"] * (tick - joint["home_tick"]) * RAD_PER_TICK)
        goals.append(angles)
    mujoco.mj_forward(model, data)
    frames = []
    for index in range(count):
        t = (index + 1) * FRAME_DT
        source_index = max(0, int(np.searchsorted(times, t, side="right") - 1))
        for name, angle in goals[source_index].items():
            data.ctrl[actuator_ids[name]] = angle
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise ValueError(f"Simulation diverged at {t:.2f}s")
        frames.append({"t": t, "qpos": [data.qpos.tolist()], "done": [False]})
    output = args.output or args.log.with_suffix(".r25_sim_goals.poses.json")
    output.write_text(json.dumps(frames), encoding="utf-8")
    print(json.dumps({"poses": str(output), "frames": len(frames),
                      "contract": str(args.contract),
                      "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
                      "recorded_policy_seconds": round(float(times[-1]), 3),
                      "simulation_seconds": round(len(frames) * FRAME_DT, 3),
                      "interpretation": "MuJoCo response to recorded motor goals, not measured root motion"},
                     indent=2))


if __name__ == "__main__":
    main()

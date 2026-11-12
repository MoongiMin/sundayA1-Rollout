"""Play a R25 MuJoCo pose file in Viser with the Week4 Sunday CAD meshes."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser

from replay_r25_goals import load_model_with_ground

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poses", type=Path)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--label", default="R25 · MuJoCo motor-goal replay")
    args = parser.parse_args()
    frames = json.loads(args.poses.read_text(encoding="utf-8"))
    if not frames:
        raise ValueError("Empty pose file")
    model = load_model_with_ground()
    data = mujoco.MjData(model)
    if any(len(row["qpos"][0]) != model.nq for row in frames):
        raise ValueError("Pose qpos dimensions do not match R25 model")
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    server.scene.set_up_direction("+z")
    server.gui.add_markdown(f"## {args.label}\n"
                            "Simulation response to goals recorded on Jetson. "
                            "This is not measured real-world root movement.")
    server.gui.add_markdown(f"**Replay:** `{args.poses.name}`\n\n"
                            f"**Duration:** {len(frames) * .02:.2f} s")
    play = server.gui.add_checkbox("Play", initial_value=True)
    speed = server.gui.add_slider("Playback speed", min=0.1, max=2.0,
                                  step=0.1, initial_value=1.0)
    slider = server.gui.add_slider("Time (s)", min=0.0,
                                   max=len(frames) * 0.02, step=0.02, initial_value=0.0)
    server.scene.add_grid("/ground", width=4, height=4, cell_size=0.1)
    handles = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = model.geom_dataid[geom]
        first_vertex, vertex_count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        first_face, face_count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        mesh = trimesh.Trimesh(
            vertices=model.mesh_vert[first_vertex:first_vertex + vertex_count].copy(),
            faces=model.mesh_face[first_face:first_face + face_count].copy(), process=False)
        handle = server.scene.add_mesh_simple(
            f"/robot/{geom}", vertices=mesh.vertices, faces=mesh.faces,
            color=(180, 180, 175), side="double")
        handles.append((geom, handle))

    @server.on_client_connect
    def connected(client):
        client.camera.position = (0.8, -1.0, 0.65)
        client.camera.look_at = (0, 0, 0.23)

    origin = np.asarray(frames[0]["qpos"][0][:2], dtype=float)
    cursor = 0.0
    previous = time.monotonic()
    print(f"R25_REPLAY_READY http://127.0.0.1:{args.port}/", flush=True)
    while True:
        begun = time.monotonic()
        elapsed = begun - previous
        previous = begun
        if play.value:
            cursor = (cursor + elapsed * speed.value) % (len(frames) * 0.02)
        else:
            cursor = slider.value
        index = min(int(cursor / 0.02), len(frames) - 1)
        data.qpos[:] = frames[index]["qpos"][0]
        data.qpos[:2] -= origin
        mujoco.mj_kinematics(model, data)
        with server.atomic():
            for geom, handle in handles:
                quaternion = np.empty(4)
                mujoco.mju_mat2Quat(quaternion, data.geom_xmat[geom])
                handle.position = data.geom_xpos[geom]
                handle.wxyz = quaternion
            if play.value:
                slider.value = round(cursor / 0.02) * 0.02
        time.sleep(max(0, 1 / 30 - (time.monotonic() - begun)))


if __name__ == "__main__":
    main()

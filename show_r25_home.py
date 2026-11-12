"""Display the exact R25 nominal pose without stepping physics or moving hardware."""
import json
import hashlib
import math
import time
from pathlib import Path
import mujoco
import numpy as np
import viser
from replay_r25_goals import HOME, MODEL, load_model_with_ground

HERE = Path(__file__).resolve().parent

def main():
    c = json.loads((HERE/'r25_contract.json').read_text())
    h = json.loads(HOME.read_text())
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == c['source_model_sha256']
    assert hashlib.sha256(HOME.read_bytes()).hexdigest() == c['source_home_sha256']
    m = load_model_with_ground()
    d = mujoco.MjData(m)
    d.qpos[:3] = [0,0,h['root_height']]
    d.qpos[3:7] = [1,0,0,0]
    for name, q in h['joint_pos'].items():
        jid = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
        d.qpos[m.jnt_qposadr[jid]] = q
    mujoco.mj_forward(m,d)
    s = viser.ViserServer(host='127.0.0.1',port=8088)
    s.scene.set_up_direction('+z')
    s.scene.add_grid('/ground',width=2,height=2,cell_size=.1)
    s.gui.add_markdown('## R25 exact simulation Home\nStatic CAD reference; no physics or motor commands.\n\n'
                       '**Visual matching gives a candidate, not measured angle calibration.**\n'
                       'Angles below are CAD coordinates, not external protractor readings.')
    view = s.gui.add_dropdown('Camera',options=('Perspective','Front (+X)','Right (-Y)','Left (+Y)','Rear (-X)','Top'),initial_value='Perspective')
    joint = s.gui.add_dropdown('Selected joint',options=c['action_joint_names'],initial_value='r_knee_roll')
    all_labels = s.gui.add_checkbox('Show all motor IDs',initial_value=False)
    info = s.gui.add_markdown('')
    s.gui.add_markdown('**Nominal root height:** 24 cm\n\n**Nominal sole-center separation:** ~14 cm\n\n'
        '|Joint|Board / ID|Home deg|\n|---|---|---:|\n'+'\n'.join(
        f"|{n}|{j['board']} / {j['id']}|{math.degrees(h['joint_pos'][n]):+.2f}|" for n,j in c['joints'].items()))
    labels={}
    for name,j in c['joints'].items():
        jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
        labels[name]=s.scene.add_label('/ids/'+name,f"{j['board']} ID {j['id']} | {name}",position=d.xanchor[jid],visible=False)
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        mid=m.geom_dataid[g]; va=m.mesh_vertadr[mid]; vn=m.mesh_vertnum[mid]; fa=m.mesh_faceadr[mid]; fn=m.mesh_facenum[mid]
        quat=np.empty(4); mujoco.mju_mat2Quat(quat,d.geom_xmat[g])
        s.scene.add_mesh_simple(f'/robot/{g}',vertices=m.mesh_vert[va:va+vn].copy(),faces=m.mesh_face[fa:fa+fn].copy(),
            position=d.geom_xpos[g],wxyz=quat,color=(140,180,215),side='double')
    cameras={'Perspective':(.85,-.85,.65),'Front (+X)':(1.,0,.3),'Right (-Y)':(0,-1.,.3),
             'Left (+Y)':(0,1.,.3),'Rear (-X)':(-1.,0,.3),'Top':(.001,0,1.2)}
    def camera(client):
        client.camera.position=cameras[view.value];client.camera.look_at=(0,0,.25)
        client.camera.up=(1,0,0) if view.value=='Top' else (0,0,1)
    @s.on_client_connect
    def connected(client):camera(client)
    @view.on_update
    def changed(event):
        for client in s.get_clients().values():camera(client)
    def refresh(event=None):
        n=joint.value;j=c['joints'][n]
        info.content=f"**{n}** — {j['board']} ID **{j['id']}**\n\nR25 Home: **{math.degrees(h['joint_pos'][n]):+.2f} deg** ({h['joint_pos'][n]:+.6f} rad)\n\nCapture the encoder tick only after physically matching this reference pose."
        for name,label in labels.items():label.visible=all_labels.value or name==n
    joint.on_update(refresh);all_labels.on_update(refresh);refresh()
    print('R25_HOME_READY http://127.0.0.1:8088/',flush=True)
    while True:time.sleep(.5)

if __name__=='__main__':main()

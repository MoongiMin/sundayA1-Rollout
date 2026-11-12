"""Offline package checks: no serial port, I2C device or motor is opened."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import r25_rollout as r
import play_sim_trajectory as playback

class OfflineTests(unittest.TestCase):
    def test_mapping_and_preserved_home(self):
        home=json.loads((ROOT/'calibration_evidence/whole_body_home_1794439281586950429.json').read_text())
        # Source capture schema is checked in the provenance audit; contract must
        # remain byte-identical to the release used by the latest downloaded log.
        self.assertTrue(home)
        self.assertEqual(hashlib.sha256((ROOT/'r25_contract.json').read_bytes()).hexdigest(),
                         '300cf9bf8f25b0d18dcb7d35a090198f9c77a05ec7514ec2290ccc5011b3a1d1')
        self.assertEqual(len({(j['board'],j['id']) for j in r.CONTRACT['joints'].values()}),22)
        self.assertEqual(len(r.CONTRACT['action_joint_names']),22)
        for j in r.CONTRACT['joints'].values():
            self.assertEqual(j['home_tick'],home['home'][j['board']][str(j['id'])]['home_tick'])

    def test_roundtrip_signed_angles(self):
        ticks={b:{} for b in r.BOARDS}
        for j in r.CONTRACT['joints'].values():ticks[j['board']][j['id']]=j['home_tick']+37
        angles=r.joint_radians(ticks)
        actual=r.goal_ticks([angles[n] for n in r.CONTRACT['action_joint_names']])
        self.assertEqual(ticks,actual)

    def test_actor_and_history(self):
        ticks={b:{j['id']:j['home_tick'] for j in r.CONTRACT['joints'].values() if j['board']==b} for b in r.BOARDS}
        q=r.joint_radians(ticks);dq={n:0. for n in q}
        imu={'angular_velocity':[0.,0.,0.],'gravity':[0.,0.,-1.]}
        history=r.R25History()
        obs=history.push(q,dq,imu,[0.]*22,0.,.06)
        self.assertEqual(obs.shape,(460,))
        action=r.Actor()(obs)
        self.assertEqual(len(action),22)
        self.assertTrue(np.isfinite(action).all())

    def test_openloop_all_500_frames(self):
        c,t=playback.load()
        self.assertEqual(len(t['frames']),500)
        self.assertEqual(t['duration_s'],10)
        self.assertEqual(c['joints'],r.CONTRACT['joints'])

    def test_shadow_never_writes_motor_commands(self):
        class Buses:
            def __init__(self,ports):pass
            def verify(self):pass
            def read(self):return {b:{j['id']:j['home_tick'] for j in r.CONTRACT['joints'].values() if j['board']==b} for b in r.BOARDS}
            def write(self,values):raise AssertionError('Shadow wrote goals')
            def torque(self,on):raise AssertionError('Shadow wrote torque')
            def close(self):pass
        class IMU:
            def __init__(self,cfg):pass
            def sample(self):return {'angular_velocity':[0.,0.,0.],'gravity':[0.,0.,-1.],'yaw':0.,'gyro_calibration':3,'accel_calibration':3}
            def close(self):pass
        actor=r.Actor()
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            (folder/'r25_contract.json').write_bytes((ROOT/'r25_contract.json').read_bytes())
            cfg=folder/'imu.json'
            cfg.write_text(json.dumps({'imu':{'bus':7,'address':40,'axes_and_quaternion_verified':True}}))
            args=SimpleNamespace(mode='shadow',command=(.06,0.,0.),experimental_fixed_body_vx=.06,
                                 ports={b:str(folder) for b in r.BOARDS},imu_config=str(cfg),
                                 enable_motors=False,supported_test=False,allow_uncalibrated_accel=False,
                                 seconds=.10,delay=0)
            with patch.object(r,'ROOT',folder),patch.object(r,'Buses',Buses),patch.object(r,'BNO055',IMU),patch.object(r,'Actor',return_value=actor):
                r.run(args)
            rows=[json.loads(line) for line in next(folder.glob('r25_shadow_*.jsonl')).read_text().splitlines()]
            self.assertTrue(any('raw_action' in row for row in rows))
            self.assertTrue(all(not row.get('motor_output',False) for row in rows))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Optional checkpoint export audit requires PyTorch')
    def test_export_matches_original_checkpoint(self):
        import torch
        path=ROOT/'provenance/r25_model_300.pt'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),r.CONTRACT['source_checkpoint_sha256'])
        state=torch.load(path,map_location='cpu',weights_only=True)['actor_state_dict']
        actor=r.Actor()
        rng=np.random.default_rng(57)
        for _ in range(5):
            obs=rng.normal(0,.1,460).astype(np.float32)
            x=(torch.from_numpy(obs).reshape(1,-1)-state['obs_normalizer._mean'])/(state['obs_normalizer._std']+.01)
            for i in (0,2,4,6):
                x=torch.nn.functional.linear(x,state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias'])
                if i!=6:x=torch.nn.functional.elu(x)
            np.testing.assert_allclose(actor(obs),x.detach().numpy().ravel(),atol=5e-5,rtol=5e-5)

if __name__=='__main__':unittest.main()

"""Jetson v13 runner. Hardware output requires explicit configuration and CLI opt-in."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

from v13_math import observation, leg_targets

ROOT = Path(__file__).resolve().parent


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(cfg, contract, output=False):
    errors = []
    def need(ok, name):
        if not ok:
            errors.append(name)
    need(bool(cfg['ports']['lower_body']), 'ports.lower_body')
    need(cfg['imu']['bus'] is not None, 'imu.bus')
    need(cfg['imu']['address'] in (0x28, 0x29), 'imu.address (40=0x28 or 41=0x29)')
    need(cfg['imu']['axes_and_quaternion_verified'] is True, 'imu.axes_and_quaternion_verified')
    need(cfg['imu']['mode'] in (8, 12), 'imu.mode (8=IMUPLUS,12=NDOF)')
    need(type(cfg['imu']['min_calibration']) is int and cfg['imu']['min_calibration'] in (2,3), 'imu.min_calibration (2 or 3)')
    need(bool(cfg['velocity']['file']), 'velocity.file (fresh body-frame velocity estimator output)')
    need(cfg['velocity']['verified'] is True, 'velocity.verified')
    need(finite(cfg['velocity']['max_speed_m_s']) and cfg['velocity']['max_speed_m_s'] > 0, 'velocity.max_speed_m_s')
    need(finite(cfg['velocity']['max_age_s']) and 0 < cfg['velocity']['max_age_s'] <= .1, 'velocity.max_age_s')
    ids = []
    for name in contract['action_joint_names']:
        joint = cfg['legs'][name]
        ids.append(joint['id'])
        need(joint['mapping_verified'] is True, name+'.mapping_verified')
        need(type(joint['sign']) is int and joint['sign'] in (-1, 1), name+'.sign')
        for field in ('reference_tick', 'reference_angle_rad', 'rad_per_tick', 'min_rad', 'max_rad', 'max_velocity_rad_s'):
            need(finite(joint[field]), name+'.'+field)
        need(type(joint['expected_homing_offset']) is int and -2047 <= joint['expected_homing_offset'] <= 2047,
             name+'.expected_homing_offset')
        if finite(joint['rad_per_tick']):
            need(joint['rad_per_tick'] > 0, name+'.rad_per_tick > 0')
        if finite(joint['min_rad']) and finite(joint['max_rad']):
            need(joint['min_rad'] < joint['max_rad'], name+'.limits')
        if finite(joint['max_velocity_rad_s']):
            need(joint['max_velocity_rad_s'] > 0, name+'.max_velocity_rad_s > 0')
    need(len(set(ids)) == 12 and set(ids) == set(range(7,13)) | set(range(19,25)), 'leg IDs')
    for name in contract['observation_joint_names']:
        if name not in cfg['legs']:
            fixed = cfg['fixed_upper_body'][name]
            need(fixed['verified'] is True and finite(fixed['position_rad']), name+'.fixed position verified')
    for key in ('start_pose_tolerance_rad', 'max_tracking_error_rad', 'max_target_change_rad', 'max_duration_s'):
        need(finite(cfg['control'][key]) and cfg['control'][key] > 0, 'control.'+key)
    need(cfg['control']['hz'] == 50, 'control.hz must remain 50 for v13')
    need(finite(cfg['control']['min_upright_cos']) and 0 < cfg['control']['min_upright_cos'] < 1,
         'control.min_upright_cos')
    if output:
        need(cfg['control']['motor_output_allowed'] is True, 'control.motor_output_allowed')
        need(cfg['control']['stop_action'] in ('disable_torque', 'hold_last'), 'control.stop_action')
        need(cfg['control']['external_stop_ready'] is True, 'control.external_stop_ready')
        need(cfg['control']['supported_start_and_fixed_arms_verified'] is True,
             'control.supported_start_and_fixed_arms_verified')
    return errors


def tick_to_rad(tick, joint):
    # Present_Position already includes device homing_offset. Do not subtract it twice.
    if not 0 <= tick <= 4095:
        raise ValueError('Only verified single-turn 0..4095 positions are supported')
    value = joint['reference_angle_rad'] + joint['sign']*(tick-joint['reference_tick'])*joint['rad_per_tick']
    if not joint['min_rad'] <= value <= joint['max_rad']:
        raise ValueError('Measured joint position outside configured physical limits')
    return value


def targets_to_ticks(targets, positions, previous, cfg, contract):
    result = {}
    for name, target, old in zip(contract['action_joint_names'], targets, previous):
        j = cfg['legs'][name]
        if not finite(target) or not j['min_rad'] <= target <= j['max_rad']:
            raise ValueError(name+': target outside measured limits')
        limit = min(cfg['control']['max_target_change_rad'], j['max_velocity_rad_s']/50.)
        if abs(target-old) > limit:
            raise ValueError(f'{name}: target change {target-old:+.6f} rad exceeds {limit:.6f} rad/step; previous={old:.6f}, requested={target:.6f}, measured={positions[name]:.6f}; requested rate={abs(target-old)*50:.3f} rad/s')
        if abs(target-positions[name]) > cfg['control']['max_tracking_error_rad']:
            raise ValueError(name+': tracking error exceeds limit')
        raw = round(j['reference_tick'] + (target-j['reference_angle_rad'])/(j['sign']*j['rad_per_tick']))
        if not 0 <= raw <= 4095:
            raise ValueError(name+': target crosses encoder wrap; unsupported')
        result[j['id']] = raw
    return result


def limit_policy_slew(targets, previous, cfg, contract):
    limited=[]
    for name,target,old in zip(contract['action_joint_names'],targets,previous):
        j=cfg['legs'][name]
        if not finite(target) or not j['min_rad']<=target<=j['max_rad']:
            raise ValueError(name+': requested policy target outside measured bounds')
        step=min(cfg['control']['max_target_change_rad'],j['max_velocity_rad_s']/50.)
        # Small rounding margin keeps the subsequent strict check stable.
        step*=cfg['control'].get('target_speed_scale',1.)*(1-1e-9)
        limited.append(old+max(-step,min(step,target-old)))
    return limited


def follow_measured_targets(targets,previous,positions,cfg,contract,memory,now):
    """Slow advancing targets as lag grows; allow retreat toward measured position."""
    hard=cfg['control']['max_tracking_error_rad']
    slow=.25*hard
    hold=.60*hard
    limited=[];details={}
    for name,target,old in zip(contract['action_joint_names'],targets,previous):
        actual=positions[name]
        if not finite(actual) or not finite(target):raise ValueError(name+': non-finite tracking input')
        error=old-actual;gap=abs(error)
        if gap>hard:raise ValueError(f'{name}: measured lag {gap:.4f} exceeds hard bound {hard:.4f}')
        delta=target-old
        advancing=delta*error>0
        scale=max(0.,min(1.,(hold-gap)/(hold-slow))) if advancing else 1.
        candidate=old+delta*scale
        if advancing:
            # Never advance the target beyond the soft hold gap in one step.
            room=max(0.,hold-gap)
            candidate=old+math.copysign(min(abs(candidate-old),room),delta)
        state='waiting' if advancing and scale==0 else 'slowing' if scale<1 else 'following'
        # Stalled mechanics must not be held indefinitely. Reset only on useful measured progress.
        if gap>=slow:
            entry=memory.get(name)
            if entry is None or entry['direction']*error<=0:
                entry={'time':now,'position':actual,'direction':1 if error>0 else -1}
                memory[name]=entry
            if (actual-entry['position'])*entry['direction']>=.005:
                entry.update(time=now,position=actual)
            elif now-entry['time']>=1.0:
                raise TimeoutError(name+': no measured progress toward held target for 1s')
        else:memory.pop(name,None)
        details[name]={'state':state,'lag_rad':gap,'advance_scale':scale}
        limited.append(candidate)
    return limited,details


def velocity_sample(settings, now=None):
    data = json.loads(Path(settings['file']).read_text(encoding='utf-8'))
    now = time.monotonic_ns() if now is None else now
    age = (now-data['monotonic_ns'])/1e9
    values = data['velocity_body_m_s']
    if not 0 <= age <= settings['max_age_s'] or len(values) != 3 or not all(finite(x) for x in values):
        raise ValueError('Velocity estimator output is stale, future-dated or invalid')
    if data.get('frame') != 'body_x_forward_y_left_z_up':
        raise ValueError('Velocity estimator frame mismatch')
    if math.sqrt(sum(x*x for x in values)) > settings['max_speed_m_s']:
        raise ValueError('Velocity estimator exceeds configured speed bound')
    return values


class LegBus:
    """Only legs. No calibration, mode, PID or EEPROM writes."""
    def __init__(self, port, baudrate, ids):
        import scservo_sdk as sdk
        self.sdk, self.ids = sdk, ids
        self.port = sdk.PortHandler(port)
        self.packet = sdk.PacketHandler(0)
        if not self.port.openPort():
            raise OSError('Cannot open leg port')
        if not self.port.setBaudRate(baudrate):
            self.port.closePort()
            raise OSError('Cannot set leg baudrate')

    def read(self, mid, addr, size=2):
        fn = self.packet.read2ByteTxRx if size == 2 else self.packet.read1ByteTxRx
        value, comm, error = fn(self.port, mid, addr)
        if comm != self.sdk.COMM_SUCCESS or error:
            raise OSError(f'Motor {mid} read {addr}: communication={comm}, device={error}')
        return value

    def positions(self):
        return {mid:self.read(mid,56) for mid in self.ids}

    def verify(self, cfg):
        for name,j in cfg['legs'].items():
            if self.read(j['id'],3) != 777:
                raise ValueError(name+': expected STS3215 model number 777')
            if self.read(j['id'],33,1) != 0:
                raise ValueError(name+': motor must already be in position mode 0')
            raw = self.read(j['id'],31)
            offset = -(raw & 2047) if raw & 2048 else raw
            if offset != j['expected_homing_offset']:
                raise ValueError(name+': stored homing offset differs from configuration')

    def goals(self, values):
        writer = self.sdk.GroupSyncWrite(self.port,self.packet,42,2)
        try:
            for mid,value in values.items():
                if not writer.addParam(mid,[value & 255,(value >> 8) & 255]):
                    raise OSError('Cannot construct sync-write')
            code = writer.txPacket()
            if code != self.sdk.COMM_SUCCESS:
                raise OSError(f'Goal transmission failed: {code}')
        finally:
            writer.clearParam()

    def torque(self, enabled):
        failures = []
        for mid in self.ids:
            comm,error = self.packet.write1ByteTxRx(self.port,mid,40,int(enabled))
            if comm != self.sdk.COMM_SUCCESS or error:
                failures.append(mid)
        if failures:
            raise OSError(f'Torque command failed for IDs {failures}')

    def close(self):
        self.port.closePort()


class BNO055:
    def __init__(self, cfg):
        from smbus2 import SMBus
        self.cfg = cfg
        self.bus = SMBus(cfg['bus'])
        self.addr = cfg['address']
        if self.reg(0) != 0xA0:
            self.close()
            raise OSError('BNO055 CHIP_ID mismatch')

    def reg(self, addr):
        return self.bus.read_byte_data(self.addr,addr)

    def initialize(self):
        # Explicit imu-init command only. Does not modify any motor.
        self.bus.write_byte_data(self.addr,0x3D,0)
        time.sleep(.03)
        for addr,value in ((0x07,0),(0x3E,0),(0x3B,0),(0x41,0x24),(0x42,0)):
            self.bus.write_byte_data(self.addr,addr,value)
        self.bus.write_byte_data(self.addr,0x3D,self.cfg['mode'])
        time.sleep(.7)

    def sample(self):
        import struct
        import numpy as np
        # Identity register remapping; mounting rotation is applied once in software.
        for addr,expected in ((0x07,0),(0x3B,0),(0x41,0x24),(0x42,0),(0x3D,self.cfg['mode'])):
            if self.reg(addr) != expected:
                raise ValueError('BNO055 register configuration mismatch; use imu-init and verify axes')
        started = time.monotonic_ns()
        data = bytes(self.bus.read_i2c_block_data(self.addr,0x14,20))
        gyro = np.array(struct.unpack('<3h',data[:6]),dtype=float)*(math.pi/180/16)
        q = np.array(struct.unpack('<4h',data[12:20]),dtype=float)/16384
        if not .95 <= np.linalg.norm(q) <= 1.05:
            raise ValueError('Invalid BNO055 quaternion')
        w,x,y,z = q/np.linalg.norm(q)
        rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                             [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                             [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        mount = np.asarray(self.cfg['R_body_from_sensor'],dtype=float)
        if mount.shape != (3,3) or not np.allclose(mount@mount.T,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(mount),1):
            raise ValueError('IMU mounting matrix must be a proper rotation')
        # Convention must be physically checked: q rotates sensor vectors into world.
        body_to_world = rotation@mount.T
        calib = self.reg(0x35)
        return {'monotonic_ns':started, 'angular_velocity':(mount@gyro).tolist(),
                'gravity':(body_to_world.T@np.array([0.,0.,-1.])).tolist(),
                'yaw':math.atan2(body_to_world[1,0],body_to_world[0,0]),
                'gyro_calibration':(calib>>4)&3,'accel_calibration':(calib>>2)&3,
                'mag_calibration':calib&3}

    def close(self):
        self.bus.close()


class Policy:
    def __init__(self,cfg):
        import onnxruntime as ort
        path = ROOT/cfg['model']['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != cfg['model']['sha256']:
            raise ValueError('Policy SHA256 mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

    def action(self,obs):
        import numpy as np
        value = self.session.run(None,{'observation':np.asarray([obs],dtype=np.float32)})[0][0].tolist()
        if len(value) != 12 or not all(finite(x) for x in value):
            raise ValueError('Invalid policy action')
        return value


def check_home_imu(imu,cfg):
    sample=imu.sample()
    if sample['gyro_calibration']<cfg['imu']['min_calibration'] or sample['accel_calibration']<cfg['imu']['min_calibration']:
        raise ValueError('IMU calibration insufficient before/during Home')
    if -sample['gravity'][2]<cfg['control']['min_upright_cos']:
        raise ValueError('Torso tilt exceeds bound before/during Home')
    if (time.monotonic_ns()-sample['monotonic_ns'])/1e9>.02:
        raise TimeoutError('IMU read stale during Home')


def move_to_home(bus,imu,cfg,log):
    # Caller owns torque cleanup, including partial enable failures.
    byid={j['id']:j for j in cfg['legs'].values()}
    start=bus.positions()
    target={mid:j['reference_tick'] for mid,j in byid.items()}
    for mid,j in byid.items():
        tick_to_rad(start[mid],j)
        tick_to_rad(target[mid],j)
    check_home_imu(imu,cfg)
    bus.goals(start)
    bus.torque(True)
    if any(bus.read(mid,40,1)!=1 for mid in byid):
        raise OSError('Home torque ON readback failed')
    # Use the existing configured velocity bounds, capped at 200 ticks/s.
    duration=max(.5,max(abs(target[mid]-start[mid])/min(200.,j['max_velocity_rad_s']/j['rad_per_tick']) for mid,j in byid.items()))
    count=math.ceil(duration/.05)
    print(f'HOME_MOVING: {count*.05:.2f}s target ramp; all 12 motors enabled',flush=True)
    previous=start
    for step in range(1,count+1):
        begun=time.monotonic()
        actual=bus.positions();check_home_imu(imu,cfg)
        for mid,j in byid.items():
            tick_to_rad(actual[mid],j)
            if abs(actual[mid]-previous[mid])*j['rad_per_tick']>cfg['control']['max_tracking_error_rad']:
                raise ValueError(f'ID {mid}: Home tracking error')
        if time.monotonic()-begun>.25:raise TimeoutError('Home read cycle stalled')
        goal={mid:round(start[mid]+(target[mid]-start[mid])*step/count) for mid in byid}
        bus.goals(goal)
        log.write(json.dumps({'phase':'home','measured_ticks':actual,'target_ticks':goal,'motor_output':True})+'\n');log.flush()
        if time.monotonic()-begun>.25:raise TimeoutError('Home IO cycle stalled')
        previous=goal
        time.sleep(max(0,.05-(time.monotonic()-begun)))
    deadline=time.monotonic()+3.;stable_since=None
    while True:
        begun=time.monotonic();actual=bus.positions();check_home_imu(imu,cfg)
        for mid,j in byid.items():tick_to_rad(actual[mid],j)
        if time.monotonic()-begun>.25:raise TimeoutError('Home settle read stalled')
        reached=all(abs(actual[mid]-target[mid])<=min(32,cfg['control']['start_pose_tolerance_rad']/j['rad_per_tick']) for mid,j in byid.items())
        if reached:
            if stable_since is None:stable_since=time.monotonic()
            if time.monotonic()-stable_since>=.3:break
        else:stable_since=None
        if time.monotonic()>deadline:raise ValueError('Home did not settle within tolerance; policy not started')
        time.sleep(.05)
    print('HOME_READY: positions settled; torque remains ON for policy handoff',flush=True)


def live(cfg,contract,log,seconds,output,auto_home=False):
    if auto_home and not output:raise ValueError("auto_home requires motor output; shadow remains read-only")
    policy = Policy(cfg)
    policy.action([0.]*contract['observation_dim'])  # warm up before the control deadline starts
    ids = [cfg['legs'][n]['id'] for n in contract['action_joint_names']]
    bus = imu = None
    armed = False
    try:
        bus = LegBus(cfg['ports']['lower_body'],cfg['baudrate'],ids)
        bus.verify(cfg)
        imu = BNO055(cfg['imu'])
        if auto_home:
            if any(bus.read(mid,40,1)!=0 for mid in ids):
                raise ValueError('Stop other controllers: all motor torque must be OFF before automatic Home')
            check_home_imu(imu,cfg)
            armed=True
            move_to_home(bus,imu,cfg,log)
        defaults = dict(zip(contract['observation_joint_names'],contract['default_joint_position_rad']))
        prev_target = [defaults[n] for n in contract['action_joint_names']]
        prev_action = [0.]*12
        tracking_memory={}
        last_q = last_time = yaw_origin = None
        start = time.monotonic()
        tick = 0
        while tick < int(seconds*50):
            deadline = start+(tick+1)*.02
            raw = bus.positions()
            sample_time = time.monotonic()
            positions = {n:tick_to_rad(raw[j['id']],j) for n,j in cfg['legs'].items()}
            imu_data = imu.sample()
            if imu_data['gyro_calibration'] < cfg['imu']['min_calibration'] or imu_data['accel_calibration'] < cfg['imu']['min_calibration']:
                raise ValueError('IMU is not sufficiently calibrated')
            if -imu_data['gravity'][2] < cfg['control']['min_upright_cos']:
                raise ValueError('Robot tilt exceeds configured limit')
            if (time.monotonic_ns()-imu_data['monotonic_ns'])/1e9 > .02:
                raise TimeoutError('IMU read took too long')
            velocity = velocity_sample(cfg['velocity'])
            if last_q is None:
                for name in positions:
                    if abs(positions[name]-defaults[name]) > cfg['control']['start_pose_tolerance_rad']:
                        raise ValueError(name+': start pose differs from configured reference; Home was not reached or was disturbed')
                last_q,last_time,yaw_origin = positions,sample_time,imu_data['yaw']
                # Establish a measured finite-difference velocity before enabling output.
                time.sleep(.02)
                start = time.monotonic()
                continue
            dt = sample_time-last_time
            if not 0 < dt <= .1:
                raise TimeoutError('Joint sample gap invalid')
            joint_velocity = {n:(positions[n]-last_q[n])/dt for n in positions}
            for n,v in joint_velocity.items():
                if abs(v) > cfg['legs'][n]['max_velocity_rad_s']:
                    raise ValueError(n+': measured speed/wrap exceeds limit')
            for n,fixed in cfg['fixed_upper_body'].items():
                positions[n]=fixed['position_rad']
                joint_velocity[n]=0.
            state = {'joint_position_rad':positions,'joint_velocity_rad_s':joint_velocity,
                     'episode_time_s':tick*.02,'heading_yaw_rad':imu_data['yaw']-yaw_origin,
                     'base_linear_velocity_body_m_s':velocity,'base_angular_velocity_body_rad_s':imu_data['angular_velocity'],
                     'projected_gravity_body':imu_data['gravity'],'command':contract['command'],
                     'previous_action':prev_action,'filtered_leg_target_rad':prev_target}
            if auto_home and tick==0:print('POLICY_INFERENCE: evaluating first live action; not yet sent to motors',flush=True)
            obs = observation(contract,state)
            action,targets = leg_targets(contract,policy.action(obs),prev_target)
            requested_targets=list(targets)
            tracking_detail={}
            try:
                if cfg['control'].get('limit_policy_slew',False):
                    targets=limit_policy_slew(targets,prev_target,cfg,contract)
                if cfg['control'].get('adaptive_tracking',False):
                    targets,tracking_detail=follow_measured_targets(targets,prev_target,positions,cfg,contract,tracking_memory,time.monotonic())
                ticks = targets_to_ticks(targets,positions,prev_target,cfg,contract)
            except (ValueError,TimeoutError) as exc:
                log.write(json.dumps({'phase':'policy_target_rejected','tick':tick,'reason':str(exc),
                    'state':state,'action':action,'previous_target_rad':prev_target,
                    'requested_target_rad':targets,'candidate_motor_command_sent':False})+'\n')
                log.flush()
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError('50 Hz deadline exceeded before motor write')
            if output:
                if not armed:
                    # Stage the current pose before enabling torque; never write calibration.
                    armed = True  # cleanup also handles a partially successful enable
                    bus.goals(raw)
                    bus.torque(True)
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Deadline exceeded during motor enable')
                bus.goals(ticks)
                if tick==0:print('POLICY_RUNNING: first v13 targets sent to all 12 legs',flush=True)
            log.write(json.dumps({'tick':tick,'state':state,'action':action,'target_rad':targets,
                                  'target_ticks':ticks,'motor_output':output,
                                  'requested_target_rad':requested_targets,
                                  'slew_limited':targets!=requested_targets,'tracking':tracking_detail})+'\n')
            log.flush()
            if time.monotonic() >= deadline:
                raise TimeoutError('50 Hz deadline exceeded after IO/logging')
            last_q = {n:positions[n] for n in cfg['legs']}
            last_time = sample_time
            prev_action,prev_target = action,targets
            tick += 1
            time.sleep(max(0,deadline-time.monotonic()))
    finally:
        exception_in_flight = sys.exc_info()[0] is not None
        cleanup_errors = []
        if bus:
            try:
                if armed and cfg['control']['stop_action'] == 'disable_torque':
                    bus.torque(False)
                    if auto_home:
                        if any(bus.read(mid,40,1)!=0 for mid in ids):raise OSError('Torque OFF readback failed')
                        print('ALL 12 torque OFF verified',flush=True)
                # hold_last means no further position writes; servo retains its last goal.
            except Exception as exc:
                cleanup_errors.append('Motor stop failed: '+str(exc))
            finally:
                bus.close()
        if imu:
            imu.close()
        if cleanup_errors:
            print('\n'.join(cleanup_errors),file=sys.stderr)
            if not exception_in_flight:
                raise RuntimeError('Motor stop could not be confirmed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['check-config','replay','monitor','imu-init','run'])
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--states',type=Path,default=ROOT/'sample_state.jsonl')
    parser.add_argument('--log',type=Path,default=Path('rollout_log.jsonl'))
    parser.add_argument('--seconds',type=float,default=5.)
    parser.add_argument('--enable-motors',action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    contract = json.loads((ROOT/'v13_io_contract.json').read_text(encoding='utf-8'))
    if not finite(args.seconds) or args.seconds <= 0:
        parser.error('seconds must be finite and positive')
    if args.enable_motors and args.mode != 'run':
        parser.error('--enable-motors is valid only for run')
    if args.mode == 'check-config':
        missing = validate(cfg,contract,True)
        print(json.dumps({'hardware_output_ready':not missing,'missing_or_invalid':missing},indent=2))
        return 2 if missing else 0
    if args.mode == 'run':
        errors = validate(cfg,contract,args.enable_motors)
        if errors:
            raise ValueError('Incomplete configuration:\n'+'\n'.join(errors))
        if args.seconds > cfg['control']['max_duration_s']:
            raise ValueError('Requested duration exceeds configured maximum')
    def interrupt(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    with args.log.open('x',encoding='utf-8') as log:
        if args.mode == 'replay':
            policy = Policy(cfg)
            with args.states.open(encoding='utf-8') as source:
                for line in source:
                    if not line.strip(): continue
                    state=json.loads(line)
                    action,targets=leg_targets(contract,policy.action(observation(contract,state)),state['filtered_leg_target_rad'])
                    log.write(json.dumps({'action':action,'target_rad':targets,'motor_output':False})+'\n')
            print('REPLAY_OK: no hardware opened')
        elif args.mode in ('monitor','imu-init'):
            bus=imu=None
            try:
                if args.mode == 'monitor' and cfg['ports']['lower_body']:
                    bus=LegBus(cfg['ports']['lower_body'],cfg['baudrate'],[j['id'] for j in cfg['legs'].values()])
                if cfg['imu']['bus'] is not None and cfg['imu']['address'] is not None:
                    imu=BNO055(cfg['imu'])
                if args.mode == 'imu-init':
                    if not imu: raise ValueError('Set imu.bus and imu.address first')
                    imu.initialize()
                    print('IMU initialized; calibrate and verify axes before run. Motors were not opened.')
                else:
                    if not bus and not imu: raise ValueError('Configure at least one sensor or leg port')
                    end=time.monotonic()+args.seconds
                    while time.monotonic()<end:
                        row={'monotonic_ns':time.monotonic_ns(),'motor_output':False}
                        if bus: row['leg_present_position_ticks']=bus.positions()
                        if imu: row['imu']=imu.sample()
                        log.write(json.dumps(row)+'\n');log.flush()
                        print(json.dumps(row),flush=True)
                        time.sleep(.05)
            finally:
                if bus: bus.close()
                if imu: imu.close()
        else:
            live(cfg,contract,log,args.seconds,args.enable_motors)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted; configured stop action attempted if motors were enabled.')
        raise SystemExit(130)

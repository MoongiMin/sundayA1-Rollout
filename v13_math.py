"""Simulator-free observation/action math. No motor transport is implemented."""
import math


def vector(value, n, label):
    if len(value) != n or any(not math.isfinite(float(x)) for x in value):
        raise ValueError(f'{label}: expected {n} finite numbers')
    return [float(x) for x in value]


def observation(contract, state):
    """State values must already be calibrated into the training root/joint frames."""
    names = contract['observation_joint_names']
    q = vector([state['joint_position_rad'][n] for n in names], len(names), 'joint positions')
    dq = vector([state['joint_velocity_rad_s'][n] for n in names], len(names), 'joint velocities')
    age = float(state['episode_time_s'])
    yaw = float(state['heading_yaw_rad'])
    if not math.isfinite(age) or age < 0 or not math.isfinite(yaw):
        raise ValueError('Invalid episode time or heading')
    theta = 2 * math.pi * ((age / contract['gait_cycle_s']) % 1)
    parts = {
        'linear_velocity': vector(state['base_linear_velocity_body_m_s'], 3, 'base linear velocity'),
        'angular_velocity': vector(state['base_angular_velocity_body_rad_s'], 3, 'base angular velocity'),
        'gravity': vector(state['projected_gravity_body'], 3, 'projected gravity'),
        'command': vector(state['command'], 3, 'command'),
        'joints': [a-b for a,b in zip(q, contract['default_joint_position_rad'])],
        'joint_velocity': [(a-b)*contract['joint_velocity_observation_scale'] for a,b in zip(dq, contract['default_joint_velocity_rad_s'])],
        'previous_action': vector(state['previous_action'], 12, 'previous action'),
        'gait_clock': [math.sin(theta), math.cos(theta)],
        'heading': [math.cos(yaw), math.sin(yaw)],
        'applied_targets': vector(state['filtered_leg_target_rad'], 12, 'filtered leg targets'),
    }
    result = []
    for field in contract['observation_fields']:
        part = parts[field['name']]
        if len(result) != field['start'] or len(part) != field['stop'] - field['start']:
            raise ValueError('Observation contract shape/order mismatch')
        result.extend(part)
    if len(result) != contract['observation_dim']:
        raise ValueError('Observation dimension mismatch')
    limit = contract['observation_clip']
    return [max(-limit, min(limit, v)) for v in result]


def leg_targets(contract, action, previous):
    action = vector(action, 12, 'action')
    previous = vector(previous, 12, 'previous target')
    bound = contract['action_clip']
    clipped = [max(-bound, min(bound, x)) for x in action]
    requested = [max(lo, min(hi, ref + contract['action_scale_rad']*a))
                 for ref,a,(lo,hi) in zip(contract['static_leg_reference_rad'], clipped,
                                        contract['leg_target_clip_rad'])]
    filtered = [p + contract['leg_filter_alpha']*(r-p) for p,r in zip(previous, requested)]
    return clipped, filtered

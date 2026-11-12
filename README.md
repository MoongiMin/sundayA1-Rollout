# Sunday A1 — R25 Jetson rollout

Runnable R25 live-policy and recorded-simulation playback package for our Sunday A1 robot. This snapshot preserves the **latest manually captured Home, confirmed motor IDs and direction signs** used in the November 11 rollout. The November 12 A1-limit-interpolated Home experiment was rejected and is **not active**.

This is the supported-robot experimental runner, not a validated autonomous walking stack. Body linear velocity is not measured, and the manually posed Home has not been independently calibrated to CAD angles. A moving support platform does not make the stationary-base assumption true.

## Contents

|File|Purpose|
|---|---|
|`r25_rollout.py`|NumPy PPO actor inference; status, shadow, saved-Home hold, live commands|
|`r25_actor.npz`|Exported actor weights and observation normalizer; no PyTorch needed on Jetson|
|`r25_contract.json`|Authoritative joint order, motor addresses/signs/Home, action scales and limits|
|`play_sim_trajectory.py`|Open-loop replay of the included 10-second simulation target sequence|
|`r25_sim_forward_10s.trajectory.json`|500 frames at 50 Hz, bound to the exact contract hash|
|`rollout.py`, `v13_math.py`, `v13_io_contract.json`|Legacy transport/IMU compatibility dependencies; **not** the R25 walking entry point|
|`monitor_joint_motion.py`|Read-only strongest-joint motion monitor with expected motion descriptions|
|`capture_whole_body_home.py`|Read-only capture of 24 encoders, including the two hands|
|`manual_joint_calibration.py`, `manual_arm_calibration.py`|Read-only range capture, labeled using the current R25 mapping|
|`board_ports.example.json`|Three board USB serials, stable device paths and captured socket metadata|
|`calibration_evidence/`|Range and Home captures used in the package; historical labels may differ|
|`provenance/`|Original checkpoint, accepted-policy/run metadata and source hashes|
|`docs/`|Mapping, conversion, limitations and rejected-experiment record|
|`replay_r25_goals.py`, `replay_r25_viser.py`, `show_r25_home.py`|Optional laptop MuJoCo/Viser review|

## Install on Jetson

The observed robot environment was Ubuntu 22.04 aarch64 / Jetson Orin Nano, Jetson Linux R36.4.7. The existing installation used Python 3.13; this package uses Python 3.10+ syntax. A new clone has not been physically retested on the robot.

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip i2c-tools
git clone https://github.com/MoongiMin/sundayA1-Rollout.git
cd sundayA1-Rollout
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/verify_package.py
python -m unittest discover -s tests -v
```

`feetech-servo-sdk` provides `scservo_sdk`; `pyserial` provides `serial`. Do **not** install the unrelated package named `serial`. NumPy runs the actor on CPU; CUDA, Isaac Sim, mjlab, ONNX Runtime and PyTorch are not required for Jetson R25 execution.

### Keep the existing robot-specific configuration

On the same Jetson, import the actual mounted-IMU configuration from the existing installation:

```bash
cp ~/Sunday_v13_Jetson_Rollout/config.confirmed.json ./config.confirmed.json
# The confirmed USB metadata normally already exists at ~/board_ports.json.
# Only when it is absent:
test -f ~/board_ports.json || cp board_ports.example.json ~/board_ports.json
```

`config.confirmed.example.json` contains an explicitly **unverified identity mount rotation**. It is not a replacement for the working IMU configuration. Never turn the verification flag on without checking the mounting transform. The runner reads only the `imu` block from `config.confirmed.json`; R25 motor mapping comes from `r25_contract.json`.

The USB serials are lower body `5B3D042991`, left arm `5B3D040665`, right arm `5AE6082486`. `/dev/ttyACM0/1/2` may change; use the `/dev/serial/by-id/` paths. Recorded USB socket locations are metadata, not the binding key.

If permissions are missing:

```bash
sudo usermod -aG dialout,i2c "$USER"
# Log out and reconnect for the group change to take effect.
```

## Commands

Run from the repository directory with the venv activated. Motor-output commands are intended for the supported robot with an accessible power cutoff. Completion or Ctrl+C attempts torque OFF; loss of communication can prevent confirmation.

### Offline checks — no motors opened

```bash
python r25_rollout.py status
python play_sim_trajectory.py inspect --seconds 10
```

`status` verifies the actor; it is not a full hardware or IMU check.

### IMU initialization

The BNO055 was detected on I2C bus **7**, address **0x28 (40)**. Mode 8 is IMUPLUS. Its live orientation and gyro remain real even when a fixed velocity input is requested.

```bash
python rollout.py imu-init --config config.confirmed.json --log "imu_init_$(date +%s).jsonl"
```

This writes IMU configuration registers only. It does not calibrate physical mounting axes or command motors. After power loss, `BNO055 register configuration mismatch` usually requires this initialization again. Gyro calibration must reach 3; the explicit `--allow-uncalibrated-accel` option relaxes the accelerometer calibration check for the bench experiment, not the gyro or mounting checks.

### Read-only live shadow

```bash
python r25_rollout.py shadow --seconds 2 --delay 5 --command forward --allow-uncalibrated-accel --experimental-fixed-body-vx 0.06
```

Reads motors/IMU and computes policy targets, but sends no goals or torque commands. It cannot validate motor tracking or prove walking stability.

### Hold the preserved Home

```bash
python r25_rollout.py home-preview --seconds 8 --delay 5 --supported-test --enable-motors --allow-uncalibrated-accel
```

Five-second delay, Home ramp, wait until all joints are within 65 ticks, eight-second hold, then torque OFF. Home settling has **no timeout** in this preserved runner: if `HOME_WAITING` persists, interrupt with Ctrl+C. The hold duration starts after Home is reached.

### Live R25 policy — 10 seconds

```bash
python r25_rollout.py run --seconds 10 --delay 5 --command forward --supported-test --enable-motors --allow-uncalibrated-accel --experimental-fixed-body-vx 0.06
```

This reproduces the fixed-velocity experiment: **0.06 m/s is a synthetic body-velocity observation**, not an IMU measurement. The forward command is a separate desired velocity. Without `--experimental-fixed-body-vx`, the velocity observation is `[0,0,0]`.

### Open-loop simulation targets — 10 seconds

```bash
python play_sim_trajectory.py run --seconds 10 --delay 5 --supported-test --enable-motors
```

Moves to the same saved Home, then replays the included targets. No live actor or IMU feedback determines the next action. Encoder reads are used for Home checks and logging. The trace is limited to 10 seconds and late frames may be skipped to follow elapsed time.

## How mapping and angles work

See [the complete 22-joint table](docs/MAPPING.md) and [mapping history and conversion](docs/CALIBRATION.md).

```text
q_rad = reference_angle_rad + sign × (tick − home_tick) × 2π / 4096
goal_tick = round(home_tick + (target_rad − reference_angle_rad) / (sign × 2π / 4096))
```

The conversion assumes direct drive and 4096 ticks per output revolution. **Recorded min/max ticks were not stretched onto R25 or A1 angle limits.** Home is an independently stored reference. Correct direction signs do not prove the saved physical Home matches the simulation Home.

## Record calibration evidence

```bash
python monitor_joint_motion.py --seconds 120 --delay 5
python manual_joint_calibration.py --all --delay 5 --seconds 360
python manual_arm_calibration.py --delay 5 --seconds 360
python capture_whole_body_home.py --delay 5
```

Range capture ends with Enter or the specified duration. These commands do not automatically apply range or Home changes. Capture summaries use `(board, motor ID)` as the stable identity. Replacing a motor, changing its firmware homing offset or remounting a horn can invalidate the stored values.

## Review in MuJoCo / Viser on a laptop

Install optional dependencies and fetch the exact upstream XML/meshes (about 34 MB):

```bash
python -m pip install -r requirements-visualization.txt
python tools/fetch_visualization_assets.py
python show_r25_home.py
```

Open http://127.0.0.1:8088/ for the static R25 Home and board/ID labels. The asset downloader uses pinned URLs and SHA256 checks; the Jetson runtime needs none of these assets.

For a downloaded run log:

```bash
python replay_r25_goals.py r25_run_TIMESTAMP.jsonl --seconds 10 --output review.poses.json
python replay_r25_viser.py review.poses.json --port 8081
```

Open http://127.0.0.1:8081/. The same replay supports `r25_sim_playback_*.jsonl`. **This simulates the model's response to the sent motor targets. It does not reconstruct measured real-world base motion, the support wire or the moving platform.** Use the exact contract matching the log; the current contract will reject earlier logs made before mapping/Home changes.

Windows PowerShell download example (outside SSH):

```powershell
scp 'sunday@192.168.18.15:/home/sunday/sundayA1-Rollout/r25_run_TIMESTAMP.jsonl' "$HOME\Downloads\"
```

## Provenance and current limitations

- R25 checkpoint: `Auspr7ng/mars-weekly-progress`, Week4 `joystick_walking_r25_300/model_300.pt`.
- Checkpoint SHA256: `3bf72dd99c3c75049db74b4d50c94a6f3ce270af6887a7c048a16130b1ba6202`.
- Contract SHA256: `300cf9bf8f25b0d18dcb7d35a090198f9c77a05ec7514ec2290ccc5011b3a1d1`.
- Observation 460 = 5 history frames × 92; output 22 joints. Other observation joints use training defaults.
- Preserved action handling: clip to ±1, 0.30 target EMA, per-joint speed/step limits, R25 soft joint limits. Motor torque magnitude/PID is not configured by the runner; torque enable/disable is used.
- Nominal 50 Hz; actual cycle time depends on hardware. Tracking lag is logged, not a stop threshold in this release. Serial reads retry timeout/corrupt responses twice, then fail.
- The real base-velocity estimator, precise absolute-angle alignment and reliable unsupported walking remain unresolved. Fixed synthetic velocity is an experiment, not a solution to estimation.
- See [dependency notes](docs/DEPENDENCIES.md) and [third-party provenance](THIRD_PARTY_NOTICES.md). No blanket license is assigned to upstream assets or policies by this repository.

한국어 요약: 최근 rollout Home과 확인한 ID·부호를 유지한 패키지입니다. A1 범위 보간 Home은 폐기했고 적용하지 않았습니다. 실제 Jetson의 `config.confirmed.json`을 가져와 사용하세요. `home-preview`는 Home 유지, `run`은 실시간 정책, `play_sim_trajectory.py run`은 open-loop 재생입니다.

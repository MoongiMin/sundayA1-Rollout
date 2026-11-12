# R25 policy I/O

The original checkpoint has a 460-input, 22-output actor MLP (256, 128, 128 hidden widths, ELU). `r25_actor.npz` contains its weights and observation normalizer. NumPy inference subtracts the exported mean and divides by exported std + 0.01 before the MLP. Training action distributions are not sampled: the deterministic actor output is used.

One observation slice has 92 values. Five history samples are stored **per term**, then the term histories are concatenated (not simply five complete 92-vectors in a row):

|Term|Values per sample|Runtime source|
|---|---:|---|
|Body angular velocity × 0.25|3|BNO055, mount-rotated|
|Projected gravity|3|BNO055 quaternion and mount rotation|
|Velocity/yaw command|3|Command ramp; not a measured velocity|
|Joint angle minus training default|27|22 real motor angles plus defaults for remaining observation joints|
|Joint velocity × 0.05|27|Encoder finite differences plus zero defaults|
|Previous clipped action|22|Previous actor output clipped to ±1|
|Gait phase sin/cos × motion gate|2|Runner phase/command state|
|Heading error sin/cos|2|Command-integrated heading target and relative IMU yaw|
|Body linear velocity|3|Zero or experimental `[fixed_vx,0,0]`; not measured|

Foot-contact sensing is not included in these actor input terms. Simulation reward/contact logic does not imply the physical actor receives a foot-contact measurement.

The target path is `clip(action,-1,1)` → reference angle plus per-joint action scale → 0.30 EMA → per-step slew from per-joint speed × 0.02 s → R25 soft limits → signed tick conversion. This preserved target shaping is part of the deployed behavior; it is not a change to motor torque/PID registers. Actual loop time may exceed 20 ms, while command/history progression uses the nominal timestep.

Open-loop playback instead reads already-recorded simulation targets. It does not recalculate the actor from hardware observations or apply the live filter again. The fixed 0.06 m/s live experiment and the prerecorded simulation command trace are not identical tests: the bundled trace's forward command ramps to approximately 0.10 m/s.

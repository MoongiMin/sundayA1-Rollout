# Mapping, signs, offsets and range evidence

## Current source of truth

`r25_contract.json` is the active motor conversion, with the November 11 capture `whole_body_home_1794439281586950429.json`. Keep this exact file with each log. The trajectory and passive monitor pin its hash. The hardware address includes the USB board; left-arm ID 1 and right-arm ID 1 are different motors.

## Corrections incorporated

- Both shoulder pitch/roll motor associations were swapped after the passive physical check. Current R25 `shoulder_roll` is arm-board ID 1; `shoulder_pitch` is ID 2.
- Both hip pitch/roll associations were corrected. Current R25 right `hip_roll` is bus ID 8, `hip_pitch` ID 7; left `hip_roll` ID 20, `hip_pitch` ID 19. Historical CLI selection numbers were sometimes different from the physical bus addresses; use the current contract, not old selection labels.
- Both ankle pitch/roll associations were swapped. Right R25 `ankle_roll` is ID 11, `ankle_pitch` ID 12; left `ankle_roll` ID 23, `ankle_pitch` ID 24.
- Both elbow-yaw signs were reversed; current sign is −1 on both arm-board ID 3 motors.
- The final passive check additionally reversed both left-shoulder signs and both right-ankle signs. The resulting signs are in [MAPPING.md](MAPPING.md), not a set of corrections to apply again.
- R25 coordinate axes differed from the earlier V73/CAD interpretation. Final physical checking superseded name-only transfer and intermediate releases.

These changes affect both encoder-to-policy observations and policy-angle-to-motor goals because they share the same contract. They are not only UI label changes. Joint names in R25 are not always anatomical: e.g. `hip_roll` describes the front/back hinge, and `ankle_roll` describes toe up/down in the current motion descriptions.

## The Home is separate from sign and ID

The formula is an offset plus a signed scale:

`q = q_home + sign * (tick - tick_home) * 2*pi/4096`.

`tick_home` is a raw encoder reading at a user-posed reference. `q_home` is the model's R25 nominal joint angle. Equating the two is a **pose alignment assumption**, not an independently measured angle calibration. A +10° model target change produces about 114 ticks relative to the saved Home, with direction determined by `sign`. It does not select 10% of the captured range.

Motor `homing_offset` EEPROM values, the runtime `home_tick`, and the model's `reference_angle_rad` are distinct quantities. This package does not rewrite EEPROM or subtract an old homing offset a second time. If offsets change, raw tick captures must be revisited.

## What min/max capture means

The user confirmed that ranges were recorded by moving joints by hand through reachable positions. They were **not** captured at independently identified A1 document angles. They are evidence of observed raw encoder spans, not proof of mechanical end stops, safe collision-free travel or angle scale.

Original arm/leg capture files are included under `calibration_evidence/`. Their names were assigned before the final mapping corrections; reconcile by board and physical ID. Range recorders in this package now label output with the current contract.

## Rejected November 12 Home experiment

The user requested an approximate visual check using A1 documentation and range endpoints. An isolated preview interpolated 18 joints with hypothetical endpoint and coordinate correspondences; four elbow joints retained the previous Home. The user reported a substantial mismatch and requested the most recent rollout Home again.

The preview did not overwrite the active contract, trajectory or EEPROM. This repository retains the old working Home. `docs/rejected_visual_home_candidate.json` and `VISUAL_HOME_REJECTED.json` are audit records only; the rejected executable is not included as a run command. Do not use those hypothetical scales as calibration.

The A1 table describes safe angular ranges, does not establish equivalence to the captured extrema, and omits both elbow-yaw limits. A documentation coordinate may also have a different axis sign or zero from R25. With direct drive and a known encoder scale, one independently identified R25 angle/tick pair fixes an offset; a second pair checks sign and scale. Visual matching can refine a candidate but does not establish absolute accuracy.

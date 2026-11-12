# Third-party provenance

- R25 policy checkpoint, metadata and CAD references originate from [Auspr7ng/mars-weekly-progress](https://github.com/Auspr7ng/mars-weekly-progress), Week4. The optional asset manifest pins commit `f403f5bad223b8628d33a0bdee7de7fe9dbd9ed1` and each asset SHA256.
- The actor NPZ is an inference export of the included R25 checkpoint. Policy provenance and hashes are retained in `provenance/` and `r25_contract.json`.
- A1 angle documentation: [Vibe Robotics docs](https://www.viberobotics.co/docs). The table is retained as a reference, not asserted to match this robot's endpoints or R25 zero frames.
- Official encoder conversion reference: [Vibe Robotics SDK math.py](https://github.com/viberobotics/vibe_robotics_sdk/blob/4c29bee4683e97d4580bee308d97d3c6c8c2418e/viberobotics/utils/math.py), `2*pi/4096` radians per tick.
- Transport/IMU and calibration utilities were developed in the original Sunday walking deployment workspace. The Feetech SDK and Python dependencies are installed separately under their own licenses.

This repository does not assign a new license to the upstream policy, CAD or documentation. Check the original projects' terms before further redistribution or commercial use.

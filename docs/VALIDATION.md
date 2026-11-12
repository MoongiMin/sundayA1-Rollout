# Package validation — 2026-11-12

No motor or IMU hardware was opened during package assembly.

- Six offline tests passed in the existing Python/NumPy/PyTorch audit environment, including comparison of exported actor output against the original PPO checkpoint.
- A fresh Python 3.13 Windows venv installed `requirements.txt` successfully (NumPy 2.5.4, pyserial 3.5, smbus2 0.5.0, feetech-servo-sdk 1.0.0). Five runtime tests passed; the optional Torch audit was skipped there because Torch is not a runtime dependency.
- `status` and the 500-frame / 10-second trajectory inspection passed in that fresh environment.
- All 22 active Home ticks match the latest included user capture by board and physical ID. Contract byte hash remains `300cf9bf8f25b0d18dcb7d35a090198f9c77a05ec7514ec2290ccc5011b3a1d1`.
- All optional model/mesh assets were downloaded from the pinned manifest and verified by SHA256. MuJoCo loaded the model successfully.
- Both the latest downloaded live-policy log (`r25_run_1794439442747988511.jsonl`) and open-loop log (`r25_sim_playback_1794439619526188371.jsonl`) produced 500-frame / 10-second MuJoCo replays using the packaged paths and matching contract.

The original logs/videos, virtualenv, downloaded CAD assets and generated replay files are not committed. CAD is fetched on demand. This validation does not certify physical pose agreement, balance, motor tracking, new-Jetson installation or the unknown current IMU mounting matrix.

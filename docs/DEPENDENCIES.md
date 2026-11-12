# Dependencies and environments

`requirements.txt` is the Jetson runtime set. `feetech-servo-sdk==1.0.0` exposes `scservo_sdk`, `pyserial==3.5` exposes `serial`, and `smbus2` accesses `/dev/i2c-7`. NumPy implements the exported MLP. No CUDA/Torch package is needed for running the exported actor.

`requirements-visualization.txt` adds MuJoCo, Viser and trimesh for laptop-side review. The local visualization environment used NumPy 2.2.6, MuJoCo 4.0.0, Viser 1.1.1 and trimesh 5.2.0. This is not a captured Jetson `pip freeze`; the complete historical Jetson package list has not been retrieved.

The original checkpoint is stored in `provenance/r25_model_300.pt`. PyTorch is only needed to audit/re-export that checkpoint, not to run this release. It is deliberately absent from the Jetson requirements.

Existing hardware configuration is imported from the working Jetson. `config.confirmed.example.json` is a placeholder; an identity sensor mount matrix must not be silently treated as the physically verified transform. The latest local motor/Home contract is available; the actual Jetson IMU block must be kept separately or supplied by the operator.

The compatibility module `rollout.py` imports `v13_math.py` and reads `v13_io_contract.json` for its `imu-init` command. These files are included so this command works in a fresh clone. Its historical V13 walking modes are not the supported entry points of this repository, and no V13 ONNX model is bundled.

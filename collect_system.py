"""Read-only system inventory. Never opens motor/IMU ports or sends commands."""
import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='system_info.json')
    args = parser.parse_args()
    result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'os': platform.platform(), 'architecture': platform.machine(),
              'python': sys.version, 'commands': {}}
    commands = [['nvidia-smi'], ['lsusb'], ['uname', '-a']]
    if sys.platform.startswith('linux'):
        commands += [['dpkg-query', '-W', 'nvidia-jetpack', 'nvidia-l4t-core']]
        release = Path('/etc/nv_tegra_release')
        result['nv_tegra_release'] = release.read_text() if release.exists() else None
        result['serial_devices'] = sorted(str(p) for pattern in ('ttyUSB*', 'ttyACM*')
                                           for p in Path('/dev').glob(pattern))
    for cmd in commands:
        if shutil.which(cmd[0]):
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=15, errors='replace')
                result['commands'][' '.join(cmd)] = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
            except subprocess.TimeoutExpired:
                result['commands'][' '.join(cmd)] = {'error': 'timeout'}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved', args.output)


if __name__ == '__main__':
    main()

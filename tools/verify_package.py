"""Verify the published package bytes without opening any hardware."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    failed=[]
    for line in (ROOT/'PACKAGE_SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected,name=line.split('  ',1)
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Invalid manifest path')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            failed.append(name)
    if failed:
        raise SystemExit('PACKAGE CHECK FAILED: '+', '.join(failed))
    print('PACKAGE_SHA256_OK: no hardware opened')

if __name__=='__main__':
    main()

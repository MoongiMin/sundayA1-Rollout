"""Fetch hash-pinned optional CAD files. No hardware or policy is executed."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT/'visualization_assets.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = (ROOT/item['path']).resolve()
        if not path.is_relative_to((ROOT/'assets').resolve()):
            raise ValueError('Asset path outside assets directory')
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']:
            print('OK',item['path'])
            continue
        request = urllib.request.Request(item['url'], headers={'User-Agent':'SundayA1-Rollout'})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError('Downloaded asset hash mismatch: '+item['path'])
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_name(path.name+'.partial')
        temp.write_bytes(payload)
        temp.replace(path)
        print('DOWNLOADED',item['path'])
    print('VISUALIZATION_ASSETS_READY')

if __name__ == '__main__':
    main()

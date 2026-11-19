"""Verify distributed files before installation or modification."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'SHA256SUMS.json').read_text())
failures = []
for name, expected in manifest.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
if failures:
    raise SystemExit('Missing/changed files: ' + ', '.join(failures))
print(f'Verified {len(manifest)} distributed files')

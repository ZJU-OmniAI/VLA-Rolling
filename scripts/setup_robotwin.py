"""Restore the pinned RoboTwin/cuRobo runtime sources without changing the uv environment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'scripts/source_manifests/robotwin.json'


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def verify(destination: Path, files: dict) -> list[str]:
    return [name for name, record in files.items()
            if not (destination / name).is_file()
            or digest(destination / name) != record['sha256']]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'src/RoboTwin')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    destination = args.destination.resolve()
    files = manifest['files']
    mismatches = verify(destination, files)
    if not mismatches:
        print(f'Verified {len(files)} RoboTwin/cuRobo source files in {destination}.')
        return
    if args.verify_only:
        parser.exit(1, f'Missing or modified source files ({len(mismatches)}):\n' + '\n'.join(mismatches[:20]) + '\n')
    conflicts = [name for name in mismatches if (destination / name).exists() or (destination / name).is_symlink()]
    if conflicts:
        parser.error('Refusing to overwrite modified local files: ' + ', '.join(conflicts[:10]))
    # Existing files must be exactly the pinned version. Assets and compiled
    # extensions can coexist with the restored sources and are never removed.
    with tempfile.TemporaryDirectory(prefix='vla-robotwin-') as work:
        checkouts = {}
        for name, spec in manifest['repositories'].items():
            checkout = Path(work) / name
            subprocess.run(['git', 'init', '-q', str(checkout)], check=True)
            subprocess.run(['git', '-C', str(checkout), 'remote', 'add', 'origin', spec['url']], check=True)
            subprocess.run(['git', '-C', str(checkout), 'fetch', '--depth=1', 'origin', spec['commit']], check=True)
            subprocess.run(['git', '-C', str(checkout), 'checkout', '-q', '--detach', 'FETCH_HEAD'], check=True)
            actual = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != spec['commit']:
                raise RuntimeError(f'Unexpected {name} revision: {actual}')
            checkouts[name] = checkout
        for name, record in files.items():
            source = checkouts[record['repository']] / record['path']
            if not source.is_file() or digest(source) != record['sha256']:
                raise RuntimeError(f'Upstream source does not match the recorded version: {name}')
        for name in mismatches:
            record = files[name]
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkouts[record['repository']] / record['path'], target)
    remaining = verify(destination, files)
    if remaining:
        raise RuntimeError(f'Source verification failed: {remaining[:10]}')
    print(f'Restored and verified {len(files)} source files in {destination}.')


if __name__ == '__main__':
    main()

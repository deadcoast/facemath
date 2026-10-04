"""Stage an explicit release inventory and seal create-only, versioned outputs."""
from __future__ import annotations

import argparse
from email.parser import Parser
from hashlib import file_digest
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tomllib
import zipfile

from verify_release import verify

SOURCE_DIRS = ('src/facemath', 'tests', 'scripts', 'docs', 'examples', '.github')
ROOT_FILES = ('pyproject.toml', 'uv.lock', 'README.md', 'AGENTS.md', 'CHANGELOG.md',
              '.gitattributes', '.gitignore', 'REPOSITORY_REVIEW.json', 'models/README.md',
              'DESIGN.md', 'premium-ui.json')
EXCLUDED = {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.git',
            '.venv', '.uv-cache', '.commit-audit', 'build', 'runs', 'packages', 'dist'}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def inventory(root: Path, evidence: list[str]) -> dict[str, Path]:
    """Allowlist roots; prune before traversal and never follow a symlink."""
    selected = {}
    roots = [root / name for name in SOURCE_DIRS if (root / name).exists()]
    for name in evidence:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to((root / 'evidence').resolve()):
            raise ValueError('selected evidence must be inside the repository evidence directory')
        if not path.is_dir():
            raise ValueError(f'evidence directory missing: {name}')
        roots.append(path)
    for path in roots:
        if path.is_symlink():
            raise ValueError(f'release symlink: {path}')
        for directory, dirs, files in os.walk(path, followlinks=False):
            base = Path(directory)
            for name in dirs + files:
                if (base / name).is_symlink():
                    raise ValueError(f'release symlink: {base / name}')
            dirs[:] = sorted(n for n in dirs if n not in EXCLUDED and not n.startswith('.')
                             and not n.endswith('.egg-info'))
            for name in sorted(files):
                p = base / name
                if ((name.startswith('.') and not base.is_relative_to(root / 'evidence'))
                        or p.suffix in {'.pyc', '.pyo'}):
                    continue
                selected[p.relative_to(root).as_posix()] = p
    for name in ROOT_FILES:
        path = root / name
        if path.is_symlink():
            raise ValueError(f'release symlink: {path}')
        if path.is_file():
            selected[name] = path
    return dict(sorted(selected.items()))


def check_wheel(wheel: Path, root: Path, version: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        expected = {p.relative_to(root / 'src').as_posix(): p for p in (root / 'src/facemath').rglob('*.py')}
        actual = {name for name in names if name.startswith('facemath/') and name.endswith('.py')}
        if len(names) != len(set(names)) or actual != set(expected):
            raise ValueError('wheel Python inventory differs from source')
        for name, path in expected.items():
            if archive.read(name) != path.read_bytes():
                raise ValueError(f'wheel Python bytes differ: {name}')
        metadata = [n for n in names if n.endswith('.dist-info/METADATA')]
        if len(metadata) != 1 or Parser().parsestr(archive.read(metadata[0]).decode()).get('Version') != version:
            raise ValueError('wheel version differs from project')


def package(root: Path, wheel: Path, stage: Path, output: Path,
            evidence: list[str], *, dry_run: bool = False) -> dict:
    root, wheel, stage, output = (p.resolve() for p in (root, wheel, stage, output))
    version = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    if not re.fullmatch(r'[0-9][A-Za-z0-9.+-]*', version) or version not in output.name or output.suffix != '.zip':
        raise ValueError('output must be a versioned .zip filename containing the project version')
    sidecar = output.with_suffix('.release.json')
    for name in (*SOURCE_DIRS, *evidence):
        if any(p.is_relative_to((root / name).resolve()) for p in (stage, output, sidecar)):
            raise ValueError('release outputs must be outside selected input directories')
    if any(p.exists() for p in (stage, output, sidecar)):
        raise FileExistsError('release stage, archive and sidecar must all be new')
    paths = inventory(root, evidence)
    check_wheel(wheel, root, version)
    paths['dist/' + wheel.name] = wheel
    selection = {'version': version, 'files': sorted(paths), 'bytes': sum(p.stat().st_size for p in paths.values())}
    if dry_run:
        return dict(selection, dry_run=True)
    stage.mkdir(parents=True, exist_ok=False)
    for name, path in paths.items():
        dest = stage / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    manifest = {'format': 'facemath.release_inventory.v1', 'version': version,
                'scope': 'staged byte consistency, not external authenticity or empirical acceptance',
                'files': {name: {'bytes': (stage / name).stat().st_size, 'sha256': digest(stage / name)}
                          for name in sorted(paths)}}
    (stage / 'RELEASE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    verify(stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted([*paths, 'RELEASE_MANIFEST.json']):
            info = zipfile.ZipInfo(f'facemath-{version}/{name}', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, (stage / name).read_bytes())
    result = dict(verify(output), artifact=str(output), bytes=output.stat().st_size,
                  stage=str(stage), wheel_sha256=digest(wheel))
    with sidecar.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', action='append', default=[], help='explicit directory relative to repository root')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(package(args.root, args.wheel, args.stage, args.output, args.evidence, dry_run=args.dry_run), indent=2))


if __name__ == '__main__':
    main()

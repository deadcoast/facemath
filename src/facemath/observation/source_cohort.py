"""Prospective, pinned PTS cohorts from bounded public tar archives.

Intake authenticates local bytes against declared pins, not the external truth of
labels or image independence. Selection never sees native observation outcomes.
"""
from __future__ import annotations

from hashlib import file_digest, sha256
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import re
from urllib.parse import urlsplit
from urllib.request import urlopen

from .contracts import ObservationError
from .reference import check_digest
from .study import read_json, write_json_new

PLAN_FORMAT = "facemath.pts_cohort_plan.v12"
RECIPE_FORMAT = "facemath.pts_dataset_recipe.v12"
SPLITS = ("development", "calibration", "evaluation")
MAX_MEMBERS = 20000
MAX_MEMBER_BYTES = 128 * 1024**2
MAX_EXPANDED_BYTES = 8 * 1024**3


class _PlainTarInfo(tarfile.TarInfo):
    """Reject extension/link headers before tarfile follows or buffers them."""
    def _proc_member(self, archive):
        # CPython documents this subclass dispatch hook in tarfile's source.
        # frombuf is insufficient: newer patch releases use _frombuf internally.
        # Validate before dispatch to PAX/GNU handlers that consume extra bytes.
        if self.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
            raise ObservationError("archive links, extensions and special members are forbidden")
        if not 0 <= self.size <= MAX_MEMBER_BYTES or (self.isdir() and self.size):
            raise ObservationError("invalid archive member size")
        return super()._proc_member(archive)


def _member_name(value: str) -> str:
    from .dataset import _name
    if not isinstance(value, str) or value.startswith('/'):
        raise ObservationError("unsafe archive member path")
    parts = value.split('/')
    for part in parts:
        _name(part)
    if PurePosixPath(value).as_posix() != value:
        raise ObservationError("noncanonical archive member path")
    return value


def validate_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get('format') != PLAN_FORMAT:
        raise ObservationError("unknown PTS cohort plan")
    for key in ('name', 'source', 'label_source', 'image_terms', 'grouping_limit', 'runtime_version', 'seed'):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ObservationError(f"cohort must declare {key}")
    _member_name(plan['candidate_prefix'])
    if plan.get('selection') not in ('sha256_seed_nul_member_blocks_v1',
                                    'sha256_seed_nul_helen_source_blocks_v1',
                                    'sha256_seed_nul_ibug_source_blocks_v1'):
        raise ObservationError("unknown prospective selection algorithm")
    counts = plan.get('split_counts', {})
    evaluation_only = plan.get('evaluation_only', False)
    if type(evaluation_only) is not bool:
        raise ObservationError("evaluation_only must be boolean")
    if set(counts) != set(SPLITS) or any(type(v) is not int or not 0 <= v <= 1000 for v in counts.values()):
        raise ObservationError("declare bounded cohort split counts")
    if evaluation_only:
        if counts['development'] or counts['calibration'] or not counts['evaluation']:
            raise ObservationError("evaluation-only cohort requires zero development/calibration and positive evaluation")
        if plan.get('protocol', {}).get('bias_mode') != 'none' or 'joint_error' in plan.get('protocol', {}):
            raise ObservationError("evaluation-only cohort cannot request a fit")
    elif not all(counts.values()):
        raise ObservationError("declare positive bounded cohort split counts")
    if type(plan.get('coordinate_origin')) is not int or plan['coordinate_origin'] not in (0, 1):
        raise ObservationError("declare PTS coordinate origin 0 or 1")
    for key, limit in (('archive', 2 * 1024**3), ('model', MAX_MEMBER_BYTES)):
        pin = plan[key]
        check_digest(pin['sha256'], key)
        if type(pin.get('bytes')) is not int or not 0 < pin['bytes'] <= limit:
            raise ObservationError(f"invalid {key} byte limit")
        url = urlsplit(pin['url'])
        if url.scheme != 'https' or not url.hostname or url.username or url.password:
            raise ObservationError("cohort URLs must be public HTTPS without credentials")
    # Dataset validation enforces the common protocol and crop bounds as well.
    if 'admission_policy' in plan.get('protocol', {}):
        raise ObservationError("published cohort cannot request empirical admission")


def _check_file(path: Path, pin: dict) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != pin['bytes']:
        raise ObservationError("cohort file size/type pin mismatch")
    with path.open('rb') as stream:
        if file_digest(stream, 'sha256').hexdigest() != pin['sha256']:
            raise ObservationError("cohort file digest pin mismatch")


def _inventory_rows(rows: list) -> dict:
    if not isinstance(rows, list) or not rows or len(rows) > MAX_MEMBERS:
        raise ObservationError("invalid archive member inventory")
    files, seen, total = {}, set(), 0
    for row in rows:
        name = _member_name(row['name'])
        if name.casefold() in seen:
            raise ObservationError("duplicate or case-aliased archive member")
        seen.add(name.casefold())
        if row['type'] not in ('file', 'directory') or type(row['bytes']) is not int or not 0 <= row['bytes'] <= MAX_MEMBER_BYTES:
            raise ObservationError("invalid archive member type or size")
        if row['type'] == 'file':
            check_digest(row['sha256'], 'archive member')
            files[name] = row
            total += row['bytes']
        elif row['bytes'] != 0:
            raise ObservationError("archive directory contains data")
    if total > MAX_EXPANDED_BYTES:
        raise ObservationError("archive exceeds expanded-byte limit")
    folded_files = {n.casefold() for n in files}
    for name in files:
        if any(parent.as_posix().casefold() in folded_files
               for parent in PurePosixPath(name).parents if parent.as_posix() != '.'):
            raise ObservationError("archive file/directory collision")
    return files


def _ibug_source_key(image: str) -> str:
    path = PurePosixPath(image)
    match = re.fullmatch(r'image_([0-9]+)(?:_([0-9]+))?', path.stem)
    if match is None:
        raise ObservationError("IBUG grouping requires image_source or image_source_face filenames")
    return (path.parent / ('image_' + str(int(match[1])))).as_posix()


def select_members(plan: dict, inventory: list) -> list[dict]:
    validate_plan(plan)
    files = _inventory_rows(inventory)
    prefix = plan['candidate_prefix'] + '/'
    candidates, image_digests = [], {}
    for name in sorted(files):
        if not name.startswith(prefix) or '/' in name[len(prefix):] or not name.endswith('.pts'):
            continue
        stem = name[:-4]
        if stem.endswith('_mirror'):
            continue
        matches = [stem + ext for ext in ('.jpg', '.png', '.jpeg') if stem + ext in files]
        if len(matches) != 1:
            raise ObservationError("PTS requires exactly one original image pair")
        image = matches[0]
        digest = files[image]['sha256']
        group = _ibug_source_key(image) if plan['selection'] == 'sha256_seed_nul_ibug_source_blocks_v1' else None
        if digest in image_digests and (group is None or image_digests[digest] != group):
            raise ObservationError("duplicate original image bytes in candidate universe")
        image_digests[digest] = group
        rank = sha256((plan['seed'] + '\0' + image).encode('utf-8')).hexdigest()
        candidates.append(dict(image=image, annotation=name, rank=rank))
    # Unpaired originals cannot disappear from the prospective universe.
    paired = {r['image'] for r in candidates}
    for name in files:
        path = PurePosixPath(name)
        if (path.parent.as_posix() == plan['candidate_prefix'] and path.suffix in ('.jpg', '.png', '.jpeg')
                and not path.stem.endswith('_mirror') and name not in paired):
            raise ObservationError("original image missing PTS annotation")
    candidate_count = len(candidates)
    candidates.sort(key=lambda r: (r['rank'], r['image']))
    if plan['selection'] in ('sha256_seed_nul_helen_source_blocks_v1', 'sha256_seed_nul_ibug_source_blocks_v1'):
        # HELEN filenames retain an upstream photograph root and face suffix.
        # Rank roots independently of their number of faces, then take one
        # deterministically ranked original per root. Never split one root.
        groups = {}
        for row in candidates:
            path = PurePosixPath(row['image'])
            if plan['selection'] == 'sha256_seed_nul_helen_source_blocks_v1':
                root, separator, index = path.stem.rpartition('_')
                if not separator or not root.isascii() or not root.isdecimal() or not index.isascii() or not index.isdecimal():
                    raise ObservationError("HELEN grouping requires numeric source_face filenames")
                key = (path.parent / str(int(root))).as_posix()
            else:
                key = _ibug_source_key(row['image'])
            if key not in groups:
                groups[key] = {**row, 'source_group': key,
                    'source_group_rank': sha256((plan['seed'] + '\0' + key).encode()).hexdigest()}
        candidates = sorted(groups.values(), key=lambda r: (r['source_group_rank'], r['source_group']))
        candidates = [{**r, 'candidate_groups': len(groups)} for r in candidates]
    required = sum(plan['split_counts'].values())
    if len(candidates) < required:
        raise ObservationError("candidate inventory is smaller than frozen cohort")
    selected, offset = [], 0
    for split in SPLITS:
        count = plan['split_counts'][split]
        for row in candidates[offset:offset + count]:
            selected.append({**row, 'split': split, 'candidate_count': candidate_count})
        offset += count
    return selected


def make_recipe(plan: dict, inventory: list) -> dict:
    selected = select_members(plan, inventory)
    files = {r['name']: r for r in inventory if r['type'] == 'file'}
    assets, images, annotations = [], [], []
    for index, row in enumerate(selected):
        names = {}
        for role in ('image', 'annotation'):
            member = files[row[role]]
            name = f'source_{index:05d}' + PurePosixPath(row[role]).suffix
            names[role] = name
            assets.append(dict(path=name, bytes=member['bytes'], sha256=member['sha256'],
                               url=plan['archive']['url'], source_member=row[role]))
        group = 'source_' + sha256(row.get('source_group', row['image']).encode()).hexdigest()
        images.append(dict(path=names['image'], annotation=names['annotation'], group_id=group,
                           lineage_ids=[group], split=row['split']))
        annotations.append(names['annotation'])
    assets.append(dict(path='face_landmarker.task', **plan['model']))
    return dict(format=RECIPE_FORMAT, label_kind='external_dataset', model_asset='face_landmarker.task',
        **{key: plan[key] for key in ('name', 'source', 'label_source', 'image_terms', 'grouping_limit',
                                     'runtime_version', 'coordinate_origin', 'box_margin', 'protocol')},
        assets=assets, images=images, annotations=annotations,
        cohort={'plan': plan, 'inventory': sorted(inventory, key=lambda r: r['name']), 'selection': selected})


def verify_recipe(recipe: dict) -> None:
    provenance = recipe['cohort']
    expected = make_recipe(provenance['plan'], provenance['inventory'])
    if recipe != expected:
        raise ObservationError("cohort recipe differs from frozen selection, pins or protocol")


def fetch_cohort(plan_path: str | Path, destination: str | Path) -> Path:
    """Explicit network boundary; failed downloads remain at the new destination."""
    plan_path, root = Path(plan_path).resolve(), Path(destination).resolve()
    plan = read_json(plan_path); validate_plan(plan)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, root/'plan.json')
    receipts = []
    try:
        for key, name in (('archive', 'source.tar.gz'), ('model', 'face_landmarker.task')):
            pin = plan[key]
            with urlopen(pin['url'], timeout=90) as response, (root/name).open('xb') as output:
                if urlsplit(response.url).scheme != 'https':
                    raise ObservationError("cohort download redirected away from HTTPS")
                count = 0
                while block := response.read(min(1024**2, pin['bytes'] + 1 - count)):
                    output.write(block); count += len(block)
                    if count > pin['bytes']:
                        raise ObservationError("cohort download exceeds byte pin")
                final_url = response.url
            _check_file(root/name, pin)
            receipts.append(dict(asset=key, final_url=final_url, **pin))
    except Exception as exc:
        write_json_new(root/'fetch.failure.json', dict(completed=False, verified_downloads=receipts,
                       exception=type(exc).__name__, detail=str(exc)))
        raise
    write_json_new(root/'fetch.json', dict(completed=True, assets=receipts))
    return root


def prepare_cohort(plan_path: str | Path, archive_path: str | Path,
                   model_path: str | Path, destination: str | Path) -> Path:
    """Two bounded streaming passes. Output retains failures; no archive extraction."""
    from .dataset import prepare_dataset
    plan_path, archive_path, model_path, root = map(lambda p: Path(p).resolve(),
        (plan_path, archive_path, model_path, destination))
    plan = read_json(plan_path); validate_plan(plan)
    _check_file(archive_path, plan['archive']); _check_file(model_path, plan['model'])
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, root/'plan.json')
    try:
        inventory, seen, expanded = [], set(), 0
        with tarfile.open(archive_path, 'r|gz', tarinfo=_PlainTarInfo) as archive:
            for member in archive:
                name = _member_name(member.name)
                if name.casefold() in seen or len(seen) >= MAX_MEMBERS:
                    raise ObservationError("duplicate archive member or member limit exceeded")
                seen.add(name.casefold())
                if not member.isfile() and not member.isdir():
                    raise ObservationError("archive links and special members are forbidden")
                if member.size < 0 or member.size > MAX_MEMBER_BYTES or (member.isdir() and member.size):
                    raise ObservationError("invalid archive member size")
                expanded += member.size
                if expanded > MAX_EXPANDED_BYTES:
                    raise ObservationError("archive exceeds expanded-byte limit")
                row = dict(name=name, bytes=member.size, type='file' if member.isfile() else 'directory')
                if member.isfile():
                    with archive.extractfile(member) as stream:
                        row['sha256'] = file_digest(stream, 'sha256').hexdigest()
                inventory.append(row)
        recipe = make_recipe(plan, inventory)
        # Validate the common dataset protocol before extracting any selected inputs.
        intake = root/'intake'; intake.mkdir()
        write_json_new(intake/'recipe.json', recipe)
        from .dataset import _recipe
        _recipe(intake/'recipe.json')
        wanted = {a['source_member']: a for a in recipe['assets'] if 'source_member' in a}
        with tarfile.open(archive_path, 'r|gz', tarinfo=_PlainTarInfo) as archive:
            for member in archive:
                if member.name in wanted:
                    asset = wanted.pop(member.name)
                    if not member.isfile() or member.size != asset['bytes']:
                        raise ObservationError("archive changed during cohort intake")
                    with archive.extractfile(member) as source, (intake/asset['path']).open('xb') as output:
                        shutil.copyfileobj(source, output, length=1024**2)
                    _check_file(intake/asset['path'], asset)
        if wanted:
            raise ObservationError("selected archive members disappeared")
        _check_file(archive_path, plan['archive'])
        shutil.copyfile(model_path, intake/'face_landmarker.task')
        result = prepare_dataset(intake/'recipe.json', root/'dataset')
    except Exception as exc:
        write_json_new(root/'intake.failure.json', dict(completed=False, exception=type(exc).__name__, detail=str(exc)))
        raise
    write_json_new(root/'intake.json', dict(completed=True, archive=plan['archive'],
                   members=len(inventory), selected=len(recipe['images']), inference_executed=False))
    return result

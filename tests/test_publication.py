import json
import os
from pathlib import Path
import subprocess
import shutil

import pytest

from facemath.observation.publication import publication_stage


def test_collision_does_not_open_or_overwrite_existing_directory(tmp_path, monkeypatch):
    import facemath.observation.publication as mod
    collision = tmp_path/'.result-collision'; collision.mkdir()
    (collision/'owned.txt').write_text('retained')
    tokens = iter(('collision', 'fresh'))
    monkeypatch.setattr(mod, 'token_hex', lambda n: next(tokens))
    stage = publication_stage(tmp_path, '.result-')
    assert stage == tmp_path/'.result-fresh' and stage.is_dir()
    assert (collision/'owned.txt').read_text() == 'retained'


def test_windows_published_output_inherits_parent_access(tmp_path):
    stage = publication_stage(tmp_path, '.result-')
    (stage/'result.json').write_text('{"value": 1}')
    target = tmp_path/'published'; stage.rename(target)
    if os.name == 'nt':
        # Query the actual DACL, not a mock of mkdir or a synthetic permission fixture.
        script = "$ErrorActionPreference='Stop'; $a=Get-Acl -LiteralPath $env:FACEMATH_ACL_TARGET; [pscustomobject]@{protected=$a.AreAccessRulesProtected;inherited=@($a.Access | Where-Object IsInherited).Count} | ConvertTo-Json -Compress"
        result = subprocess.run([shutil.which('pwsh') or 'powershell', '-NoProfile', '-Command', script],
            env={**os.environ, 'FACEMATH_ACL_TARGET': str(target)}, capture_output=True, text=True, check=True)
        acl = json.loads(result.stdout)
        assert acl['protected'] is False and acl['inherited'] > 0
    else:
        sibling = tmp_path/'ordinary'; sibling.mkdir()
        assert stage.exists() is False
        assert target.stat().st_mode == sibling.stat().st_mode
    assert json.loads((target/'result.json').read_text()) == {'value': 1}


@pytest.mark.parametrize('prefix', ['', '../escape', 'a/b', 'a\\b'])
def test_prefix_cannot_escape_destination_parent(tmp_path, prefix):
    with pytest.raises(ValueError): publication_stage(tmp_path, prefix)

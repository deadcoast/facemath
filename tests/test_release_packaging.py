"""Distribution selection tests; fixture wheels are not runtime qualification."""
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest


@pytest.fixture
def packager(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('facemath_release_test',scripts/'package_release.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path):
    root=tmp_path/'project';root.mkdir()
    (root/'pyproject.toml').write_text('[project]\nname="facemath"\nversion="0.10.1"\n')
    (root/'src/facemath').mkdir(parents=True)
    (root/'src/facemath/__init__.py').write_bytes(b'__version__ = "0.10.1"\n')
    for name in ('.venv/secret','.uv-cache/secret','.commit-audit/secret','runs/secret',
                 'packages/old.zip','dist/old.whl','src/facemath.egg-info/PKG-INFO',
                 'tests/__pycache__/cached.pyc','tests/.venv/secret','tests/real.py',
                 'evidence/current/acquisition/.runner.lock','evidence/current/workspace/.workspace-writer.lock',
                 'evidence/old/historical.json'):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
    wheel=tmp_path/'facemath-0.10.1-py3-none-any.whl'
    with zipfile.ZipFile(wheel,'w') as z:
        z.write(root/'src/facemath/__init__.py','facemath/__init__.py')
        z.writestr('facemath-0.10.1.dist-info/METADATA','Name: facemath\nVersion: 0.10.1\n')
    return root,wheel,tmp_path/'stage',tmp_path/'facemath-0.10.1.zip'


def test_explicit_selection_dry_run_and_sealed_extraction(packager,project,tmp_path):
    root,wheel,stage,out=project
    plan=packager.package(*project,['evidence/current'],dry_run=True)
    assert not stage.exists() and not out.exists()
    assert set(plan['files'])=={'pyproject.toml','src/facemath/__init__.py','tests/real.py',
                              'evidence/current/acquisition/.runner.lock',
                              'evidence/current/workspace/.workspace-writer.lock','dist/'+wheel.name}
    result=packager.package(*project,['evidence/current'])
    assert result['verified'] and packager.verify(out)['verified']
    with zipfile.ZipFile(out) as z:z.extractall(tmp_path/'extracted')
    assert packager.verify(tmp_path/'extracted/facemath-0.10.1')['verified']
    assert (stage/'evidence/current/acquisition/.runner.lock').read_bytes()==b'fixture'
    before=out.read_bytes()
    with pytest.raises(FileExistsError):packager.package(*project,['evidence/current'])
    assert out.read_bytes()==before


@pytest.mark.parametrize('existing',['archive','sidecar','stage'])
def test_existing_outputs_are_never_replaced(packager,project,existing):
    root,wheel,stage,out=project
    target={'archive':out,'sidecar':out.with_suffix('.release.json'),'stage':stage}[existing]
    target.write_bytes(b'keep')
    with pytest.raises(FileExistsError):packager.package(*project,[])
    assert target.read_bytes()==b'keep'


def test_wrong_wheel_and_unsafe_selection_fail_before_stage(packager,project):
    root,wheel,stage,out=project
    with pytest.raises(ValueError,match='evidence'):packager.package(*project,['../outside'])
    with pytest.raises(ValueError,match='outside selected'):
        packager.package(root,wheel,root/'tests/stage',out,[])
    (root/'src/facemath/__init__.py').write_bytes(b'changed source')
    with pytest.raises(ValueError,match='bytes differ'):packager.package(*project,[])
    assert not stage.exists() and not out.exists()


def test_wheel_metadata_accepts_standard_crlf_headers(packager,project,tmp_path):
    root,wheel,stage,out=project
    windows_wheel=tmp_path/'windows-metadata.whl'
    with zipfile.ZipFile(wheel) as original,zipfile.ZipFile(windows_wheel,'w') as changed:
        for name in original.namelist():
            data=original.read(name)
            if name.endswith('/METADATA'):data=data.replace(b'\n',b'\r\n')
            changed.writestr(name,data)
    assert packager.package(root,windows_wheel,stage,out,[],dry_run=True)['version']=='0.10.1'

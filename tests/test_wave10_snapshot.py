from io import BytesIO
from pathlib import Path
import stat
import struct
import zipfile
import warnings
import pytest

from facemath.observation.deployment import pack_snapshot,unpack_snapshot
from facemath.observation.contracts import ObservationError


def archive(names,symlink=False):
    out=BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with zipfile.ZipFile(out,'w') as z:
            for name in names:
                i=zipfile.ZipInfo(name)
                # Bypass ZipInfo's constructor normalization when writing this
                # intentionally malformed fixture; inspect both raw headers below.
                i.filename=name
                if symlink:i.external_attr=(stat.S_IFLNK|0o777)<<16
                z.writestr(i,b'original raw bytes')
    data=out.getvalue()
    with zipfile.ZipFile(BytesIO(data)) as z:
        central=z.start_dir
        for name,info in zip(names,z.infolist()):
            encoded=name.encode('utf-8')
            offset=info.header_offset
            size=struct.unpack_from('<H',data,offset+26)[0]
            assert data[offset+30:offset+30+size]==encoded
            size,extra,comment=struct.unpack_from('<HHH',data,central+28)
            assert data[central+46:central+46+size]==encoded
            assert info.orig_filename==name
            central+=46+size+extra+comment
    return data


@pytest.mark.parametrize('name',['../escape','/escape','a/../../escape','a//b','a/./b','a\\b','a\x00b','C:/a','a:ads','file.','dir /file','CON','nul.txt','a/COM1.log','a/LPT9','folder/'])
def test_unsafe_portable_archive_names_rejected(tmp_path,name):
    with pytest.raises(ObservationError):unpack_snapshot(archive(['valid/first',name]),tmp_path)
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('names',[['a','a/b'],['A/b','a']])
def test_file_directory_collision_rejected_before_writing(tmp_path,names):
    with pytest.raises(ObservationError,match='collision'):unpack_snapshot(archive(names),tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_benign_slash_path_is_not_malformed(tmp_path):
    unpack_snapshot(archive(['a/b']),tmp_path)
    assert (tmp_path/'a/b').read_bytes()==b'original raw bytes'


@pytest.mark.parametrize('names',[['same','same'],['same','SAME']])
def test_duplicate_casefold_paths_rejected(tmp_path,names):
    with pytest.raises(ObservationError):unpack_snapshot(archive(names),tmp_path)


def test_symlink_entry_rejected(tmp_path):
    with pytest.raises(ObservationError):unpack_snapshot(archive(['link'],True),tmp_path)


def test_invalid_zip_and_wrong_byte_type_rejected(tmp_path):
    for data in (b'not a zip','not bytes'):
        with pytest.raises(ObservationError):unpack_snapshot(data,tmp_path)


def test_snapshot_exact_bytes_deterministic_and_no_overwrite(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'a').mkdir()
    (source/'a/data.bin').write_bytes(bytes(range(256)));(source/'report.json').write_text('{"v":1}')
    a=pack_snapshot(source);b=pack_snapshot(source);assert a==b
    out=tmp_path/'out';out.mkdir();unpack_snapshot(a,out)
    assert (out/'a/data.bin').read_bytes()==(source/'a/data.bin').read_bytes()
    with pytest.raises(ObservationError):unpack_snapshot(a,out)


def test_extraction_budget_rejects_before_writing(tmp_path,monkeypatch):
    import facemath.observation.deployment as mod
    monkeypatch.setattr(mod,'MAX_FILES',1)
    with pytest.raises(ObservationError,match='budget'):unpack_snapshot(archive(['a','b']),tmp_path)
    assert list(tmp_path.iterdir())==[]

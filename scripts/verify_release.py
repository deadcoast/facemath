"""Standard-library verification of a released archive or extracted folder.

This checks the shipped inventory, not external authorship or semantic validity.
Run facemath verify/replay for numerical evidence. No archive extraction or pickle
loading is performed by the ZIP route.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path,PurePosixPath
import stat
import zipfile


def verify(path:Path)->dict:
    if path.is_file():
        with zipfile.ZipFile(path) as z:
            names=z.namelist()
            candidates=[n for n in names if n.endswith('/RELEASE_MANIFEST.json')]
            if len(candidates)!=1:raise ValueError('expected one release inventory')
            manifest_name=candidates[0];prefix=manifest_name[:-len('RELEASE_MANIFEST.json')]
            records={}
            for info in z.infolist():
                name=info.filename
                if (name in records or '\\' in name or '..' in PurePosixPath(name).parts
                        or not name.startswith(prefix) or PurePosixPath(name).is_absolute()
                        or stat.S_ISLNK(info.external_attr>>16) or info.is_dir()):
                    raise ValueError('unsafe or duplicate release member')
                records[name]=info
            manifest=json.loads(z.read(manifest_name))
            expected={prefix+n for n in manifest['files']}|{manifest_name}
            if set(records)!=expected:raise ValueError('release file inventory differs')
            for name,entry in manifest['files'].items():
                data=z.read(prefix+name)
                if len(data)!=entry['bytes'] or sha256(data).hexdigest()!=entry['sha256']:
                    raise ValueError('release member changed: '+name)
            result={'verified':True,'files':len(manifest['files']),'version':manifest['version'],
                'archive_sha256':sha256(path.read_bytes()).hexdigest(),'kind':'archive'}
    else:
        manifest=json.loads((path/'RELEASE_MANIFEST.json').read_text())
        for name,entry in manifest['files'].items():
            p=path/name
            if p.is_symlink() or not p.resolve().is_relative_to(path.resolve()):raise ValueError('unsafe release path')
            data=p.read_bytes()
            if len(data)!=entry['bytes'] or sha256(data).hexdigest()!=entry['sha256']:
                raise ValueError('release member changed: '+name)
        result={'verified':True,'files':len(manifest['files']),'version':manifest['version'],'kind':'extracted_files'}
    result['scope']='shipped-byte consistency, not external authenticity or numerical replay'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path',type=Path,nargs='?',default=Path('.'));a=p.parse_args()
    print(json.dumps(verify(a.path),indent=2))

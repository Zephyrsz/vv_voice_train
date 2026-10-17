"""Pinned author-source snapshots, with safe extraction and per-file receipts."""
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from .common import resolve, sha256, write_json


def _lock(cfg):
    lock = json.loads(resolve(cfg, cfg.get('upstream_lock', 'upstream.lock.json')).read_text())
    if lock.get('schema_version') != 1 or not lock.get('sources'):
        raise ValueError('Invalid upstream lock')
    for name, source in lock['sources'].items():
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name) or not re.fullmatch(r'[0-9a-f]{40}', source['revision']):
            raise ValueError('Source name or locked revision is invalid')
        if not re.fullmatch(r'https://github.com/[\w.-]+/[\w.-]+', source['repository']):
            raise ValueError('Source repository must be an explicit GitHub author repository')
    return lock


def _patches(cfg, source):
    values=[]
    for patch in source.get('patches', []):
        path=resolve(cfg, patch['path'])
        digest=sha256(path)
        if digest != patch['sha256']:
            raise ValueError(f'Patch sha256 mismatch: {path}')
        values.append({'path':patch['path'],'sha256':digest})
    return values


def _inventory(root):
    # Python imports may create caches after fetching; caches are not source files.
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != '.source-receipt.json' and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def inspect_sources(cfg):
    vendor=resolve(cfg,cfg['paths']['vendor']); result={}
    for name, source in _lock(cfg)['sources'].items():
        root=vendor/name
        receipt=json.loads((root/'.source-receipt.json').read_text())
        if receipt.get('revision') != source['revision'] or receipt.get('repository') != source['repository']:
            raise ValueError(f'{name}: locked source revision/repository mismatch')
        if receipt.get('patches',[]) != _patches(cfg,source):
            raise ValueError(f'{name}: source patch mismatch')
        if _inventory(root) != receipt.get('files'):
            raise ValueError(f'{name}: source file hash/inventory mismatch')
        result[name]={'path':str(root),'revision':source['revision'],'files_count':len(receipt['files']),'patches':receipt.get('patches',[])}
    return {'status':'source-ready','sources':result}


def fetch_sources(cfg):
    vendor=resolve(cfg,cfg['paths']['vendor']);vendor.mkdir(parents=True,exist_ok=True)
    lock=_lock(cfg)
    for name, source in lock['sources'].items():
        destination=vendor/name
        if destination.exists():
            # Never overwrite an existing modified checkout.
            one=dict(cfg)
            receipt=json.loads((destination/'.source-receipt.json').read_text())
            if receipt.get('revision')!=source['revision'] or receipt.get('repository')!=source['repository'] or receipt.get('patches',[])!=_patches(cfg,source) or receipt.get('files')!=_inventory(destination):
                raise ValueError(f'{name}: existing snapshot revision/hash mismatch; move it aside explicitly')
            continue
        owner_repo=source['repository'].removeprefix('https://github.com/')
        url=f'https://codeload.github.com/{owner_repo}/tar.gz/{source["revision"]}'
        with urllib.request.urlopen(url,timeout=90) as response:
            archive=response.read(100*1024*1024+1)
        if len(archive)>100*1024*1024:raise ValueError('Source archive exceeds 100 MiB')
        with tempfile.TemporaryDirectory(prefix=f'.{name}-',dir=vendor) as temporary:
            root=Path(temporary)/'source';root.mkdir()
            with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
                expected=f'{owner_repo.split("/")[-1]}-{source["revision"]}'
                for member in tar.getmembers():
                    parts=PurePosixPath(member.name).parts
                    if not parts or parts[0]!=expected or '..' in parts or PurePosixPath(member.name).is_absolute():
                        raise ValueError('Archive root does not match locked revision or unsafe path')
                    if member.isdir():continue
                    if not member.isfile():raise ValueError('Source archive contains unsupported link/device')
                    target=root.joinpath(*parts[1:]);target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
            pristine=_inventory(root)
            patches=_patches(cfg,source)
            for patch in patches:
                subprocess.run(['patch','--batch','-p1','-i',str(resolve(cfg,patch['path']))],cwd=root,check=True,capture_output=True,text=True)
            write_json(root/'.source-receipt.json',{'schema_version':1,'repository':source['repository'],'revision':source['revision'], 'archive_sha256':hashlib.sha256(archive).hexdigest(),'pristine_files':pristine,'files':_inventory(root),'patches':patches})
            shutil.move(str(root),str(destination))
    return inspect_sources(cfg)

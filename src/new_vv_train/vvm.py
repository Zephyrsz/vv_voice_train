"""Talk-only VVM format 2 assembly and defensive archive inspection."""
from __future__ import annotations
import json
import os
from pathlib import Path, PurePosixPath
import stat
import uuid
import zipfile

from .common import resolve, sha256, write_json

MODELS = ('predict_duration', 'predict_intonation', 'decode')


def _uint(value, label):
    if type(value) is not int or not 0 <= value <= 2**32-1:
        raise ValueError(f'{label} must be a uint32')
    return value


def _uuid(value, label):
    try: return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError) as e: raise ValueError(f'Invalid {label} UUID: {value}') from e


def _safe(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or '..' in p.parts or '.' in p.parts or str(p) != name:
        raise ValueError(f'Unsafe archive path: {name}')
    return name


def inspect_vvm(path):
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)): raise ValueError('Duplicate archive file paths')
        for i in infos:
            _safe(i.filename)
            if stat.S_ISLNK(i.external_attr >> 16): raise ValueError('Archive path is a symlink')
            if i.file_size > 4*1024**3: raise ValueError('Archive member too large')
        def read_json(name):
            _safe(name)
            if name not in names: raise ValueError(f'Missing archive file: {name}')
            if z.getinfo(name).file_size > 4*1024**2: raise ValueError('Metadata too large')
            return json.loads(z.read(name))
        m = read_json('manifest.json')
        if m.get('vvm_format_version') != 2: raise ValueError('Only VVM format 2 is supported')
        model_id = _uuid(m.get('id'), 'model')
        metas = read_json(m.get('metas_filename', ''))
        if not isinstance(metas, list) or not metas: raise ValueError('Empty speaker metadata')
        domains = set(m) & {'streaming_talk','singing_teacher','frame_decode'}
        if domains or 'talk' not in m: raise ValueError('Candidate must contain only talk domain')
        talk = m['talk']
        filenames = []
        for model in MODELS:
            entry = talk.get(model, {})
            name = _safe(entry.get('filename',''))
            if entry.get('type') != 'onnx' or name not in names or not z.getinfo(name).file_size:
                raise ValueError(f'Missing or invalid ONNX file: {model}')
            filenames.append(name)
        if len(set(filenames)) != 3: raise ValueError('ONNX file paths must be distinct')
        ids, speaker_ids = [], []
        for speaker in metas:
            speaker_id = _uuid(speaker.get('speaker_uuid'), 'speaker')
            if speaker_id == model_id or speaker_id in speaker_ids: raise ValueError('Duplicate model/speaker UUID')
            speaker_ids.append(speaker_id)
            if not speaker.get('name') or not isinstance(speaker.get('version'), str): raise ValueError('Invalid speaker metadata')
            if not speaker.get('styles'): raise ValueError('Speaker has no styles')
            for style in speaker['styles']:
                style_id = _uint(style.get('id'), 'public style ID')
                if style_id in ids: raise ValueError(f'Duplicate public style ID: {style_id}')
                if style.get('type', 'talk') != 'talk' or not style.get('name'): raise ValueError('Invalid talk style')
                ids.append(style_id)
        mapping = talk.get('style_id_to_inner_voice_id')
        if not isinstance(mapping, dict) or set(mapping) != {str(i) for i in ids}:
            raise ValueError('Style mapping must exactly match metadata style IDs')
        for k, v in mapping.items(): _uint(v, 'inner voice ID')
        return {'status': 'static-validated', 'path': str(path.resolve()), 'sha256': sha256(path),
                'format_version': 2, 'model_uuid': model_id, 'speaker_uuids': speaker_ids,
                'style_ids': sorted(ids), 'mapping': mapping, 'manifest': m, 'metas': metas}


def existing_style_ids(directory):
    """Existing product archives may contain other domains; read metadata only."""
    result = {}
    directory = Path(directory)
    if not directory.is_dir(): raise FileNotFoundError(f'Existing model directory missing: {directory}')
    for path in sorted(directory.glob('*.vvm')):
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read('manifest.json'))
            name = _safe(manifest['metas_filename'])
            if z.getinfo(name).file_size > 4*1024**2: raise ValueError('Existing VVM metadata too large')
            for speaker in json.loads(z.read(name)):
                for style in speaker['styles']:
                    i = _uint(style['id'], 'existing style ID')
                    if i in result: raise ValueError(f'Duplicate existing public style ID: {i}')
                    result[i] = str(path)
    return result


def package(cfg):
    from .export import validate_onnx
    run = resolve(cfg, cfg['paths']['run'])
    receipt_path = run/'onnx/export_receipt.json'
    if not receipt_path.is_file(): raise FileNotFoundError(f'Missing export receipt: {receipt_path}; run export first')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('sources', {}).get('status') != 'source-ready' or not receipt.get('native_artifacts'):
        raise ValueError('Export receipt requires verified author sources and native artifact hashes')
    for name, digest in receipt['native_artifacts'].items():
        _safe(name)
        artifact=(run/name).resolve()
        if not artifact.is_relative_to(run.resolve()) or not artifact.is_file() or sha256(artifact)!=digest:
            raise ValueError(f'Export native artifact changed or missing: {name}')
    profile_path = resolve(cfg, cfg['feature_profile'])
    if receipt.get('status') != 'exported' or receipt.get('profile_sha256') != sha256(profile_path):
        raise ValueError('Export receipt/profile mismatch; re-export actual trained models')
    for name in MODELS:
        p = run/'onnx'/f'{name}.onnx'
        if not p.is_file() or receipt.get('files', {}).get(name) != sha256(p):
            raise ValueError(f'Export receipt hash mismatch or missing file: {name}')
    validation = validate_onnx(cfg)
    voices_path = resolve(cfg, cfg['paths']['corpus'])/'speakers.json'
    voices = json.loads(voices_path.read_text())['voices']
    if not voices: raise ValueError('Voice registry is empty')
    public_ids = [_uint(v['public_style_id'], 'public style ID') for v in voices]
    if len(set(public_ids)) != len(public_ids): raise ValueError('Duplicate public style ID in voice registry')
    for v in voices:
        i = _uint(v['inner_voice_id'], 'inner voice ID')
        if i >= receipt.get('speaker_size',0): raise ValueError(f'Inner voice ID {i} outside exported speaker table')
    existing = cfg.get('core',{}).get('existing_models')
    if existing:
        collisions = set(public_ids) & set(existing_style_ids(resolve(cfg, existing)))
        if collisions: raise ValueError(f'Public style ID collision with existing models: {sorted(collisions)}')
    output = run/'vvm'
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output/'identity.json'
    options = cfg.get('package',{})
    if identity_path.exists(): identity = json.loads(identity_path.read_text())
    else:
        identity = {'model_uuid': options.get('model_uuid', str(uuid.uuid4())),
                    'speaker_uuids': {v['speaker_key']: options.get('speaker_uuids',{}).get(v['speaker_key'], str(uuid.uuid4())) for v in voices}}
        write_json(identity_path, identity)
    groups = {}
    for v in voices:
        key=v['speaker_key']
        if key not in groups:
            groups[key]={'name':v['name'], 'speaker_uuid':identity['speaker_uuids'][key],
                         'version':options.get('version','0.0.1'), 'styles':[]}
        groups[key]['styles'].append({'name':v['style_name'], 'id':v['public_style_id'], 'type':'talk'})
    manifest={'vvm_format_version':2,'id':identity['model_uuid'],'metas_filename':'metas.json',
              'talk': {n:{'type':'onnx','filename':n+'.onnx'} for n in MODELS}}
    manifest['talk']['style_id_to_inner_voice_id']={str(v['public_style_id']):v['inner_voice_id'] for v in voices}
    path=output/'candidate.vvm'
    temp=output/'candidate.vvm.tmp'
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED) as z:
            z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False))
            z.writestr('metas.json',json.dumps(list(groups.values()),ensure_ascii=False))
            for n in MODELS: z.write(run/'onnx'/f'{n}.onnx',n+'.onnx')
        result=inspect_vvm(temp)
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)
    result.update(path=str(path), export_receipt_sha256=sha256(receipt_path),
                  onnx_validation=validation, status='packaged-static-validated')
    write_json(output/'package_receipt.json',result)
    return result

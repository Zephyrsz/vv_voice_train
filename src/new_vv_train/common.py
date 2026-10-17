"""Shared paths, profiles and reproducible JSON receipts."""

import hashlib
import json
import math
import os
from pathlib import Path

import yaml


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def resolve(cfg, value):
    if value is None:
        raise ValueError('Missing required path')
    p = Path(value).expanduser()
    return (p if p.is_absolute() else Path(cfg['_root']) / p).resolve()


def interpreter_path(cfg, value):
    """Preserve a virtualenv's python symlink when launching child processes."""
    p = Path(value).expanduser()
    return Path(os.path.abspath(p if p.is_absolute() else Path(cfg['_root']) / p))


def load_project(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict):
        raise ValueError('Project config must be an object')
    root = Path(cfg.get('root', '..'))
    cfg['_root'] = str((path.parent / root).resolve())
    cfg['_config_path'] = str(path)
    cfg['profile'] = json.loads(resolve(cfg, cfg['feature_profile']).read_text(encoding='utf-8'))
    validate_profile(cfg['profile'])
    return cfg


def validate_profile(profile):
    if profile.get('sample_rate') != 24000 or profile.get('hop_length') != 256:
        raise ValueError('Core talk requires sample_rate=24000 and hop_length=256')
    phonemes = profile.get('phonemes', [])
    if not phonemes or len(set(phonemes)) != len(phonemes) or phonemes[0] != 'pau':
        raise ValueError('Invalid ordered phoneme vocabulary')
    if profile.get('f0', {}).get('representation', 'ln_hz_zero_unvoiced') != 'ln_hz_zero_unvoiced':
        raise ValueError('F0 must be ln_hz_zero_unvoiced')
    spec = profile.get('spectrogram', {})
    if spec and (spec.get('log', 'natural') != 'natural' or spec.get('center', False)):
        raise ValueError('Use author HiFi-GAN natural-log spectrum with center=false')
    for key, val in profile.get('f0', {}).items():
        if isinstance(val, float) and not math.isfinite(val):
            raise ValueError(f'Invalid profile field {key}')


def profile_hash(cfg):
    return hashlib.sha256(json.dumps(cfg['profile'], sort_keys=True, allow_nan=False).encode()).hexdigest()


def record(cfg, stage, report):
    result = dict(report, stage=stage, profile_sha256=profile_hash(cfg))
    write_json(resolve(cfg, cfg['paths']['run']) / 'reports' / f'{stage}.json', result)
    return result

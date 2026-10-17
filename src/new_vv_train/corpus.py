"""Validate source recordings and produce non-destructive 24 kHz copies."""

import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .common import record, resolve, sha256, write_json


def init_corpus(cfg):
    root = resolve(cfg, cfg['paths']['corpus'])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Corpus directory already contains files: {root}')
    for folder in ('raw', 'splits', 'alignment_input', 'corrections/full_context'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    example = dict(utterance_id='newvoice_000001', audio='raw/newvoice_000001.wav', text='今日はいい天気ですね。', normalized_text='今日はいい天気ですね。', reading='キョーワイーテンキデスネ', speaker_key='new_voice', style_key='neutral', session_id='session_001', language='ja')
    (root / 'manifest.example.jsonl').write_text(json.dumps(example, ensure_ascii=False) + '\n', encoding='utf-8')
    (root / 'manifest.jsonl').touch()
    for split in ('train', 'valid', 'test'):
        (root / 'splits' / f'{split}.txt').touch()
    write_json(root / 'speakers.json', dict(schema_version=1, voices=[dict(speaker_key='new_voice', style_key='neutral', inner_voice_id=0, public_style_id=1000000, name='新しい声', style_name='ノーマル')]))
    write_json(root / 'dataset_info.json', dict(schema_version=1, purpose='personal_research', source='self_recorded', recorded=False, note='Record actual source, microphone/session information and voice authorization here.'))
    return dict(status='empty-corpus-created', path=str(root), next='Add real WAVs and manifest entries, then assign every ID to one split. See DATA_FORMAT.md.')


def read_manifest(cfg):
    root = resolve(cfg, cfg['paths']['corpus'])
    rows = []
    for line in (root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError('Manifest rows must be objects')
        uid = row.get('utterance_id', '')
        if not isinstance(uid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', uid):
            raise ValueError(f'Invalid utterance_id {uid!r}')
        path = (root / row['audio']).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Audio is outside corpus: {uid}')
        if not row.get('text', '').strip() or row.get('language') != 'ja':
            raise ValueError(f'Nonempty Japanese text required: {uid}')
        rows.append(row)
    if not rows:
        raise ValueError('Corpus has no recordings')
    if len({r['utterance_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate utterance_id')
    return root, rows


def read_voices(cfg):
    path = resolve(cfg, cfg['paths']['corpus']) / 'speakers.json'
    voices = json.loads(path.read_text(encoding='utf-8'))['voices']
    if not voices:
        raise ValueError('Empty voice registry')
    keys = [(v['speaker_key'], v['style_key']) for v in voices]
    inner = [v['inner_voice_id'] for v in voices]
    public = [v['public_style_id'] for v in voices]
    if len(set(keys)) != len(keys) or len(set(inner)) != len(inner) or len(set(public)) != len(public):
        raise ValueError('Duplicate voice keys/IDs')
    if sorted(inner) != list(range(len(inner))) or any(type(v) is not int or not 0 <= v <= 0xFFFFFFFF for v in public):
        raise ValueError('Invalid voice ID ranges; inner IDs must be contiguous from 0')
    return voices


def read_splits(root, rows):
    known = {r['utterance_id'] for r in rows}
    result, seen = {}, set()
    for split in ('train', 'valid', 'test'):
        ids = (root / 'splits' / f'{split}.txt').read_text(encoding='utf-8').split()
        if not ids or len(ids) != len(set(ids)) or set(ids) - known:
            raise ValueError(f'Empty/duplicate/unknown IDs in {split}')
        if seen.intersection(ids):
            raise ValueError('ID appears in multiple splits')
        seen.update(ids)
        result[split] = ids
    if seen != known:
        raise ValueError('Every utterance must have exactly one split')
    return result


def validate_corpus(cfg):
    root, rows = read_manifest(cfg)
    voices = read_voices(cfg)
    keys = {(v['speaker_key'], v['style_key']) for v in voices}
    splits = read_splits(root, rows)
    split_by_id = {uid: split for split, ids in splits.items() for uid in ids}
    hashes, texts, source_groups = {}, {}, {}
    items = []
    for row in rows:
        uid = row['utterance_id']
        if (row.get('speaker_key'), row.get('style_key')) not in keys:
            raise ValueError(f'Unknown voice: {uid}')
        path = root / row['audio']
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        if not len(audio) or not np.isfinite(audio).all() or rate < 8000:
            raise ValueError(f'Empty/non-finite/invalid WAV: {uid}')
        h = sha256(path)
        if h in hashes:
            raise ValueError(f'duplicate audio: {uid} and {hashes[h]}')
        hashes[h] = uid
        # Normalized text prevents trivial whitespace differences hiding leakage.
        text = re.sub(r'\s+', '', row.get('normalized_text', row['text']))
        if text in texts and texts[text] != split_by_id[uid]:
            raise ValueError(f'Duplicate text across splits: {uid}')
        texts[text] = split_by_id[uid]
        source = row.get('source_recording_id')
        if source:
            if source in source_groups and source_groups[source] != split_by_id[uid]:
                raise ValueError(f'Same source recording across splits: {source}')
            source_groups[source] = split_by_id[uid]
        peak = float(np.max(np.abs(audio)))
        if peak >= 0.999:
            raise ValueError(f'Clipped recording requires review: {uid}')
        if peak == 0:
            raise ValueError(f'All-silent recording: {uid}')
        items.append(dict(utterance_id=uid, sha256=h, seconds=len(audio) / rate, sample_rate=rate, channels=audio.shape[1], peak=peak, split=split_by_id[uid]))
    return record(cfg, 'validate-corpus', dict(status='validated', utterances=len(items), seconds=sum(i['seconds'] for i in items), items=items))


def prepare_audio(cfg):
    validate_corpus(cfg)
    root, rows = read_manifest(cfg)
    out = resolve(cfg, cfg['paths']['derived']) / 'audio24k'
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for row in rows:
        y, sr = sf.read(root / row['audio'], dtype='float32', always_2d=True)
        if y.shape[1] != 1:
            raise ValueError(f'Mono required; manually select channel for {row["utterance_id"]}')
        ratio = Fraction(24000, sr)
        y = resample_poly(y[:, 0], ratio.numerator, ratio.denominator)
        if np.abs(y).max() >= 1:
            raise ValueError('Resampling produced clipping; review recording gain')
        target = out / (row['utterance_id'] + '.wav')
        sf.write(target, y, 24000, subtype='FLOAT')
        items.append(dict(utterance_id=row['utterance_id'], source_sha256=sha256(root / row['audio']), output_sha256=sha256(target)))
    return record(cfg, 'prepare-audio', dict(status='prepared', utterances=len(items), items=items))

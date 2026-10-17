"""Import external alignments; refuse wrong sequences or time bases."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import record, resolve, sha256
from .corpus import read_manifest


def parse_alignment(text, expected, vocabulary, duration, unit='seconds'):
    if unit not in ('seconds', 'htk'):
        raise ValueError('Alignment unit must be seconds or htk')
    scale = 1. if unit == 'seconds' else 1e-7
    spans = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError('Alignment needs start end phoneme')
        start, end = float(parts[0]) * scale, float(parts[1]) * scale
        phoneme = parts[2]
        # Only endpoint silence aliases accepted, as in the author's convert().
        if phoneme in ('sil', 'silB', 'silE'):
            phoneme = 'pau'
        if phoneme not in vocabulary:
            raise ValueError(f'unknown alignment phoneme: {phoneme}')
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            raise ValueError('Invalid alignment time')
        previous = spans[-1][1] if spans else 0.
        if abs(start - previous) > 0.00011:
            raise ValueError('Alignment must be continuous from zero')
        spans.append((previous, end, phoneme))
    if [p for _, _, p in spans] != expected:
        raise ValueError('Alignment phoneme sequence differs from frontend')
    if not spans or abs(spans[-1][1] - duration) > max(1 / 24000, .00011):
        raise ValueError('Alignment end differs from audio duration')
    spans[-1] = (spans[-1][0], duration, spans[-1][2])
    return spans


def import_alignment(cfg):
    _, rows = read_manifest(cfg)
    derived = resolve(cfg, cfg['paths']['derived'])
    inputs = resolve(cfg, cfg['alignment']['input'])
    out = derived / 'phoneme_list'
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for row in rows:
        uid = row['utterance_id']
        data = json.loads((derived / 'frontend' / f'{uid}.json').read_text())
        audio = derived / 'audio24k' / f'{uid}.wav'
        info = sf.info(audio)
        spans = parse_alignment((inputs / f'{uid}.lab').read_text(), data['phonemes'], cfg['profile']['phonemes'], info.duration, cfg['alignment'].get('unit', 'seconds'))
        target = out / f'{uid}.lab'
        target.write_text('\n'.join(f'{start:.4f}\t{end:.4f}\t{p}' for start, end, p in spans) + '\n')
        items.append(dict(utterance_id=uid, phonemes=len(spans), audio_sha256=sha256(audio), frontend_sha256=sha256(derived / 'frontend' / f'{uid}.json'), alignment_sha256=sha256(target)))
    return record(cfg, 'import-alignment', dict(status='imported', utterances=len(items), items=items))

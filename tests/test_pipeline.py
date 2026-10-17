"""Real Japanese frontend and feature extraction on explicitly synthetic test audio."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from new_vv_train.common import load_project, resolve, sha256
from new_vv_train.corpus import prepare_audio
from new_vv_train.frontend import frontend
from new_vv_train.alignment import import_alignment
from new_vv_train.features import extract_features, load_feature


def test_japanese_data_stages_with_author_sources(tmp_path):
    for dependency in ('torch', 'pyopenjtalk', 'pyworld', 'librosa'):
        pytest.importorskip(dependency)
    root = Path(__file__).resolve().parents[1]
    if not (root / 'vendor/vv_core_inference').is_dir():
        pytest.skip('Run fetch-sources for author integration test')
    cfg = load_project(root / 'configs/project.yaml')
    dictionary = resolve(cfg, cfg['frontend']['dictionary'])
    if not (dictionary / 'sys.dic').is_file():
        pytest.skip('Configured Japanese dictionary is unavailable')
    cfg['frontend']['dictionary'] = str(dictionary)
    cfg['paths'].update(corpus=str(tmp_path / 'corpus'), derived=str(tmp_path / 'derived'), run=str(tmp_path / 'run'), vendor=str(root / 'vendor'))
    cfg['alignment']['input'] = str(tmp_path / 'alignments')
    corpus = tmp_path / 'corpus'
    (corpus / 'raw').mkdir(parents=True)
    (corpus / 'splits').mkdir()
    sentences = ['今日はいい天気ですね。', '新しい本を読みました。', '明日は東京に行きます。']
    rows = []
    for index, (split, text) in enumerate(zip(('train', 'valid', 'test'), sentences)):
        uid = f'test{index}'
        # A tone proves data handling only. Equal synthetic boundaries are not speech alignment.
        waveform = .1 * np.sin(2 * np.pi * (160 + index * 30) * np.arange(48000) / 48000)
        sf.write(corpus / f'raw/{uid}.wav', waveform, 48000, subtype='PCM_24')
        rows.append(dict(utterance_id=uid, audio=f'raw/{uid}.wav', text=text, speaker_key='new_voice', style_key='neutral', language='ja'))
        (corpus / f'splits/{split}.txt').write_text(uid + '\n')
    (corpus / 'manifest.jsonl').write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows))
    (corpus / 'speakers.json').write_text(json.dumps({'voices': [dict(speaker_key='new_voice', style_key='neutral', inner_voice_id=0, public_style_id=1000000, name='test', style_name='normal')]}))
    prepare_audio(cfg)
    report = frontend(cfg)
    assert len(report['items']) == 3
    (tmp_path / 'alignments').mkdir()
    for row in rows:
        uid = row['utterance_id']
        data = json.loads((tmp_path / f'derived/frontend/{uid}.json').read_text())
        phonemes = data['phonemes']
        assert phonemes[0] == phonemes[-1] == 'pau'
        assert all(len(values) == len(phonemes) for values in data['accents'].values())
        bounds = np.linspace(0, 1, len(phonemes) + 1)
        (tmp_path / f'alignments/{uid}.lab').write_text('\n'.join(f'{bounds[i]:.8f} {bounds[i + 1]:.8f} {p}' for i, p in enumerate(phonemes)))
    import_alignment(cfg)
    result = extract_features(cfg)
    assert len(result['items']) == 3
    for row in rows:
        uid = row['utterance_id']
        f0, rate = load_feature(tmp_path / f'derived/f0/{uid}.npz')
        spec, spec_rate = load_feature(tmp_path / f'derived/spec/{uid}.npz')
        assert rate == spec_rate == 93.75
        assert f0.shape == (93, 1) and spec.shape == (93, 80)
        assert np.isfinite(spec).all() and np.isfinite(f0).all()
        item = next(item for item in result['items'] if item['utterance_id'] == uid)
        assert item['audio_sha256'] == sha256(tmp_path / f'derived/audio24k/{uid}.wav')

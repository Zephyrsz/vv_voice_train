import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from new_vv_train.corpus import validate_corpus, prepare_audio, init_corpus
from new_vv_train.alignment import parse_alignment, import_alignment
from new_vv_train.features import log_f0, save_feature, load_feature, frame_phonemes


@pytest.fixture
def cfg(tmp_path):
    corpus = tmp_path / 'corpus'
    (corpus / 'raw').mkdir(parents=True)
    (corpus / 'splits').mkdir()
    rows = []
    for i, split in enumerate(('train', 'valid', 'test')):
        uid = f'u{i}'
        sf.write(corpus / f'raw/{uid}.wav', np.sin(np.arange(4800) * (0.08 + i * 0.02)) * 0.1, 48000)
        rows.append(dict(utterance_id=uid, audio=f'raw/{uid}.wav', text=f'テスト{i}', speaker_key='new', style_key='neutral', language='ja'))
        (corpus / f'splits/{split}.txt').write_text(uid + '\n')
    (corpus / 'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    (corpus / 'speakers.json').write_text(json.dumps({'voices': [dict(speaker_key='new', style_key='neutral', inner_voice_id=0, public_style_id=1000000, name='new', style_name='normal')]}))
    return {'_root': str(tmp_path), 'paths': {'corpus': 'corpus', 'derived': 'derived', 'run': 'run'}, 'alignment': {'input': 'align', 'unit': 'seconds'}, 'profile': {'phonemes': ['pau', 'k', 'o'], 'sample_rate': 24000, 'hop_length': 256}}


def test_corpus_resampling_preserves_amplitude_and_original(cfg):
    report = validate_corpus(cfg)
    assert report['utterances'] == 3
    original = Path(cfg['_root']) / 'corpus/raw/u0.wav'
    before = original.read_bytes()
    prepare_audio(cfg)
    y, rate = sf.read(Path(cfg['_root']) / 'derived/audio24k/u0.wav')
    assert rate == 24000 and len(y) == 2400
    assert 0.08 < max(abs(y)) < 0.12
    assert original.read_bytes() == before


def test_corpus_rejects_audio_duplicate_across_splits(cfg):
    root = Path(cfg['_root']) / 'corpus'
    (root / 'raw/u1.wav').write_bytes((root / 'raw/u0.wav').read_bytes())
    with pytest.raises(ValueError, match='duplicate audio'):
        validate_corpus(cfg)


def test_corpus_rejects_path_escape(cfg):
    p = Path(cfg['_root']) / 'corpus/manifest.jsonl'
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    rows[0]['audio'] = '../escape.wav'
    p.write_text('\n'.join(json.dumps(x) for x in rows))
    with pytest.raises(ValueError, match='outside'):
        validate_corpus(cfg)


def test_alignment_htk_conversion_and_repeated_phonemes():
    spans = parse_alignment('0 1000000 pau\n1000000 2000000 o\n2000000 3000000 o\n3000000 4000000 pau', ['pau', 'o', 'o', 'pau'], ['pau', 'o'], 0.4, 'htk')
    assert spans[1][:2] == pytest.approx((0.1, 0.2))
    assert spans[1][2] == 'o'
    with pytest.raises(ValueError, match='sequence'):
        parse_alignment('0 .1 pau\n.1 .2 o\n.2 .4 pau', ['pau', 'o', 'o', 'pau'], ['pau', 'o'], .4)


def test_alignment_rejects_gap_and_unknown():
    with pytest.raises(ValueError, match='continuous'):
        parse_alignment('0 .1 pau\n.2 .4 o', ['pau', 'o'], ['pau', 'o'], .4)
    with pytest.raises(ValueError, match='unknown'):
        parse_alignment('0 .4 xyz', ['xyz'], ['pau'], .4)


def test_import_requires_frontend_and_writes_canonical_labels(cfg):
    prepare_audio(cfg)
    root = Path(cfg['_root'])
    (root / 'align').mkdir()
    (root / 'derived/frontend').mkdir()
    for i in range(3):
        (root / f'align/u{i}.lab').write_text('0 .02 pau\n.02 .04 k\n.04 .08 o\n.08 .1 pau')
        (root / f'derived/frontend/u{i}.json').write_text(json.dumps({'phonemes': ['pau', 'k', 'o', 'pau']}))
    import_alignment(cfg)
    assert (root / 'derived/phoneme_list/u0.lab').read_text().splitlines()[2].split() == ['0.0400', '0.0800', 'o']


def test_f0_and_feature_storage_are_log_hz_no_pickle(tmp_path):
    assert np.allclose(log_f0(np.array([0., 100., 440.])), [0., 4.605170186, 6.086774727])
    with pytest.raises(ValueError):
        log_f0(np.array([np.nan]))
    p = tmp_path / 'f0.npz'
    save_feature(p, np.array([[0.], [4.6]], np.float32), 93.75)
    a, rate = load_feature(p)
    assert rate == 93.75 and a.shape == (2, 1)


def test_frame_onehot_uses_profile_order():
    spans = [(0., .02, 'pau'), (.02, .06, 'o'), (.06, .1, 'pau')]
    result = frame_phonemes(spans, ['pau', 'k', 'o'], 10, 100.)
    assert result.shape == (10, 3)
    assert result.argmax(axis=1).tolist() == [0, 0, 2, 2, 2, 2, 0, 0, 0, 0]


def test_init_corpus_creates_editable_templates_without_fake_audio(tmp_path):
    config = {'_root': str(tmp_path), 'paths': {'corpus': 'corpus', 'run': 'run'}, 'profile': {'phonemes': ['pau']}}
    init_corpus(config)
    root = tmp_path / 'corpus'
    assert (root / 'manifest.jsonl').read_text() == ''
    assert (root / 'manifest.example.jsonl').is_file()
    assert not list((root / 'raw').glob('*.wav'))
    assert json.loads((root / 'speakers.json').read_text())['voices'][0]['inner_voice_id'] == 0
    (root / 'manifest.jsonl').write_text('my recording')
    with pytest.raises(FileExistsError):
        init_corpus(config)
    assert (root / 'manifest.jsonl').read_text() == 'my recording'

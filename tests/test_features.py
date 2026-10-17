import numpy as np
import pytest

from new_vv_train.frontend import accents_from_contexts, validate_vocabulary
from new_vv_train.features import author_spectrogram, validate_feature_bundle


def test_author_accent_marks_are_phoneme_aligned():
    contexts = [dict(a1='xx', a2='xx', a3='xx'), dict(a1='-1', a2='1', a3='2'), dict(a1='-1', a2='1', a3='2'), dict(a1='0', a2='2', a3='1'), dict(a1='xx', a2='xx', a3='xx')]
    result = accents_from_contexts(contexts)
    assert result['start_accent'] == [0, 0, 0, 1, 0]
    assert result['end_accent'] == [0, 0, 0, 1, 0]
    assert result['start_accent_phrase'] == [0, 1, 1, 0, 0]
    assert result['end_accent_phrase'] == [0, 0, 0, 1, 0]


def test_reordered_vocabulary_rejected():
    with pytest.raises(ValueError, match='order'):
        validate_vocabulary(['pau', 'o', 'a'], ['pau', 'a', 'o'])


def test_bundle_rejects_timebase_and_bad_onehot():
    bundle = {'f0': (np.zeros((10, 1)), 93.75), 'phoneme': (np.tile([1., 0.], (10, 1)), 93.75), 'spec': (np.ones((10, 3)), 93.75), 'silence': (np.zeros((10, 1), bool), 93.75), 'volume': (np.ones((10, 1)), 93.75)}
    validate_feature_bundle(bundle, 2, 3)
    bundle['f0'] = (np.zeros((10, 1)), 100.)
    with pytest.raises(ValueError, match='rate'):
        validate_feature_bundle(bundle, 2, 3)
    bundle['f0'] = (np.zeros((10, 1)), 93.75)
    bundle['phoneme'][0][:] = 0
    with pytest.raises(ValueError, match='one-hot'):
        validate_feature_bundle(bundle, 2, 3)


def test_spectrogram_preserves_absolute_gain_and_hop(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('librosa')
    # Minimal actual author function snapshot supplied to the adapter, not a mocked output.
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / 'vendor/hifi-gan/hifi_gan/meldataset.py'
    if not source.exists():
        pytest.skip('Pinned source fixture not available')
    profile = dict(sample_rate=24000, hop_length=256, spectrogram=dict(n_fft=1024, win_length=1024, num_mels=80, fmin=0, fmax=12000, center=False))
    y = np.sin(np.arange(24000, dtype=np.float32) * .08) * .1
    a = author_spectrogram(source, y, profile)
    b = author_spectrogram(source, y * .5, profile)
    assert a.shape == (93, 80)
    assert np.median((a - b)[a > -8]) == pytest.approx(np.log(2), abs=.01)
    assert np.isfinite(a).all()

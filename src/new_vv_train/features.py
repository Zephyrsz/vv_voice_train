"""Explicit time bases and author-compatible acoustic targets."""

from pathlib import Path

import numpy as np


def log_f0(hz):
    hz = np.asarray(hz)
    if not np.isfinite(hz).all() or (hz < 0).any():
        raise ValueError('F0 Hz must be finite and nonnegative')
    out = np.zeros_like(hz, dtype=np.float32)
    voiced = hz > 0
    out[voiced] = np.log(hz[voiced])
    return out


def save_feature(path, array, rate):
    path = Path(path)
    array = np.asarray(array)
    if not array.size or not np.isfinite(array).all() or rate <= 0:
        raise ValueError('Invalid feature array or rate')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, array=array, rate=np.float64(rate))


def load_feature(path):
    with np.load(path, allow_pickle=False) as d:
        array, rate = d['array'], float(d['rate'])
    if not array.size or not np.isfinite(array).all() or not np.isfinite(rate) or rate <= 0:
        raise ValueError(f'Invalid feature: {path}')
    return array, rate


def frame_phonemes(spans, vocabulary, length, rate):
    times = np.arange(length) / rate
    indexes = np.searchsorted([e for _, e, _ in spans], times, side='right')
    if not spans or (indexes >= len(spans)).any():
        raise ValueError('Feature frames extend outside alignment')
    ids = np.array([vocabulary.index(p) for _, _, p in spans], dtype=np.int64)
    return np.eye(len(vocabulary), dtype=np.float32)[ids[indexes]]


def author_spectrogram(source, audio, profile):
    """Run original author function with two API-only compatibility fixes.

    librosa now requires keyword args; torch requires return_complex. Keep
    sqrt(sum(real/imag**2)+1e-9), original reflect padding and natural log.
    The source snapshot stays unmodified and its hash is in the receipt.
    """
    import ast
    import librosa
    import torch
    tree = ast.parse(Path(source).read_text())
    names = {'dynamic_range_compression_torch', 'spectral_normalize_torch', 'mel_spectrogram'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names:
        raise ValueError('Pinned HiFi-GAN spectral functions not found')

    class ModernStft(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' and node.func.attr == 'stft':
                node.keywords = [kw for kw in node.keywords if kw.arg != 'return_complex']
                node.keywords.append(ast.keyword(arg='return_complex', value=ast.Constant(True)))
                return ast.Call(func=ast.Attribute(value=ast.Name(id='torch', ctx=ast.Load()), attr='view_as_real', ctx=ast.Load()), args=[node], keywords=[])
            return node

    def mel(*args, **kwargs):
        if args:
            sr, n_fft, num_mels, fmin, fmax = args
            kwargs = dict(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        return librosa.filters.mel(**kwargs)

    module = ast.fix_missing_locations(ModernStft().visit(ast.Module(body=functions, type_ignores=[])))
    scope = dict(torch=torch, librosa_mel_fn=mel, mel_basis={}, hann_window={})
    exec(compile(module, str(source), 'exec'), scope)
    spec = profile['spectrogram']
    with torch.no_grad():
        out = scope['mel_spectrogram'](torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0), spec['n_fft'], spec['num_mels'], profile['sample_rate'], profile['hop_length'], spec['win_length'], spec['fmin'], spec['fmax'], center=False)
    return out[0].T.numpy().astype(np.float32)


def validate_feature_bundle(bundle, phoneme_size, spec_dim):
    length = len(bundle['spec'][0])
    for key, dim in [('f0', 1), ('phoneme', phoneme_size), ('spec', spec_dim), ('silence', 1), ('volume', 1)]:
        array, rate = bundle[key]
        if rate != 93.75:
            raise ValueError(f'Wrong feature rate: {key}')
        if array.shape != (length, dim) or not np.isfinite(array).all():
            raise ValueError(f'Invalid feature shape/values: {key}')
    onehot = bundle['phoneme'][0]
    if not np.all((onehot == 0) | (onehot == 1)) or not np.all(onehot.sum(1) == 1):
        raise ValueError('phoneme must be one-hot')
    if (bundle['f0'][0] < 0).any() or (bundle['volume'][0] < 0).any():
        raise ValueError('Negative F0/volume')
    if not np.all((bundle['silence'][0] == 0) | (bundle['silence'][0] == 1)):
        raise ValueError('Invalid silence mask')


def extract_features(cfg):
    import json
    import pyworld
    import soundfile as sf
    from .alignment import parse_alignment
    from .common import record, resolve, sha256, write_json, profile_hash
    from .corpus import read_manifest
    from .frontend import source_vocabulary
    from .upstream import inspect_sources
    inspect_sources(cfg)
    vocabulary = source_vocabulary(cfg)
    _, rows = read_manifest(cfg)
    derived = resolve(cfg, cfg['paths']['derived'])
    author_source = resolve(cfg, cfg['paths']['vendor']) / 'hifi-gan/hifi_gan/meldataset.py'
    rate = 93.75
    f0_cfg = cfg['profile']['f0']
    items = []
    for row in rows:
        uid = row['utterance_id']
        audio_path = derived / 'audio24k' / f'{uid}.wav'
        y, sr = sf.read(audio_path)
        if sr != 24000 or y.ndim != 1:
            raise ValueError('Prepared audio must be 24k mono')
        frontend_path = derived / 'frontend' / f'{uid}.json'
        frontend_data = json.loads(frontend_path.read_text())
        labels_path = derived / 'phoneme_list' / f'{uid}.lab'
        spans = parse_alignment(labels_path.read_text(), frontend_data['phonemes'], vocabulary, len(y) / sr)
        spec = author_spectrogram(author_source, y.astype(np.float32), cfg['profile'])
        length = len(spec)
        frame_period = 1000 / rate
        hz, times = pyworld.harvest(y.astype(np.float64), sr, f0_floor=f0_cfg['floor_hz'], f0_ceil=f0_cfg['ceil_hz'], frame_period=frame_period)
        hz = pyworld.stonemask(y.astype(np.float64), hz, times, sr)
        indexes = np.rint(np.arange(length) / rate / (frame_period / 1000)).astype(int)
        if (indexes >= len(hz)).any():
            raise ValueError('F0 timebase shorter than spectrum')
        phoneme = frame_phonemes(spans, vocabulary, length, rate)
        lf0 = log_f0(hz[indexes])[:, None]
        # The Core zeros unvoiced mora pitch. Retain measured Hz for diagnostics.
        unvoiced_ids = [vocabulary.index(p) for p in ('pau', 'A', 'I', 'U', 'E', 'O', 'cl') if p in vocabulary]
        lf0[np.isin(phoneme.argmax(1), unvoiced_ids)] = 0
        frames = [y[i * 256:(i + 1) * 256] for i in range(length)]
        volume = np.array([np.sqrt(np.mean(a ** 2)) for a in frames], np.float32)[:, None]
        silence = (phoneme.argmax(1) == vocabulary.index('pau'))[:, None]
        bundle = dict(f0=(lf0, rate), phoneme=(phoneme, rate), spec=(spec, rate), silence=(silence, rate), volume=(volume, rate))
        validate_feature_bundle(bundle, len(vocabulary), cfg['profile']['spectrogram']['num_mels'])
        for name, (array, feature_rate) in bundle.items():
            save_feature(derived / name / f'{uid}.npz', array, feature_rate)
        save_feature(derived / 'f0_hz' / f'{uid}.npz', hz[indexes][:, None], rate)
        items.append(dict(utterance_id=uid, frames=length, audio_sha256=sha256(audio_path), alignment_sha256=sha256(labels_path), frontend_sha256=sha256(frontend_path), feature_sha256={name:sha256(derived/name/f'{uid}.npz') for name in bundle}))
    write_json(derived / 'feature_profile.json', cfg['profile'])
    return record(cfg, 'extract-features', dict(status='extracted-needs-review', profile_sha256=profile_hash(cfg), spectral_source_sha256=sha256(author_source), api_patches=['librosa_mel_keyword_arguments', 'torch_stft_complex_to_real_view'], items=items))

"""Report individual prerequisites without conflating them with training success."""

import importlib.metadata
import json
import platform
import sys

from .common import resolve


def doctor(cfg):
    from .upstream import inspect_sources
    from .training import validate_base
    from .corpus import validate_corpus
    checks, errors = {}, {}
    for key, fn in [('source_ready', inspect_sources), ('data_ready', validate_corpus)]:
        try:
            result = fn(cfg)
            checks[key] = result.get('status') not in ('missing', 'invalid', 'not-ready')
            checks[key + '_details'] = result
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            checks[key] = False
            errors[key] = str(exc)
    if cfg.get('training', {}).get('mode', 'scratch') == 'scratch':
        checks['base_ready'] = False
        checks['initialization'] = 'scratch-no-pretrained-base'
    else:
        try:
            checks['base_details'] = validate_base(cfg)
            checks['base_ready'] = True
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            checks['base_ready'] = False
            errors['base_ready'] = str(exc)
    training_python = cfg.get('training', {}).get('python', sys.executable)
    checks['training_python_exists'] = resolve(cfg, training_python).is_file()
    core = cfg.get('core', {})
    checks['core_python_exists'] = resolve(cfg, core.get('python', 'missing')).is_file()
    checks['core_assets_exist'] = resolve(cfg, core.get('assets', 'missing')).is_dir()
    versions = {}
    for package in ('numpy', 'torch', 'librosa', 'pyworld', 'pyopenjtalk', 'onnx', 'onnxruntime'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    ready = checks['source_ready'] and checks['data_ready'] and checks['training_python_exists'] and (cfg.get('training', {}).get('mode', 'scratch') == 'scratch' or checks['base_ready'])
    return dict(checks, status='prerequisites-present-untrained' if ready else 'not-ready', errors=errors, python=sys.version.split()[0], platform=platform.platform(), packages=versions, note='No GPU training, native synthesis or Core compatibility is implied by prerequisite checks.')

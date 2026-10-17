"""Explicit stage CLI; failures return nonzero and do not become success receipts."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import load_project


def main(argv=None):
    parser = argparse.ArgumentParser(description='Research toolkit for native VOICEVOX author modules')
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[2] / 'configs/project.yaml'))
    sub = parser.add_subparsers(dest='command', required=True)
    commands = ('doctor', 'fetch-sources', 'init-corpus', 'validate-corpus', 'prepare-audio', 'frontend', 'import-alignment', 'extract-features', 'materialize', 'validate-base', 'evaluate-native', 'export', 'validate-onnx', 'package', 'validate-core', 'validate-service')
    for command in commands:
        sub.add_parser(command)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('module', choices=('duration', 'intonation', 'acoustic', 'vocoder'))
    train_parser.add_argument('--dry-run', action='store_true')
    inspect = sub.add_parser('inspect-vvm')
    inspect.add_argument('path')
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect-vvm':
            from .vvm import inspect_vvm
            result = inspect_vvm(Path(args.path))
        else:
            cfg = load_project(args.config)
            if args.command == 'doctor':
                from .doctor import doctor
                result = doctor(cfg)
            elif args.command == 'fetch-sources':
                from .upstream import fetch_sources
                result = fetch_sources(cfg)
            elif args.command in ('init-corpus', 'validate-corpus', 'prepare-audio'):
                from .corpus import validate_corpus, prepare_audio, init_corpus
                result = {'init-corpus': init_corpus, 'validate-corpus': validate_corpus, 'prepare-audio': prepare_audio}[args.command](cfg)
            elif args.command == 'frontend':
                from .frontend import frontend
                result = frontend(cfg)
            elif args.command == 'import-alignment':
                from .alignment import import_alignment
                result = import_alignment(cfg)
            elif args.command == 'extract-features':
                from .features import extract_features
                result = extract_features(cfg)
            elif args.command in ('materialize', 'validate-base', 'train'):
                from .training import materialize, validate_base, train
                if args.command == 'train':
                    result = train(cfg, args.module, dry_run=args.dry_run)
                else:
                    result = (materialize if args.command == 'materialize' else validate_base)(cfg)
            elif args.command in ('export', 'validate-onnx', 'evaluate-native'):
                from .export import export_models, validate_onnx, evaluate_native
                result = {'export': export_models, 'validate-onnx': validate_onnx, 'evaluate-native': evaluate_native}[args.command](cfg)
            elif args.command == 'package':
                from .vvm import package
                result = package(cfg)
            else:
                from .core_validation import validate_core, validate_service
                result = (validate_core if args.command == 'validate-core' else validate_service)(cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if result.get('status') == 'not-ready' else 0
    except (ValueError, RuntimeError, OSError, KeyError, ImportError, subprocess.CalledProcessError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

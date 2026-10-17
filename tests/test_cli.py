import json
import subprocess
import sys
from pathlib import Path

import pytest

from new_vv_train.doctor import doctor


def test_doctor_reports_missing_material_without_readiness(tmp_path):
    cfg = {'_root': str(tmp_path), 'paths': {'corpus': 'absent', 'vendor': 'vendor', 'run': 'run', 'base': None}, 'profile': {'sample_rate': 24000, 'hop_length': 256, 'phonemes': ['pau']}, 'training': {'mode': 'finetune', 'python': sys.executable}, 'core': {'version': '0.17.0', 'python': 'missing-python', 'assets': 'missing-assets'}}
    result = doctor(cfg)
    assert result['source_ready'] is False
    assert result['data_ready'] is False
    assert result['base_ready'] is False
    assert result['status'] == 'not-ready'


def test_cli_help_exposes_real_stage_commands():
    result = subprocess.run([sys.executable, '-m', 'new_vv_train', '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'import-alignment' in result.stdout
    assert 'validate-core' in result.stdout


def test_cli_missing_corpus_is_nonzero(tmp_path):
    root = Path(__file__).resolve().parents[1]
    import yaml
    config = yaml.safe_load((root / 'configs/project.yaml').read_text())
    config['root'] = str(root)
    config['paths']['corpus'] = str(tmp_path / 'absent')
    config['paths']['run'] = str(tmp_path / 'run')
    config_path = tmp_path / 'project.yaml'
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run([sys.executable, '-m', 'new_vv_train', '--config', str(config_path), 'validate-corpus'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'manifest.jsonl' in result.stderr

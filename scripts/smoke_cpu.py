"""Exercise actual upstream CPU trainers with synthetic tones, never a voice/base.

Run from new_vv_train: .venv/bin/python scripts/smoke_cpu.py
Only the final diagnostic report survives; all synthetic checkpoints are temporary.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from new_vv_train.common import load_project, resolve, write_json
from new_vv_train.corpus import prepare_audio
from new_vv_train.frontend import frontend
from new_vv_train.alignment import import_alignment
from new_vv_train.features import extract_features
from new_vv_train.training import materialize, train


def run():
    root = Path(__file__).resolve().parents[1]
    cfg = load_project(root / 'configs/project.yaml')
    report = {'fixture': 'synthetic-tones-equal-boundaries', 'voice_quality_claim': False, 'modules': {}}
    try:
        with tempfile.TemporaryDirectory(prefix='native-vv-smoke-') as temp:
            temp = Path(temp)
            cfg['paths'].update(corpus=str(temp / 'corpus'), derived=str(temp / 'derived'), run=str(temp / 'run'))
            cfg['alignment']['input'] = str(temp / 'alignments')
            cfg['training'].update(mode='scratch', modules={})
            for module in ('duration', 'intonation'):
                cfg['training']['modules'][module] = {'overrides': {'train': {'use_gpu': False, 'batch_size': 1, 'log_iteration': 1, 'snapshot_iteration': 1, 'stop_iteration': 10}}}
            cfg['training']['modules']['duration']['overrides'].update(dataset={'phoneme_num': 16}, network={'hidden_size_list': [16], 'kernel_size_list': [3]})
            cfg['training']['modules']['intonation']['overrides'].update(dataset={'sampling_length': 16}, network={'encoder_hidden_size': 16, 'encoder_layer_num': 1, 'ar_encoder_hidden_size': 16, 'ar_encoder_layer_num': 1})
            cfg['training']['modules']['acoustic'] = {'overrides': {'network': {'hidden_size': 16, 'block_num': 1}, 'train': {'use_gpu': False, 'use_amp': False, 'batch_size': 1, 'eval_batch_size': 1, 'stop_epoch': 1, 'scheduler': None}}}
            corpus = temp / 'corpus'
            (corpus / 'raw').mkdir(parents=True)
            (corpus / 'splits').mkdir()
            (temp / 'alignments').mkdir()
            rows = []
            sentences = ['今日はいい天気ですね。', '新しい本を読みました。', '明日は東京に行きます。']
            for index, (split, text) in enumerate(zip(('train', 'valid', 'test'), sentences)):
                uid = f'synthetic{index}'
                sf.write(corpus / f'raw/{uid}.wav', .1 * np.sin(2 * np.pi * (160 + index * 30) * np.arange(48000) / 48000), 48000, subtype='PCM_24')
                rows.append(dict(utterance_id=uid, audio=f'raw/{uid}.wav', text=text, speaker_key='test', style_key='neutral', language='ja'))
                (corpus / f'splits/{split}.txt').write_text(uid + '\n')
            (corpus / 'manifest.jsonl').write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows))
            write_json(corpus / 'speakers.json', {'voices': [dict(speaker_key='test', style_key='neutral', inner_voice_id=0, public_style_id=1000000, name='synthetic test', style_name='test')]})
            prepare_audio(cfg)
            frontend(cfg)
            for row in rows:
                uid = row['utterance_id']
                phonemes = json.loads((temp / f'derived/frontend/{uid}.json').read_text())['phonemes']
                bounds = np.linspace(0, 1, len(phonemes) + 1)
                (temp / f'alignments/{uid}.lab').write_text('\n'.join(f'{bounds[i]:.8f} {bounds[i+1]:.8f} {p}' for i, p in enumerate(phonemes)))
            import_alignment(cfg)
            extract_features(cfg)
            materialize(cfg)
            report['native_dataset_loading'] = 'all-four-modules-passed'
            for module in ('duration', 'intonation', 'acoustic'):
                try:
                    result = train(cfg, module)
                    report['modules'][module] = {'status': result['status'], 'model_sha256': result['model_sha256']}
                except Exception as exc:
                    log = temp / f'run/logs/{module}.log'
                    report['modules'][module] = {'status': 'failed', 'error': str(exc), 'log': log.read_text() if log.exists() else ''}
                print(module, report['modules'][module]['status'], flush=True)
            report['vocoder'] = 'CUDA trainer not run on this CPU smoke'
    finally:
        destination = root / 'runs/diagnostics/cpu-smoke.json'
        write_json(destination, report)
        print(destination)
    return 0 if len(report['modules']) == 3 and all(r['status'] == 'trained' for r in report['modules'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(run())

"""Materialize author dataset views and invoke the real native training entrypoints."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .common import resolve, sha256, write_json, profile_hash, interpreter_path
from .corpus import read_manifest, read_splits, read_voices, validate_corpus
from .upstream import inspect_sources

MODULES={'duration':'yukarin_s','intonation':'yukarin_sa','acoustic':'yukarin_sosoa','vocoder':'hifi-gan'}
FEATURES=('f0','volume','phoneme','spec','silence')
ACCENTS=('start_accent','end_accent','start_accent_phrase','end_accent_phrase')


def _module_options(cfg,module):
    options=cfg.get('training',{}).get('modules',{}).get(module,{})
    return {'config':options} if isinstance(options,str) else options


def _python(cfg,module):
    return str(interpreter_path(cfg,_module_options(cfg,module).get('python',cfg['training']['python'])))


def _env(cfg):
    vendor=resolve(cfg,cfg['paths']['vendor'])
    env={key:os.environ[key] for key in ('PATH','HOME','TMPDIR','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS') if key in os.environ}
    env['WANDB_MODE']='disabled'
    env['WANDB_DISABLE_CODE']='true'
    env['PYTHONPATH']=os.pathsep.join(str(vendor/n) for n in ('pytorch-trainer','yukarin_s','yukarin_sa','yukarin_sosoa','hifi-gan','old_yukarin_sosoa','espnet_pytorch_library'))
    return env


def _merge(a,b):
    for key,value in b.items():
        if isinstance(value,dict) and isinstance(a.get(key),dict):_merge(a[key],value)
        else:a[key]=value
    return a


def _template(cfg,module):
    options=_module_options(cfg,module)
    default=Path(__file__).resolve().parents[2]/'configs/modules'/f'{module}.{"json" if module=="vocoder" else "yaml"}'
    path=resolve(cfg,options['config']) if options.get('config') else default
    data=yaml.safe_load(path.read_text())
    return _merge(data,options.get('overrides',{}))


def _inventory(root):
    return {p.relative_to(root).as_posix():sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='materialization.json'}


def _validate_native_config(cfg,module,path):
    if module=='vocoder':return
    script='import json,sys,yaml; from '+MODULES[module]+'.config import Config; Config.from_dict(yaml.safe_load(open(sys.argv[1]))); print("native-config-valid")'
    subprocess.run([_python(cfg,module),'-c',script,str(path)],env=_env(cfg),check=True,capture_output=True,text=True)


def _validate_native_dataset(cfg,module,path):
    if module=='vocoder':
        script = """import sys,json,numpy as np
from argparse import Namespace
from pathlib import Path
from hifi_gan.meldataset import get_dataset_filelist,MelDataset
from hifi_gan.env import AttrDict
v=Path(sys.argv[1]); h=AttrDict(json.loads((v/'config.json').read_text()))
a=Namespace(input_training_file=str(v/'train.txt'),input_validation_file=str(v/'valid.txt'),input_wavs_dir=str(v/'data/audio24k'),input_f0_dir=str(v/'vocoder_f0'),input_mels_dir=str(v/'mels'),fine_tuning=True,input_wavs_npy=False)
for files in get_dataset_filelist(a):
 dataset=MelDataset(files,h.segment_size,h.n_fft,h.num_mels,h.hop_size,h.win_size,h.sampling_rate,h.fmin,h.fmax,split=False,shuffle=False,fine_tuning=True)
 sample=dataset[0]
 assert sample[0].numel()>0 and sample[1].numel()>0
print('native-dataset-valid')
"""
    else:
        script="""import sys,yaml
from MODULE.config import Config
from MODULE.dataset import create_dataset
config=Config.from_dict(yaml.safe_load(open(sys.argv[1])))
datasets=create_dataset(config.dataset)
for key in ('train','test'):
 assert len(datasets[key])>0
 sample=datasets[key][0]
 assert sample['speaker_id'].item()>=0
print('native-dataset-valid')
""".replace('MODULE',MODULES[module])
    argument=path.parent if module=='vocoder' else path
    subprocess.run([_python(cfg,module),'-c',script,str(argument)],env=_env(cfg),check=True,capture_output=True,text=True)


def materialize(cfg):
    validate_corpus(cfg)
    root,rows=read_manifest(cfg);splits=read_splits(root,rows);voices=read_voices(cfg)
    derived=resolve(cfg,cfg['paths']['derived']); run=resolve(cfg,cfg['paths']['run'])
    if json.loads((derived/'feature_profile.json').read_text()) != cfg['profile']:
        raise ValueError('Derived feature profile differs from project profile')
    from .frontend import source_vocabulary
    from .alignment import parse_alignment
    from .features import frame_phonemes
    vocabulary=source_vocabulary(cfg)
    extraction=json.loads((run/'reports/extract-features.json').read_text())
    if extraction['profile_sha256']!=profile_hash(cfg):raise ValueError('Feature extraction profile changed')
    feature_items={item['utterance_id']:item for item in extraction['items']}
    active=splits['train']+splits['valid']; row_by_id={r['utterance_id']:r for r in rows}
    all_voices=sorted(voices,key=lambda v:v['inner_voice_id'])
    speakers={f'{v["speaker_key"]}::{v["style_key"]}':[] for v in all_voices}
    for uid in active:
        r=row_by_id[uid]; key=f'{r["speaker_key"]}::{r["style_key"]}'
        if key not in speakers:raise ValueError(f'Unknown voice {key}')
        speakers[key].append(uid)
    p=len(cfg['profile']['phonemes']);spec_cfg=cfg['profile']['spectrogram'];d=int(spec_cfg['num_mels']);rate=24000/256
    arrays={}
    # Validate every input before writing outputs, including final-test feature integrity.
    for uid in row_by_id:
        for feature in FEATURES:
            path=derived/feature/f'{uid}.npz'
            if not path.exists():raise FileNotFoundError(path)
        lengths=[]; item={}
        for feature in FEATURES:
            with np.load(derived/feature/f'{uid}.npz',allow_pickle=False) as archive:
                if set(archive.files)!={'array','rate'}:raise ValueError(f'Invalid SamplingData npz: {uid}/{feature}')
                array=archive['array'];actual_rate=float(archive['rate'])
            dim=p if feature=='phoneme' else d if feature=='spec' else 1
            if array.ndim!=2 or array.shape[1]!=dim or not len(array) or not np.isfinite(array).all() or actual_rate!=rate:
                raise ValueError(f'Invalid feature shape/rate/value: {uid}/{feature}')
            if feature in ('f0','volume') and (array<0).any():raise ValueError(f'Negative {feature}: {uid}')
            if feature=='phoneme' and (not np.isin(array,[0,1]).all() or not np.all(array.sum(axis=1)==1)):
                raise ValueError(f'Phoneme must be one-hot: {uid}')
            lengths.append(len(array));item[feature]=array
        if len(set(lengths))!=1:raise ValueError(f'Feature frame mismatch: {uid}')
        label=derived/'phoneme_list'/f'{uid}.lab'; count=len(label.read_text().splitlines())
        front=derived/'frontend'/f'{uid}.json'
        wav=derived/'audio24k'/f'{uid}.wav'
        extraction_item=feature_items[uid]
        for key,path in [('audio',wav),('alignment',label),('frontend',front)]:
            if extraction_item[key+'_sha256']!=sha256(path):raise ValueError(f'Stale feature input: {uid}/{key}')
        for feature in FEATURES:
            if extraction_item.get('feature_sha256',{}).get(feature)!=sha256(derived/feature/f'{uid}.npz'):raise ValueError(f'Stale/corrupt feature output: {uid}/{feature}')
        spans=parse_alignment(label.read_text(),json.loads(front.read_text())['phonemes'],vocabulary,sf.info(wav).duration)
        expected_phoneme=frame_phonemes(spans,vocabulary,len(item['phoneme']),rate)
        if not np.array_equal(expected_phoneme,item['phoneme']):raise ValueError(f'Phoneme frames differ from alignment: {uid}')
        if count==0:raise ValueError(f'Empty phoneme alignment: {uid}')
        for accent in ACCENTS:
            values=np.loadtxt(derived/accent/f'{uid}.txt',dtype=np.int64,ndmin=1)
            if values.size!=count or not np.isin(values,[0,1]).all():raise ValueError(f'Accent alignment mismatch: {uid}/{accent}')
        wav=derived/'audio24k'/f'{uid}.wav'; info=sf.info(wav)
        if info.samplerate!=24000 or info.channels!=1 or info.frames==0:raise ValueError(f'Invalid prepared wav: {uid}')
        arrays[uid]=item
    inspect_sources(cfg)
    result={}
    for module in MODULES:
        view=run/'native'/module
        if view.exists():shutil.rmtree(view)
        view.mkdir(parents=True)
        write_json(view/'speaker_dict.json',speakers)  # insertion order IS upstream embedding ID
        write_json(view/'splits.json',{'train':splits['train'],'valid':splits['valid']})
        for uid in active:
            for directory,ext in [('phoneme_list','lab')]+[(a,'txt') for a in ACCENTS]:
                target=view/'data'/directory/f'{uid}.{ext}';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(derived/directory/f'{uid}.{ext}',target)
            for feat in FEATURES:
                target=view/'data'/feat/f'{uid}.npy';target.parent.mkdir(parents=True,exist_ok=True)
                np.save(target,{'array':arrays[uid][feat],'rate':rate},allow_pickle=True)
            for split_name in ['train','valid']:
                if uid not in splits[split_name]:continue
                for feat in FEATURES:
                    target=view/split_name/feat/f'{uid}.npy';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(view/'data'/feat/f'{uid}.npy',target)
                for directory,ext in [('phoneme_list','lab')]+[(a,'txt') for a in ACCENTS]:
                    target=view/split_name/directory/f'{uid}.{ext}';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(view/'data'/directory/f'{uid}.{ext}',target)
            wav=view/'data/audio24k'/f'{uid}.wav';wav.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(derived/'audio24k'/f'{uid}.wav',wav)
            vf0=view/'vocoder_f0'/f'{uid}.npy';vf0.parent.mkdir(parents=True,exist_ok=True);np.save(vf0,{'array':arrays[uid]['f0'][:,0],'rate':rate})
            mel=view/'mels'/f'{uid}.npy';mel.parent.mkdir(parents=True,exist_ok=True);np.save(mel,arrays[uid]['spec'].T)
        config=_template(cfg,module)
        if module!='vocoder':
            dataset=config['dataset'];dataset.update(speaker_dict_path=str(view/'speaker_dict.json'),test_num=len(splits['valid']))
            config['network']['speaker_size']=len(voices)
            dataset['num_speaker' if module=='acoustic' else 'speaker_size']=len(voices)
            if module!='acoustic':config['network']['phoneme_size']=p
            features=('phoneme_list',) if module=='duration' else ('phoneme_list',*ACCENTS,'f0','volume') if module=='intonation' else FEATURES+('phoneme_list',)
            for feat in features:
                key=feat+'_list_glob' if feat in ACCENTS else feat+'_glob'
                if feat=='phoneme_list':key='phoneme_list_glob'
                ext='lab' if feat=='phoneme_list' else 'txt' if feat in ACCENTS else 'npy'
                dataset[key]=str(view/'data'/feat/f'*.{ext}')
                if module!='duration':dataset['valid_'+key]=str(view/'valid'/feat/f'*.{ext}')
            if module!='duration':
                dataset.update(valid_speaker_dict_path=str(view/'speaker_dict.json'),valid_num=len(splits['valid']),valid_trial_num=1)
            if module=='acoustic':config['network'].update(input_feature_size=p+1,output_size=d)
            path=view/'config.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False));_validate_native_config(cfg,module,path)
        else:
            config.update(sampling_rate=24000,hop_size=256,num_mels=d,n_fft=spec_cfg['n_fft'],win_size=spec_cfg['win_length'],fmin=spec_cfg['fmin'],fmax=spec_cfg['fmax'],with_hn=True)
            if np.prod(config['upsample_rates'])!=256:raise ValueError('Vocoder upsample factor must equal hop_length')
            write_json(view/'config.json',config)
            path=view/'config.json'
            for split in ['train','valid']:(view/f'{split}.txt').write_text(''.join(f'{uid}|{uid}\n' for uid in splits[split]))
        _validate_native_dataset(cfg,module,path)
        write_json(view/'materialization.json',{'status':'ready','profile_sha256':profile_hash(cfg),'inputs':_inputs(cfg),'files':_inventory(view)})
        result[module]=str(view)
    report={'status':'ready','modules':result,'speaker_order':list(speakers),'final_test_excluded':splits['test'],'profile_sha256':profile_hash(cfg),'native_dataset_loading':'verified'}
    write_json(run/'reports/materialize.json',report)
    return report


def validate_base(cfg):
    base=resolve(cfg,cfg['paths'].get('base'));manifest=json.loads((base/'base.json').read_text());components=manifest.get('components',{})
    # Inspect component hashes before completeness so corrupted weights get specific diagnostics.
    for module,component in components.items():
        for key in ['weights','config']:
            if key not in component:
                if key=='weights':raise ValueError(f'{module}: missing weights')
                continue
            path=(base/component[key]).resolve()
            if not path.is_relative_to(base):raise ValueError('Base component escapes base directory')
            expected=component.get(key+'_sha256',component.get('sha256') if key=='weights' else None)
            if not expected or sha256(path)!=expected:raise ValueError(f'{module}/{key}: sha256 missing or mismatch')
            if path.suffix in ('.onnx','.vvm'):raise ValueError('Inference weights cannot initialize training')
    if set(components)!=set(MODULES):raise ValueError('Base requires all four training components')
    if manifest.get('origin') not in ('self_trained','licensed','owned'):raise ValueError('Base origin/rights are required')
    for key in ('feature_profile','voice_registry','rights_record'):
        path=(base/manifest[key]).resolve()
        if not path.is_relative_to(base) or not path.is_file():raise ValueError(f'Missing/unsafe base {key}')
        expected=manifest.get(key+'_sha256')
        if not expected or sha256(path)!=expected:raise ValueError(f'{key}: sha256 missing or mismatch')
    if json.loads((base/manifest['feature_profile']).read_text())!=cfg['profile']:raise ValueError('Base feature profile mismatch')
    bv=json.loads((base/manifest['voice_registry']).read_text())['voices'];voices=read_voices(cfg)
    if [(v['speaker_key'],v['style_key'],v['inner_voice_id']) for v in bv] != [(v['speaker_key'],v['style_key'],v['inner_voice_id']) for v in voices]:
        raise ValueError('Base voice registry mismatch; adding/remapping speakers is unsupported')
    sources=inspect_sources(cfg)
    if manifest.get('sources')!=sources['sources']:raise ValueError('Base source revisions/patches mismatch')
    for module,component in components.items():
        config=yaml.safe_load((base/component['config']).read_text());target=_template(cfg,module)
        # Generated network dimensions are part of base compatibility; paths/training counts are not.
        p=len(cfg['profile']['phonemes']);d=cfg['profile']['spectrogram']['num_mels']
        if module!='vocoder':
            network=target['network'];network['speaker_size']=len(voices)
            if module=='acoustic':network.update(input_feature_size=p+1,output_size=d)
            else:network['phoneme_size']=p
            if config.get('network')!=network:raise ValueError(f'{module}: base network config mismatch')
        else:
            for key in ('upsample_rates','upsample_kernel_sizes','upsample_initial_channel','resblock','resblock_kernel_sizes','resblock_dilation_sizes'):
                if config.get(key)!=target[key]:raise ValueError(f'vocoder: base network config mismatch ({key})')
            if config.get('sampling_rate')!=24000 or config.get('hop_size')!=256 or config.get('num_mels')!=d or not config.get('with_hn'):raise ValueError('Vocoder base feature contract mismatch')
    return {'status':'base-ready','base':str(base),'manifest':manifest}


def _inputs(cfg):
    paths=[resolve(cfg,cfg['paths']['corpus']),resolve(cfg,cfg['paths']['derived'])]
    return {str(p):sha256(p) for root in paths for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def train(cfg,module,dry_run=False):
    if module not in MODULES:raise ValueError(f'Unknown training module: {module}')
    run=resolve(cfg,cfg['paths']['run']);view=run/'native'/module
    receipt=json.loads((view/'materialization.json').read_text())
    if receipt.get('status')!='ready' or receipt.get('files')!=_inventory(view):raise ValueError('Materialization files/hash changed; materialize again')
    if receipt.get('profile_sha256') is not None and receipt['profile_sha256']!=profile_hash(cfg):raise ValueError('Materialization profile changed')
    if receipt.get('inputs') is not None and receipt['inputs']!=_inputs(cfg):raise ValueError('Corpus/derived inputs changed; materialize again')
    inspect_sources(cfg)
    python=_python(cfg,module);config=view/f'config.{"json" if module=="vocoder" else "yaml"}';output=run/'native-output'/module
    if not Path(python).is_file():raise FileNotFoundError(python)
    mode=cfg['training'].get('mode','scratch')
    if mode not in ('scratch','finetune','resume'):raise ValueError('Training mode must be scratch, finetune or resume')
    if mode=='finetune':
        base=validate_base(cfg);component=base['manifest']['components'][module];weights=str(Path(base['base'])/component['weights'])
        if module!='vocoder':
            parsed=yaml.safe_load(config.read_text());parsed['train']['pretrained_predictor_path']=weights;parsed['train']['weight_initializer']=None
            config=run/'initialization'/f'{module}.yaml';config.parent.mkdir(parents=True,exist_ok=True);config.write_text(yaml.safe_dump(parsed))
    previous=run/'reports'/f'train-{module}.json'
    if mode=='resume':
        if not previous.exists():raise ValueError('Resume requires the same run and prior training receipt')
        old=json.loads(previous.read_text())
        if old.get('materialization_sha256')!=sha256(view/'materialization.json') or old.get('config_sha256')!=sha256(config):raise ValueError('Resume data/config changed')
        patterns=['g_*','do_*'] if module=='vocoder' else ['snapshot.pth'] if module=='acoustic' else ['trainer_*.pth']
        for pattern in patterns:
            if not list(output.glob(pattern)):raise FileNotFoundError(f'Missing complete resume state {pattern}')
    elif output.exists() and any(output.iterdir()):raise ValueError('Native output already exists; use resume or a new run')
    source=resolve(cfg,cfg['paths']['vendor'])/MODULES[module]
    command=[python,str(source/'train.py')]
    if module=='vocoder':
        command+=['--config',str(config),'--checkpoint_path',str(output),'--input_wavs_dir',str(view/'data/audio24k'),'--input_f0_dir',str(view/'vocoder_f0'),'--input_mels_dir',str(view/'mels'),'--input_training_file',str(view/'train.txt'),'--input_validation_file',str(view/'valid.txt'),'--fine_tuning','True']
        if mode=='finetune':command+=['--pretrained_generator_path',weights]
        for key in ['training_epochs','checkpoint_interval','stdout_interval','summary_interval','validation_interval']:
            if key in _module_options(cfg,module):command+=['--'+key,str(_module_options(cfg,module)[key])]
    else:command += [str(config),str(output)]
    report={'status':'dry-run' if dry_run else 'running','module':module,'mode':mode,'command':command,'config_sha256':sha256(config),'materialization_sha256':sha256(view/'materialization.json')}
    if dry_run:return report
    output.mkdir(parents=True,exist_ok=True);log=run/'logs'/f'{module}.log';log.parent.mkdir(parents=True,exist_ok=True)
    prior={p.name:sha256(p) for p in output.glob('g_*' if module=='vocoder' else 'predictor_*.pth')}
    write_json(previous,report)
    try:
        with log.open('w') as f:subprocess.run(command,cwd=source,env=_env(cfg),stdout=f,stderr=subprocess.STDOUT,check=True)
        checkpoints=list(output.glob('g_*' if module=='vocoder' else 'predictor_*.pth'))
        fresh=[p for p in checkpoints if p.stat().st_size>0 and prior.get(p.name)!=sha256(p)]
        if not fresh:raise FileNotFoundError(f'No fresh predictor checkpoint produced: {output}')
        selected=max(fresh,key=lambda p:p.stat().st_mtime_ns)
        model=run/'models'/module;model.mkdir(parents=True,exist_ok=True)
        destination=model/'model.pth'
        # Check actual state dictionaries for finite tensors; HiFi generator checkpoint is unwrapped.
        script='''import sys,torch
obj=torch.load(sys.argv[1],map_location="cpu",weights_only=True)
if "generator" in obj: obj=obj["generator"]
if not isinstance(obj,dict) or not obj or not all(isinstance(v,torch.Tensor) and torch.isfinite(v).all() for v in obj.values()):raise ValueError("Invalid/nonfinite predictor state_dict")
torch.save({"generator":obj} if sys.argv[3]=="vocoder" else obj,sys.argv[2])
'''
        subprocess.run([python,'-c',script,str(selected),str(destination),module],env=_env(cfg),check=True,capture_output=True,text=True)
        shutil.copyfile(config,model/f'config.{"json" if module=="vocoder" else "yaml"}')
        report.update(status='trained',checkpoint=str(selected),model=str(destination),model_sha256=sha256(destination),log=str(log))
        write_json(previous,report);return report
    except Exception as exc:
        report.update(status='failed',error=str(exc),returncode=getattr(exc,'returncode',None),log=str(log));write_json(previous,report);raise

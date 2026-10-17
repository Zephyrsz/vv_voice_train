"""Pinned upstream ONNX export and executable Core talk contract checks.

Torch and ONNX dependencies are lazy; export may use a separate interpreter.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import resolve, sha256, write_json

MODELS = ('predict_duration','predict_intonation','decode')
INTONATION_VECTORS = ('vowel_phoneme_list','consonant_phoneme_list','start_accent_list',
                     'end_accent_list','start_accent_phrase_list','end_accent_phrase_list')


def interpreter_path(cfg, value):
    # Dereferencing bin/python changes the virtualenv selected by CPython.
    path=Path(value).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else Path(cfg['_root'])/path))


def _profile(cfg):
    p = resolve(cfg,cfg['feature_profile'])
    profile = json.loads(p.read_text())
    if profile.get('sample_rate') != 24000 or profile.get('hop_length') != 256:
        raise ValueError('Export profile requires sample_rate=24000 and hop_length=256')
    phonemes = profile.get('phonemes',profile.get('phoneme_list'))
    if not isinstance(phonemes,list) or not phonemes or len(set(phonemes)) != len(phonemes):
        raise ValueError('Export profile needs an ordered unique phoneme list')
    return profile,phonemes


def check_signature(name, signature, phoneme_size):
    expected = {
        'predict_duration': ([('phoneme_list','int64',[None]),('speaker_id','int64',[1])], [('phoneme_length','float32',[None])]),
        'predict_intonation': ([('length','int64',[])] + [(n,'int64',[None]) for n in INTONATION_VECTORS] + [('speaker_id','int64',[1])], [('f0_list','float32',[None])]),
        'decode': ([('f0','float32',[None,1]),('phoneme','float32',[None,phoneme_size]),('speaker_id','int64',[1])], [('wave','float32',[None])]),
    }[name]
    for kind,want in zip(('inputs','outputs'),expected):
        got=signature.get(kind,[])
        if len(got)!=len(want) or {v['name'] for v in got}!={v[0] for v in want}:
            raise ValueError(f'{name} signature {kind} names mismatch')
        by_name={v['name']:v for v in got}
        for n,dtype,shape in want:
            actual=by_name[n]
            if actual['dtype']!=dtype or len(actual['shape'])!=len(shape):
                raise ValueError(f'{name} signature mismatch for {n}: {actual}')
            for g,w in zip(actual['shape'],shape):
                if w is None:
                    if isinstance(g,int): raise ValueError(f'{name} requires dynamic sequence axis: {n}')
                elif g!=w: raise ValueError(f'{name} signature dimension mismatch: {n}')


def _signature(model):
    import onnx
    types={onnx.TensorProto.INT64:'int64',onnx.TensorProto.FLOAT:'float32'}
    def value(v):
        t=v.type.tensor_type
        return {'name':v.name,'dtype':types.get(t.elem_type,str(t.elem_type)),
                'shape':[d.dim_value if d.HasField('dim_value') else d.dim_param or None for d in t.shape.dim]}
    return {'inputs':[value(v) for v in model.graph.input], 'outputs':[value(v) for v in model.graph.output]}


def _inputs(name,length,phonemes,speaker):
    import numpy as np
    sid=np.array([speaker],dtype=np.int64)
    vowel=next((phonemes.index(v) for v in ['a','o','i'] if v in phonemes),0)
    if name=='predict_duration':
        return {'phoneme_list': np.arange(length,dtype=np.int64)%len(phonemes),'speaker_id':sid}
    if name=='predict_intonation':
        values={'length':np.array(length,dtype=np.int64)}
        for n in INTONATION_VECTORS:
            values[n]=np.full(length,vowel if n=='vowel_phoneme_list' else -1 if n=='consonant_phoneme_list' else 0,dtype=np.int64)
        values['start_accent_phrase_list'][0]=1
        values['end_accent_phrase_list'][-1]=1
        values['speaker_id']=sid
        return values
    onehot=np.zeros((length,len(phonemes)),dtype=np.float32)
    onehot[:,vowel]=1
    f0=np.full((length,1),np.log(150),dtype=np.float32)
    f0[::7]=0
    return {'f0':f0,'phoneme':onehot,'speaker_id':sid}


def validate_onnx(cfg):
    run=resolve(cfg,cfg['paths']['run'])
    for name in MODELS:
        if not (run/'onnx'/f'{name}.onnx').is_file():
            raise FileNotFoundError(f'Missing ONNX model: {run}/onnx/{name}.onnx; run export first')
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError('ONNX validation requires numpy, onnx and onnxruntime in this interpreter') from e
    _,phonemes=_profile(cfg)
    corpus=resolve(cfg,cfg['paths']['corpus'])/'speakers.json'
    voices=json.loads(corpus.read_text())['voices']
    speakers=sorted({v['inner_voice_id'] for v in voices})
    if not speakers: raise ValueError('Empty voice registry')
    reports={}
    for name in MODELS:
        path=run/'onnx'/f'{name}.onnx'
        model=onnx.load(str(path),load_external_data=False)
        if any(v.data_location==onnx.TensorProto.EXTERNAL for v in model.graph.initializer):
            raise ValueError('VVM ONNX must embed weights; external data is unsupported')
        onnx.checker.check_model(model)
        signature=_signature(model)
        check_signature(name,signature,len(phonemes))
        session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
        checks=[]
        for speaker in speakers:
            for length in ([5,11,23] if name!='decode' else [16,31,48]):
                inp=_inputs(name,length,phonemes,speaker)
                out=session.run(None,inp)[0]
                expected=length*256 if name=='decode' else length
                if out.dtype!=np.float32 or out.shape!=(expected,) or not np.isfinite(out).all():
                    raise ValueError(f'{name} runtime output invalid at length={length}, speaker={speaker}: {out.shape}/{out.dtype}')
                if name=='predict_duration' and (out<=0).any(): raise ValueError('Nonpositive predicted phoneme duration')
                if name=='decode' and not np.any(out): raise ValueError('Decode output is completely silent')
                checks.append({'length':length,'speaker_id':speaker,'output_length':out.size,
                               'min':float(out.min()),'max':float(out.max())})
        reports[name]={'sha256':sha256(path),'signature':signature,'runtime_checks':checks}
    result={'status':'onnx-runtime-validated','models':reports,'onnx_version':onnx.__version__,
            'onnxruntime_version':ort.__version__,'profile_sha256':sha256(resolve(cfg,cfg['feature_profile']))}
    write_json(run/'onnx/validation.json',result)
    return result


def evaluate_native(cfg):
    """Run the pinned PyTorch forwarders on a real text and save an honest WAV report."""
    run=resolve(cfg,cfg['paths']['run'])
    for module in ('duration','intonation','acoustic','vocoder'):
        directory=run/'models'/module
        for filename in ('model.pth','config.json' if module=='vocoder' else 'config.yaml'):
            p=directory/filename
            if not p.is_file() or not p.stat().st_size:
                raise FileNotFoundError(f'Missing native evaluation artifact: {p}')
    _profile(cfg)
    from .upstream import inspect_sources
    sources=inspect_sources(cfg)
    output=run/'native-evaluation'; output.mkdir(parents=True,exist_ok=True)
    (output/'receipt.json').unlink(missing_ok=True)
    (output/'report.json').unlink(missing_ok=True)
    request=output/'request.json'; write_json(request,{'config':cfg,'output':str(output)})
    python=str(interpreter_path(cfg,cfg.get('export',{}).get('python',sys.executable)))
    env=os.environ.copy(); env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
    with (output/'evaluation.log').open('w') as log:
        process=subprocess.run([python,'-m','new_vv_train.export','--evaluate-worker',str(request)],stdout=log,stderr=subprocess.STDOUT,env=env,check=False)
    report_path=output/'report.json'
    if process.returncode or not report_path.is_file():
        raise RuntimeError(f'Native PyTorch evaluation failed (exit {process.returncode}); inspect {output}/evaluation.log')
    report=json.loads(report_path.read_text()); report['sources']=sources; report['python']=python
    write_json(output/'receipt.json',report)
    return report


def export_models(cfg):
    run=resolve(cfg,cfg['paths']['run'])
    for module in ('duration','intonation','acoustic','vocoder'):
        directory=run/'models'/module
        for filename in ('model.pth','config.json' if module=='vocoder' else 'config.yaml'):
            p=directory/filename
            if not p.is_file() or not p.stat().st_size: raise FileNotFoundError(f'Missing trained export artifact: {p}')
    _profile(cfg)
    from .upstream import inspect_sources
    source_receipt=inspect_sources(cfg)
    out=run/'onnx'
    out.mkdir(parents=True,exist_ok=True)
    # Invalidate an older success receipt before attempting replacement.
    (out/'export_receipt.json').unlink(missing_ok=True)
    python=str(interpreter_path(cfg,cfg.get('export',{}).get('python',sys.executable)))
    request=out/'export_request.json'
    write_json(request,cfg)
    env=os.environ.copy()
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
    with (out/'export.log').open('w') as log:
        completed=subprocess.run([python,'-m','new_vv_train.export','--worker',str(request)],
                                 stdout=log,stderr=subprocess.STDOUT,env=env,check=False)
    if completed.returncode:
        raise RuntimeError(f'Author export failed (exit {completed.returncode}); inspect {out}/export.log')
    if not (out/'worker_receipt.json').is_file(): raise RuntimeError('Export worker produced no receipt')
    result=json.loads((out/'worker_receipt.json').read_text())
    if result.get('status')!='exported': raise RuntimeError('Export worker did not complete')
    result['sources']=source_receipt
    result['python']=python
    # Validate in the export interpreter as well; its worker writes validation.
    for n in MODELS:
        if result.get('files',{}).get(n)!=sha256(out/f'{n}.onnx'): raise ValueError('Export artifact changed after worker')
    write_json(out/'export_receipt.json',result)
    return result


def _vendor_paths(cfg):
    from .upstream import inspect_sources
    receipt=inspect_sources(cfg)
    for name,source in receipt['sources'].items():
        # The compiled frontend is installed separately and its version is reported.
        if name != 'pyopenjtalk': sys.path.insert(0,source['path'])


def _worker(cfg):
    """Load only source paths from verified, pinned local checkouts."""
    _vendor_paths(cfg)
    import numpy as np
    import torch
    import yaml
    from yukarin_s.config import Config as DurationConfig
    from yukarin_s.network.predictor import create_predictor as duration_predictor
    from yukarin_sa.config import Config as IntonationConfig
    from yukarin_sa.network.predictor import create_predictor as intonation_predictor
    from vv_core_inference.make_yukarin_s_forwarder import WrapperYukarinS
    from vv_core_inference.make_yukarin_sa_forwarder import WrapperYukarinSa
    from vv_core_inference.make_yukarin_sosoa_forwarder import make_yukarin_sosoa_wrapper
    from vv_core_inference.make_decode_forwarder import make_hifigan_wrapper
    from vv_core_inference.utility import remove_weight_norm
    run=resolve(cfg,cfg['paths']['run'])
    out=run/'onnx'
    profile,phonemes=_profile(cfg)
    from vv_core_inference.acoustic_feature_extractor import OjtPhoneme
    if phonemes!=list(OjtPhoneme.phoneme_list): raise ValueError('Profile phoneme order differs from locked author wrapper')
    options=cfg.get('export',{})
    device=options.get('device','cpu')
    opset=int(options.get('opset',17))
    models={}
    sizes=[]
    for name,Config,create,Wrapper in [('duration',DurationConfig,duration_predictor,WrapperYukarinS),('intonation',IntonationConfig,intonation_predictor,WrapperYukarinSa)]:
        directory=run/'models'/name
        config=Config.from_dict(yaml.safe_load((directory/'config.yaml').read_text()))
        sizes.append(config.network.speaker_size)
        if config.network.phoneme_size!=len(phonemes): raise ValueError('Native phoneme vocabulary size differs from export profile')
        net=create(config.network)
        net.load_state_dict(torch.load(directory/'model.pth',map_location=device,weights_only=True))
        net.eval().to(device)
        if name=='intonation': net.apply(remove_weight_norm)
        models['predict_'+name]=Wrapper(net).eval()
    acoustic_config=yaml.safe_load((run/'models/acoustic/config.yaml').read_text())
    if acoustic_config['network']['input_feature_size']!=len(phonemes)+1 or acoustic_config['network']['output_size']!=profile['spectrogram']['num_mels']:
        raise ValueError('Acoustic input/output dimensions differ from feature profile')
    acoustic=make_yukarin_sosoa_wrapper(run/'models/acoustic',device)
    sizes.append(acoustic.speaker_embedder.num_embeddings)
    if len(set(sizes))!=1: raise ValueError('Native modules have different speaker table sizes')
    vocoder_config=json.loads((run/'models/vocoder/config.json').read_text())
    if vocoder_config.get('sampling_rate')!=24000 or int(np.prod(vocoder_config['upsample_rates']))!=256:
        raise ValueError('Vocoder configuration must produce 24000 Hz / hop 256')
    for key,profile_key in [('num_mels','num_mels'),('n_fft','n_fft'),('win_size','win_length'),('fmin','fmin'),('fmax','fmax')]:
        if vocoder_config.get(key)!=profile['spectrogram'][profile_key]: raise ValueError(f'Vocoder {key} differs from feature profile')
    stochastic=bool(vocoder_config.get('with_hn',False))
    if stochastic and options.get('nsf_policy','reject')!='allow-stochastic':
        raise ValueError('NSF uses random excitation; set export.nsf_policy=allow-stochastic to accept finite-output validation without waveform parity')
    vocoder=make_hifigan_wrapper(run/'models/vocoder',device)
    class WaveOnly(torch.nn.Module):
        def __init__(self,acoustic,vocoder):
            super().__init__(); self.acoustic=acoustic; self.vocoder=vocoder
        def forward(self,f0,phoneme,speaker_id):
            return self.vocoder(self.acoustic(f0,phoneme,speaker_id),f0)
    models['decode']=WaveOnly(acoustic,vocoder).eval()
    numerical={}
    import onnxruntime as ort
    speaker=json.loads((resolve(cfg,cfg['paths']['corpus'])/'speakers.json').read_text())['voices'][0]['inner_voice_id']
    for name,wrapper in models.items():
        inp=_inputs(name,16,phonemes,speaker)
        args=tuple(torch.from_numpy(v).to(device) for v in inp.values())
        names=list(inp)
        output_name={'predict_duration':'phoneme_length','predict_intonation':'f0_list','decode':'wave'}[name]
        dynamic={n:{0:'sequence'} for n in names if n not in ('length','speaker_id')}
        dynamic[output_name]={0:'samples' if name=='decode' else 'sequence'}
        exported=torch.jit.script(wrapper) if name=='predict_intonation' else wrapper
        torch.onnx.export(exported,args,str(out/f'{name}.onnx'),opset_version=opset,
                          input_names=names,output_names=[output_name],dynamic_axes=dynamic,
                          do_constant_folding=True,dynamo=False)
        session=ort.InferenceSession(str(out/f'{name}.onnx'),providers=['CPUExecutionProvider'])
        comparisons=[]
        for length in [7,16,29] if name!='decode' else [16,31,48]:
            values=_inputs(name,length,phonemes,speaker)
            with torch.no_grad(): native=wrapper(*(torch.from_numpy(v).to(device) for v in values.values())).cpu().numpy()
            onnx_out=session.run(None,values)[0]
            if native.shape!=onnx_out.shape or not np.isfinite(native).all() or not np.isfinite(onnx_out).all():
                raise ValueError(f'{name} native/ONNX shape or finite comparison failed at {length}')
            parity=not(name=='decode' and stochastic)
            if parity and not np.allclose(native,onnx_out,rtol=float(options.get('rtol',1e-3)),atol=float(options.get('atol',1e-4))):
                raise ValueError(f'{name} native/ONNX numerical mismatch at {length}')
            comparisons.append({'length':length,'parity_checked':parity,
                                'max_abs_error':float(np.max(np.abs(native-onnx_out))) if parity else None})
        numerical[name]=comparisons
    validation=validate_onnx(cfg)
    result={'status':'exported','speaker_size':sizes[0], 'opset':opset,'torch_version':torch.__version__,
            'profile_sha256':sha256(resolve(cfg,cfg['feature_profile'])),
            'files':{n:sha256(out/f'{n}.onnx') for n in MODELS},'numerical_comparisons':numerical,
            'stochastic_nsf':stochastic,'nsf_policy':options.get('nsf_policy','reject'), 'validation':validation,
            'native_artifacts':{str(p.relative_to(run)):sha256(p) for p in (run/'models').rglob('*') if p.is_file()}}
    write_json(out/'worker_receipt.json',result)


def _evaluate_worker(cfg):
    _vendor_paths(cfg['config'])
    import numpy as np, torch, yaml, soundfile as sf, pyopenjtalk, importlib.metadata
    from vv_core_inference.forwarder import Forwarder
    from vv_core_inference.make_yukarin_s_forwarder import make_yukarin_s_forwarder
    from vv_core_inference.make_yukarin_sa_forwarder import make_yukarin_sa_forwarder
    from vv_core_inference.make_decode_forwarder import make_decode_forwarder
    c=cfg['config']; run=resolve(c,c['paths']['run']); out=Path(cfg['output'])
    sample=c.get('validation',{}).get('text','こんにちは。')
    speaker=int(json.loads((resolve(c,c['paths']['corpus'])/'speakers.json').read_text())['voices'][0]['inner_voice_id'])
    duration=make_yukarin_s_forwarder(run/'models/duration','cpu')
    intonation=make_yukarin_sa_forwarder(run/'models/intonation','cpu')
    decode=make_decode_forwarder(run/'models/acoustic',run/'models/vocoder','cpu')
    forwarder=Forwarder(duration,intonation,decode)
    wave, intermediates=forwarder.forward(sample,speaker,speaker,return_intermediate_results=True)
    wave=np.asarray(wave,dtype=np.float32).reshape(-1)
    if not wave.size or not np.isfinite(wave).all() or not np.any(wave): raise ValueError('Native wave output is empty, non-finite or completely silent')
    wav_path=out/'native.wav'; sf.write(wav_path,wave,24000,subtype='PCM_16')
    write_json(out/'report.json',{'status':'native-synthesis-validated','text':sample,'speaker_id':speaker,
        'pyopenjtalk_version':importlib.metadata.version('pyopenjtalk'),'pyopenjtalk_module':pyopenjtalk.__file__,
        'frontend_pin_verified':False,'quality_claim':False,'wave_path':str(wav_path),'wave_samples':int(wave.size),'wave_peak':float(np.max(np.abs(wave))),
        'intermediate_shapes':{k:{n:list(np.asarray(v).shape) for n,v in x.items()} for k,x in intermediates.items()}})


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1] not in ('--worker','--evaluate-worker'): raise SystemExit('Internal worker requires --worker or --evaluate-worker <request.json>')
    request=json.loads(Path(sys.argv[2]).read_text())
    _evaluate_worker(request) if sys.argv[1]=='--evaluate-worker' else _worker(request)

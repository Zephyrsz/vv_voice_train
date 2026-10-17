"""Real isolated Core and existing-service candidate validation, never hot reload."""
from __future__ import annotations
import array
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import wave

from .common import resolve, sha256, write_json
from .vvm import inspect_vvm, existing_style_ids
from .export import interpreter_path


def check_wav(data):
    try:
        with wave.open(io.BytesIO(data),'rb') as w:
            frames,rate,channels,width=w.getnframes(),w.getframerate(),w.getnchannels(),w.getsampwidth()
            raw=w.readframes(frames)
    except (wave.Error,EOFError) as e: raise ValueError('Invalid Core WAV output') from e
    if frames<=0 or not raw: raise ValueError('Core WAV output is empty')
    if rate!=24000 or channels!=1 or width!=2: raise ValueError('Core WAV must be 24000 Hz mono 16-bit PCM')
    if len(raw)!=frames*channels*width: raise ValueError('Core WAV contains truncated sample data')
    samples=array.array('h',raw)
    if not any(samples): raise ValueError('Core WAV output is completely silent')
    return {'frames':frames,'sample_rate':rate,'channels':channels,'sample_width':width,
            'duration_seconds':frames/rate,'peak_pcm':max(abs(x) for x in samples)}


_WORKER = r'''
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback
cfg=json.loads(Path(sys.argv[1]).read_text())
report={"status":"failed", "mode":cfg["mode"], "python":sys.executable,"python_version":sys.version}
try:
    import voicevox_core
    from voicevox_core.blocking import Onnxruntime,OpenJtalk,Synthesizer,VoiceModelFile
    report["core_version"]=importlib.metadata.version("voicevox_core")
    report["core_module"]=voicevox_core.__file__
    report["core_binaries"]=[str(p) for p in Path(voicevox_core.__file__).parent.rglob("*") if p.suffix in (".so",".pyd",".dylib")]
    report["runtime_path"]=cfg["runtime"]
    version=Path(cfg["assets"])/"onnxruntime/VERSION_NUMBER"
    report["runtime_version"]=version.read_text().strip() if version.exists() else None
    report["dictionary_path"]=cfg["dictionary"]
    if report["core_version"]!=cfg["core_version"]:
        raise ValueError("Configured Core version differs from actual loaded binary: "+report["core_version"])
    if cfg["mode"]=="service":
        sys.path.insert(0,cfg["service_path"])
        from local_engine import LocalVoicevoxEngine
        engine=LocalVoicevoxEngine(Path(cfg["assets"]),Path(cfg["model_dir"]),cpu_num_threads=2)
        speakers=engine.get_speakers()
        visible={style["id"] for s in speakers for style in s["styles"]}
        if not set(cfg["style_ids"]).issubset(visible): raise ValueError("Candidate styles not visible through get_speakers")
        report["visible_style_ids"]=sorted(visible)
        outputs=[]
        for style in cfg["style_ids"]:
            wav=engine.synthesize(cfg["text"],style,{})
            output=Path(cfg["output"])/("service-"+str(style)+".wav")
            output.write_bytes(wav)
            outputs.append(str(output))
        # Exercise an existing voice to verify lazy loading and coexistence.
        existing=sorted(visible-set(cfg["style_ids"]))
        if existing:
            output=Path(cfg["output"])/"service-existing.wav"
            output.write_bytes(engine.synthesize(cfg["text"],existing[0],{}))
            outputs.append(str(output))
        report["wavs"]=outputs
    else:
        runtime=Onnxruntime.load_once(filename=cfg["runtime"])
        synth=Synthesizer(runtime,OpenJtalk(cfg["dictionary"]),acceleration_mode="CPU",cpu_num_threads=2)
        with VoiceModelFile.open(cfg["candidate"]) as model:
            report["opened_model_id"]=str(model.id)
            synth.load_voice_model(model)
        outputs=[]
        for style in cfg["style_ids"]:
            query=synth.create_audio_query(cfg["text"],style)
            query.output_sampling_rate=24000
            query.output_stereo=False
            query.validate()
            wav=synth.synthesis(query,style)
            output=Path(cfg["output"])/("core-"+str(style)+".wav")
            output.write_bytes(wav)
            outputs.append(str(output))
        report["wavs"]=outputs
    report["status"]="synthesized"
except Exception as e:
    report["error_type"]=type(e).__name__
    report["error"]=str(e)
    report["traceback"]=traceback.format_exc()
finally:
    Path(cfg["report"]).write_text(json.dumps(report,ensure_ascii=False,indent=2))
if report["status"]!="synthesized": sys.exit(1)
'''


def _validate(cfg,mode):
    run=resolve(cfg,cfg['paths']['run'])
    candidate=run/'vvm/candidate.vvm'
    if not candidate.is_file(): raise FileNotFoundError(f'Missing candidate vvm: {candidate}; run package first')
    info=inspect_vvm(candidate)
    core=cfg.get('core',{})
    python=interpreter_path(cfg,core.get('python',''))
    if not core.get('python') or not python.is_file(): raise FileNotFoundError(f'Core Python interpreter missing: {python}')
    assets=resolve(cfg,core.get('assets','../vv_tts_ws_server/assets'))
    runtime=resolve(cfg,core['runtime']) if core.get('runtime') else assets/'onnxruntime/lib/libvoicevox_onnxruntime.1.17.3.dylib'
    dictionary=resolve(cfg,core['dictionary']) if core.get('dictionary') else assets/'dict/open_jtalk_dic_utf_8-1.11'
    if not runtime.is_file(): raise FileNotFoundError(f'Core Runtime binary missing: {runtime}')
    if not (dictionary/'sys.dic').is_file(): raise FileNotFoundError(f'Core dictionary missing: {dictionary}')
    output=run/'validation'/mode
    output.mkdir(parents=True,exist_ok=True)
    report_path=output/'worker_report.json'
    report_path.unlink(missing_ok=True)
    request={'mode':mode,'candidate':str(candidate),'style_ids':info['style_ids'],
             'core_version':core.get('version','0.17.0'),'assets':str(assets),'runtime':str(runtime),
             'dictionary':str(dictionary),'text':cfg.get('validation',{}).get('text','こんにちは。'),
             'output':str(output),'report':str(report_path)}
    # Candidate and existing model files stay untouched; links are isolated.
    with tempfile.TemporaryDirectory(prefix='new-vv-candidate-') as temp:
        if mode=='service':
            existing=resolve(cfg,core.get('existing_models',str(assets/'models')))
            ids=existing_style_ids(existing)
            collisions=set(info['style_ids']) & set(ids)
            if collisions: raise ValueError(f'Candidate public style ID collision: {sorted(collisions)}')
            model_dir=Path(temp)/'models'
            model_dir.mkdir()
            for i,p in enumerate(sorted(existing.glob('*.vvm'))): (model_dir/f'existing-{i}.vvm').symlink_to(p.resolve())
            (model_dir/'candidate.vvm').symlink_to(candidate.resolve())
            service=resolve(cfg,core.get('service_path',str(assets.parent)))
            if not (service/'local_engine.py').is_file(): raise FileNotFoundError(f'Service adapter missing: {service}/local_engine.py')
            request.update(model_dir=str(model_dir),service_path=str(service))
        request_path=output/'request.json'
        write_json(request_path,request)
        env=os.environ.copy()
        env.pop('PYTHONPATH',None)
        with (output/'validation.log').open('w') as log:
            try:
                process=subprocess.run([str(python),'-I','-c',_WORKER,str(request_path)],
                                       stdout=log,stderr=subprocess.STDOUT,env=env,
                                       timeout=cfg.get('validation',{}).get('timeout_seconds',300),check=False)
            except subprocess.TimeoutExpired as e:
                write_json(output/'receipt.json',{'status':'failed','error':'Core subprocess timeout','candidate_sha256':sha256(candidate)})
                raise RuntimeError(f'{mode} validation timed out; inspect {output}/validation.log') from e
    if not report_path.is_file(): raise RuntimeError(f'{mode} subprocess produced no report (exit {process.returncode}); inspect {output}/validation.log')
    report=json.loads(report_path.read_text())
    report.update(candidate_sha256=sha256(candidate),runtime_sha256=sha256(runtime),returncode=process.returncode)
    for binary in report.get('core_binaries',[]):
        if Path(binary).is_file(): report.setdefault('core_binary_sha256',{})[binary]=sha256(binary)
    if process.returncode or report.get('status')!='synthesized':
        write_json(output/'receipt.json',report)
        raise RuntimeError(f'{mode} validation failed: {report.get("error","subprocess failed")}; inspect {output}/receipt.json')
    wavs=[]
    try:
        for path in report.get('wavs',[]):
            p=Path(path)
            wavs.append({'path':str(p),'sha256':sha256(p),**check_wav(p.read_bytes())})
        if not wavs: raise ValueError('Core produced no WAV outputs')
    except (ValueError,FileNotFoundError) as e:
        report.update(status='failed',error=str(e))
        write_json(output/'receipt.json',report)
        raise
    report.update(status='service-validated' if mode=='service' else 'core-validated',wavs=wavs,
                  target=core.get('target','existing-service-core'),quality_claim=False)
    write_json(output/'receipt.json',report)
    return report


def validate_core(cfg):
    return _validate(cfg,'core')


def validate_service(cfg):
    return _validate(cfg,'service')

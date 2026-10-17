import json, subprocess, sys
from pathlib import Path
import pytest
import soundfile as sf
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from new_vv_train import upstream, training


def project(tmp_path):
    root = tmp_path
    corpus = root/'corpora'/'new_voice_v1'; derived=corpus/'derived'
    (derived/'audio24k').mkdir(parents=True); (derived/'frontend').mkdir(); (derived/'phoneme_list').mkdir(); (derived/'f0').mkdir(); (derived/'volume').mkdir()
    (derived/'phoneme').mkdir(); (derived/'spec').mkdir(); (derived/'silence').mkdir()
    (derived/'start_accent').mkdir(); (derived/'end_accent').mkdir(); (derived/'start_accent_phrase').mkdir(); (derived/'end_accent_phrase').mkdir()
    (corpus/'splits').mkdir(); (corpus/'raw').mkdir()
    rows=[]
    for stem,speaker in [('a','voice_a'),('b','voice_a'),('c','voice_a')]:
        rows.append({'utterance_id':stem,'audio':f'raw/{stem}.wav','speaker_key':speaker,'style_key':'neutral','text':stem,'language':'ja'})
        sf.write(corpus/'raw'/f'{stem}.wav', 0.1*np.sin(np.arange(1024,dtype=np.float32)*(1+ord(stem)/100)), 24000)
        sf.write(derived/'audio24k'/f'{stem}.wav', np.zeros(1024,dtype=np.float32),24000)
        for d in ('phoneme_list','f0','volume','phoneme','spec','silence','start_accent','end_accent','start_accent_phrase','end_accent_phrase'):
            p=derived/d/f'{stem}.lab' if d in ('phoneme_list',) else derived/d/f'{stem}.txt' if 'accent' in d else derived/d/f'{stem}.npz'
            p.write_text('0.0000 0.0213 pau\n0.0213 0.0427 a\n' if d=='phoneme_list' else '0 0\n')
    (corpus/'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    (corpus/'speakers.json').write_text(json.dumps({'schema_version':1,'voices':[{'speaker_key':'voice_a','style_key':'neutral','inner_voice_id':0,'public_style_id':10000}]}))
    for name,ids in [('train',['a']),('valid',['b']),('test',['c'])]: (corpus/'splits'/f'{name}.txt').write_text('\n'.join(ids)+'\n')
    cfg = {'_root':str(root), 'paths':{'corpus':'corpora/new_voice_v1','derived':'corpora/new_voice_v1/derived','vendor':'vendor','run':'runs/first_run','base':None},'training':{'mode':'scratch','python':sys.executable,'modules':{}},'profile':json.loads((Path(__file__).parents[1]/'configs/feature_profile.json').read_text())}
    (derived/'feature_profile.json').write_text(json.dumps(cfg['profile']))
    for stem in ('a','b','c'):
        (derived/'frontend'/f'{stem}.json').parent.mkdir(exist_ok=True); (derived/'frontend'/f'{stem}.json').write_text(json.dumps({'phonemes':['pau','a']}))
    return cfg


def test_materialize_preserves_upstream_speaker_mapping(tmp_path):
    cfg=project(tmp_path)
    # make valid npz payloads for required adapters
    import numpy as np
    d=tmp_path/'corpora/new_voice_v1/derived'
    for stem in ('a','b','c'):
        for feat in ('f0','volume','phoneme','spec','silence'):
            array=np.ones((4,len(cfg['profile']['phonemes']) if feat=='phoneme' else cfg['profile']['spectrogram']['num_mels'] if feat=='spec' else 1),dtype=np.float32)
            if feat=='phoneme':
                array[:]=0;array[:2,0]=1;array[2:,cfg['profile']['phonemes'].index('a')]=1
            if feat=='silence':array=np.zeros((4,1),dtype=bool)
            np.savez(d/feat/f'{stem}.npz', array=array, rate=93.75)
    cfg['paths']['vendor']=str(Path(__file__).parents[1]/'vendor')
    cfg['upstream_lock']=str(Path(__file__).parents[1]/'upstream.lock.json')
    lock=json.loads(Path(cfg['upstream_lock']).read_text())
    for source in lock['sources'].values():
        for patch in source['patches']:
            patch['path']=str(Path(__file__).parents[1]/patch['path'])
    # use root aliases preserving receipt patch paths by copying patch files
    import shutil
    shutil.copytree(Path(__file__).parents[1]/'patches',tmp_path/'patches')
    _report={'profile_sha256':training.profile_hash(cfg),'items':[]}
    for uid in ('a','b','c'):
        wav=d/'audio24k'/f'{uid}.wav'; lab=d/'phoneme_list'/f'{uid}.lab'; front=d/'frontend'/f'{uid}.json'
        _report['items'].append({'utterance_id':uid,'audio_sha256':training.sha256(wav),'alignment_sha256':training.sha256(lab),'frontend_sha256':training.sha256(front),'feature_sha256':{f:training.sha256(d/f/f'{uid}.npz') for f in ('f0','volume','phoneme','spec','silence')}})
    training.write_json(Path(cfg['_root'])/'runs/first_run/reports/extract-features.json',_report)
    out=training.materialize(cfg)
    speaker=json.loads((Path(out['modules']['duration'])/'speaker_dict.json').read_text())
    assert list(speaker) == ['voice_a::neutral']
    assert speaker['voice_a::neutral'] == ['a','b']
    assert out['status']=='ready'


def test_materialize_rejects_missing_derived_data(tmp_path):
    cfg=project(tmp_path)
    (tmp_path/'corpora/new_voice_v1/derived/f0/a.npz').unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError): training.materialize(cfg)


def test_train_dry_run_never_reports_success(tmp_path):
    cfg=project(tmp_path); cfg['training']['modules']={'duration':{'command':[sys.executable,'-c','raise SystemExit(9)']}}
    # bypass data stage by injecting pre-materialized config
    cfg['training']['modules']['duration']['config']='configs/modules/duration.yaml'
    with pytest.raises(FileNotFoundError): training.train(cfg,'duration',dry_run=True)


def test_validate_base_checks_hash(tmp_path):
    cfg=project(tmp_path); base=tmp_path/'bases'/'x'; base.mkdir(parents=True)
    (base/'base.json').write_text(json.dumps({'schema_version':1,'components':{'duration':{'weights':'model.pth','sha256':'00'}}}))
    (base/'model.pth').write_bytes(b'actual'); cfg['paths']['base']='bases/x'
    with pytest.raises(ValueError, match='sha256'): training.validate_base(cfg)


def snapshot(cfg, name='yukarin_s'):
    from new_vv_train.common import sha256
    root=Path(cfg['_root']); source=root/'vendor'/name; source.mkdir(parents=True)
    (source/'train.py').write_text('raise SystemExit(7)\n')
    (root/'upstream.lock.json').write_text(json.dumps({'schema_version':1,'sources':{name:{'repository':'https://github.com/Hiroshiba/yukarin_s','revision':'a'*40,'patches':[]}}}))
    (source/'.source-receipt.json').write_text(json.dumps({'revision':'a'*40,'repository':'https://github.com/Hiroshiba/yukarin_s','patches':[], 'files':{'train.py':sha256(source/'train.py')}}))
    return source


def test_source_receipt_rejects_locked_revision_mismatch(tmp_path):
    cfg=project(tmp_path); source=snapshot(cfg)
    r=json.loads((source/'.source-receipt.json').read_text()); r['revision']='b'*40
    (source/'.source-receipt.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='revision'):upstream.inspect_sources(cfg)


def test_source_receipt_rejects_tampered_file(tmp_path):
    cfg=project(tmp_path); source=snapshot(cfg); (source/'train.py').write_text('tampered')
    with pytest.raises(ValueError,match='hash'):upstream.inspect_sources(cfg)


def configured_run(tmp_path):
    cfg=project(tmp_path); source=snapshot(cfg)
    native=tmp_path/'runs/first_run/native/duration'; native.mkdir(parents=True)
    (native/'config.yaml').write_text('train: {}\n')
    (native/'materialization.json').write_text(json.dumps({'status':'ready','files':{'config.yaml':__import__('hashlib').sha256((native/'config.yaml').read_bytes()).hexdigest()}}))
    return cfg,source,native


def test_train_propagates_real_subprocess_failure(tmp_path):
    cfg,source,native=configured_run(tmp_path)
    with pytest.raises(subprocess.CalledProcessError): training.train(cfg,'duration')
    report=json.loads((tmp_path/'runs/first_run/reports/train-duration.json').read_text())
    assert report['status']=='failed' and report['returncode']==7


def test_train_requires_fresh_checkpoint(tmp_path):
    cfg,source,native=configured_run(tmp_path); (source/'train.py').write_text('pass\n')
    from new_vv_train.common import sha256
    r=json.loads((source/'.source-receipt.json').read_text());r['files']['train.py']=sha256(source/'train.py');(source/'.source-receipt.json').write_text(json.dumps(r))
    with pytest.raises(FileNotFoundError,match='checkpoint'): training.train(cfg,'duration')


def test_prepared_dry_run_does_not_launch_or_create_completion(tmp_path):
    cfg,source,native=configured_run(tmp_path)
    result=training.train(cfg,'duration',dry_run=True)
    assert result['status']=='dry-run'
    assert not (tmp_path/'runs/first_run/models/duration/model.pth').exists()


def test_materialize_exact_native_splits_and_hifigan_shapes(tmp_path):
    cfg=project(tmp_path)
    package=Path(__file__).parents[1]
    cfg['paths']['vendor']=str(package/'vendor');cfg['upstream_lock']=str(package/'upstream.lock.json')
    import shutil,numpy as np
    shutil.copytree(package/'patches',tmp_path/'patches')
    d=tmp_path/'corpora/new_voice_v1/derived'
    for stem in ('a','b','c'):
        for feat in ('f0','volume','phoneme','spec','silence'):
            array=np.ones((4,len(cfg['profile']['phonemes']) if feat=='phoneme' else cfg['profile']['spectrogram']['num_mels'] if feat=='spec' else 1),dtype=np.float32)
            if feat=='phoneme':
                array[:]=0;array[:2,0]=1;array[2:,cfg['profile']['phonemes'].index('a')]=1
            if feat=='silence':array=np.zeros((4,1),dtype=bool)
            np.savez(d/feat/f'{stem}.npz',array=array,rate=93.75)
    _report={'profile_sha256':training.profile_hash(cfg),'items':[]}
    for uid in ('a','b','c'):
        wav=d/'audio24k'/f'{uid}.wav'; lab=d/'phoneme_list'/f'{uid}.lab'; front=d/'frontend'/f'{uid}.json'
        _report['items'].append({'utterance_id':uid,'audio_sha256':training.sha256(wav),'alignment_sha256':training.sha256(lab),'frontend_sha256':training.sha256(front),'feature_sha256':{f:training.sha256(d/f/f'{uid}.npz') for f in ('f0','volume','phoneme','spec','silence')}})
    training.write_json(Path(cfg['_root'])/'runs/first_run/reports/extract-features.json',_report)
    report=training.materialize(cfg)
    assert report['native_dataset_loading']=='verified'
    v=Path(report['modules']['vocoder'])
    assert np.load(v/'vocoder_f0/a.npy',allow_pickle=True).item()['array'].shape==(4,)
    assert np.load(v/'mels/a.npy').shape==(80,4)
    assert not (v/'mels/c.npy').exists()
    for module in ('duration','intonation','acoustic'):
        assert json.loads((Path(report['modules'][module])/'splits.json').read_text())=={'train':['a'],'valid':['b']}

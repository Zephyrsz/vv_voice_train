"""Compatibility tests use synthetic contracts; they do not assert voice quality."""
import io
import json
import tempfile
import unittest
import uuid
import wave
import zipfile
from pathlib import Path

from new_vv_train.export import export_models, check_signature, validate_onnx, evaluate_native
from new_vv_train.vvm import inspect_vvm, package
from new_vv_train.core_validation import check_wav, validate_core


class ExportContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = {'_root': str(self.root), 'paths': {'run': 'runs/test', 'corpus': 'corpus'},
                    'feature_profile': 'profile.json', 'core': {}}

    def tearDown(self):
        self.tmp.cleanup()

    def archive(self, change=None, extra=None):
        manifest = {'vvm_format_version': 2, 'id': str(uuid.uuid4()), 'metas_filename': 'metas.json',
                    'talk': {k: {'type': 'onnx', 'filename': k+'.onnx'} for k in
                             ['predict_duration', 'predict_intonation', 'decode']}}
        manifest['talk']['style_id_to_inner_voice_id'] = {'1000000': 0}
        metas = [{'name': 'Test', 'speaker_uuid': str(uuid.uuid4()), 'version': '0.0.1',
                  'styles': [{'name': 'Neutral', 'id': 1000000, 'type': 'talk'}]}]
        if change: change(manifest, metas)
        path = self.root/'test.vvm'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('manifest.json', json.dumps(manifest))
            z.writestr('metas.json', json.dumps(metas))
            for n in ['predict_duration', 'predict_intonation', 'decode']: z.writestr(n+'.onnx', b'fixture')
            if extra: z.writestr(extra, b'danger')
        return path

    def test_inspect_valid_talk_metadata(self):
        result = inspect_vvm(self.archive())
        self.assertEqual(result['style_ids'], [1000000])
        self.assertEqual(result['format_version'], 2)

    def test_reject_duplicate_style_ids(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate.*style'):
            inspect_vvm(self.archive(lambda _, m: m[0]['styles'].append(dict(m[0]['styles'][0]))))

    def test_reject_incomplete_mapping(self):
        with self.assertRaisesRegex(ValueError, 'mapping'):
            inspect_vvm(self.archive(lambda m, _: m['talk'].update(style_id_to_inner_voice_id={})))

    def test_reject_archive_path_escape(self):
        with self.assertRaisesRegex(ValueError, 'path'):
            inspect_vvm(self.archive(extra='../escape'))

    def test_reject_invalid_uuid(self):
        with self.assertRaisesRegex(ValueError, 'UUID'):
            inspect_vvm(self.archive(lambda m, _: m.update(id='not-a-uuid')))

    def test_package_requires_actual_export_receipt(self):
        with self.assertRaisesRegex((ValueError, FileNotFoundError), 'export'):
            package(self.cfg)
        self.assertFalse(list(self.root.rglob('*.vvm')))

    def test_export_rejects_missing_checkpoint_before_optional_dependencies(self):
        with self.assertRaisesRegex(FileNotFoundError, 'model.pth'):
            export_models(self.cfg)

    def test_native_evaluation_rejects_missing_checkpoint(self):
        with self.assertRaisesRegex(FileNotFoundError,'model.pth'):
            evaluate_native(self.cfg)
        self.assertFalse((self.root/'runs/test/native-evaluation/receipt.json').exists())

    def test_signature_rejects_wrong_types_and_fixed_sequence(self):
        good = {'inputs': [{'name': 'phoneme_list', 'dtype': 'int64', 'shape': ['N']},
                           {'name': 'speaker_id', 'dtype': 'int64', 'shape': [1]}],
                'outputs': [{'name': 'phoneme_length', 'dtype': 'float32', 'shape': ['N']}]}
        check_signature('predict_duration', good, 45)
        good['inputs'][0]['dtype'] = 'float32'
        with self.assertRaisesRegex(ValueError, 'signature'): check_signature('predict_duration', good, 45)
        good['inputs'][0].update(dtype='int64', shape=[12])
        with self.assertRaisesRegex(ValueError, 'dynamic'): check_signature('predict_duration', good, 45)

    def test_core_requires_candidate_package(self):
        with self.assertRaisesRegex(FileNotFoundError, 'vvm'):
            validate_core(self.cfg)

    def test_interpreter_path_keeps_virtualenv_symlink(self):
        from new_vv_train.export import interpreter_path
        executable=self.root/'venv/bin/python'
        executable.parent.mkdir(parents=True)
        executable.symlink_to('/usr/bin/python3')
        self.assertEqual(interpreter_path(self.cfg,'venv/bin/python'),executable)

    def test_core_output_must_be_nonempty_24k_mono_pcm(self):
        def wav(frames, rate=24000):
            b=io.BytesIO()
            with wave.open(b, 'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(frames)
            return b.getvalue()
        with self.assertRaisesRegex(ValueError, 'empty'): check_wav(wav(b''))
        with self.assertRaisesRegex(ValueError, '24000'): check_wav(wav(b'\x01\x00'*8, 44100))
        with self.assertRaisesRegex(ValueError, 'silent'): check_wav(wav(b'\x00\x00'*8))
        self.assertEqual(check_wav(wav(b'\x01\x00'*8))['frames'], 8)

class ExecutableOnnxContracts(ExportContracts):
    def setUp(self):
        super().setUp()
        try:
            import onnx
            import onnxruntime
        except ImportError:
            self.skipTest('optional ONNX runtime dependencies not installed')
        self.phonemes=['pau','a','i']
        (self.root/'profile.json').write_text(json.dumps({'sample_rate':24000,'hop_length':256,'phonemes':self.phonemes}))
        (self.root/'corpus').mkdir()
        (self.root/'corpus/speakers.json').write_text(json.dumps({'voices':[{'speaker_key':'test','style_key':'neutral','public_style_id':1000000,'inner_voice_id':0,'name':'Test','style_name':'Neutral'}]}))
        self.out=self.root/'runs/test/onnx'
        self.out.mkdir(parents=True)
        self.make_models()

    def make_models(self):
        import onnx
        from onnx import helper as h, TensorProto as T
        def vi(n,t,shape): return h.make_tensor_value_info(n,t,shape)
        def save(name,inputs,output,nodes,inits):
            model=h.make_model(h.make_graph(nodes,name,inputs,[output],initializer=inits),opset_imports=[h.make_opsetid('',17)])
            model.ir_version=9
            onnx.save(model,str(self.out/(name+'.onnx')))
        scalar=h.make_tensor('value',T.FLOAT,[],[0.1])
        zero=h.make_tensor('zero',T.FLOAT,[],[0.0])
        duration_nodes=[h.make_node('Cast',['phoneme_list'],['x'],to=T.FLOAT),h.make_node('Mul',['x','zero'],['y']),h.make_node('Add',['y','value'],['phoneme_length'])]
        save('predict_duration',[vi('phoneme_list',T.INT64,['N']),vi('speaker_id',T.INT64,[1])],vi('phoneme_length',T.FLOAT,['N']),duration_nodes,[scalar,zero])
        names=['vowel_phoneme_list','consonant_phoneme_list','start_accent_list','end_accent_list','start_accent_phrase_list','end_accent_phrase_list']
        save('predict_intonation',[vi('length',T.INT64,[])]+[vi(n,T.INT64,['M']) for n in names]+[vi('speaker_id',T.INT64,[1])],vi('f0_list',T.FLOAT,['M']),[h.make_node('Cast',['vowel_phoneme_list'],['x'],to=T.FLOAT),h.make_node('Mul',['x','zero'],['y']),h.make_node('Add',['y','value'],['f0_list'])],[scalar,zero])
        save('decode',[vi('f0',T.FLOAT,['T',1]),vi('phoneme',T.FLOAT,['T',3]),vi('speaker_id',T.INT64,[1])],vi('wave',T.FLOAT,['S']),[h.make_node('Reshape',['f0','flat'],['x']),h.make_node('Tile',['x','repeat'],['wave'])],[h.make_tensor('flat',T.INT64,[1],[-1]),h.make_tensor('repeat',T.INT64,[1],[256])])

    def receipt(self):
        from new_vv_train.common import sha256
        artifact=self.root/'runs/test/models/duration/model.pth'; artifact.parent.mkdir(parents=True,exist_ok=True); artifact.write_bytes(b'synthetic fixture')
        receipt={'status':'exported','speaker_size':1,'profile_sha256':sha256(self.root/'profile.json'),'files':{n:sha256(self.out/(n+'.onnx')) for n in ['predict_duration','predict_intonation','decode']},'sources':{'status':'source-ready'},'native_artifacts':{'models/duration/model.pth':sha256(artifact)}}
        (self.out/'export_receipt.json').write_text(json.dumps(receipt))

    def test_runtime_checks_multiple_dynamic_lengths(self):
        result=validate_onnx(self.cfg)
        self.assertEqual([r['output_length'] for r in result['models']['decode']['runtime_checks']],[4096,7936,12288])

    def test_packaged_candidate_preserves_public_to_inner_mapping(self):
        self.receipt()
        result=package(self.cfg)
        inspected=inspect_vvm(result['path'])
        self.assertEqual(inspected['mapping'],{'1000000':0})
        self.assertEqual(set(inspected['manifest']['talk']),{'predict_duration','predict_intonation','decode','style_id_to_inner_voice_id'})

    def test_package_rejects_changed_onnx_after_export(self):
        self.receipt()
        with (self.out/'decode.onnx').open('ab') as f: f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'): package(self.cfg)

    def test_package_rejects_changed_native_weights_after_export(self):
        self.receipt()
        (self.root/'runs/test/models/duration/model.pth').write_bytes(b'new weights')
        with self.assertRaisesRegex(ValueError,'native artifact'): package(self.cfg)

    def test_package_rejects_collision_with_existing_models(self):
        self.receipt()
        existing=self.root/'existing'; existing.mkdir()
        archive=self.archive(); archive.rename(existing/'existing.vvm')
        self.cfg['core']['existing_models']='existing'
        with self.assertRaisesRegex(ValueError,'collision'): package(self.cfg)

    def test_runtime_rejects_decode_fixed_length(self):
        import onnx
        model=onnx.load(str(self.out/'decode.onnx'))
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_value=16
        onnx.save(model,str(self.out/'decode.onnx'))
        with self.assertRaisesRegex(ValueError,'dynamic'): validate_onnx(self.cfg)

    def test_package_rejects_unverified_source_receipt(self):
        self.receipt()
        r=json.loads((self.out/'export_receipt.json').read_text()); r.pop('sources')
        (self.out/'export_receipt.json').write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'source'): package(self.cfg)

if __name__ == '__main__': unittest.main()

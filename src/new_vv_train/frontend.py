"""Japanese frontend using the author's full-context parser."""

import ast
import importlib.util
import json
from pathlib import Path

from .common import record, resolve, sha256, write_json
from .corpus import read_manifest


ACCENTS = ('start_accent', 'end_accent', 'start_accent_phrase', 'end_accent_phrase')


def load_source_module(name, path):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_vocabulary(actual, expected):
    if list(actual) != list(expected):
        raise ValueError('Phoneme vocabulary order differs from author/Core')


def source_vocabulary(cfg):
    source = resolve(cfg, cfg['paths']['vendor']) / 'yukarin_sosoa/yukarin_sosoa/data/phoneme.py'
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'OjtPhoneme':
            for item in node.body:
                if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'phoneme_list' for t in item.targets):
                    vocabulary = ast.literal_eval(item.value)
                    validate_vocabulary(vocabulary, cfg['profile']['phonemes'])
                    return list(vocabulary)
    raise ValueError('Could not find author OjtPhoneme vocabulary')


def accents_from_contexts(contexts):
    result = {name: [] for name in ACCENTS}
    type1 = False
    for ctx in contexts:
        end = ctx['a1'] == '0'
        if ctx['a2'] == '1':
            type1 = end
        result['start_accent'].append(int((ctx['a2'] == '1' and type1) or (ctx['a2'] == '2' and not type1)))
        result['end_accent'].append(int(end))
        result['start_accent_phrase'].append(int(ctx['a2'] == '1'))
        result['end_accent_phrase'].append(int(ctx['a3'] == '1'))
    return result


def frontend(cfg):
    from .upstream import inspect_sources
    inspect_sources(cfg)
    vocabulary = source_vocabulary(cfg)
    parser = resolve(cfg, cfg['paths']['vendor']) / 'vv_core_inference/vv_core_inference/full_context_label.py'
    import os
    dictionary = resolve(cfg, cfg.get('frontend', {}).get('dictionary', str(resolve(cfg, cfg['core']['assets']) / 'dict/open_jtalk_dic_utf_8-1.11')))
    if not (dictionary / 'sys.dic').is_file():
        raise FileNotFoundError(f'Frontend dictionary missing: {dictionary}')
    # Set before importing pyopenjtalk to avoid automatic network dictionary download.
    os.environ['OPEN_JTALK_DICT_DIR'] = str(dictionary)
    author = load_source_module('_vv_author_labels', parser)
    import pyopenjtalk
    # Tools may run repeatedly in one process; construct from explicit dictionary.
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=str(dictionary).encode('utf-8'))
    derived = resolve(cfg, cfg['paths']['derived'])
    corpus, rows = read_manifest(cfg)
    items = []
    for row in rows:
        uid = row['utterance_id']
        # A reviewed fullcontext override regenerates all downstream arrays consistently.
        override = corpus / 'corrections/full_context' / f'{uid}.lab'
        if override.exists():
            labels = override.read_text().splitlines()
            utterance = author.Utterance.from_phonemes([author.Phoneme.from_label(l) for l in labels])
        else:
            data = jtalk.run_frontend(row.get('normalized_text', row['text']))
            labels = jtalk.make_label(data)
            utterance = author.Utterance.from_phonemes([author.Phoneme.from_label(l) for l in labels])
        phonemes = utterance.phonemes
        symbols = [p.phoneme for p in phonemes]
        for i in (0, len(symbols) - 1):
            if symbols[i] in ('sil', 'silB', 'silE'):
                symbols[i] = 'pau'
        if any(p not in vocabulary for p in symbols):
            raise ValueError(f'Unsupported frontend phonemes in {uid}: {symbols}')
        arrays = accents_from_contexts([p.contexts for p in phonemes])
        structured = dict(utterance_id=uid, phonemes=symbols, contexts=[p.contexts for p in phonemes], accents=arrays, text=row.get('normalized_text', row['text']), reading_review=row.get('reading'), source='reviewed_full_context' if override.exists() else 'automatic', pyopenjtalk_version=getattr(pyopenjtalk, '__version__', 'unknown'), parser_sha256=sha256(parser), dictionary_path=str(dictionary), dictionary_sys_sha256=sha256(dictionary / 'sys.dic'))
        target = derived / 'frontend' / f'{uid}.json'
        write_json(target, structured)
        lab = derived / 'full_context' / f'{uid}.lab'
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text('\n'.join(p.label for p in phonemes) + '\n')
        for name, values in arrays.items():
            p = derived / name / f'{uid}.txt'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(' '.join(map(str, values)) + '\n')
        items.append(dict(utterance_id=uid, phonemes=len(symbols), frontend_sha256=sha256(target)))
    return record(cfg, 'frontend', dict(status='generated-needs-review', items=items))

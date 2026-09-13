#!/usr/bin/env python3
"""Prepare/evaluate the frozen source challenge; solve with unchanged trigram.py."""
import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import random
import statistics
from pathlib import Path
import numpy as np
from diagnose import c, v, read_pickle
from trigram import counts, score


def prepare(root, data, out):
    assert not (out/'private/truth.pkl').exists(), 'Do not replace an existing freeze'
    src = pickle.loads(gzip.decompress((root/'edition_preflight.pkl.gz').read_bytes()))
    overlap = json.loads((root/'edition_overlap.json').read_text())
    assert all(not hits for fam in overlap['hits'].values() for hits in fam.values())
    old_path = data/'coverage_models/models.pkl'
    assert c.sha_file(old_path) == '1bf2db4056028510a8b7f9a7eb7b168eeda20ea93f9cc090c7973091bbc87463'
    old = pickle.loads(old_path.read_bytes())
    old_truth = read_pickle(data/'result/checkpoint.pkl')['truth']['model_truth']
    root_seed = os.urandom(32)
    order = ['A', 'B']; random.Random(c.sseed(root_seed, 'new_model_alias')).shuffle(order)
    models = {new:old[prev] for new,prev in zip(['A','B'],order)}
    model_truth = {new:old_truth[prev] for new,prev in zip(['A','B'],order)}
    for new,prev in zip(['A','B'],order):
        assert np.array_equal(models[new]['lp'], old[prev]['lp'])
        assert models[new]['train_uni'] == old[prev]['train_uni']
    c.atomic_pickle(models, out/'public/models.pkl')
    names = sorted(src['sources']); random.Random(c.sseed(root_seed,'new_source_alias')).shuffle(names)
    truth = {'root_hex':root_seed.hex(), 'model_truth':model_truth, 'group_truth':{}, 'cases':{}}
    source_meta = {}
    for i,name in enumerate(names):
        group = f'X03{i:02d}'
        source = src['sources'][name]
        streams = [tuple(z['words']) for z in source['variants'].values()]
        assert len(set(streams)) == 1
        words = list(streams[0]); assert len(words)>=1056
        family = 'JM-Jona-1598-v01' if name=='jona' else 'JM-Hiob-17c-v01'
        fit,audit,start = c.choose_span(words,family)
        n2fit = c.shuffle_words(fit, family, 'fit'); n2audit = c.shuffle_words(audit,family,'audit')
        public, private = [], []
        for ki in range(32):
            key = c.make_key(root_seed, 'EDITION03', group, ki); oracle = c.invert_permutation(key)
            case = {'group':group,'key_index':ki}
            for dest, plain in [('fit_cipher',fit),('audit_cipher',audit),('n2_fit_cipher',n2fit),('n2_audit_cipher',n2audit)]:
                case[dest] = c.encrypt(plain,key)
                assert c.decode_words(case[dest],oracle)==plain
            public.append(case)
            private.append({'key_index':ki,'fit_truth':fit,'audit_truth':audit,
                            'n2_fit_truth':n2fit,'n2_audit_truth':n2audit,'oracle':oracle})
        c.atomic_json({'group':group,'cases':public},out/f'public/{group}.json')
        truth['group_truth'][group] = {'family':name,'language':'yiddish'}
        truth['cases'][group] = {'segment_start':start,'rows':private}
        source_meta[name] = {'pdf_sha256':source['sha256'], 'words':len(words),
                             'word_stream_sha256':hashlib.sha256('\n'.join(words).encode()).hexdigest(),
                             'span_start_zero_based':start}
    c.atomic_pickle(truth,out/'private/truth.pkl')
    manifest = {'status':'FROZEN_BEFORE_MODEL_SCORES', 'sources':source_meta,
                'model_sha256':c.sha_file(out/'public/models.pkl'),
                'truth_commitment_sha256':c.sha_file(out/'private/truth.pkl'),
                'ciphertext_sha256':{g:c.sha_file(out/f'public/{g}.json') for g in truth['cases']},
                'extraction_checkpoint_sha256':c.sha_file(root/'edition_preflight.pkl.gz'),
                'overlap_checkpoint_sha256':c.sha_file(root/'edition_overlap.json'),
                'protocol_sha256':c.sha_file(Path(__file__).with_name('EDITION_CHALLENGE_FREEZE.md')),
                'solver_sha256':c.sha_file(Path(__file__).with_name('trigram.py')),
                'native_sha256':c.sha_file(Path(__file__).with_name('native_trigram.so')),
                'target_loaded':False,'restarts':32,'keys_per_group':32,
                'scope':'TWO_YIDDISH_EXTERNAL_CHALLENGES_NOT_BALANCED_QUALIFICATION'}
    c.atomic_json(manifest,out/'FREEZE.json')
    print(json.dumps(manifest,indent=2))


def evaluate(out):
    freeze = json.loads((out/'FREEZE.json').read_text())
    assert c.sha_file(out/'private/truth.pkl')==freeze['truth_commitment_sha256']
    assert c.sha_file(out/'public/models.pkl')==freeze['model_sha256']
    truth = read_pickle(out/'private/truth.pkl')
    models = pickle.loads((out/'public/models.pkl').read_bytes())
    y = next(m for m,t in truth['model_truth'].items() if t=='yiddish'); g = 'B' if y=='A' else 'A'
    families, allrows = [], []
    for group,meta in sorted(truth['group_truth'].items()):
        rs=[json.loads(l) for l in (out/f'solutions/{group}/rows.jsonl').read_text().splitlines()]
        assert len(rs)==32 and [r['key_index'] for r in rs]==list(range(32))
        pcs=json.loads((out/f'public/{group}.json').read_text())['cases']
        for row,pc,tr in zip(rs,pcs,truth['cases'][group]['rows']):
            assert row['key_index']==pc['key_index']==tr['key_index']
            for arm,ck,tk in [('primary','audit_cipher','audit_truth'),('n1','audit_cipher','audit_truth'),('n2','n2_audit_cipher','n2_audit_truth')]:
                for mid in ['A','B']:
                    rec=c.rec_score(tr[tk],pc[ck],row[arm][mid]['mapping'])
                    row[arm][mid].update(rec)
                    row[arm][mid]['m0_pass']=rec['atom_recovery']>=.90 and rec['word_recovery']>=.80
                row[arm]['contrast']=v.z_ind_from_arm(row[arm],y,g)
                assert row[arm]['contrast']['z_ind'] is not None
                assert math.isfinite(row[arm]['contrast']['z_ind'])
            # Prospective bounding test, performed only after blind outputs.
            oracle_arm={}
            for mid in ['A','B']:
                obs=row['primary'][mid]
                true_fit=score(counts(pc['fit_cipher']),models[mid]['lp'],tr['oracle'])
                true_audit=score(counts(pc['audit_cipher']),models[mid]['lp'],tr['oracle'])
                obs['returned_minus_true_fit']=obs['fit_objective']-true_fit
                obs['returned_minus_true_audit']=obs['score']-true_audit
                oracle_arm[mid]={'score':true_audit,'null_mean':obs['null_mean'],'null_sd':obs['null_sd']}
            row['oracle_contrast']=v.z_ind_from_arm(oracle_arm,y,g)
        fam={**meta,'group':group,'arms':{}}
        for arm in ['primary','n1','n2']:
            contrasts=[r[arm]['contrast'] for r in rs]
            z=statistics.median(x['z_ind'] for x in contrasts)
            fam['arms'][arm]={'median_z_ind':z,'median_D':statistics.median(x['D'] for x in contrasts),
                              'median_independent_null_sd':statistics.median(x['denom'] for x in contrasts),
                              'call':v.call(z),
                              'yiddish_m0_passes':sum(r[arm][y]['m0_pass'] for r in rs),
                              'german_m0_passes':sum(r[arm][g]['m0_pass'] for r in rs),
                              'median_yiddish_atom_recovery':statistics.median(r[arm][y]['atom_recovery'] for r in rs),
                              'median_yiddish_word_recovery':statistics.median(r[arm][y]['word_recovery'] for r in rs)}
        fam['bounding']={'yiddish_returned_beats_true_fit':sum(r['primary'][y]['returned_minus_true_fit']>1e-10 for r in rs),
                          'median_yiddish_returned_minus_true_fit':statistics.median(r['primary'][y]['returned_minus_true_fit'] for r in rs),
                          'oracle_median_D':statistics.median(r['oracle_contrast']['D'] for r in rs),
                          'oracle_median_null_sd':statistics.median(r['oracle_contrast']['denom'] for r in rs),
                          'oracle_median_z_ind':statistics.median(r['oracle_contrast']['z_ind'] for r in rs)}
        p=fam['arms']['primary']; fam['core_challenge_pass']=p['yiddish_m0_passes']>=29 and p['german_m0_passes']<29 and p['call']=='yiddish'
        families.append(fam);allrows.extend(rs)
    result={'status':'EXTERNAL_CHALLENGE_PASS_LIMITED' if all(f['core_challenge_pass'] for f in families) else 'EXTERNAL_CHALLENGE_FAIL',
            'families':families,'family_label_null_sd':0.0,
            'label_null_effect':0.0,'label_null_interpretation':'Degenerate: all families are Yiddish; no balanced accuracy inference.',
            'bounds_complete':True,'target_loaded':False,'c7_state':'SEALED',
            'limitations':['Edited secondary text; no original-script qualification.',
                           'Different Romanisation; not Penn-equivalent.',
                           'No independent double-reader transcription audit.',
                           'No matched fresh German family panel; cannot issue L qualification.']}
    c.atomic_json(result,out/'RESULT.json')
    c.atomic_pickle({'summary':result,'rows':allrows},out/'result_checkpoint.pkl')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','evaluate'])
    ap.add_argument('--root',type=Path);ap.add_argument('--data',type=Path)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.mode=='prepare':prepare(a.root,a.data,a.out)
    else:evaluate(a.out)

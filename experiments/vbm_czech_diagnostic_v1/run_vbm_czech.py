# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import concurrent.futures, hashlib, json, statistics, sys

sys.path.insert(0,'experiments/amadi_residuals_v1')
sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
sys.path.insert(0,'experiments/vbm_hmm_v2')
sys.path.insert(0,'experiments/vbm_amadi_homophone_v3')
sys.path.insert(0,'experiments/vbm_discriminative_v4')
sys.path.insert(0,'experiments/vbm_key_transfer_v6')

import vbm_structure_v1 as s0
import vbm_hmm_v2 as b
import vbm_hmm_moment_v2 as m
import vbm_amadi_q0_v3 as q3
import vbm_discriminative_v4 as v4
import vbm_key_transfer_v6 as v6

NS='VBMCZECHDIAGV1'
for mod in (b, m.b, q3.b, v4.b, v6.b):
    mod.NS=NS
q3.NS=NS
v4.NS=NS
v6.NS=NS
v6.q3.NS=NS
v6.q3.b.NS=NS

CZ_COMMIT='798f89716ae5a96e86042df7d394d56787e2e213'
CZ_BASE=f'https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-CAC/{CZ_COMMIT}'
CZ_URLS={
    'train':f'{CZ_BASE}/cs_cac-ud-train.conllu',
    'dev':f'{CZ_BASE}/cs_cac-ud-dev.conllu',
    'test':f'{CZ_BASE}/cs_cac-ud-test.conllu',
}
LANGS=['bavarian','german','czech']


def load_czech_lm():
    blobs={}
    seq={}
    for split,url in CZ_URLS.items():
        raw=s0.get(url)
        blobs[split]=hashlib.sha256(raw).hexdigest()
        seq[split]=s0.parse_conllu(raw)
    train=seq['train']
    control=seq['dev']+seq['test']
    lm=b.build_lm('czech',train,control)
    meta={
        'commit':CZ_COMMIT,
        'sha256':blobs,
        'train_sentences':len(train),
        'control_sentences':len(control),
        'train_chars':sum(len(s0.norm(x)) for x in train),
        'control_chars':sum(len(s0.norm(x)) for x in control),
    }
    return lm,meta


def load_lms():
    inherited=b.load_lms()
    cz,meta=load_czech_lm()
    return {'bavarian':inherited['bavarian'],'german':inherited['german'],'czech':cz},meta


def compact_control(z):
    cand={x['language']:{
        'score':float(x['score_mean']),
        'score_A':float(x['score_A']),
        'score_B':float(x['score_B']),
        'score_gap':float(x['score_gap']),
        'delta_vs_surface':float(x['delta']),
        'recovery_diag':None if x['recovery_diag'] is None else float(x['recovery_diag']),
    } for x in z['candidates']}
    return {
        'truth':z['truth'],
        'core_regime':z['core_regime'],
        'bridge_schedule':z['bridge_schedule'],
        'winner_A':z['winner_A'],
        'winner_B':z['winner_B'],
        'winner_mean':z['winner_mean'],
        'margin_mean':float(z['margin_mean']),
        'truth_delta_vs_surface':float(z['truth_delta']),
        'truth_score_gap':float(z['truth_score_gap']),
        'truth_recovery_diag':None if z['truth_recovery_diag'] is None else float(z['truth_recovery_diag']),
        'qualified':bool(z['qualified']),
        'null_score':float(z['null_score']),
        'null_model':z['null_model'],
        'fit_events':int(z['fit_events']),
        'hold_events':int(z['hold_events']),
        'candidates':cand,
    }


def q0(lms):
    tests=[('UNIFORM','FLAT'),('FREQ_PROP','CYCLE')]
    rows=[]
    for core,sched in tests:
        z=v4.one_control(lms,'czech',core,sched,18000,7000,40)
        c=compact_control(z)
        rows.append(c)
        print('CZ_Q0 '+json.dumps(c,sort_keys=True),flush=True)
    passed=all(x['qualified'] for x in rows)
    return {'pass':passed,'rows':rows}


def score_fold(k,folios,labs,lms):
    tr=[f for i,f in enumerate(folios) if i%6!=k]
    ho=[f for i,f in enumerate(folios) if i%6==k]
    trseq=v6.flatten_folios(tr)
    hoseq=v6.flatten_folios(ho)
    null=v4.best_null(trseq,hoseq)
    candidates=[]
    for la in LANGS:
        r=m.paired_fit_moment(trseq,hoseq,lms[la],f'{NS}:FIT:F{k}:{la}',None,40)
        candidates.append({
            'language':la,
            'score':float(r['score']),
            'score_A':float(r['A_eval']['score']),
            'score_B':float(r['B_eval']['score']),
            'score_gap':float(r['score_gap']),
            'decode_agreement':float(r['decode_agreement']),
            'converged':bool(r['converged']),
        })
    candidates.sort(key=lambda x:(-x['score'],x['language']))
    win=candidates[0]
    cz=next(x for x in candidates if x['language']=='czech')
    bg=max(x['score'] for x in candidates if x['language'] in ('bavarian','german'))
    row={
        'fold':k,
        'hold_folios':[labs[i] for i in range(len(labs)) if i%6==k],
        'train_folios':len(tr),
        'hold_folio_count':len(ho),
        'train_events':int(sum(len(q) for q in trseq)),
        'hold_events':int(sum(len(q) for q in hoseq)),
        'winner':win['language'],
        'winner_score':float(win['score']),
        'surface_null_model':null['model'],
        'surface_null_score':float(null['score']),
        'winner_delta_vs_surface':float(win['score']-null['score']),
        'czech_delta_vs_surface':float(cz['score']-null['score']),
        'czech_delta_vs_best_bg':float(cz['score']-bg),
        'candidates':candidates,
    }
    return row


def q1(lms):
    folios,labs,meta=v6.target_folios()
    folios,labs=v6.balanced_hash_order(folios,labs,6)
    rows=[]
    # Q0 has already compiled the numba kernels; three folds at a time keeps CPU/memory bounded.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures=[ex.submit(score_fold,k,folios,labs,lms) for k in range(6)]
        for fut in concurrent.futures.as_completed(futures):
            row=fut.result();rows.append(row)
            print('CZ_FIT '+json.dumps(row,sort_keys=True),flush=True)
    rows.sort(key=lambda x:x['fold'])
    wins={la:sum(r['winner']==la for r in rows) for la in LANGS}
    cz_scores=[next(x['score'] for x in r['candidates'] if x['language']=='czech') for r in rows]
    bg_scores=[max(x['score'] for x in r['candidates'] if x['language'] in ('bavarian','german')) for r in rows]
    out={
        'meta':meta,
        'folds':rows,
        'latent_win_counts':wins,
        'median_czech_score':float(statistics.median(cz_scores)),
        'median_best_bg_score':float(statistics.median(bg_scores)),
        'median_czech_delta_vs_best_bg':float(statistics.median(r['czech_delta_vs_best_bg'] for r in rows)),
        'median_czech_delta_vs_surface':float(statistics.median(r['czech_delta_vs_surface'] for r in rows)),
        'median_winner_delta_vs_surface':float(statistics.median(r['winner_delta_vs_surface'] for r in rows)),
        'czech_beats_bg_folds':sum(r['czech_delta_vs_best_bg']>0 for r in rows),
        'czech_beats_surface_folds':sum(r['czech_delta_vs_surface']>0 for r in rows),
        'any_latent_beats_surface_folds':sum(r['winner_delta_vs_surface']>0 for r in rows),
    }
    return out


def main():
    lms,czmeta=load_lms()
    print('CZ_SOURCE '+json.dumps(czmeta,sort_keys=True),flush=True)
    z0=q0(lms)
    if not z0['pass']:
        out={'namespace':NS,'status':'CLOSED_AT_Q0','q0':z0,'czech_source':czmeta,'voynich_fit_opened':False}
        print('RESULT_JSON '+json.dumps(out,sort_keys=True),flush=True)
        return
    z1=q1(lms)
    out={'namespace':NS,'status':'COMPLETE_EXPLORATORY','q0':z0,'q1':z1,'czech_source':czmeta,'voynich_fit_opened':True,
         'interpretation_guardrail':'FIT is consumed and VBM is closed on latent-vs-stable-surface identifiability; relative language rank is not plaintext evidence.'}
    print('RESULT_JSON '+json.dumps(out,sort_keys=True),flush=True)

if __name__=='__main__':
    main()

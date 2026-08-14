from __future__ import annotations
import argparse, csv, hashlib, json, math, pickle, re
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import u5_verbose_recognition as u5

EXPECTED_RECORD_SHA="dbf87cf5525e065da881b06a26c9d411543ff8ef3f5f8e15a9e4b557808f1174"
EXPECTED_THRESHOLD=0.9997460219719421
PRIMARY_N=2731
FAMILY_REPS=["ZLZI","TTII","TTVE","VDRB-1","GCGI"]


def sha256(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def fit_frozen_instrument(naibbe_repo:Path, cache:Path):
    """Reconstruct U5-B entirely before any Voynich file is opened."""
    lengths=u5.parse_role_lengths(naibbe_repo/'references'/'naibbe_tables.csv')
    cache.mkdir(parents=True,exist_ok=True)
    devsrc={}
    for iso in ('la','it'):
        devsrc[iso]=u5.normalize(u5.strip_gutenberg(u5.fetch(u5.TRAIN_URLS[iso],cache/f'train_{iso}.txt')))
    rows=[]
    for iso in ('la','it'):
        for i,(plain,start) in enumerate(u5.source_chunks(devsrc[iso],60,f'dev-{iso}')):
            group='fit' if i<40 else 'calibration'
            surfs=u5.make_sample(plain,lengths,u5.stable_seed('u5b-dev',iso,i))
            for fam,toks in surfs.items():
                x,_=u5.features(toks)
                rows.append((group,1 if fam=='positive' else 0,fam,x))
    def matrix(group):
        z=[r for r in rows if r[0]==group]
        return np.vstack([r[3] for r in z]),np.array([r[1] for r in z]),[r[2] for r in z]
    Xf,yf,_=matrix('fit');Xc,yc,fc=matrix('calibration')
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight='balanced',solver='liblinear',random_state=20260814,max_iter=1000))
    clf.fit(Xf,yf);pc=clf.predict_proba(Xc)[:,1]
    valid=[]
    for t in sorted(set(pc.tolist()),reverse=True):
        m=u5.metrics(yc,pc,t,fc)
        if m['recall']>0 and m['precision']>=0.95 and max(m['per_family_fpr'].values())<=0.05:
            valid.append((m['recall'],t,m))
    if not valid: raise RuntimeError('frozen U5-B calibration no longer yields any admissible threshold')
    valid.sort(key=lambda x:(x[0],x[1]),reverse=True)
    _,thr,cal=valid[0]
    if not math.isclose(thr,EXPECTED_THRESHOLD,rel_tol=0.0,abs_tol=1e-12):
        raise RuntimeError(f'pre-target instrument reproduction failed: threshold {thr:.17g} != {EXPECTED_THRESHOLD:.17g}')
    if cal['precision']!=1.0 or cal['recall']!=1.0 or cal['fp']!=0 or cal['tp']!=40:
        raise RuntimeError(f'pre-target calibration metrics drifted: {cal}')
    return clf,float(thr),cal


def split_blocks(tokens,n):
    k=len(tokens)//n
    return [tokens[i*n:(i+1)*n] for i in range(k)],tokens[k*n:]


def prob(clf,tokens):
    x,_=u5.features(tokens)
    return float(clf.predict_proba(x.reshape(1,-1))[0,1])


def score_primary(clf,thr,tokens):
    blocks,rem=split_blocks(tokens,PRIMARY_N)
    rows=[];nullrows=[]
    for i,b in enumerate(blocks):
        p=prob(clf,b);rows.append({'block':i,'n_tokens':len(b),'probability':p,'call':p>=thr})
        transforms={
          'type_recode':u5.null_type_recode(b,u5.stable_seed('u5c',i,'type_recode')),
          'global_char_reshuffle':u5.null_global_shuffle(b,u5.stable_seed('u5c',i,'global_char_reshuffle')),
          'token_internal_shuffle':u5.null_internal_shuffle(b,u5.stable_seed('u5c',i,'token_internal_shuffle')),
        }
        for fam,z in transforms.items():
            q=prob(clf,z);nullrows.append({'block':i,'family':fam,'probability':q,'call':q>=thr})
    V=sum(r['call'] for r in rows)
    null_counts={fam:sum(r['call'] for r in nullrows if r['family']==fam) for fam in ('type_recode','global_char_reshuffle','token_internal_shuffle')}
    if V>=11 and all(v==0 for v in null_counts.values()): verdict='PASS_COMPATIBLE_FROZEN_VERBOSE'
    elif V<=2: verdict='FAIL_INCOMPATIBLE_FROZEN_VERBOSE'
    else: verdict='ABSTAIN_UNRESOLVED'
    return {'block_size_tokens':PRIMARY_N,'n_blocks':len(blocks),'remainder_tokens':len(rem),'positive_blocks':V,'positive_fraction':V/len(blocks) if blocks else None,'null_positive_counts':null_counts,'formal_verdict':verdict,'blocks':rows,'target_derived_nulls':nullrows,'remainder_probability':prob(clf,rem) if len(rem)>=100 else None,'remainder_call':(prob(clf,rem)>=thr) if len(rem)>=100 else None}


def score_scale(clf,thr,tokens,n):
    blocks,rem=split_blocks(tokens,n);ps=[prob(clf,b) for b in blocks]
    return {'block_size_tokens':n,'n_blocks':len(blocks),'positive_blocks':sum(p>=thr for p in ps),'positive_fraction':sum(p>=thr for p in ps)/len(ps) if ps else None,'probabilities':ps,'remainder_tokens':len(rem)}


def natkey(x):
    return [int(y) if y.isdigit() else y.lower() for y in re.split(r'(\d+)',str(x))]


def transcriber_tokens(slim,rep,page_order):
    out=[];lines_present=0;lines_total=0
    pages=slim.get('pages',{})
    for fol in page_order:
        lines=pages.get(fol,{})
        for lk in sorted(lines,key=natkey):
            lines_total+=1
            txt=lines[lk].get('t',{}).get(rep,'')
            if txt:
                toks=[t for t in str(txt).split() if t and t!='*']
                if toks:
                    out.extend(toks);lines_present+=1
    return out,lines_present,lines_total


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--records',type=Path,required=True)
    ap.add_argument('--slim',type=Path,required=True)
    ap.add_argument('--naibbe-repo',type=Path,required=True)
    ap.add_argument('--u5b-result',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)

    # FIREWALL: reproduce frozen classifier before even hashing/reading target files.
    clf,thr,cal=fit_frozen_instrument(a.naibbe_repo,a.out/'instrument_source_cache')
    frozen=json.loads(a.u5b_result.read_text(encoding='utf-8'))
    if frozen.get('formal_verdict')!='PASS_RECOGNITION_CALIBRATION' or not math.isclose(float(frozen['threshold']),thr,abs_tol=1e-12,rel_tol=0):
        raise RuntimeError('frozen U5-B result does not match reproduced instrument')
    instrument_check={'status':'PASS','threshold':thr,'calibration_metrics':cal,'u5b_locked_metrics':frozen['locked_metrics'],'target_opened_at_check':False}
    (a.out/'U5C_INSTRUMENT_REPRODUCTION.json').write_text(json.dumps(instrument_check,indent=2,sort_keys=True),encoding='utf-8')
    print('U5C_FIREWALL_PASS',json.dumps(instrument_check,sort_keys=True),flush=True)

    # Target may open only after the preceding assertions.
    got=sha256(a.records)
    if got!=EXPECTED_RECORD_SHA: raise RuntimeError(f'canonical target SHA mismatch: {got}')
    with a.records.open('rb') as f:records=pickle.load(f)
    if len(records)!=37465:raise RuntimeError(f'canonical target count mismatch: {len(records)}')
    tokens=[str(r.get('token','')) for r in records if str(r.get('token',''))]
    if len(tokens)!=37465:raise RuntimeError(f'nonempty canonical token count changed: {len(tokens)}')
    primary=score_primary(clf,thr,tokens)
    scales={str(n):score_scale(clf,thr,tokens,n) for n in (2048,4096)}
    scales['whole']={'n_tokens':len(tokens),'probability':prob(clf,tokens),'call':prob(clf,tokens)>=thr}

    # Pre-specified representation sensitivities, never used to replace primary verdict.
    slim=json.loads(a.slim.read_text(encoding='utf-8'))
    page_order=[];seen=set()
    for r in records:
        fol=str(r.get('folio',''))
        if fol and fol not in seen:seen.add(fol);page_order.append(fol)
    reps={}
    for rep in FAMILY_REPS:
        rt,lp,lt=transcriber_tokens(slim,rep,page_order)
        sc=score_scale(clf,thr,rt,PRIMARY_N) if len(rt)>=PRIMARY_N else {'block_size_tokens':PRIMARY_N,'n_blocks':0,'positive_blocks':0,'positive_fraction':None,'probabilities':[],'remainder_tokens':len(rt)}
        sc.update({'representation':rep,'tokens':len(rt),'lines_present':lp,'lines_considered':lt,'line_coverage':lp/lt if lt else None})
        reps[rep]=sc

    result={'schema':'frontier-u5-c-v0.1','target_opened':True,'claim_scope':'compatibility with frozen FRESH-VERBOSE reusable unigram/prefix/suffix architecture only','canonical_records_sha256':got,'canonical_records':len(records),'frozen_threshold':thr,'u5b_locked_metrics':frozen['locked_metrics'],'primary':primary,'scale_sensitivities':scales,'transliteration_sensitivities':reps,'interpretation_guard':'PASS is not proof of encryption, Naibbe, language, plaintext, date, place, or provenance.'}
    (a.out/'U5C_VOYNICH_COMPATIBILITY_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8')
    md=['# U5-C Voynich fresh-codebook compatibility result','',f'Formal primary verdict: **{primary["formal_verdict"]}**','',f'- primary positive blocks: **{primary["positive_blocks"]}/{primary["n_blocks"]}**',f'- target-derived null positive counts: `{primary["null_positive_counts"]}`',f'- frozen U5-B threshold: `{thr:.16g}`',f'- U5-B locked recognition: recall {frozen["locked_metrics"]["recall"]:.3f}, precision {frozen["locked_metrics"]["precision"]:.3f}, FP {frozen["locked_metrics"]["fp"]}/400','', '## Guardrail','','This result is compatibility with the frozen reusable unigram/prefix/suffix fresh-codebook architecture. It is not proof that the manuscript is encrypted, that Naibbe was used, or that any plaintext/source language/provenance has been identified.','', '## Primary block probabilities']
    for r in primary['blocks']:md.append(f'- block {r["block"]:02d}: p={r["probability"]:.8f} call={int(r["call"])}')
    md += ['', '## Representation sensitivities']
    for rep,s in reps.items():md.append(f'- {rep}: {s["positive_blocks"]}/{s["n_blocks"]} positive; tokens={s["tokens"]}; line coverage={s["line_coverage"]:.3f}' if s['line_coverage'] is not None else f'- {rep}: unavailable')
    (a.out/'U5C_RESULT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('U5C_FINAL',json.dumps({'formal_verdict':primary['formal_verdict'],'positive_blocks':primary['positive_blocks'],'n_blocks':primary['n_blocks'],'null_positive_counts':primary['null_positive_counts'],'whole_probability':scales['whole']['probability']},sort_keys=True),flush=True)

if __name__=='__main__':main()

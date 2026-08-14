from __future__ import annotations
import argparse, hashlib, json, math, pickle, random, re
from pathlib import Path
import numpy as np
import u5_verbose_recognition as u5
import u5_target_score as u5c

THR=u5c.EXPECTED_THRESHOLD
N=2731
EMPTY={None,"","∅","EMPTY","-","None","nan"}
ART="abcdefghijklmnopqrst"


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:8],"big") & 0x7fffffffffffffff


def words(text):
    raw=re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+",text)
    out=[]
    for w in raw:
        z=u5.normalize(w)
        if z: out.append(z.lower())
    return out


def blocks(tokens,n=N):
    k=len(tokens)//n
    return [tokens[i*n:(i+1)*n] for i in range(k)]


def probability(clf,tokens):
    x,_=u5.features(tokens)
    return float(clf.predict_proba(x.reshape(1,-1))[0,1])


def score_blocks(clf,tokens):
    bs=blocks(tokens)
    ps=[probability(clf,b) for b in bs]
    return {"tokens":len(tokens),"complete_blocks":len(bs),"positive_blocks":sum(p>=THR for p in ps),"positive_fraction":sum(p>=THR for p in ps)/len(ps) if ps else None,"probabilities":ps,"remainder_tokens":len(tokens)-len(bs)*N}


def unique_strings(rng,count,lengths):
    used=set();out=[]
    while len(out)<count:
        L=rng.choice(lengths);s="".join(rng.choice(ART) for _ in range(L))
        if s not in used:used.add(s);out.append(s)
    return out


def root_affix_block(i):
    rng=random.Random(stable_seed('u5d-root-affix',i))
    prefixes=unique_strings(rng,16,(1,2,3));roots=unique_strings(rng,128,(2,3,4,5));suffixes=unique_strings(rng,24,(1,2,3))
    rorder=list(range(128));rng.shuffle(rorder)
    weights=[1/((rank+1)**1.1) for rank in range(128)]
    # Map the shuffled root index to a fixed rank weight.
    root_weights=[0.0]*128
    for rank,idx in enumerate(rorder): root_weights[idx]=weights[rank]
    pclass=[rng.randrange(4) for _ in roots];sclass=[rng.randrange(4) for _ in roots]
    out=[]
    for _ in range(N):
        ridx=rng.choices(range(128),weights=root_weights,k=1)[0]
        parts=[]
        if rng.random()<0.70:
            candidates=prefixes[pclass[ridx]*4:(pclass[ridx]+1)*4]
            parts.append(rng.choice(candidates))
        parts.append(roots[ridx])
        if rng.random()<0.80:
            candidates=suffixes[sclass[ridx]*6:(sclass[ridx]+1)*6]
            parts.append(rng.choice(candidates))
        out.append("".join(parts))
    return out


def comp_text(v):
    if v in EMPTY:return ""
    s=str(v)
    return "" if s in EMPTY else s


def pgcs_recombine(records,block_index):
    recs=records[block_index*N:(block_index+1)*N]
    cols={k:[comp_text(r.get(k)) for r in recs] for k in ('prefix','gallows','core','suffix')}
    for k,z in cols.items():
        rng=random.Random(stable_seed('u5d-pgcs',block_index,k));rng.shuffle(z)
    out=[]
    for j in range(len(recs)):
        t="".join(cols[k][j] for k in ('prefix','gallows','core','suffix'))
        out.append(t if t else 'x')
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--records',type=Path,required=True)
    ap.add_argument('--naibbe-repo',type=Path,required=True)
    ap.add_argument('--u5b-result',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)

    # Exact frozen classifier reconstruction; target result is already known, but no retuning is allowed.
    clf,thr,cal=u5c.fit_frozen_instrument(a.naibbe_repo,a.cache)
    if not math.isclose(thr,THR,abs_tol=1e-12,rel_tol=0):raise RuntimeError('threshold drift')
    frozen=json.loads(a.u5b_result.read_text())
    if frozen.get('formal_verdict')!='PASS_RECOGNITION_CALIBRATION':raise RuntimeError('U5-B not qualified')

    # A: held-out direct natural word surfaces.
    locked={}
    for iso in ('la','it'):
        path=a.naibbe_repo/u5.TEST_FILES[iso]
        locked[iso]=score_blocks(clf,words(path.read_text(encoding='utf-8',errors='ignore')))
    dev={}
    for iso in ('la','it'):
        path=a.cache/f'train_{iso}.txt'
        dev[iso]=score_blocks(clf,words(u5.strip_gutenberg(path.read_text(encoding='utf-8',errors='ignore'))))
    nat_pos=sum(v['positive_blocks'] for v in locked.values());nat_n=sum(v['complete_blocks'] for v in locked.values());F_nat=nat_pos/nat_n if nat_n else None

    # B: 100 independent non-message morphology blocks.
    morph_ps=[]
    for i in range(100):morph_ps.append(probability(clf,root_affix_block(i)))
    morph_pos=sum(p>=THR for p in morph_ps);F_morph=morph_pos/100

    # C: target-derived component marginal recombination.
    if u5c.sha256(a.records)!=u5c.EXPECTED_RECORD_SHA:raise RuntimeError('target record SHA drift')
    with a.records.open('rb') as f:records=pickle.load(f)
    if len(records)!=37465:raise RuntimeError('record count drift')
    pgcs_ps=[probability(clf,pgcs_recombine(records,i)) for i in range(13)]
    V_pgcs=sum(p>=THR for p in pgcs_ps)

    if F_nat<=0.05 and F_morph<=0.05 and V_pgcs<=2:
        verdict='SPECIFICITY_SURVIVES'
    elif F_nat>=0.20 or F_morph>=0.20 or V_pgcs>=11:
        verdict='GENERIC_COMPOSITION_CONFOUND'
    else:
        verdict='SPECIFICITY_AMBIGUOUS'

    result={
      'schema':'frontier-u5-d-specificity-v0.1','status':'POSTHOC_INTERPRETATION_CHALLENGE','confirmatory_u5c_verdict_unchanged':'PASS_COMPATIBLE_FROZEN_VERBOSE',
      'frozen_threshold':thr,'formal_specificity_verdict':verdict,
      'heldout_natural_language':locked,'development_author_natural_language_descriptive':dev,'F_nat':F_nat,
      'root_affix':{'blocks':100,'positive_blocks':morph_pos,'positive_fraction':F_morph,'probabilities':morph_ps},
      'pgcs_component_recombination':{'blocks':13,'positive_blocks':V_pgcs,'positive_fraction':V_pgcs/13,'probabilities':pgcs_ps},
      'interpretation':('U5-C remains a compatibility result but the feature family is not specific to verbose cipher architecture; it also accepts a non-message compositional control.' if verdict=='GENERIC_COMPOSITION_CONFOUND' else 'Specificity challenge did not produce a large generic-composition confound at the frozen bands.' if verdict=='SPECIFICITY_SURVIVES' else 'Specificity remains ambiguous under the frozen post-hoc bands.'),
      'guardrail':'This post-hoc panel does not retroactively alter the confirmatory U5-C block calls and tests no plaintext, language, key or provenance.'
    }
    (a.out/'U5D_SPECIFICITY_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8')
    md=['# U5-D post-hoc specificity challenge','',f'Formal specificity verdict: **{verdict}**','',f'- held-out direct natural-language positive fraction: **{F_nat:.4f}** ({nat_pos}/{nat_n})',f'- non-message ROOT+AFFIX positive fraction: **{F_morph:.4f}** ({morph_pos}/100)',f'- PGCS component-recombination positives: **{V_pgcs}/13**','', 'Confirmatory U5-C remains `PASS_COMPATIBLE_FROZEN_VERBOSE`; U5-D adjudicates only how specifically that result can be interpreted.']
    (a.out/'U5D_RESULT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('U5D_FINAL',json.dumps({'formal_specificity_verdict':verdict,'F_nat':F_nat,'F_morph':F_morph,'V_pgcs':V_pgcs,'nat_pos':nat_pos,'nat_n':nat_n},sort_keys=True),flush=True)

if __name__=='__main__':main()

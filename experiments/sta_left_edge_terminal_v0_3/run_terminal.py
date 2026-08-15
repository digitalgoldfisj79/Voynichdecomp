#!/usr/bin/env python3
"""Terminal STA left-edge discriminator v0.3.
Binding protocol: PREREG_20260815.md.
"""
import argparse, collections, hashlib, importlib.util, json, math, random
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'sta_boundary_return_discriminator_v0_1'/'run_discriminator.py'
spec=importlib.util.spec_from_file_location('parent',PARENT)
pmod=importlib.util.module_from_spec(spec); spec.loader.exec_module(pmod)
p=pmod.p

SEED=20260815
NFOLD=5
ALPHA_GLOBAL=5.0
ALPHA_LOCAL=20.0
NNEG=2000
NPOS=30
PLANT_Q=0.20
SOURCE_SHA='81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17'


def fold_of(folio): return p.fold_of(folio)
def eligible(x): return len(x['tokens'])>=6

def norm(v):
    a=np.asarray(v,dtype=float); s=float(a.sum())
    if s<=0: return np.ones(len(a),dtype=float)/len(a)
    return a/s

def fit_position_model(lines):
    corpus_start=np.zeros(2); corpus_mid=np.zeros(3)
    gs=collections.defaultdict(lambda:np.zeros(2)); gm=collections.defaultdict(lambda:np.zeros(3))
    ls=collections.defaultdict(lambda:np.zeros(2)); lm=collections.defaultdict(lambda:np.zeros(3))
    for x in lines:
        if not eligible(x): continue
        t=x['tokens']; n=len(t); sec=x['section']; lb=p.lbucket(n)
        for pos in (0,1):
            tok=t[pos]; corpus_start[pos]+=1; gs[tok][pos]+=1; ls[(sec,lb,tok)][pos]+=1
        for pos in range(2,n-2):
            tok=t[pos]; cat=0 if pos==2 else (1 if pos==3 else 2)
            corpus_mid[cat]+=1; gm[tok][cat]+=1; lm[(sec,lb,tok)][cat]+=1
    return {'corpus_start':corpus_start,'corpus_mid':corpus_mid,'gs':gs,'gm':gm,'ls':ls,'lm':lm}

def probs_start(model,sec,lb,tok):
    base=norm(model['corpus_start'])
    g=model['gs'].get(tok,np.zeros(2)); gp=(g+ALPHA_GLOBAL*base)/(g.sum()+ALPHA_GLOBAL)
    l=model['ls'].get((sec,lb,tok),np.zeros(2)); return (l+ALPHA_LOCAL*gp)/(l.sum()+ALPHA_LOCAL)

def probs_mid(model,sec,lb,tok):
    base=norm(model['corpus_mid'])
    g=model['gm'].get(tok,np.zeros(3)); gp=(g+ALPHA_GLOBAL*base)/(g.sum()+ALPHA_GLOBAL)
    l=model['lm'].get((sec,lb,tok),np.zeros(3)); return (l+ALPHA_LOCAL*gp)/(l.sum()+ALPHA_LOCAL)

def ordered_assignments(counter,weight_fn):
    vals=[]; total=0.0
    for a,ca in counter.items():
        if ca<=0: continue
        for b,cb in counter.items():
            mult=ca*(cb-(1 if a==b else 0))
            if mult<=0: continue
            w=float(mult)*float(weight_fn(a,b))
            if w>0:
                vals.append((a,b,w)); total+=w
    if total<=0: raise RuntimeError('zero assignment mass')
    return [(a,b,w/total) for a,b,w in vals]

def line_components(x,model):
    t=x['tokens']; n=len(t); sec=x['section']; lb=p.lbucket(n)
    sc=collections.Counter(t[:2]); mc=collections.Counter(t[2:n-2])
    def sw(a,b):
        pa=probs_start(model,sec,lb,a); pb=probs_start(model,sec,lb,b)
        return pa[0]*pb[1]
    def mw(a,b):
        pa=probs_mid(model,sec,lb,a); pb=probs_mid(model,sec,lb,b)
        # Conditional odds after fixing the MID multiset and category totals.
        return (pa[0]/max(pa[2],1e-9))*(pb[1]/max(pb[2],1e-9))
    return ordered_assignments(sc,sw), ordered_assignments(mc,mw)

def line_outcome_probs(x,model):
    sd,md=line_components(x,model)
    q=np.zeros((2,2),dtype=float)
    for a,b,ps in sd:
        for c,d,pm in md:
            q[int(a==c),int(b==d)] += ps*pm
    q/=q.sum()
    return q

def convolve_total(line_qs):
    dp=np.array([1.0])
    for q in line_qs:
        pk=np.array([q[0,0],q[1,0]+q[0,1],q[1,1]],dtype=float)
        nd=np.zeros(len(dp)+2)
        nd[:len(dp)] += dp*pk[0]
        nd[1:1+len(dp)] += dp*pk[1]
        nd[2:2+len(dp)] += dp*pk[2]
        dp=nd
    dp/=dp.sum(); return dp

def null_summary(pmf,line_qs,obs_total,obs02,obs13):
    k=np.arange(len(pmf),dtype=float); mu=float((k*pmf).sum()); var=float((((k-mu)**2)*pmf).sum()); sd=math.sqrt(max(var,0.0))
    tail=float(pmf[int(obs_total):].sum()) if obs_total<len(pmf) else 0.0
    e02=float(sum(q[1,:].sum() for q in line_qs)); e13=float(sum(q[:,1].sum() for q in line_qs))
    z=(obs_total-mu)/sd if sd>0 else 0.0; ratio=obs_total/mu if mu>0 else None
    gate=bool(z>=2.58 and tail<=.01 and ratio is not None and ratio>=1.10 and obs02>e02 and obs13>e13)
    return {'observed_total':int(obs_total),'expected_total':mu,'null_sd':sd,'ratio':float(ratio) if ratio is not None else None,
            'z':float(z),'exact_one_sided_p':tail,'observed_L0_L2':int(obs02),'expected_L0_L2':e02,
            'observed_L1_L3':int(obs13),'expected_L1_L3':e13,'primary_gate':gate}

def build_crossfit(lines):
    models={}
    for f in range(NFOLD): models[f]=fit_position_model([x for x in lines if fold_of(x['folio'])!=f])
    eligible_lines=[]; line_qs=[]
    for x in lines:
        if not eligible(x): continue
        f=fold_of(x['folio']); eligible_lines.append(x); line_qs.append(line_outcome_probs(x,models[f]))
    return models,eligible_lines,line_qs

def observed_counts(lines):
    a=b=0
    for x in lines:
        t=x['tokens']; a+=int(t[0]==t[2]); b+=int(t[1]==t[3])
    return a+b,a,b

def sample_assignment(dist,rng):
    u=rng.random(); c=0.0
    for a,b,pr in dist:
        c+=pr
        if u<=c: return a,b
    return dist[-1][0],dist[-1][1]

def sample_line(x,model,rng):
    t=list(x['tokens']); n=len(t); sd,md=line_components(x,model)
    a,b=sample_assignment(sd,rng); c,d=sample_assignment(md,rng)
    t[0],t[1]=a,b
    remaining=list(t[2:n-2]);
    # remove one occurrence of selected L2/L3 tokens from the frozen MID multiset
    remaining.remove(c); remaining.remove(d); rng.shuffle(remaining)
    t[2:n-2]=[c,d]+remaining
    return {'folio':x['folio'],'section':x['section'],'tokens':tuple(t)}

def sample_corpus(eligible_lines,models,seed):
    rng=random.Random(seed); return [sample_line(x,models[fold_of(x['folio'])],rng) for x in eligible_lines]

def plant_corpus(corpus,seed,q=PLANT_Q):
    rng=random.Random(seed); out=[]; swaps=0
    for x in corpus:
        t=list(x['tokens']); n=len(t)
        for src,target in ((0,2),(1,3)):
            if t[src]==t[target] or rng.random()>=q: continue
            cand=[k for k in range(4,n-2) if t[k]==t[src]]
            if not cand: continue
            k=cand[0]; t[target],t[k]=t[k],t[target]; swaps+=1
        out.append({'folio':x['folio'],'section':x['section'],'tokens':tuple(t)})
    return out,swaps

def draw_outcomes(line_qs,rng):
    o02=o13=0
    for q in line_qs:
        u=rng.random(); c=0.0; chosen=(0,0)
        for i,j in ((0,0),(1,0),(0,1),(1,1)):
            c+=q[i,j]
            if u<=c: chosen=(i,j); break
        o02+=chosen[0]; o13+=chosen[1]
    return o02+o13,o02,o13

def negative_calibration(pmf,line_qs,summary,n=NNEG):
    rng=random.Random(SEED+30000); passes=0
    e02=summary['expected_L0_L2']; e13=summary['expected_L1_L3']; mu=summary['expected_total']; sd=summary['null_sd']
    for _ in range(n):
        ot,o02,o13=draw_outcomes(line_qs,rng); tail=float(pmf[ot:].sum()); ratio=ot/mu if mu>0 else 0.0; z=(ot-mu)/sd if sd>0 else 0.0
        passes+=int(z>=2.58 and tail<=.01 and ratio>=1.10 and o02>e02 and o13>e13)
    frac=passes/n; return {'n':n,'passes':passes,'fraction':frac,'pass':bool(frac<=.02)}

def positive_calibration(eligible_lines,models,pmf,line_qs,summary,n=NPOS):
    passes=0; swaps=[]; rows=[]
    for r in range(n):
        syn=sample_corpus(eligible_lines,models,SEED+40000+r); planted,ns=plant_corpus(syn,SEED+50000+r)
        ot,o02,o13=observed_counts(planted); s=null_summary(pmf,line_qs,ot,o02,o13); passes+=int(s['primary_gate']); swaps.append(ns)
        rows.append({'swaps':ns,'z':s['z'],'p':s['exact_one_sided_p'],'ratio':s['ratio'],'gate':s['primary_gate']})
    frac=passes/n; ms=float(np.median(swaps)); return {'n':n,'q':PLANT_Q,'passes':passes,'fraction':frac,'median_swaps':ms,
        'pass':bool(ms>=20 and frac>=.80),'replicates':rows}

def diagnostics(lines,line_qs):
    tok_obs=collections.Counter(); tok_exp=collections.Counter(); sec_obs=collections.Counter(); sec_exp=collections.Counter()
    for x,q in zip(lines,line_qs):
        t=x['tokens']; sec=x['section']
        for src,target,prob in ((0,2,float(q[1,:].sum())),(1,3,float(q[:,1].sum()))):
            if t[src]==t[target]: tok_obs[t[src]]+=1; sec_obs[sec]+=1
            sec_exp[sec]+=prob
    # Expected token-specific decomposition is not identifiable from q alone; report observed drivers only.
    top=[{'token':k,'observed_matches':v} for k,v in tok_obs.most_common(20)]
    secs={s:{'observed':sec_obs[s],'expected':sec_exp[s],'residual':sec_obs[s]-sec_exp[s]} for s in sorted(set(sec_obs)|set(sec_exp))}
    return {'top_observed_match_tokens':top,'section_residuals':secs}

def toy_exactness():
    # Two explicit line outcome distributions; DP must equal brute-force polynomial product.
    q1=np.array([[.4,.1],[.2,.3]]); q2=np.array([[.5,.2],[.1,.2]])
    pmf=convolve_total([q1,q2])
    p1=np.array([.4,.3,.3]); p2=np.array([.5,.3,.2]); brute=np.convolve(p1,p2)
    return bool(np.max(np.abs(pmf-brute))<1e-12 and abs(pmf.sum()-1)<1e-12)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('rf1b',nargs='?',default='/tmp/RF1b.txt'); ap.add_argument('--build-check',action='store_true'); args=ap.parse_args()
    if args.build_check:
        print('BUILD_CHECK_PASS' if toy_exactness() else 'BUILD_CHECK_FAIL'); raise SystemExit(0 if toy_exactness() else 1)
    sections=p.load_sections(); raw,lines,parse_audit=p.parse_rf(args.rf1b,sections); sha=hashlib.sha256(raw).hexdigest(); inherited=pmod.score(lines)
    validation={'source_sha256':sha,'header_ok':raw.startswith(b'#=IVTFF STA1 2.0'),'folios':len({x['folio'] for x in lines}),
        'segments':len(lines),'tokens':sum(len(x['tokens']) for x in lines),'eligible_clean_left_lines':sum(eligible(x) for x in lines),
        'parser':parse_audit,'anchors':{'E2_N0':inherited['E2_N0'],'E2_N1':inherited['E2_N1'],'E2_N3':inherited['E2_N3']}}
    validation['pass']=bool(validation['header_ok'] and sha==SOURCE_SHA and 1.16<=inherited['E2_N0']<=1.20 and 1.05<=inherited['E2_N1']<=1.09 and 1.02<=inherited['E2_N3']<=1.07)
    result={'metadata':{'seed':SEED,'folds':NFOLD,'alpha_global':ALPHA_GLOBAL,'alpha_local':ALPHA_LOCAL,'negative_n':NNEG,'positive_n':NPOS,'plant_q':PLANT_Q},'validation':validation}
    if not validation['pass']:
        result['verdict']='INSTRUMENT_FAIL_TERMINAL'; result['reason']='SOURCE_OR_ANCHOR_VALIDATION_FAIL'
    else:
        models,elines,line_qs=build_crossfit(lines); pmf=convolve_total(line_qs); ot,o02,o13=observed_counts(elines)
        target=null_summary(pmf,line_qs,ot,o02,o13); c1=negative_calibration(pmf,line_qs,target); c2=positive_calibration(elines,models,pmf,line_qs,target)
        calibration={'C0_toy_exactness':toy_exactness(),'C1_negative':c1,'C2_positive':c2}; calibration['pass']=bool(calibration['C0_toy_exactness'] and c1['pass'] and c2['pass'])
        result['calibration']=calibration; result['target']=target; result['diagnostics']=diagnostics(elines,line_qs)
        if not calibration['pass']: verdict='INSTRUMENT_FAIL_TERMINAL'
        elif target['primary_gate']: verdict='LEFT_EDGE_ORDERING_RESIDUAL_SURVIVES'
        else: verdict='EXACT_LEFT_POSITION_PLUS_LINE_INVENTORY_SUFFICIENT'
        result['verdict']=verdict
    out=Path('results/sta_left_edge_terminal_v0_3'); out.mkdir(parents=True,exist_ok=True)
    (out/'RESULTS_20260815.json').write_text(json.dumps(result,indent=2)+'\n')
    md=['# STA left-edge terminal discriminator v0.3','',f"Validation: **{'PASS' if validation['pass'] else 'FAIL'}**",f"Verdict: **{result['verdict']}**"]
    if 'target' in result:
        t=result['target']; md += ['',f"Eligible clean-left lines: **{validation['eligible_clean_left_lines']}**",'', '|quantity|value|','|---|---:|',
            f"|observed total|{t['observed_total']}|",f"|expected total|{t['expected_total']:.3f}|",f"|ratio|{t['ratio']:.4f}|",f"|z|{t['z']:.3f}|",f"|exact one-sided p|{t['exact_one_sided_p']:.6g}|",
            f"|L0→L2 obs / exp|{t['observed_L0_L2']} / {t['expected_L0_L2']:.3f}|",f"|L1→L3 obs / exp|{t['observed_L1_L3']} / {t['expected_L1_L3']:.3f}|",f"|primary gate|{t['primary_gate']}|",'',
            f"Calibration pass: **{result['calibration']['pass']}**",f"Negative gate fraction: {result['calibration']['C1_negative']['fraction']:.4f}",f"Planted gate fraction: {result['calibration']['C2_positive']['fraction']:.4f}; median swaps={result['calibration']['C2_positive']['median_swaps']:.1f}."]
    (out/'RESULTS_20260815.md').write_text('\n'.join(md)+'\n'); print('\n'.join(md))

if __name__=='__main__': main()

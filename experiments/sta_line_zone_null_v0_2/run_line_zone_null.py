#!/usr/bin/env python3
"""STA within-line / within-zone short-lag discriminator v0.2.

Binding protocol: PREREG_20260815.md
This is a conditional randomisation test. It preserves each individual line's
N1 START/MID/END token multisets exactly and randomises only order within them.
"""
import argparse, collections, hashlib, importlib.util, json, math, random
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'sta_boundary_return_discriminator_v0_1'/'run_discriminator.py'
spec=importlib.util.spec_from_file_location('boundary_parent',PARENT)
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
p=b.p

SEED=20260815
NPERM=20000
N_NULL_PSEUDO=200
N_PLANTED=30
PLANT_Q=0.10
SOURCE_SHA='81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17'
LAGS=(1,2,3,4)


def med(xs): return float(np.median(np.asarray(xs,dtype=float)))
def qtile(xs,q): return float(np.quantile(np.asarray(xs,dtype=float),q))

def boundary_starts(n,lag):
    return [i for i in range(max(0,n-lag)) if i<2 or i+lag>=n-2]

def counts_by_lag(lines):
    out={lag:0 for lag in LAGS}; left2=right2=0; events={lag:0 for lag in LAGS}
    for x in lines:
        t=x['tokens']; n=len(t)
        for lag in LAGS:
            for i in boundary_starts(n,lag):
                j=i+lag; events[lag]+=1; out[lag]+=int(t[i]==t[j])
                if lag==2:
                    if i<2: left2+=int(t[i]==t[j])
                    if j>=n-2: right2+=int(t[i]==t[j])
    return {'lag':out,'events':events,'lag2_left':left2,'lag2_right':right2}

def line_zone_counters(tokens):
    return [collections.Counter(tokens[int(i)] for i in g) for g in p.groups_for(len(tokens),'N1')]

def shuffle_tokens(tokens,rng):
    t=list(tokens); n=len(t)
    for g in p.groups_for(n,'N1'):
        pos=[int(i) for i in g]; vals=[t[i] for i in pos]; rng.shuffle(vals)
        for i,v in zip(pos,vals): t[i]=v
    return tuple(t)

def shuffle_corpus_once(lines,seed):
    rng=random.Random(seed); out=[]
    for x in lines:
        out.append({'folio':x['folio'],'section':x['section'],'tokens':shuffle_tokens(x['tokens'],rng)})
    return out

def preservation_audit(lines,n_lines=100,n_rep=20):
    # deterministic spread over corpus, not the first n lines only
    if not lines:return False
    idx=np.linspace(0,len(lines)-1,min(n_lines,len(lines)),dtype=int)
    for r in range(n_rep):
        rng=random.Random(SEED+50000+r)
        for ii in idx:
            t=lines[int(ii)]['tokens']; before=line_zone_counters(t); after=line_zone_counters(shuffle_tokens(t,rng))
            if before!=after:return False
    return True

def permutation_counts(lines,nperm,seed):
    """Joint uniform LZ permutation distribution for lags 1..4 plus lag-2 sides."""
    rng=np.random.default_rng(seed)
    arr={lag:np.zeros(nperm,dtype=np.int32) for lag in LAGS}
    left2=np.zeros(nperm,dtype=np.int32); right2=np.zeros(nperm,dtype=np.int32)
    for x in lines:
        t=x['tokens']; n=len(t)
        if n<2: continue
        vocab={tok:k for k,tok in enumerate(sorted(set(t)))}
        ids=np.asarray([vocab[tok] for tok in t],dtype=np.int32)
        perm=np.broadcast_to(np.arange(n,dtype=np.int32),(nperm,n)).copy()
        for g in p.groups_for(n,'N1'):
            pos=np.asarray(g,dtype=np.int32)
            if len(pos)<=1: continue
            keys=rng.random((nperm,len(pos)))
            order=np.argsort(keys,axis=1)
            vals=perm[:,pos].copy()
            perm[:,pos]=np.take_along_axis(vals,order,axis=1)
        tok=ids[perm]
        for lag in LAGS:
            starts=boundary_starts(n,lag)
            if not starts: continue
            s=np.asarray(starts,dtype=np.int32)
            eq=(tok[:,s]==tok[:,s+lag])
            arr[lag]+=eq.sum(axis=1,dtype=np.int32)
            if lag==2:
                lm=np.asarray([i<2 for i in starts],dtype=bool)
                rm=np.asarray([i+2>=n-2 for i in starts],dtype=bool)
                if lm.any(): left2+=eq[:,lm].sum(axis=1,dtype=np.int32)
                if rm.any(): right2+=eq[:,rm].sum(axis=1,dtype=np.int32)
    return {'lag':arr,'lag2_left':left2,'lag2_right':right2}

def summarize_obs(obs,ref):
    stats={}; zperm={}
    for lag in LAGS:
        a=np.asarray(ref['lag'][lag],dtype=float); mu=float(a.mean()); sd=float(a.std(ddof=1)); o=float(obs['lag'][lag])
        z=(o-mu)/sd if sd>0 else 0.0; zarr=(a-mu)/sd if sd>0 else np.zeros_like(a)
        p1=float((1+np.count_nonzero(a>=o))/(len(a)+1))
        stats[lag]={'observed':int(o),'events':int(obs['events'][lag]),'null_mean':mu,'null_sd':sd,
                    'ratio':float(o/mu) if mu>0 else None,'z':float(z),'p_one_sided':p1}
        zperm[lag]=zarr
    z2=stats[2]['z']; max_perm=np.maximum.reduce([zperm[k] for k in LAGS])
    pmax=float((1+np.count_nonzero(max_perm>=z2))/(len(max_perm)+1))
    other=max(stats[k]['z'] for k in (1,3,4)); contrast=float(z2-other)
    cperm=zperm[2]-np.maximum.reduce([zperm[1],zperm[3],zperm[4]])
    pspec=float((1+np.count_nonzero(cperm>=contrast))/(len(cperm)+1))
    full=bool(z2>=2.58 and stats[2]['p_one_sided']<=.01 and pmax<=.01 and contrast>=1.0 and pspec<=.01)
    side={}
    for name,key in (('left','lag2_left'),('right','lag2_right')):
        a=np.asarray(ref[key],dtype=float); mu=float(a.mean()); sd=float(a.std(ddof=1)); o=float(obs[key])
        side[name]={'observed':int(o),'null_mean':mu,'null_sd':sd,'ratio':float(o/mu) if mu>0 else None,
                    'z':float((o-mu)/sd) if sd>0 else 0.0,
                    'p_one_sided':float((1+np.count_nonzero(a>=o))/(len(a)+1))}
    return {'lags':{str(k):stats[k] for k in LAGS},'p_maxT_lag2':pmax,'lag2_specificity_contrast':contrast,
            'p_specificity':pspec,'full_lag2_specific_gate':full,'sides_lag2':side}

def null_pseudo_control(ref,lines,n=N_NULL_PSEUDO):
    pseudo=permutation_counts(lines,n,SEED+60000); passes=0; z2s=[]; contrasts=[]; pmaxs=[]; pspecs=[]
    for r in range(n):
        obs={'lag':{lag:int(pseudo['lag'][lag][r]) for lag in LAGS},
             'events':counts_by_lag(lines)['events'],
             'lag2_left':int(pseudo['lag2_left'][r]),'lag2_right':int(pseudo['lag2_right'][r])}
        s=summarize_obs(obs,ref); passes+=int(s['full_lag2_specific_gate']); z2s.append(s['lags']['2']['z'])
        contrasts.append(s['lag2_specificity_contrast']); pmaxs.append(s['p_maxT_lag2']); pspecs.append(s['p_specificity'])
    frac=passes/n
    return {'n':n,'gate_passes':passes,'gate_fraction':frac,'median_z2':med(z2s),'p99_z2':qtile(z2s,.99),
            'median_contrast':med(contrasts),'median_pmax':med(pmaxs),'median_pspecificity':med(pspecs),'pass':bool(frac<=.05)}

def line_match_counts(tokens):
    out={lag:0 for lag in LAGS}; n=len(tokens)
    for lag in LAGS:
        for i in boundary_starts(n,lag): out[lag]+=int(tokens[i]==tokens[i+lag])
    return out

def target_zone_positions(n,j):
    for g in p.groups_for(n,'N1'):
        gg=[int(i) for i in g]
        if j in gg:return gg
    raise RuntimeError('target position not in N1 group')

def plant_line_lag2(tokens,rng,q=PLANT_Q):
    t=list(tokens); n=len(t); used=set(); swaps=0
    order=boundary_starts(n,2); rng.shuffle(order)
    for i in order:
        j=i+2
        if i in used or j in used or t[i]==t[j] or rng.random()>=q: continue
        src=t[i]; before=line_match_counts(t); candidates=[]
        for k in target_zone_positions(n,j):
            if k==j or k in used or t[k]!=src: continue
            tt=t.copy(); tt[j],tt[k]=tt[k],tt[j]; after=line_match_counts(tt)
            d2=after[2]-before[2]
            if d2<=0: continue
            collateral=sum(abs(after[L]-before[L]) for L in (1,3,4))
            candidates.append((-d2,collateral,k,tt))
        if not candidates: continue
        candidates.sort(key=lambda z:(z[0],z[1],z[2])); _,_,k,tt=candidates[0]
        if line_zone_counters(t)!=line_zone_counters(tt): raise RuntimeError('plant changed line-zone inventory')
        t=tt; used.update((i,j,k)); swaps+=1
    return tuple(t),swaps

def planted_control(lines,ref,n=N_PLANTED):
    passes=0; swaps_all=[]; summaries=[]
    events=counts_by_lag(lines)['events']
    for r in range(n):
        base=shuffle_corpus_once(lines,SEED+70000+r); rng=random.Random(SEED+71000+r); planted=[]; ns=0
        for x in base:
            before=line_zone_counters(x['tokens']); tt,k=plant_line_lag2(x['tokens'],rng,PLANT_Q); ns+=k
            if before!=line_zone_counters(tt): raise RuntimeError('planted corpus changed line-zone inventory')
            planted.append({'folio':x['folio'],'section':x['section'],'tokens':tt})
        obs=counts_by_lag(planted); s=summarize_obs(obs,ref); passes+=int(s['full_lag2_specific_gate']); swaps_all.append(ns)
        summaries.append({'swaps':ns,'z2':s['lags']['2']['z'],'contrast':s['lag2_specificity_contrast'],'pass':s['full_lag2_specific_gate']})
    frac=passes/n; ms=med(swaps_all)
    return {'n':n,'q':PLANT_Q,'gate_passes':passes,'gate_fraction':frac,'median_swaps':ms,
            'median_z2':med([x['z2'] for x in summaries]),'median_contrast':med([x['contrast'] for x in summaries]),
            'pass':bool(frac>=.80 and ms>=10),'replicates':summaries}

def exact_identity_diagnostics(lines,lag=2):
    obs=collections.Counter(); exp=collections.Counter(); sec_obs=collections.Counter(); sec_exp=collections.Counter()
    for x in lines:
        t=x['tokens']; n=len(t); groups=p.groups_for(n,'N1'); gmap={int(pos):gi for gi,g in enumerate(groups) for pos in g}
        for i in boundary_starts(n,lag):
            j=i+lag; a=t[i]
            if t[i]==t[j]: obs[a]+=1; sec_obs[x['section']]+=1
            ga,gb=gmap[i],gmap[j]; A=groups[ga]; B=groups[gb]
            ca=collections.Counter(t[int(k)] for k in A); cb=collections.Counter(t[int(k)] for k in B)
            if ga==gb:
                m=len(A); den=m*(m-1)
                if den:
                    for tok,c in ca.items(): exp[tok]+=c*(c-1)/den
                    sec_exp[x['section']]+=sum(c*(c-1) for c in ca.values())/den
            else:
                den=len(A)*len(B)
                for tok,c in ca.items(): exp[tok]+=c*cb.get(tok,0)/den
                sec_exp[x['section']]+=sum(c*cb.get(tok,0) for tok,c in ca.items())/den
    toks=sorted(set(obs)|set(exp),key=lambda k:(-(obs[k]-exp[k]),k))[:20]
    secs=sorted(set(sec_obs)|set(sec_exp))
    return {'top_identity_excess':[{'token':k,'observed':obs[k],'expected':exp[k],'excess':obs[k]-exp[k]} for k in toks],
            'section_residuals':{s:{'observed':sec_obs[s],'expected':sec_exp[s],'excess':sec_obs[s]-sec_exp[s]} for s in secs}}

def adjudicate(cal,target):
    if not cal['pass']: return 'INSTRUMENT_FAIL'
    z2=target['lags']['2']['z']; p2=target['lags']['2']['p_one_sided']; pm=target['p_maxT_lag2']
    if z2<2.58 or p2>.01 or pm>.01: return 'LINE_ZONE_INVENTORY_SUFFICIENT_FOR_LAG2'
    if target['lag2_specificity_contrast']<1.0 or target['p_specificity']>.01:
        return 'SHORT_RANGE_ORDERING_RESIDUAL_NOT_LAG2_SPECIFIC'
    return 'LAG2_SEQUENTIAL_RESIDUAL_SUPPORTED'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('rf1b',nargs='?',default='/tmp/RF1b.txt'); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    nperm=2000 if args.smoke else NPERM; nnull=30 if args.smoke else N_NULL_PSEUDO; nplant=5 if args.smoke else N_PLANTED
    sections=p.load_sections(); raw,lines,parse_audit=p.parse_rf(args.rf1b,sections); sha=hashlib.sha256(raw).hexdigest(); inherited=b.score(lines)
    validation={'source_sha256':sha,'header_ok':raw.startswith(b'#=IVTFF STA1 2.0'),'folios':len({x['folio'] for x in lines}),
                'segments':len(lines),'tokens':sum(len(x['tokens']) for x in lines),'parser':parse_audit,
                'anchors':{'E2_N0':inherited['E2_N0'],'E2_N1':inherited['E2_N1'],'E2_N3':inherited['E2_N3']}}
    validation['pass']=bool(validation['header_ok'] and sha==SOURCE_SHA and 1.16<=inherited['E2_N0']<=1.20 and 1.05<=inherited['E2_N1']<=1.09 and 1.02<=inherited['E2_N3']<=1.07)
    result={'metadata':{'seed':SEED,'nperm':nperm,'n_null_pseudo':nnull,'n_planted':nplant,'plant_q':PLANT_Q},'validation':validation}
    if not validation['pass']:
        result['verdict']='INSTRUMENT_FAIL'; result['reason']='SOURCE_OR_ANCHOR_VALIDATION_FAIL'
    else:
        c0=preservation_audit(lines)
        ref=permutation_counts(lines,nperm,SEED+10000)
        obs=counts_by_lag(lines); target=summarize_obs(obs,ref)
        c1=null_pseudo_control(ref,lines,nnull)
        c2=planted_control(lines,ref,nplant)
        cal={'C0_exact_line_zone_preservation':c0,'C1_null_pseudo':c1,'C2_planted_identity':c2,
             'pass':bool(c0 and c1['pass'] and c2['pass'])}
        result['calibration']=cal; result['target']=target; result['diagnostics']=exact_identity_diagnostics(lines,2)
        result['verdict']=adjudicate(cal,target)
    outdir=Path('results/sta_line_zone_null_v0_2'); outdir.mkdir(parents=True,exist_ok=True); stem='SMOKE' if args.smoke else 'RESULTS'
    (outdir/f'{stem}_20260815.json').write_text(json.dumps(result,indent=2)+'\n')
    md=['# STA within-line / within-zone short-lag discriminator v0.2','',f"Mode: **{'SMOKE' if args.smoke else 'PRIMARY'}**",f"Validation: **{'PASS' if validation['pass'] else 'FAIL'}**",f"Verdict: **{result.get('verdict')}**"]
    if 'target' in result:
        md+=['','|lag|events|observed|null mean|ratio|z|p(one-sided)|','|---|---:|---:|---:|---:|---:|---:|']
        for lag in LAGS:
            s=result['target']['lags'][str(lag)]; md.append(f"|{lag}|{s['events']}|{s['observed']}|{s['null_mean']:.3f}|{s['ratio']:.4f}|{s['z']:.3f}|{s['p_one_sided']:.6g}|")
        md+=['',f"Lag-2 maxT p: **{result['target']['p_maxT_lag2']:.6g}**",f"Lag-2 specificity contrast: **{result['target']['lag2_specificity_contrast']:.3f}**",f"Specificity p: **{result['target']['p_specificity']:.6g}**",f"Full lag-2-specific gate: **{result['target']['full_lag2_specific_gate']}**",'',f"Calibration pass: **{result['calibration']['pass']}**",f"Null pseudo gate fraction: {result['calibration']['C1_null_pseudo']['gate_fraction']:.3f}",f"Planted gate fraction: {result['calibration']['C2_planted_identity']['gate_fraction']:.3f}; median swaps={result['calibration']['C2_planted_identity']['median_swaps']:.1f}"]
    (outdir/f'{stem}_20260815.md').write_text('\n'.join(md)+'\n'); print('\n'.join(md))

if __name__=='__main__': main()

#!/usr/bin/env python3
"""STA fixed-marginal boundary diagonal test v0.1.

Binding protocol: PREREG_20260815.md.
This is a conditional identity-diagonal detector, not a generator fit.
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
NFOLD=5
K_PRIMARY=8
NPERM=20000
NBOOT=5000
N_N1=200
N_TEMPLATE=50
N_PLANTED=30
PLANTED_Q=0.03
SOURCE_SHA='81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17'


def med(xs): return float(np.median(np.asarray(xs,dtype=float)))
def qtile(xs,q): return float(np.quantile(np.asarray(xs,dtype=float),q))

def boundary_event(i,j,n): return i<2 or j>=n-2

def line_fold(x): return p.fold_of(x['folio'])

def lag_events(lines,lag=2,boundary_only=True,zmap=None):
    out=[]
    for li,x in enumerate(lines):
        t=x['tokens']; n=len(t); sec=x['section']
        if n<=lag: continue
        for i in range(n-lag):
            j=i+lag
            if boundary_only and not boundary_event(i,j,n): continue
            out.append({'line':li,'folio':x['folio'],'fold':line_fold(x),'section':sec,'n':n,
                        'lb':p.lbucket(n),'si':p.edge_pos(i,n),'tj':p.edge_pos(j,n),
                        'source':t[i],'target':t[j],'eq':t[i]==t[j],
                        'z':None if zmap is None else zmap[li],
                        'left':i<2,'right':j>=n-2})
    return out

def panel_key(ev,panel):
    if panel=='D0': return (ev['section'],ev['si'],ev['tj'])
    if panel=='D1': return (ev['section'],ev['lb'],ev['si'],ev['tj'])
    if panel=='D2': return (ev['section'],ev['n'],ev['si'],ev['tj'])
    if panel=='D3': return (ev['section'],ev['lb'],ev['si'],ev['tj'],ev['z'])
    raise ValueError(panel)

def stratum_moments(rows):
    N=len(rows); r=collections.Counter(e['source'] for e in rows); c=collections.Counter(e['target'] for e in rows)
    obs=sum(e['eq'] for e in rows)
    if N==0: return obs,0.0,0.0,r,c
    S=sum(r[a]*c.get(a,0) for a in r)
    mu=S/N
    if N<2:
        return obs,float(mu),0.0,r,c
    same=sum(r[a]*(r[a]-1)*c.get(a,0)*(c.get(a,0)-1) for a in r)
    sq=sum((r[a]*c.get(a,0))**2 for a in r)
    diff=S*S-sq
    efact=(same+diff)/(N*(N-1))
    var=max(0.0,efact+mu-mu*mu)
    return obs,float(mu),float(var),r,c

def panel_summary(events,panel,zmap_required=False):
    strata=collections.defaultdict(list)
    for e in events:
        if zmap_required and e['z'] is None: raise RuntimeError('D3 event lacks cross-fitted Z')
        strata[panel_key(e,panel)].append(e)
    observed=0; expected=0.0; variance=0.0; residual_by_folio=collections.defaultdict(lambda:[0.0,0]); residual_by_fold=collections.Counter()
    audit=[]
    for key,rows in strata.items():
        obs,mu,var,r,c=stratum_moments(rows); observed+=obs; expected+=mu; variance+=var
        N=len(rows)
        for e in rows:
            pi=c.get(e['source'],0)/N if N else 0.0
            rr=(1.0 if e['eq'] else 0.0)-pi
            residual_by_folio[e['folio']][0]+=rr; residual_by_folio[e['folio']][1]+=1
            residual_by_fold[e['fold']]+=rr
        audit.append((key,N,r,c))
    ratio=observed/expected if expected>0 else None
    z=(observed-expected)/math.sqrt(variance) if variance>0 else None
    return {'panel':panel,'events':len(events),'strata':len(strata),'observed':int(observed),'expected':float(expected),
            'excess':float(observed-expected),'ratio':float(ratio) if ratio is not None else None,
            'variance':float(variance),'z':float(z) if z is not None else None,
            'residual_by_folio':{k:[float(a),int(n)] for k,(a,n) in residual_by_folio.items()},
            'residual_by_fold':{str(k):float(residual_by_fold[k]) for k in range(NFOLD)},
            '_strata':strata,'_audit':audit}

def bootstrap_panel(summary,nboot=NBOOT,seed=SEED+70000):
    items=list(summary['residual_by_folio'].values())
    if not items: return {'mean_residual':None,'bootstrap95':[None,None],'positive_folds':0}
    point=sum(a for a,n in items)/sum(n for a,n in items)
    rng=random.Random(seed); boots=[]
    for _ in range(nboot):
        samp=[items[rng.randrange(len(items))] for __ in range(len(items))]
        den=sum(n for a,n in samp); boots.append(sum(a for a,n in samp)/den if den else 0.0)
    pos=sum(v>0 for v in summary['residual_by_fold'].values())
    return {'mean_residual':float(point),'bootstrap95':[qtile(boots,.025),qtile(boots,.975)],'positive_folds':int(pos)}

def permutation_p(summary,nperm=NPERM,seed=SEED+80000,chunk=200):
    strata=[]
    for key,rows in summary['_strata'].items():
        if len(rows)<=1: continue
        toks=sorted({e['source'] for e in rows}|{e['target'] for e in rows})
        enc={t:i for i,t in enumerate(toks)}
        s=np.asarray([enc[e['source']] for e in rows],dtype=np.int32)
        t=np.asarray([enc[e['target']] for e in rows],dtype=np.int32)
        strata.append((s,t))
    obs=summary['observed']; ge=0; done=0; rng=np.random.default_rng(seed)
    while done<nperm:
        m=min(chunk,nperm-done); totals=np.zeros(m,dtype=np.int32)
        for s,t in strata:
            N=len(t)
            # random-key permutations: independent uniform permutation in each row (ties negligible)
            keys=rng.random((m,N)); order=np.argsort(keys,axis=1)
            totals += (t[order]==s[None,:]).sum(axis=1)
        ge += int((totals>=obs).sum()); done += m
    return {'nperm':int(nperm),'ge':int(ge),'p_one_sided':float((ge+1)/(nperm+1))}

def marginal_audit(summary,seed=SEED+81000):
    rng=random.Random(seed)
    for key,N,r,c in summary['_audit']:
        rows=summary['_strata'][key]
        src=[e['source'] for e in rows]; tar=[e['target'] for e in rows]
        before_s=collections.Counter(src); before_t=collections.Counter(tar)
        rng.shuffle(tar)
        if before_s!=collections.Counter(src) or before_t!=collections.Counter(tar): return False
        if before_s!=r or before_t!=c or N!=len(rows): return False
    return True

def analytic_gate(summary):
    return bool(summary['ratio'] is not None and summary['z'] is not None and summary['ratio']>=1.10 and summary['z']>=2.58)

def full_gate(summary,perm,boot):
    return bool(analytic_gate(summary) and perm['p_one_sided']<=.01 and boot['bootstrap95'][0] is not None and boot['bootstrap95'][0]>0 and boot['positive_folds']>=4)

def public_summary(s):
    return {k:v for k,v in s.items() if not k.startswith('_') and k not in ('residual_by_folio',)}

# -------- exact-equivalent fast k-modes used only for D3 cross-fitting --------
def kmodes_fast(lines,K):
    feats=[b.boundary_feature(x) for x in lines]; counts=collections.Counter(feats); uniq=sorted(counts)
    if not uniq:return [],[]
    K=min(K,len(uniq)); mx=max(counts.values()); prot=[min(v for v in uniq if counts[v]==mx)]
    while len(prot)<K:
        bestd=-1; best=None
        for v in uniq:
            if v in prot: continue
            d=min(b.hamming(v,z) for z in prot)
            if d>bestd or (d==bestd and (best is None or v<best)): bestd=d; best=v
        prot.append(best)
    assign=None
    for _ in range(30):
        new=[]
        for v in feats:
            ds=[b.hamming(v,z) for z in prot]; new.append(min(range(K),key=lambda z:(ds[z],z)))
        if assign==new: break
        assign=new
        for z in range(K):
            members=[feats[i] for i,a in enumerate(assign) if a==z]
            if not members: continue
            nv=[]
            for col in zip(*members):
                c=collections.Counter(col); mm=max(c.values()); nv.append(min(k for k,v in c.items() if v==mm))
            prot[z]=tuple(nv)
    return prot,assign

def crossfit_z(lines,K=K_PRIMARY):
    zmap={}
    for f in range(NFOLD):
        train_idx=[i for i,x in enumerate(lines) if line_fold(x)!=f]
        test_idx=[i for i,x in enumerate(lines) if line_fold(x)==f]
        train=[lines[i] for i in train_idx]; prot,_=kmodes_fast(train,K)
        if not prot: raise RuntimeError('no prototypes')
        for i in test_idx:
            feat=b.boundary_feature(lines[i]); ds=[b.hamming(feat,z) for z in prot]
            zmap[i]=min(range(len(prot)),key=lambda z:(ds[z],z))
    if len(zmap)!=len(lines): raise RuntimeError('incomplete crossfit Z')
    return zmap

def evaluate_panel(lines,panel,lag=2,boundary_only=True,zmap=None,do_perm=False,perm_seed=SEED+80000,boot_seed=SEED+70000):
    ev=lag_events(lines,lag,boundary_only,zmap); s=panel_summary(ev,panel,panel=='D3'); boot=bootstrap_panel(s,NBOOT,boot_seed)
    perm=permutation_p(s,NPERM,perm_seed) if do_perm else None
    return s,boot,perm

def n1_controls(lines,n=N_N1):
    zs=[]; ratios=[]; gates=[]
    for r in range(n):
        sh=b.n1_shuffle(lines,random.Random(SEED+900000+r)); s=panel_summary(lag_events(sh,2,True), 'D2')
        zs.append(s['z'] if s['z'] is not None else 0.0); ratios.append(s['ratio'] if s['ratio'] is not None else 1.0); gates.append(analytic_gate(s))
    return {'n':n,'median_z':med(zs),'p99_z':qtile(zs,.99),'gate_fraction':sum(gates)/n,
            'median_ratio':med(ratios),'zs':zs,
            'basic_pass':bool(sum(gates)/n<=.05 and -.5<=med(zs)<=.5)}

def plant_identity(lines,q,seed):
    rng=random.Random(seed); out=[]
    for x in lines:
        t=list(x['tokens']); n=len(t)
        for j in range(2,n):
            i=j-2
            if boundary_event(i,j,n) and rng.random()<q: t[j]=t[i]
        out.append({'folio':x['folio'],'section':x['section'],'tokens':tuple(t)})
    return out

def synthetic_controls(lines,smoke=False):
    fits=b.fit_fold_models(lines,K_PRIMARY)
    nt=4 if smoke else N_TEMPLATE; np_=4 if smoke else N_PLANTED
    neg={'D2':0,'D3':0}; pos={'D2':0,'D3':0}; pos_ratios=[]; neg_ratios=[]
    for r in range(max(nt,np_)):
        syn=b.generate_oof(fits,'B2',3000+r)
        if r<nt:
            z=crossfit_z(syn,K_PRIMARY)
            s2=panel_summary(lag_events(syn,2,True),'D2'); s3=panel_summary(lag_events(syn,2,True,z),'D3',True)
            neg['D2']+=analytic_gate(s2); neg['D3']+=analytic_gate(s3); neg_ratios.append([s2['ratio'],s3['ratio']])
        if r<np_:
            pl=plant_identity(syn,PLANTED_Q,SEED+950000+r); z=crossfit_z(pl,K_PRIMARY)
            s2=panel_summary(lag_events(pl,2,True),'D2'); s3=panel_summary(lag_events(pl,2,True,z),'D3',True)
            pos['D2']+=analytic_gate(s2); pos['D3']+=analytic_gate(s3); pos_ratios.append([s2['ratio'],s3['ratio']])
    negfrac={k:neg[k]/nt for k in neg}; posfrac={k:pos[k]/np_ for k in pos}
    passed=all(v<=.05 for v in negfrac.values()) and all(v>=.80 for v in posfrac.values())
    return {'template_only_n':nt,'template_only_gate_fraction':negfrac,'planted_n':np_,'planted_q':PLANTED_Q,
            'planted_gate_fraction':posfrac,'negative_ratios':neg_ratios,'planted_ratios':pos_ratios,'pass':bool(passed)}

def side_summary(events,side):
    ev=[e for e in events if e[side]]; s=panel_summary(ev,'D2'); return public_summary(s)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('rf1b',nargs='?',default='/tmp/RF1b.txt'); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    sections=p.load_sections(); raw,lines,parse_audit=p.parse_rf(args.rf1b,sections); sha=hashlib.sha256(raw).hexdigest()
    inherited=b.score(lines)
    validation={'source_sha256':sha,'header_ok':raw.startswith(b'#=IVTFF STA1 2.0'),'folios':len({x['folio'] for x in lines}),
                'segments':len(lines),'tokens':sum(len(x['tokens']) for x in lines),'parser':parse_audit,
                'anchors':{'E2_N0':inherited['E2_N0'],'E2_N1':inherited['E2_N1'],'E2_N3':inherited['E2_N3']}}
    validation['pass']=bool(validation['header_ok'] and sha==SOURCE_SHA and 1.16<=inherited['E2_N0']<=1.20 and 1.05<=inherited['E2_N1']<=1.09 and 1.02<=inherited['E2_N3']<=1.07)
    result={'metadata':{'seed':SEED,'nperm':(1000 if args.smoke else NPERM),'nboot':(500 if args.smoke else NBOOT),'K':K_PRIMARY},'validation':validation}
    if not validation['pass']:
        result['verdict']='INSTRUMENT_FAIL'; result['reason']='SOURCE_OR_ANCHOR_VALIDATION_FAIL'
    else:
        # smoke temporarily patches workloads, without changing primary thresholds
        global NPERM,NBOOT
        if args.smoke: NPERM,NBOOT=1000,500
        z8=crossfit_z(lines,K_PRIMARY)
        panels={}
        for ix,panel in enumerate(('D0','D1','D2','D3')):
            z=z8 if panel=='D3' else None; s,boot,perm=evaluate_panel(lines,panel,2,True,z,do_perm=panel in ('D2','D3'),perm_seed=SEED+81000+ix,boot_seed=SEED+71000+ix)
            panels[panel]={'summary':public_summary(s),'bootstrap':boot,'permutation':perm,'analytic_gate':analytic_gate(s),
                           'full_gate':full_gate(s,perm,boot) if perm is not None else None,'marginal_audit':marginal_audit(s,SEED+82000+ix)}
        result['panels']=panels
        n1=n1_controls(lines,10 if args.smoke else N_N1); n1['real_exceeds_p99']=bool(panels['D2']['summary']['z']>n1['p99_z'])
        syn=synthetic_controls(lines,args.smoke)
        c0=all(panels[k]['marginal_audit'] for k in panels)
        calibration={'C0_marginal_audit':c0,'C1_N1':n1,'C2_C3_synthetic':syn,
                     'pass':bool(c0 and n1['basic_pass'] and n1['real_exceeds_p99'] and syn['pass'])}
        result['calibration']=calibration
        # hostile specificity and side diagnostics
        specs={}
        for lag in (1,3,4):
            s=panel_summary(lag_events(lines,lag,True),'D2'); specs[str(lag)]=public_summary(s)
        ev2=lag_events(lines,2,True); result['specificity_lags']=specs
        result['side_diagnostics']={'left':side_summary(ev2,'left'),'right':side_summary(ev2,'right')}
        # K sensitivities, diagnostics only
        ks={}
        if not args.smoke:
            for K in (4,12):
                zk=crossfit_z(lines,K); s=panel_summary(lag_events(lines,2,True,zk),'D3',True); boot=bootstrap_panel(s,NBOOT,SEED+73000+K)
                ks[str(K)]={'summary':public_summary(s),'bootstrap':boot,'analytic_gate':analytic_gate(s)}
        result['K_sensitivity']=ks
        if not calibration['pass']:
            verdict='INSTRUMENT_FAIL'
        elif not panels['D2']['full_gate']:
            verdict='FIXED_MARGINALS_EXPLAIN_LAG2_DIAGONAL'
        elif not panels['D3']['full_gate']:
            verdict='LATENT_TEMPLATE_MIXTURE_NOT_EXCLUDED'
        else:
            verdict='IDENTITY_DIAGONAL_ENRICHED'
        result['verdict']=verdict
    outdir=Path('results/sta_fixed_marginal_diagonal_v0_1'); outdir.mkdir(parents=True,exist_ok=True); stem='SMOKE' if args.smoke else 'RESULTS'
    (outdir/f'{stem}_20260815.json').write_text(json.dumps(result,indent=2)+'\n')
    md=['# STA fixed-marginal boundary diagonal v0.1','',f"Mode: **{'SMOKE' if args.smoke else 'PRIMARY'}**",f"Validation: **{'PASS' if validation['pass'] else 'FAIL'}**",f"Verdict: **{result.get('verdict')}**"]
    if 'panels' in result:
        md+=['','|panel|events|observed|expected|ratio|z|perm p|full gate|','|---|---:|---:|---:|---:|---:|---:|---|']
        for q in ('D0','D1','D2','D3'):
            d=result['panels'][q]; s=d['summary']; pp='-' if d['permutation'] is None else f"{d['permutation']['p_one_sided']:.6g}"
            md.append(f"|{q}|{s['events']}|{s['observed']}|{s['expected']:.2f}|{s['ratio']:.4f}|{s['z'] if s['z'] is not None else float('nan'):.3f}|{pp}|{d['full_gate']}|")
        md+=['',f"Calibration pass: **{result['calibration']['pass']}**",f"N1 controls: median z={result['calibration']['C1_N1']['median_z']:.3f}; p99 z={result['calibration']['C1_N1']['p99_z']:.3f}; gate fraction={result['calibration']['C1_N1']['gate_fraction']:.3f}.",f"Synthetic template-only gate fractions: `{result['calibration']['C2_C3_synthetic']['template_only_gate_fraction']}`.",f"Synthetic q=0.03 planted gate fractions: `{result['calibration']['C2_C3_synthetic']['planted_gate_fraction']}`."]
    (outdir/f'{stem}_20260815.md').write_text('\n'.join(md)+'\n'); print('\n'.join(md))

if __name__=='__main__': main()

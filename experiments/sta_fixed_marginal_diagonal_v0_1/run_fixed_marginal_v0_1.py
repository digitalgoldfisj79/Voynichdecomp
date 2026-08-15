#!/usr/bin/env python3
"""STA fixed-marginal boundary diagonal test v0.1. See PREREG_20260815.md."""
import argparse, collections, hashlib, importlib.util, json, math, random
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'sta_boundary_return_discriminator_v0_1'/'run_discriminator.py'
spec=importlib.util.spec_from_file_location('boundary_parent',PARENT)
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
p=b.p

SEED=20260815; NFOLD=5; K_PRIMARY=8
NPERM=20000; NBOOT=5000; N_N1=200; N_TEMPLATE=50; N_PLANTED=30; PLANTED_Q=.03
SOURCE_SHA='81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17'

def med(x): return float(np.median(np.asarray(x,dtype=float)))
def qtile(x,q): return float(np.quantile(np.asarray(x,dtype=float),q))
def boundary_event(i,j,n): return i<2 or j>=n-2
def fold_of_line(x): return p.fold_of(x['folio'])

def lag_events(lines,lag=2,boundary_only=True,zmap=None):
    out=[]
    for li,x in enumerate(lines):
        t=x['tokens']; n=len(t)
        for i in range(max(0,n-lag)):
            j=i+lag
            if boundary_only and not boundary_event(i,j,n): continue
            out.append({'line':li,'folio':x['folio'],'fold':fold_of_line(x),'section':x['section'],'n':n,'lb':p.lbucket(n),
                        'si':p.edge_pos(i,n),'tj':p.edge_pos(j,n),'source':t[i],'target':t[j],'eq':t[i]==t[j],
                        'z':None if zmap is None else zmap[li],'left':i<2,'right':j>=n-2})
    return out

def key(e,panel):
    if panel=='D0': return (e['section'],e['si'],e['tj'])
    if panel=='D1': return (e['section'],e['lb'],e['si'],e['tj'])
    if panel=='D2': return (e['section'],e['n'],e['si'],e['tj'])
    if panel=='D3': return (e['section'],e['lb'],e['si'],e['tj'],e['z'])
    raise ValueError(panel)

def stratum_moments(rows):
    N=len(rows); r=collections.Counter(e['source'] for e in rows); c=collections.Counter(e['target'] for e in rows); obs=sum(e['eq'] for e in rows)
    if not N: return obs,0.,0.,r,c
    S=sum(r[a]*c.get(a,0) for a in r); mu=S/N
    if N<2: return obs,float(mu),0.,r,c
    same=sum(r[a]*(r[a]-1)*c.get(a,0)*(c.get(a,0)-1) for a in r)
    sq=sum((r[a]*c.get(a,0))**2 for a in r); diff=S*S-sq
    efact=(same+diff)/(N*(N-1)); var=max(0.,efact+mu-mu*mu)
    return obs,float(mu),float(var),r,c

def summarize(events,panel):
    strata=collections.defaultdict(list)
    for e in events:
        if panel=='D3' and e['z'] is None: raise RuntimeError('D3 requires cross-fitted Z')
        strata[key(e,panel)].append(e)
    obs=0; exp=0.; var=0.; byfolio=collections.defaultdict(lambda:[0.,0]); byfold=collections.Counter(); audit=[]
    for k,rows in strata.items():
        o,m,v,r,c=stratum_moments(rows); obs+=o; exp+=m; var+=v; N=len(rows); audit.append((k,N,r,c))
        for e in rows:
            resid=(1. if e['eq'] else 0.)-c.get(e['source'],0)/N
            byfolio[e['folio']][0]+=resid; byfolio[e['folio']][1]+=1; byfold[e['fold']]+=resid
    ratio=obs/exp if exp>0 else None; z=(obs-exp)/math.sqrt(var) if var>0 else None
    return {'panel':panel,'events':len(events),'strata':len(strata),'observed':int(obs),'expected':float(exp),'excess':float(obs-exp),
            'ratio':None if ratio is None else float(ratio),'variance':float(var),'z':None if z is None else float(z),
            'residual_by_folio':{f:[float(a),int(n)] for f,(a,n) in byfolio.items()},
            'residual_by_fold':{str(f):float(byfold[f]) for f in range(NFOLD)},'_strata':strata,'_audit':audit}

def public(s): return {k:v for k,v in s.items() if not k.startswith('_') and k!='residual_by_folio'}

def bootstrap(s,nboot,seed):
    items=list(s['residual_by_folio'].values())
    if not items:return {'mean_residual':None,'bootstrap95':[None,None],'positive_folds':0}
    point=sum(a for a,n in items)/sum(n for a,n in items); rng=random.Random(seed); vals=[]
    for _ in range(nboot):
        samp=[items[rng.randrange(len(items))] for __ in range(len(items))]; den=sum(n for a,n in samp)
        vals.append(sum(a for a,n in samp)/den if den else 0.)
    return {'mean_residual':float(point),'bootstrap95':[qtile(vals,.025),qtile(vals,.975)],
            'positive_folds':sum(v>0 for v in s['residual_by_fold'].values())}

def permutation_p(s,nperm,seed,chunk=200):
    pieces=[]; fixed=0
    for rows in s['_strata'].values():
        if len(rows)<=1:
            fixed+=sum(e['eq'] for e in rows); continue
        toks=sorted({e['source'] for e in rows}|{e['target'] for e in rows}); enc={t:i for i,t in enumerate(toks)}
        src=np.asarray([enc[e['source']] for e in rows],dtype=np.int32); tar=np.asarray([enc[e['target']] for e in rows],dtype=np.int32)
        pieces.append((src,tar))
    rng=np.random.default_rng(seed); ge=0; done=0
    while done<nperm:
        m=min(chunk,nperm-done); totals=np.full(m,fixed,dtype=np.int32)
        for src,tar in pieces:
            order=np.argsort(rng.random((m,len(tar))),axis=1); totals+=(tar[order]==src[None,:]).sum(axis=1)
        ge+=int((totals>=s['observed']).sum()); done+=m
    return {'nperm':nperm,'ge':ge,'p_one_sided':float((ge+1)/(nperm+1))}

def marginal_audit(s,seed):
    rng=random.Random(seed)
    for k,N,r,c in s['_audit']:
        rows=s['_strata'][k]; src=[e['source'] for e in rows]; tar=[e['target'] for e in rows]
        if collections.Counter(src)!=r or collections.Counter(tar)!=c or len(rows)!=N:return False
        before=collections.Counter(tar); rng.shuffle(tar)
        if collections.Counter(tar)!=before:return False
    return True

def analytic_gate(s): return bool(s['ratio'] is not None and s['z'] is not None and s['ratio']>=1.10 and s['z']>=2.58)
def full_gate(s,perm,boot): return bool(analytic_gate(s) and perm['p_one_sided']<=.01 and boot['bootstrap95'][0] is not None and boot['bootstrap95'][0]>0 and boot['positive_folds']>=4)

def crossfit_z(lines,K=K_PRIMARY):
    zmap={}
    for f in range(NFOLD):
        train=[x for x in lines if fold_of_line(x)!=f]; prot,_=b.kmodes(train,K)
        if not prot: raise RuntimeError('no prototypes')
        for i,x in enumerate(lines):
            if fold_of_line(x)!=f: continue
            feat=b.boundary_feature(x); ds=[b.hamming(feat,z) for z in prot]
            zmap[i]=min(range(len(prot)),key=lambda z:(ds[z],z))
    if len(zmap)!=len(lines): raise RuntimeError('incomplete crossfit Z')
    return zmap

def eval_panel(lines,panel,zmap,nperm,nboot,do_perm,offset):
    s=summarize(lag_events(lines,2,True,zmap),panel); bt=bootstrap(s,nboot,SEED+71000+offset)
    pm=permutation_p(s,nperm,SEED+81000+offset) if do_perm else None
    return {'summary':public(s),'bootstrap':bt,'permutation':pm,'analytic_gate':analytic_gate(s),
            'full_gate':None if pm is None else full_gate(s,pm,bt),'marginal_audit':marginal_audit(s,SEED+82000+offset)}

def n1_controls(lines,n):
    zs=[]; rs=[]; gs=[]
    for r in range(n):
        sh=b.n1_shuffle(lines,random.Random(SEED+900000+r)); s=summarize(lag_events(sh,2,True),'D2')
        zs.append(0. if s['z'] is None else s['z']); rs.append(1. if s['ratio'] is None else s['ratio']); gs.append(analytic_gate(s))
    return {'n':n,'median_z':med(zs),'p99_z':qtile(zs,.99),'gate_fraction':sum(gs)/n,'median_ratio':med(rs),'zs':zs,
            'basic_pass':bool(sum(gs)/n<=.05 and -.5<=med(zs)<=.5)}

def plant(lines,q,seed):
    rng=random.Random(seed); out=[]
    for x in lines:
        t=list(x['tokens']); n=len(t)
        for j in range(2,n):
            if boundary_event(j-2,j,n) and rng.random()<q:t[j]=t[j-2]
        out.append({'folio':x['folio'],'section':x['section'],'tokens':tuple(t)})
    return out

def synthetic_controls(lines,nneg,npos):
    fits=b.fit_fold_models(lines,K_PRIMARY); neg={'D2':0,'D3':0}; pos={'D2':0,'D3':0}; nr=[]; pr=[]
    for r in range(max(nneg,npos)):
        syn=b.generate_oof(fits,'B2',3000+r)
        if r<nneg:
            z=crossfit_z(syn,K_PRIMARY); s2=summarize(lag_events(syn,2,True),'D2'); s3=summarize(lag_events(syn,2,True,z),'D3')
            neg['D2']+=analytic_gate(s2); neg['D3']+=analytic_gate(s3); nr.append([s2['ratio'],s3['ratio']])
        if r<npos:
            pl=plant(syn,PLANTED_Q,SEED+950000+r); z=crossfit_z(pl,K_PRIMARY); s2=summarize(lag_events(pl,2,True),'D2'); s3=summarize(lag_events(pl,2,True,z),'D3')
            pos['D2']+=analytic_gate(s2); pos['D3']+=analytic_gate(s3); pr.append([s2['ratio'],s3['ratio']])
    nf={k:neg[k]/nneg for k in neg}; pf={k:pos[k]/npos for k in pos}
    return {'template_only_n':nneg,'template_only_gate_fraction':nf,'planted_n':npos,'planted_q':PLANTED_Q,'planted_gate_fraction':pf,
            'negative_ratios':nr,'planted_ratios':pr,'pass':bool(all(v<=.05 for v in nf.values()) and all(v>=.80 for v in pf.values()))}

def side_diag(events,side): return public(summarize([e for e in events if e[side]],'D2'))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('rf1b',nargs='?',default='/tmp/RF1b.txt'); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    nperm=1000 if args.smoke else NPERM; nboot=500 if args.smoke else NBOOT; nn1=10 if args.smoke else N_N1; nneg=4 if args.smoke else N_TEMPLATE; npos=4 if args.smoke else N_PLANTED
    sections=p.load_sections(); raw,lines,pa=p.parse_rf(args.rf1b,sections); sha=hashlib.sha256(raw).hexdigest(); inherited=b.score(lines)
    val={'source_sha256':sha,'header_ok':raw.startswith(b'#=IVTFF STA1 2.0'),'folios':len({x['folio'] for x in lines}),'segments':len(lines),'tokens':sum(len(x['tokens']) for x in lines),'parser':pa,
         'anchors':{'E2_N0':inherited['E2_N0'],'E2_N1':inherited['E2_N1'],'E2_N3':inherited['E2_N3']}}
    val['pass']=bool(val['header_ok'] and sha==SOURCE_SHA and 1.16<=inherited['E2_N0']<=1.20 and 1.05<=inherited['E2_N1']<=1.09 and 1.02<=inherited['E2_N3']<=1.07)
    result={'metadata':{'seed':SEED,'nperm':nperm,'nboot':nboot,'n_N1':nn1,'n_template':nneg,'n_planted':npos,'K':K_PRIMARY},'validation':val}
    if not val['pass']:
        result['verdict']='INSTRUMENT_FAIL'; result['reason']='SOURCE_OR_ANCHOR_VALIDATION_FAIL'
    else:
        z8=crossfit_z(lines,K_PRIMARY); panels={}
        for ix,q in enumerate(('D0','D1','D2','D3')):panels[q]=eval_panel(lines,q,z8 if q=='D3' else None,nperm,nboot,q in ('D2','D3'),ix)
        result['panels']=panels
        c1=n1_controls(lines,nn1); c1['real_exceeds_p99']=bool(panels['D2']['summary']['z']>c1['p99_z'])
        syn=synthetic_controls(lines,nneg,npos); c0=all(panels[q]['marginal_audit'] for q in panels)
        cal={'C0_marginal_audit':c0,'C1_N1':c1,'C2_C3_synthetic':syn,'pass':bool(c0 and c1['basic_pass'] and c1['real_exceeds_p99'] and syn['pass'])}; result['calibration']=cal
        specs={}
        for lag in (1,3,4):specs[str(lag)]=public(summarize(lag_events(lines,lag,True),'D2'))
        result['specificity_lags']=specs; ev2=lag_events(lines,2,True); result['side_diagnostics']={'left':side_diag(ev2,'left'),'right':side_diag(ev2,'right')}
        ks={}
        if not args.smoke:
            for K in (4,12):
                zk=crossfit_z(lines,K); s=summarize(lag_events(lines,2,True,zk),'D3'); ks[str(K)]={'summary':public(s),'bootstrap':bootstrap(s,nboot,SEED+73000+K),'analytic_gate':analytic_gate(s)}
        result['K_sensitivity']=ks
        if not cal['pass']: verdict='INSTRUMENT_FAIL'
        elif not panels['D2']['full_gate']: verdict='FIXED_MARGINALS_EXPLAIN_LAG2_DIAGONAL'
        elif not panels['D3']['full_gate']: verdict='LATENT_TEMPLATE_MIXTURE_NOT_EXCLUDED'
        else: verdict='IDENTITY_DIAGONAL_ENRICHED'
        result['verdict']=verdict
    out=Path('results/sta_fixed_marginal_diagonal_v0_1'); out.mkdir(parents=True,exist_ok=True); stem='SMOKE' if args.smoke else 'RESULTS'
    (out/f'{stem}_20260815.json').write_text(json.dumps(result,indent=2)+'\n')
    md=['# STA fixed-marginal boundary diagonal v0.1','',f"Mode: **{'SMOKE' if args.smoke else 'PRIMARY'}**",f"Validation: **{'PASS' if val['pass'] else 'FAIL'}**",f"Verdict: **{result.get('verdict')}**"]
    if 'panels' in result:
        md+=['','|panel|events|observed|expected|ratio|z|perm p|full gate|','|---|---:|---:|---:|---:|---:|---:|---|']
        for q in ('D0','D1','D2','D3'):
            d=result['panels'][q]; s=d['summary']; pp='-' if d['permutation'] is None else f"{d['permutation']['p_one_sided']:.6g}"
            zz=float('nan') if s['z'] is None else s['z']; rr=float('nan') if s['ratio'] is None else s['ratio']
            md.append(f"|{q}|{s['events']}|{s['observed']}|{s['expected']:.2f}|{rr:.4f}|{zz:.3f}|{pp}|{d['full_gate']}|")
        md+=['',f"Calibration pass: **{result['calibration']['pass']}**",f"N1: median z={result['calibration']['C1_N1']['median_z']:.3f}; p99 z={result['calibration']['C1_N1']['p99_z']:.3f}; gate fraction={result['calibration']['C1_N1']['gate_fraction']:.3f}.",f"Template-only gate fractions: `{result['calibration']['C2_C3_synthetic']['template_only_gate_fraction']}`.",f"Planted q=.03 gate fractions: `{result['calibration']['C2_C3_synthetic']['planted_gate_fraction']}`."]
    (out/f'{stem}_20260815.md').write_text('\n'.join(md)+'\n'); print('\n'.join(md))

if __name__=='__main__': main()

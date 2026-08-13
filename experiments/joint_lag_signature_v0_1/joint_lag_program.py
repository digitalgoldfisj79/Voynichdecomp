#!/usr/bin/env python3
import json, math, os, sys, hashlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

SEED = 20260813
DATA_DEFAULT = '/mnt/data/joint_lag/voynich_transcriptions_slim.json'
UNIQUE_ELIGIBLE = ['GCGA','VDRB-1','TTVE','TTIA','ZLZB','ZLZI','TTLI','VDRB','FFSG','FFSG-2','RGVN','PCCA']
ALIASES = {'GCGI':'GCGA','PCCI':'PCCA','TTII':'TTIA','FFSG-1':'FFSG'}
METRICS = ['E1','E2','E4','N1','BRIDGE','CHAIN5','ISO_ABA']

def is_ed1(a,b):
    if a == b: return False
    la, lb = len(a), len(b)
    if abs(la-lb) > 1: return False
    if la == lb:
        return sum(c1 != c2 for c1,c2 in zip(a,b)) == 1
    if la > lb:
        a,b,la,lb = b,a,lb,la
    i=j=diff=0
    while i<la and j<lb:
        if a[i]==b[j]: i+=1; j+=1
        else:
            diff += 1; j += 1
            if diff>1: return False
    return True

def load_frame(path, frame):
    obj=json.load(open(path,encoding='utf-8'))
    lines=[]
    for page,p in obj['pages'].items():
        def line_key(x):
            try: return (0,int(x))
            except: return (1,str(x))
        for lid in sorted(p,key=line_key):
            rec=p[lid]
            s=rec.get('t',{}).get(frame)
            if s is None: continue
            toks=s.split()
            if toks: lines.append((page,str(lid),toks))
    return lines

def prepare(raw_lines, top_sets=None):
    out=[]
    for page,lid,toks in raw_lines:
        n=len(toks)
        eq=np.zeros((n,n),dtype=np.bool_)
        ed=np.zeros((n,n),dtype=np.bool_)
        for i in range(n):
            eq[i,i]=True
            for j in range(i+1,n):
                e=(toks[i]==toks[j]); eq[i,j]=eq[j,i]=e
                if not e:
                    d=is_ed1(toks[i],toks[j]); ed[i,j]=ed[j,i]=d
        tops={}
        if top_sets:
            for k,ss in top_sets.items():
                tops[k]=np.array([t in ss for t in toks],dtype=np.bool_)
        out.append((page,lid,toks,eq,ed,tops))
    return out

def denominators(prep):
    d={m:0 for m in METRICS}
    even=odd=start=interior=0
    for _,_,t,_,_,_ in prep:
        n=len(t)
        d['E1']+=max(0,n-1); d['N1']+=max(0,n-1)
        d['E2']+=max(0,n-2); d['BRIDGE']+=max(0,n-2)
        d['E4']+=max(0,n-4); d['CHAIN5']+=max(0,n-4); d['ISO_ABA']+=max(0,n-4)
        if n>=3:
            m=n-2
            even += (m+1)//2; odd += m//2
            start += 1; interior += max(0,m-1)
    return d, {'even':even,'odd':odd,'start':start,'interior':interior}

def count_metrics(prep, rng=None, permute=False, topks=()):
    c={m:0 for m in METRICS}
    phase={'even':0,'odd':0,'start':0,'interior':0}
    tk={int(k):0 for k in topks}
    for _,_,t,eq,ed,tops in prep:
        n=len(t)
        if n==0: continue
        p=rng.permutation(n) if permute else np.arange(n)
        if n>=2:
            e1=eq[p[:-1],p[1:]]
            c['E1'] += int(e1.sum())
            c['N1'] += int(ed[p[:-1],p[1:]].sum())
        if n>=3:
            e2=eq[p[:-2],p[2:]]
            c['E2'] += int(e2.sum())
            bridge=e2 & ed[p[:-2],p[1:-1]]
            c['BRIDGE'] += int(bridge.sum())
            phase['even'] += int(e2[::2].sum())
            phase['odd'] += int(e2[1::2].sum())
            phase['start'] += int(e2[0])
            if len(e2)>1: phase['interior'] += int(e2[1:].sum())
            for k in topks:
                mask=~tops[int(k)][p[:-2]]
                tk[int(k)] += int((e2 & mask).sum())
        if n>=5:
            e4=eq[p[:-4],p[4:]]
            c['E4'] += int(e4.sum())
            e02=eq[p[:-4],p[2:-2]]
            chain=e02 & e4
            c['CHAIN5'] += int(chain.sum())
            iso=e02 & (~eq[p[:-4],p[1:-3]]) & (~e4)
            c['ISO_ABA'] += int(iso.sum())
    return c,phase,tk

def summarize_scalar(actual, arr):
    arr=np.asarray(arr,dtype=float)
    mu=float(arr.mean()); sd=float(arr.std(ddof=1)) if len(arr)>1 else float('nan')
    ratio=float(actual/mu) if mu>0 else (float('inf') if actual>0 else float('nan'))
    z=float((actual-mu)/sd) if sd>0 else float('nan')
    p2=float((1+np.sum(np.abs(arr-mu)>=abs(actual-mu)))/(len(arr)+1)) if len(arr)>0 else float('nan')
    return {'actual':float(actual),'null_mean':mu,'null_sd':sd,'ratio':ratio,'z':z,'mc_p2':p2}

def analyze_raw(raw_lines, nperm, seed, topks=(5,20,50), include_top=True):
    freq=Counter(t for _,_,ts in raw_lines for t in ts)
    top_sets={k:set(x for x,_ in freq.most_common(k)) for k in topks} if include_top else {}
    prep=prepare(raw_lines,top_sets)
    den, pden=denominators(prep)
    actual,aphase,atk=count_metrics(prep,topks=topks if include_top else ())
    sims={m:[] for m in METRICS}; sph={k:[] for k in aphase}; stk={k:[] for k in atk}
    rng=np.random.default_rng(seed)
    for _ in range(nperm):
        cc,pp,kk=count_metrics(prep,rng=rng,permute=True,topks=topks if include_top else ())
        for m in METRICS: sims[m].append(cc[m])
        for k in pp: sph[k].append(pp[k])
        for k in kk: stk[k].append(kk[k])
    res={'n_lines':len(prep),'n_tokens':sum(len(x[2]) for x in prep),'nperm':nperm,'metrics':{},'phase':{},'topk':{},'top_types':freq.most_common(50)}
    for m in METRICS:
        s=summarize_scalar(actual[m],sims[m]); s['denom']=den[m]; s['rate']=actual[m]/den[m] if den[m] else float('nan'); s['null_rate']=s['null_mean']/den[m] if den[m] else float('nan'); res['metrics'][m]=s
    for k in aphase:
        s=summarize_scalar(aphase[k],sph[k]); s['denom']=pden[k]; s['rate']=aphase[k]/pden[k] if pden[k] else float('nan'); s['null_rate']=s['null_mean']/pden[k] if pden[k] else float('nan'); res['phase'][k]=s
    if pden['even'] and pden['odd']:
        obs=aphase['even']/pden['even']-aphase['odd']/pden['odd']
        arr=np.array(sph['even'])/pden['even']-np.array(sph['odd'])/pden['odd']
        res['phase']['parity_contrast']=summarize_scalar(obs,arr)
        r1=res['phase']['even']['ratio']; r2=res['phase']['odd']['ratio']; res['phase']['parity_contrast']['ratio_fold']=max(r1,r2)/min(r1,r2) if min(r1,r2)>0 else float('inf')
    if pden['start'] and pden['interior']:
        obs=aphase['start']/pden['start']-aphase['interior']/pden['interior']
        arr=np.array(sph['start'])/pden['start']-np.array(sph['interior'])/pden['interior']
        res['phase']['boundary_contrast']=summarize_scalar(obs,arr)
        r1=res['phase']['start']['ratio']; r2=res['phase']['interior']['ratio']; res['phase']['boundary_contrast']['ratio_fold']=max(r1,r2)/min(r1,r2) if min(r1,r2)>0 else float('inf')
    for k in atk:
        res['topk'][str(k)]=summarize_scalar(atk[k],stk[k])
    return res

def shuffled_copy(raw_lines, seed):
    rng=np.random.default_rng(seed); out=[]
    for page,lid,t in raw_lines:
        a=list(t); rng.shuffle(a); out.append((page,lid,a))
    return out

def choose_sites(lines, span, frac, seed):
    rng=np.random.default_rng(seed)
    cand=[]
    for li,(_,_,t) in enumerate(lines):
        for i in range(max(0,len(t)-span+1)): cand.append((li,i))
    rng.shuffle(cand); target=max(1,int(frac*len(cand))); chosen=[]; occ=defaultdict(set)
    for li,i in cand:
        ss=set(range(i,i+span))
        if occ[li].isdisjoint(ss):
            chosen.append((li,i)); occ[li].update(ss)
            if len(chosen)>=target: break
    return chosen

def mutate1(s):
    if not s: return '~'
    repl='~' if s[0] != '~' else '^'
    return repl+s[1:]

def inject(raw, kind, frac, seed):
    out=[(p,l,list(t)) for p,l,t in raw]
    span=5 if kind=='parity' else 3
    sites=choose_sites(out,span,frac,seed)
    for li,i in sites:
        t=out[li][2]; a=t[i]
        if kind=='aba': t[i+2]=a
        elif kind=='bridge': t[i+2]=a; t[i+1]=mutate1(a)
        elif kind=='parity': t[i+2]=a; t[i+4]=a
    return out,len(sites)

def control_suite(path):
    raw=load_frame(path,'ZLZI')
    c0=[]
    for r in range(20):
        pseudo=shuffled_copy(raw, SEED+1000+r)
        rr=analyze_raw(pseudo,200,SEED+2000+r,include_top=False)
        c0.append({m:rr['metrics'][m] for m in ['E2','N1','BRIDGE','CHAIN5','E4']})
    c0_summary={}
    for m in ['E2','N1','BRIDGE','CHAIN5']:
        ratios=[x[m]['ratio'] for x in c0 if math.isfinite(x[m]['ratio'])]
        zs=[x[m]['z'] for x in c0 if math.isfinite(x[m]['z'])]
        c0_summary[m]={'mean_ratio':float(np.mean(ratios)) if ratios else float('nan'),'n_abs_z_ge2':sum(abs(z)>=2 for z in zs),'n_valid':len(ratios)}
        c0_summary[m]['pass']=(len(ratios)==20 and 0.95<=c0_summary[m]['mean_ratio']<=1.05 and c0_summary[m]['n_abs_z_ge2']<=2)
    positives={}
    base=shuffled_copy(raw,SEED+3000)
    for j,kind in enumerate(['aba','bridge','parity']):
        inj,n=inject(base,kind,0.02,SEED+3100+j)
        rr=analyze_raw(inj,500,SEED+3200+j,include_top=False)
        positives[kind]={'n_sites':n,'metrics':rr['metrics']}
    p1=positives['aba']['metrics']['E2']; pass1=(p1['z']>=3 and p1['ratio']>=1.15)
    pb=positives['bridge']['metrics']; pass2=all(pb[m]['z']>=3 and pb[m]['ratio']>=1.15 for m in ['E2','BRIDGE'])
    pp=positives['parity']['metrics']; pass3=all(pp[m]['z']>=3 and pp[m]['ratio']>=1.15 for m in ['E2','E4','CHAIN5'])
    validity={'E2': c0_summary['E2']['pass'] and pass1,'N1': c0_summary['N1']['pass'],'BRIDGE': c0_summary['BRIDGE']['pass'] and pass2,'CHAIN5': c0_summary['CHAIN5']['pass'] and pass3,'E4': pass3,'ISO_ABA': c0_summary['E2']['pass'] and pass1,'PHASE': c0_summary['E2']['pass'] and pass1,'TOPK': c0_summary['E2']['pass'] and pass1}
    return {'c0_replicates':c0,'c0_summary':c0_summary,'positive':positives,'positive_pass':{'C1_ABA':pass1,'C2_BRIDGE':pass2,'C3_PARITY':pass3},'validity':validity}

def frame_worker(args):
    path,frame,nperm,seed=args
    raw=load_frame(path,frame)
    return frame,analyze_raw(raw,nperm,seed,include_top=False)

def decisions(controls, cross, ref):
    frames=list(cross); out={}
    vals=[cross[f]['metrics']['N1'] for f in frames]
    npos=sum(x['ratio']>1 for x in vals); nstrong=sum(x['ratio']>=1.10 and x['z']>=2 for x in vals); med=float(np.median([x['ratio'] for x in vals]))
    h0n=controls['validity']['N1'] and npos>=math.ceil(.8*len(vals)) and nstrong>=math.ceil((2/3)*len(vals)) and med>=1.10
    out['H0_N1']={'verdict':'PASS' if h0n else 'FAIL','npos':npos,'nstrong':nstrong,'nframes':len(vals),'median_ratio':med}
    vals2=[cross[f]['metrics']['E2'] for f in frames]; n2=sum(x['ratio']>=1.10 and x['z']>=2 for x in vals2)
    h0e=controls['validity']['E2'] and n2>=math.ceil(.8*len(vals2))
    out['H0_E2']={'verdict':'PASS' if h0e else 'FAIL','nstrong':n2,'nframes':len(vals2),'median_ratio':float(np.median([x['ratio'] for x in vals2]))}
    b=ref['metrics']['BRIDGE']; bvals=[cross[f]['metrics']['BRIDGE'] for f in frames]
    bpos=sum(x['ratio']>1 for x in bvals); bstrong=sum(x['z']>=2 for x in bvals); bmed=float(np.median([x['ratio'] for x in bvals]))
    support=controls['validity']['BRIDGE'] and b['ratio']>=1.20 and b['z']>=3 and bpos>=math.ceil((2/3)*len(bvals)) and bmed>=1.10
    strong=support and bstrong>=math.ceil((2/3)*len(bvals))
    fals=controls['validity']['BRIDGE'] and b['ratio']<=1.10 and abs(b['z'])<2 and bmed<=1.05
    out['H1_BRIDGE']={'verdict':'STRONG_SUPPORT' if strong else ('SUPPORT' if support else ('FALSIFIED' if fals else 'UNRESOLVED')),'reference':b,'cross_npos':bpos,'cross_nstrong':bstrong,'cross_median':bmed}
    e4=ref['metrics']['E4']; ch=ref['metrics']['CHAIN5']
    sup=controls['validity']['E4'] and controls['validity']['CHAIN5'] and e4['ratio']>=1.15 and e4['z']>=2 and ch['ratio']>=1.25 and ch['z']>=2
    fal=controls['validity']['E4'] and controls['validity']['CHAIN5'] and e4['ratio']<=1.10 and ch['ratio']<=1.10 and (abs(e4['z'])<2 and abs(ch['z'])<2)
    out['H2_PARITY_CHAIN']={'verdict':'SUPPORT' if sup else ('FALSIFIED' if fal else 'UNRESOLVED'),'E4':e4,'CHAIN5':ch}
    iso=ref['metrics']['ISO_ABA']; sup3=h0e and out['H2_PARITY_CHAIN']['verdict']=='FALSIFIED' and controls['validity']['ISO_ABA'] and iso['ratio']>=1.15 and iso['z']>=3
    out['H3_ISOLATED_CLOSURE']={'verdict':'SUPPORT' if sup3 else 'NOT_SUPPORTED','ISO_ABA':iso}
    pc=ref['phase']['parity_contrast']; bc=ref['phase']['boundary_contrast']
    h4=controls['validity']['PHASE'] and ((abs(pc['z'])>=3 and pc['ratio_fold']>=1.25) or (abs(bc['z'])>=3 and bc['ratio_fold']>=1.25))
    out['H4_LINE_PHASE']={'verdict':'SUPPORT' if h4 else 'UNSUPPORTED','parity':pc,'boundary':bc,'phase_cells':{k:ref['phase'][k] for k in ['even','odd','start','interior']}}
    k20=ref['topk']['20']; k50=ref['topk']['50']
    sup5=controls['validity']['TOPK'] and (k20['ratio']<1.05 or abs(k20['z'])<2)
    fal5=controls['validity']['TOPK'] and k50['ratio']>=1.15 and k50['z']>=2
    out['H5_LEXICAL_CARRIER']={'verdict':'SUPPORT' if sup5 else ('FALSIFIED' if fal5 else 'UNRESOLVED'),'K5':ref['topk']['5'],'K20':k20,'K50':k50}
    return out

def fmt(x):
    if x is None or (isinstance(x,float) and not math.isfinite(x)): return 'NA'
    return f'{x:.3f}' if isinstance(x,float) else str(x)

def write_md(out,path):
    c=out['controls']; d=out['decisions']; ref=out['reference_ZLZI']; cross=out['cross_frames']; L=[]
    L += ['# Joint lag-signature programme v0.1 — results','',f"Seed: {SEED}. Tight null: **within-line multiset-preserving permutation**. Reference ZLZI n={ref['n_tokens']:,} tokens, {ref['n_lines']:,} lines; 2,000 permutations.",'']
    L += ['## Controls','', '| control/stat | result |','|---|---|']
    for m,s in c['c0_summary'].items(): L.append(f"| C0 {m} | mean ratio {fmt(s['mean_ratio'])}; |z|>=2 in {s['n_abs_z_ge2']}/20; {'PASS' if s['pass'] else 'FAIL'} |")
    for k,v in c['positive_pass'].items(): L.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    L += ['',f"Valid downstream statistics: `{json.dumps(c['validity'],sort_keys=True)}`",'']
    L += ['## Primary decisions','', '| hypothesis | verdict |','|---|---|']
    for k,v in d.items(): L.append(f"| {k} | **{v['verdict']}** |")
    L += ['','## Reference-frame metrics','', '| metric | actual | null | ratio | z |','|---|---:|---:|---:|---:|']
    for m in METRICS:
        x=ref['metrics'][m]; L.append(f"| {m} | {fmt(x['actual'])} | {fmt(x['null_mean'])} | {fmt(x['ratio'])} | {fmt(x['z'])} |")
    L += ['','## Cross-frame identical-statistic replication','', '| frame | tokens | E2 ratio (z) | N1 ratio (z) | BRIDGE ratio (z) |','|---|---:|---:|---:|---:|']
    for f in UNIQUE_ELIGIBLE:
        r=cross[f]; a=r['metrics']['E2']; b=r['metrics']['N1']; q=r['metrics']['BRIDGE']; L.append(f"| {f} | {r['n_tokens']:,} | {fmt(a['ratio'])} ({fmt(a['z'])}) | {fmt(b['ratio'])} ({fmt(b['z'])}) | {fmt(q['ratio'])} ({fmt(q['z'])}) |")
    L += ['','## H2/H3: persistence versus isolated closure','']
    for m in ['E2','E4','CHAIN5','ISO_ABA']:
        x=ref['metrics'][m]; L.append(f"- {m}: ratio {fmt(x['ratio'])}, z {fmt(x['z'])}.")
    L += ['','## H4: line phase / boundary','']
    for k in ['even','odd','start','interior']:
        x=ref['phase'][k]; L.append(f"- {k}: enrichment ratio {fmt(x['ratio'])}, z {fmt(x['z'])}.")
    pc=ref['phase']['parity_contrast']; bc=ref['phase']['boundary_contrast']; L.append(f"- parity contrast: z {fmt(pc['z'])}, enrichment fold-contrast {fmt(pc['ratio_fold'])}."); L.append(f"- boundary contrast: z {fmt(bc['z'])}, enrichment fold-contrast {fmt(bc['ratio_fold'])}.")
    L += ['','## H5: high-frequency lexical carriers','']
    for k in ['5','20','50']:
        x=ref['topk'][k]; L.append(f"- mask top {k} repeated endpoint types: E2 event ratio {fmt(x['ratio'])}, z {fmt(x['z'])}.")
    L += ['','## Interpretation discipline','', '- A PASS is only a pass of the preregistered statistical implication. It does not identify plaintext, cipher, hoax, or scribal intent.', '- The July G\' sufficiency result concerned the older loose/page-null profile and is not treated as explaining a residual that survives this tighter within-line null.', '- Cross-frame decisions use 12 unique transcription contents; four byte-identical aliases are excluded from replication counts by preregistered amendment.','']
    open(path,'w',encoding='utf-8').write('\n'.join(L))

def main():
    path=sys.argv[1] if len(sys.argv)>1 else DATA_DEFAULT
    outdir=sys.argv[2] if len(sys.argv)>2 else '/mnt/data/joint_lag/out'
    os.makedirs(outdir,exist_ok=True)
    print('STAGE controls',flush=True)
    controls=control_suite(path)
    json.dump(controls,open(os.path.join(outdir,'controls.json'),'w'),indent=2)
    print('controls validity',controls['validity'],flush=True)
    print('STAGE cross-frame',flush=True)
    cross={}; args=[]
    for i,f in enumerate(UNIQUE_ELIGIBLE): args.append((path,f,500,SEED+10000+i*1009))
    with ProcessPoolExecutor(max_workers=min(4,os.cpu_count() or 2)) as ex:
        futs={ex.submit(frame_worker,a):a[1] for a in args}
        for fut in as_completed(futs):
            f,r=fut.result(); cross[f]=r; print(' frame',f,'done',flush=True)
    print('STAGE reference ZLZI 2000',flush=True)
    ref=analyze_raw(load_frame(path,'ZLZI'),2000,SEED+50000,include_top=True)
    dec=decisions(controls,cross,ref)
    out={'programme':'joint_lag_signature_v0.1','seed':SEED,'unique_frames':UNIQUE_ELIGIBLE,'aliases':ALIASES,'controls':controls,'cross_frames':cross,'reference_ZLZI':ref,'decisions':dec}
    jpath=os.path.join(outdir,'RESULTS_joint_lag_signature_v0_1_20260813.json')
    json.dump(out,open(jpath,'w'),indent=2)
    mpath=os.path.join(outdir,'RESULTS_joint_lag_signature_v0_1_20260813.md'); write_md(out,mpath)
    print('DECISIONS')
    for k,v in dec.items(): print(k,v['verdict'])
    print('WROTE',jpath,mpath)
if __name__=='__main__': main()

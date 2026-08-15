#!/usr/bin/env python3
import collections, functools, hashlib, json, math, os, random, re, sys
from pathlib import Path
import numpy as np

SEED=20260815
NPERM=1000
NREP=30
NFOLD=5
STA_WORD=re.compile(r'(?:[A-Z][0-9a-z])+\Z')
STA_MEMBER=re.compile(r'[A-Z][0-9a-z]')
LOCUS=re.compile(r'^<([^>.]+)\.([^,>]+),([^>]+)>\s*(.*)$')
LBUCKETS=((2,5,'2-5'),(6,9,'6-9'),(10,14,'10-14'),(15,10**9,'15+'))
CATS_LEN=('short','mid','long')

@functools.lru_cache(maxsize=500000)
def is_ed1(a,b):
    if a==b:return False
    la,lb=len(a),len(b)
    if abs(la-lb)>1:return False
    if la==lb:return sum(x!=y for x,y in zip(a,b))==1
    if la>lb:a,b=b,a;la,lb=lb,la
    i=j=d=0
    while i<la and j<lb:
        if a[i]==b[j]:i+=1;j+=1
        else:
            d+=1;j+=1
            if d>1:return False
    return True

def ed_kind(a,b):
    if not is_ed1(a,b):return None
    if len(b)>len(a):return 'acc'
    if len(b)<len(a):return 'red'
    # equal-length substitution
    i=next(k for k,(x,y) in enumerate(zip(a,b)) if x!=y)
    return 'sub_first' if i < len(a)/2 else 'sub_second'

def lenclass(a,b):
    m=(len(a)+len(b))/2
    return 'short' if m<=4 else ('mid' if m<=6 else 'long')

def lbucket(n):
    for lo,hi,s in LBUCKETS:
        if lo<=n<=hi:return s
    return '15+'

def coarse_pos(i,n):
    if n<5:return 'L2' if i<min(2,n) else 'R2'
    if i<2:return 'L2'
    if i>=n-2:return 'R2'
    return 'M'

def edge_pos(i,n):
    dl=i; dr=n-1-i
    if dl<=3 and (dl<=dr):return f'L{dl}'
    if dr<=3:return f'R{dr}'
    return 'M'

def groups_for(n,null):
    if null=='N0': return [np.arange(n,dtype=np.int32)] if n else []
    if null=='N1':
        if n<5:
            a=np.arange(min(2,n),dtype=np.int32); b=np.arange(min(2,n),n,dtype=np.int32)
            return [g for g in (a,b) if len(g)]
        return [np.arange(0,2,dtype=np.int32),np.arange(2,n-2,dtype=np.int32),np.arange(n-2,n,dtype=np.int32)]
    raise ValueError(null)

def mode(vals):
    c=collections.Counter(vals)
    return c.most_common(1)[0][0] if c else 'UNKNOWN'

def load_sections(path='enriched_records.json'):
    obj=json.load(open(path,encoding='utf-8')); by=collections.defaultdict(list)
    for r in obj['records']:
        if r.get('section'):by[r['folio']].append(r['section'])
    return {f:mode(v) for f,v in by.items()}

def sta_family(word):
    if not STA_WORD.fullmatch(word):return None
    mem=STA_MEMBER.findall(word)
    if ''.join(mem)!=word:return None
    return ''.join(x[0] for x in mem)

def parse_rf(path,sections):
    raw=Path(path).read_bytes(); text=raw.decode('utf-8')
    lines=[]; invalid=0; eligible_words=0; p0_loci=0
    for ln in text.splitlines():
        m=LOCUS.match(ln)
        if not m:continue
        folio,lid,u,txt=m.groups()
        if not u.endswith('P0'):continue
        p0_loci+=1
        # drawing interruptions are hard breaks. Within each piece, dots are certain word boundaries.
        for piece in txt.split('<->'):
            cur=[]
            for w in piece.strip().split('.'):
                w=w.strip()
                if not w:continue
                fam=sta_family(w)
                if fam is None:
                    invalid+=1
                    if len(cur)>=2:lines.append({'folio':folio,'section':sections.get(folio,'UNKNOWN'),'tokens':tuple(cur)})
                    cur=[]
                else:
                    cur.append(fam); eligible_words+=1
            if len(cur)>=2:lines.append({'folio':folio,'section':sections.get(folio,'UNKNOWN'),'tokens':tuple(cur)})
    return raw,lines,{'p0_loci':p0_loci,'eligible_words':eligible_words,'invalid_words_or_breaks':invalid}

def relation_prob(tokens,idx1,idx2,rel,groups):
    # exact expectation after independently permuting within fixed groups
    gmap={int(p):gi for gi,g in enumerate(groups) for p in g}
    g1=groups[gmap[idx1]]; g2=groups[gmap[idx2]]
    if gmap[idx1]==gmap[idx2]:
        m=len(g1)
        if m<2:return 0.0
        num=0
        for a in g1:
            for b in g1:
                if a!=b and rel(tokens[int(a)],tokens[int(b)]):num+=1
        return num/(m*(m-1))
    num=0
    for a in g1:
        for b in g2:
            if rel(tokens[int(a)],tokens[int(b)]):num+=1
    return num/(len(g1)*len(g2))

def observed_counts(lines):
    out=collections.Counter()
    for x in lines:
        t=x['tokens']; n=len(t)
        for i in range(n-1):
            a,b=t[i],t[i+1]
            if is_ed1(a,b):
                out['ED1_whole']+=1; out['LEN_'+lenclass(a,b)]+=1
                k=ed_kind(a,b); out['DIR_'+k]+=1
            if n>=6 and 2<=i<=n-4 and is_ed1(a,b): out['ED1_N3']+=1
        for i in range(n-2):
            if t[i]==t[i+2]:
                out['E2_whole']+=1
                if n>=7 and 2<=i<=n-5:out['E2_N3']+=1
    return out

def exact_null_means(lines,null):
    out=collections.Counter()
    for x in lines:
        t=x['tokens']; n=len(t); groups=groups_for(n,null)
        for i in range(n-1):
            out['ED1_whole']+=relation_prob(t,i,i+1,is_ed1,groups)
            for c in CATS_LEN:
                out['LEN_'+c]+=relation_prob(t,i,i+1,lambda a,b,c=c:is_ed1(a,b) and lenclass(a,b)==c,groups)
            out['DIR_acc']+=relation_prob(t,i,i+1,lambda a,b:is_ed1(a,b) and len(b)>len(a),groups)
            out['DIR_red']+=relation_prob(t,i,i+1,lambda a,b:is_ed1(a,b) and len(b)<len(a),groups)
            out['DIR_sub_first']+=relation_prob(t,i,i+1,lambda a,b:ed_kind(a,b)=='sub_first',groups)
            out['DIR_sub_second']+=relation_prob(t,i,i+1,lambda a,b:ed_kind(a,b)=='sub_second',groups)
            if n>=6 and 2<=i<=n-4:
                out['ED1_N3']+=relation_prob(t,i,i+1,is_ed1,groups)
        for i in range(n-2):
            eq=lambda a,b:a==b
            out['E2_whole']+=relation_prob(t,i,i+2,eq,groups)
            if n>=7 and 2<=i<=n-5:out['E2_N3']+=relation_prob(t,i,i+2,eq,groups)
    return out

def ratios_exact(lines):
    obs=observed_counts(lines); n0=exact_null_means(lines,'N0'); n1=exact_null_means(lines,'N1')
    def rr(k,mu):return obs[k]/mu[k] if mu[k]>0 else None
    out={
      'E2_N0':rr('E2_whole',n0),'E2_N1':rr('E2_whole',n1),'E2_N3':rr('E2_N3',n1),
      'ED1_N0':rr('ED1_whole',n0),'ED1_N1':rr('ED1_whole',n1),'ED1_N3':rr('ED1_N3',n1),
    }
    for c in CATS_LEN:out['LEN_'+c]=rr('LEN_'+c,n0)
    ar_obs=obs['DIR_acc']/obs['DIR_red'] if obs['DIR_red'] else None
    ar_null=n0['DIR_acc']/n0['DIR_red'] if n0['DIR_red'] else None
    ss_obs=obs['DIR_sub_first']/obs['DIR_sub_second'] if obs['DIR_sub_second'] else None
    ss_null=n0['DIR_sub_first']/n0['DIR_sub_second'] if n0['DIR_sub_second'] else None
    out['direction']={
      'acc_red_obs':ar_obs,'acc_red_null':ar_null,
      'acc_red_abs_logdev':abs(math.log(ar_obs/ar_null)) if ar_obs and ar_null else None,
      'subsite_obs':ss_obs,'subsite_null':ss_null,
      'subsite_abs_logdev':abs(math.log(ss_obs/ss_null)) if ss_obs and ss_null else None}
    out['counts']={k:float(v) for k,v in obs.items()}
    out['null_N0']={k:float(v) for k,v in n0.items()}
    out['null_N1']={k:float(v) for k,v in n1.items()}
    return out

def prep_perm_line(tokens):
    n=len(tokens); ids=np.arange(n,dtype=np.int32)
    ed=np.zeros((n,n),dtype=np.int8); lc=np.full((n,n),-1,dtype=np.int8); dk=np.full((n,n),-1,dtype=np.int8)
    cmap={'short':0,'mid':1,'long':2}; dmap={'acc':0,'red':1,'sub_first':2,'sub_second':3}
    for i,a in enumerate(tokens):
        for j,b in enumerate(tokens):
            if i!=j and is_ed1(a,b):
                ed[i,j]=1; lc[i,j]=cmap[lenclass(a,b)]; dk[i,j]=dmap[ed_kind(a,b)]
    return {'tokens':tokens,'n':n,'ed':ed,'lc':lc,'dk':dk}

def permutation_distributions(lines,null,nperm,seed,detail_n3=False):
    rng=np.random.default_rng(seed)
    keys=['E2','ED1','LEN_short','LEN_mid','LEN_long','DIR_acc','DIR_red','DIR_sub_first','DIR_sub_second']
    if detail_n3:keys+=['E2_N3','ED1_N3']
    arr={k:np.zeros(nperm,dtype=np.float64) for k in keys}
    for x in lines:
        p=prep_perm_line(x['tokens']); n=p['n']
        if n<2:continue
        perm=np.broadcast_to(np.arange(n,dtype=np.int32),(nperm,n)).copy()
        for g in groups_for(n,null):
            if len(g)>1:
                order=np.argsort(rng.random((nperm,len(g))),axis=1); vals=perm[:,g].copy(); perm[:,g]=np.take_along_axis(vals,order,axis=1)
        tok=np.array(p['tokens'],dtype=object)[perm]
        if n>=3:
            eq=tok[:,:-2]==tok[:,2:]; arr['E2']+=eq.sum(axis=1)
            if detail_n3 and n>=7:
                starts=np.arange(2,n-4); arr['E2_N3']+=(tok[:,starts]==tok[:,starts+2]).sum(axis=1)
        e=p['ed'][perm[:,:-1],perm[:,1:]]; arr['ED1']+=e.sum(axis=1)
        lc=p['lc'][perm[:,:-1],perm[:,1:]]; dk=p['dk'][perm[:,:-1],perm[:,1:]]
        for ci,c in enumerate(CATS_LEN):arr['LEN_'+c]+=(lc==ci).sum(axis=1)
        for di,d in enumerate(('acc','red','sub_first','sub_second')):arr['DIR_'+d]+=(dk==di).sum(axis=1)
        if detail_n3 and n>=6:
            starts=np.arange(2,n-3); arr['ED1_N3']+=p['ed'][perm[:,starts],perm[:,starts+1]].sum(axis=1)
    return arr

def stat(obs,sim):
    a=np.asarray(sim,dtype=float); mu=float(a.mean()); sd=float(a.std(ddof=1)); ratio=float(obs/mu) if mu>0 else None; z=float((obs-mu)/sd) if sd>0 else None
    return {'observed':float(obs),'null_mean':mu,'null_sd':sd,'ratio':ratio,'z':z}

def real_stats(lines):
    obs=observed_counts(lines)
    n0=permutation_distributions(lines,'N0',NPERM,SEED+1000,False)
    n1=permutation_distributions(lines,'N1',NPERM,SEED+2000,True)
    out={
      'E2_N0':stat(obs['E2_whole'],n0['E2']), 'E2_N1':stat(obs['E2_whole'],n1['E2']), 'E2_N3':stat(obs['E2_N3'],n1['E2_N3']),
      'ED1_N0':stat(obs['ED1_whole'],n0['ED1']), 'ED1_N1':stat(obs['ED1_whole'],n1['ED1']), 'ED1_N3':stat(obs['ED1_N3'],n1['ED1_N3'])}
    for c in CATS_LEN:out['LEN_'+c]=stat(obs['LEN_'+c],n0['LEN_'+c])
    # correct ratio-distribution directional test
    ar=np.divide(n0['DIR_acc'],n0['DIR_red'],out=np.full(NPERM,np.nan),where=n0['DIR_red']>0)
    ss=np.divide(n0['DIR_sub_first'],n0['DIR_sub_second'],out=np.full(NPERM,np.nan),where=n0['DIR_sub_second']>0)
    aro=obs['DIR_acc']/obs['DIR_red'] if obs['DIR_red'] else float('nan'); sso=obs['DIR_sub_first']/obs['DIR_sub_second'] if obs['DIR_sub_second'] else float('nan')
    def rs(o,a):
        a=a[np.isfinite(a)]; mu=float(a.mean()); sd=float(a.std(ddof=1)); return {'observed_ratio':float(o),'null_ratio_mean':mu,'null_ratio_sd':sd,'z':float((o-mu)/sd) if sd>0 else None}
    out['direction']={'acc_red':rs(aro,ar),'subsite':rs(sso,ss)}
    return out

def fold_of(f):return int(hashlib.sha256(f.encode()).hexdigest()[:8],16)%NFOLD

def build_tables(train,model):
    tables=collections.defaultdict(collections.Counter)
    for x in train:
        t=x['tokens']; n=len(t); sec=x['section']; lb=lbucket(n)
        for i,tok in enumerate(t):
            tables[('GLOBAL',)][tok]+=1; tables[('SEC',sec)][tok]+=1; tables[('LB',sec,lb)][tok]+=1
            if model=='S1':
                p=coarse_pos(i,n); tables[('P',sec,p)][tok]+=1; tables[('FULL',sec,lb,p)][tok]+=1
            elif model=='S2':
                p=edge_pos(i,n); tables[('P',sec,p)][tok]+=1; tables[('FULL',sec,lb,p)][tok]+=1
    return tables

def choose_counter(tables,sec,lb,p,model):
    cand=[]
    if model in ('S1','S2'):cand += [('FULL',sec,lb,p),('P',sec,p)]
    cand += [('LB',sec,lb),('SEC',sec),('GLOBAL',)]
    for k in cand:
        c=tables.get(k)
        if c and sum(c.values())>=30 and len(c)>=5:return c
    return tables[('GLOBAL',)]

def sample_counter(c,rng):
    toks=list(c); weights=list(c.values()); return rng.choices(toks,weights=weights,k=1)[0]

def generate_oof(lines,model,rep):
    byfold={f:[] for f in range(NFOLD)}
    for x in lines:byfold[fold_of(x['folio'])].append(x)
    out=[]
    for f in range(NFOLD):
        train=[x for g in range(NFOLD) if g!=f for x in byfold[g]]; test=byfold[f]
        tables=build_tables(train,model); rng=random.Random(SEED+100000*rep+1000*f+{'S0':0,'S1':100,'S2':200}[model])
        for x in test:
            n=len(x['tokens']); sec=x['section']; lb=lbucket(n); g=[]
            for i in range(n):
                p=None
                if model=='S1':p=coarse_pos(i,n)
                elif model=='S2':p=edge_pos(i,n)
                c=choose_counter(tables,sec,lb,p,model); g.append(sample_counter(c,rng))
            out.append({'folio':x['folio'],'section':sec,'tokens':tuple(g)})
    return out

def med(xs):return float(np.median(np.asarray(xs,dtype=float)))
def qtile(xs,q):return float(np.quantile(np.asarray(xs,dtype=float),q))

def main():
    rf=sys.argv[1] if len(sys.argv)>1 else '/tmp/RF1b.txt'
    sections=load_sections(); raw,lines,pa=parse_rf(rf,sections)
    folios=sorted({x['folio'] for x in lines}); overlap=sum(f in sections for f in folios)/len(folios) if folios else 0
    validation={'header_ok':raw.startswith(b'#=IVTFF STA1 2.0'),'folios':len(folios),'segments':len(lines),'tokens':sum(len(x['tokens']) for x in lines),'section_overlap':overlap}
    validation['pass']=validation['header_ok'] and validation['folios']>=200 and validation['tokens']>=25000 and overlap>=.95
    result={'metadata':{'seed':SEED,'nperm':NPERM,'nrep':NREP,'nfold':NFOLD,'source_sha256':hashlib.sha256(raw).hexdigest(),'parser':pa},'validation':validation}
    if not validation['pass']:
        result['verdict']='INCONCLUSIVE'; result['reason']='SOURCE_OR_ALIGNMENT_VALIDATION_FAILED'
    else:
        real=real_stats(lines); result['real']=real
        # gates
        lg=lambda x:abs(math.log(x)) if x and x>0 else float('inf')
        r1=real['E2_N0']['ratio']>=1.10 and real['E2_N0']['z']>=2 and lg(real['E2_N1']['ratio'])<lg(real['E2_N0']['ratio']) and lg(real['E2_N3']['ratio'])<lg(real['E2_N0']['ratio']) and ((real['E2_N1']['ratio']<1.10 or abs(real['E2_N1']['z'])<2) or (real['E2_N3']['ratio']<1.10 or abs(real['E2_N3']['z'])<2))
        r2=real['ED1_N0']['ratio']>=1.10 and real['ED1_N0']['z']>=2 and lg(real['ED1_N3']['ratio'])<lg(real['ED1_N0']['ratio']) and (real['ED1_N3']['ratio']<1.10 or abs(real['ED1_N3']['z'])<2)
        long=real['LEN_long']; r3_power=long['observed']>=60; r3=(long['ratio']>=1.15 and long['z']>=2) if r3_power else None
        r4=abs(real['direction']['acc_red']['z'])<2 and abs(real['direction']['subsite']['z'])<2
        result['representation_gates']={'R1_E2':bool(r1),'R2_ED1':bool(r2),'R3_long':('PASS' if r3 else 'FAIL') if r3_power else 'UNDERPOWERED','R4_direction_absent':bool(r4)}
        # exact real targets for fair synthetic scoring
        realx=ratios_exact(lines); result['real_exact']=realx
        models={}
        for model in ('S0','S1','S2'):
            reps=[]
            for r in range(NREP):
                syn=generate_oof(lines,model,r); reps.append(ratios_exact(syn))
            keys=('E2_N0','E2_N1','E2_N3','ED1_N0','ED1_N1','ED1_N3','LEN_short','LEN_mid','LEN_long')
            agg={'median':{k:med([q[k] for q in reps if q[k] is not None]) for k in keys},'p10_p90':{k:{'p10':qtile([q[k] for q in reps if q[k] is not None],.1),'p90':qtile([q[k] for q in reps if q[k] is not None],.9)} for k in keys}}
            md_ar=med([q['direction']['acc_red_abs_logdev'] for q in reps if q['direction']['acc_red_abs_logdev'] is not None]); md_ss=med([q['direction']['subsite_abs_logdev'] for q in reps if q['direction']['subsite_abs_logdev'] is not None])
            agg['direction']={'median_acc_red_abs_logdev':md_ar,'median_subsite_abs_logdev':md_ss}
            m=agg['median']; rx=realx
            # F1 exact target matching and attenuation ordering only if R1 observed
            f1=abs(m['E2_N0']-rx['E2_N0'])<=.05 and abs(m['E2_N1']-rx['E2_N1'])<=.05 and abs(m['E2_N3']-rx['E2_N3'])<=.05
            if r1:
                f1=f1 and (lg(m['E2_N1'])<lg(m['E2_N0'])) and (lg(m['E2_N3'])<lg(m['E2_N0']))
            f2=abs(m['ED1_N0']-rx['ED1_N0'])<=.05 and abs(m['ED1_N3']-rx['ED1_N3'])<=.05
            f3=(abs(m['LEN_long']-rx['LEN_long'])<=.07 and m['LEN_long']>=1.15) if r3_power else None
            f4=(md_ar<=.20 and md_ss<=.20 and r4)
            agg['fingerprints']={'F1_E2':bool(f1),'F2_ED1':bool(f2),'F3_long':bool(f3) if f3 is not None else 'NOT_SCORED','F4_direction':bool(f4)}
            models[model]={'aggregate':agg,'replicates':reps}
        result['models']=models
        s2=models['S2']['aggregate']['fingerprints']; scored=[s2['F1_E2'],s2['F2_ED1'],s2['F4_direction']] + ([] if s2['F3_long']=='NOT_SCORED' else [s2['F3_long']])
        if not r1 or not r2:
            verdict='REPRESENTATION_SENSITIVE'
        elif not r4:
            verdict='REPRESENTATION_SENSITIVE'
        elif all(scored):
            verdict='STA_POSITIONAL_MODEL_SUFFICIENT'
        else:
            verdict='STA_RESIDUAL_SURVIVES'
        result['verdict']=verdict
    outdir=Path('results/sta_boundary_robustness_v0_1'); outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'RESULTS_20260815.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    md=['# STA boundary/local-dependence robustness v0.1 — results','',f"Verdict: **{result['verdict']}**.",f"Source SHA-256: `{result['metadata']['source_sha256']}`.",f"Validation: **{'PASS' if result['validation']['pass'] else 'FAIL'}**; {result['validation']['tokens']:,} eligible STA-family tokens in {result['validation']['segments']:,} segments / {result['validation']['folios']} folios; section overlap {result['validation']['section_overlap']:.3f}."]
    if result['validation']['pass']:
        md += ['','## Real STA-family corpus','|metric|N0|N1|N3|','|---|---:|---:|---:|']
        real=result['real']; md.append(f"|exact lag-2|{real['E2_N0']['ratio']:.3f} (z={real['E2_N0']['z']:.2f})|{real['E2_N1']['ratio']:.3f} (z={real['E2_N1']['z']:.2f})|{real['E2_N3']['ratio']:.3f} (z={real['E2_N3']['z']:.2f})|")
        md.append(f"|adjacent ED1|{real['ED1_N0']['ratio']:.3f} (z={real['ED1_N0']['z']:.2f})|{real['ED1_N1']['ratio']:.3f} (z={real['ED1_N1']['z']:.2f})|{real['ED1_N3']['ratio']:.3f} (z={real['ED1_N3']['z']:.2f})|")
        md += ['','### ED1 by mean STA-family token length','|class|observed|ratio|z|','|---|---:|---:|---:|']
        for c in CATS_LEN:
            q=real['LEN_'+c]; md.append(f"|{c}|{q['observed']:.0f}|{q['ratio']:.3f}|{q['z']:.2f}|")
        d=real['direction']; md += ['',f"Direction: accretion/reduction z={d['acc_red']['z']:.2f}; substitution-site z={d['subsite']['z']:.2f}.",f"Representation gates: `{result['representation_gates']}`.",'','## Held-out memoryless models','|model|E2 N0|E2 N1|E2 N3|ED1 N0|ED1 N3|long ED1|fingerprints|','|---|---:|---:|---:|---:|---:|---:|---|']
        rx=result['real_exact']; md.append(f"|REAL|{rx['E2_N0']:.3f}|{rx['E2_N1']:.3f}|{rx['E2_N3']:.3f}|{rx['ED1_N0']:.3f}|{rx['ED1_N3']:.3f}|{rx['LEN_long']:.3f}|target|")
        for model in ('S0','S1','S2'):
            a=result['models'][model]['aggregate']; m=a['median']; md.append(f"|{model}|{m['E2_N0']:.3f}|{m['E2_N1']:.3f}|{m['E2_N3']:.3f}|{m['ED1_N0']:.3f}|{m['ED1_N3']:.3f}|{m['LEN_long']:.3f}|`{a['fingerprints']}`|")
        md += ['','S2 is the strongest zero-memory arm: section + line-length bucket + exact edge-coordinate class, with no P70 state and no neighbour/lookback information.','', '## Guardrail','STA-family is a correlated representation of the same manuscript transcription, not an independent corpus. A surviving residual upgrades representation robustness; it does not identify language, semantics, cipher, or a specific copying mechanism.']
    (outdir/'RESULTS_20260815.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))

if __name__=='__main__':main()

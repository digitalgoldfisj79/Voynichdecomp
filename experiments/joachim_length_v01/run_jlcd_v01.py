#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, math, pickle, re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np

SEED=20260829
PRIMARY_REPS=500
CONTROL_REPS=100
THRESHOLDS=tuple(range(2,9))
MULTI=('ckh','cth','cph','cfh','ikh','ith','iph','ifh','ch','sh')

@dataclass
class Occ:
    folio:str; currier:str; section:str; token:str; pos:str; linebin:str
    prev_token:str|None; next_token:str|None


def sha256(p:Path)->str:
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def clean(s:str)->list[str]:
    s=re.sub(r'<!.*?>','',s)
    for x in ('<%>','<$>','<->'): s=s.replace(x,'')
    s=re.sub(r'<[^>]*>','',s); out=[]
    for x in re.split(r'[\s\.,]+',s.strip()):
        if not x or any(c in x for c in "[]{}?@'/:;0123456789"): continue
        x=re.sub('[^a-z]','',x.lower())
        if x: out.append(x)
    return out

def parse(src:Path,smap:dict[str,str])->tuple[list[Occ],dict]:
    out=[]; cur='UNK'; pages=0; lines=0
    for raw in src.read_text(encoding='utf-8',errors='replace').splitlines():
        if not raw.startswith('<'): continue
        h=re.match(r'^<([^>]+)>\s*<!\s*(.*?)>\s*$',raw)
        if h and '.' not in h.group(1):
            m=re.search(r'\$L=([^\s>]+)',h.group(2)); cur=m.group(1) if m else 'UNK'; pages+=1; continue
        m=re.match(r'^<([^>]+)>\s*(.*)$',raw)
        if not m or ',' not in m.group(1) or '.' not in m.group(1): continue
        left,code=m.group(1).rsplit(',',1)
        if 'P' not in code: continue
        f=left.split('.',1)[0]; ts=clean(m.group(2)); n=len(ts)
        if n<2: continue
        lines+=1
        linebin='S' if n<=6 else ('M' if n<=10 else 'L')
        for i,t in enumerate(ts):
            if i==0: pos='I'
            elif i==n-1: pos='F'
            else:
                q=min(2,int(3*i/max(1,n-1))); pos=f'M{q}'
            out.append(Occ(f,cur,smap.get(f,'UNK'),t,pos,linebin,ts[i-1] if i else None,ts[i+1] if i+1<n else None))
    audit={'pages':pages,'lines':lines,'tokens':len(out),'folios':len({x.folio for x in out}),
           'currier_tokens':dict(collections.Counter(x.currier for x in out)),
           'sections':dict(collections.Counter(x.section for x in out))}
    return out,audit

def units(t:str,rep:str)->tuple[str,...]:
    if rep=='char': return tuple(t)
    out=[]; i=0
    while i<len(t):
        q=next((u for u in MULTI if t.startswith(u,i)),None)
        if q: out.append(q); i+=len(q)
        else: out.append(t[i]); i+=1
    return tuple(out)

def context_key(o:Occ,rep:str)->tuple:
    if o.prev_token is None: pl=0; pe='^'
    else:
        pu=units(o.prev_token,rep); pl=min(len(pu),9); pe=pu[-1]
    if o.next_token is None: nl=0; ne='$'
    else:
        nu=units(o.next_token,rep); nl=min(len(nu),9); ne=nu[0]
    return (pe,pl,ne,nl)

def mi_xy(x:list[int],y:list[int])->float:
    n=len(x)
    if n<2: return 0.0
    cx=collections.Counter(x); cy=collections.Counter(y); cxy=collections.Counter(zip(x,y)); v=0.0
    for (a,b),c in cxy.items():
        p=c/n; v += p*math.log2((c*n)/(cx[a]*cy[b]))
    return v

def cmi(groups:list[tuple[np.ndarray,np.ndarray]])->float:
    N=sum(len(x) for x,_ in groups)
    if N==0: return float('nan')
    s=0.0
    for x,y in groups:
        if len(x)<2: continue
        s += len(x)/N * mi_xy(x.tolist(),y.tolist())
    return s

def prepare_groups(occs:list[Occ],rep:str,stripset:frozenset[str],scope:str='full')->tuple[list[tuple[np.ndarray,np.ndarray]],dict]:
    ctxmap={}; nextctx=0; raw=collections.defaultdict(list); core_global=collections.defaultdict(collections.Counter)
    for o in occs:
        if scope!='full' and o.currier!=scope: continue
        u=units(o.token,rep); core=tuple(a for a in u if a not in stripset)
        if not core or len(core)==len(u): continue
        L=len(u); core_global[core][L]+=1
        ck=context_key(o,rep)
        if ck not in ctxmap: ctxmap[ck]=nextctx; nextctx+=1
        z=(core,o.currier,o.section,o.pos,o.linebin)
        raw[z].append((L,ctxmap[ck]))
    eligible_cores={c for c,h in core_global.items() if len(h)>=2 and sum(h.values())>=6}
    groups=[]; rows=0; strata=0
    for z,v in raw.items():
        if z[0] not in eligible_cores or len(v)<4: continue
        xs=[a for a,b in v]
        if len(set(xs))<2: continue
        arrx=np.asarray(xs,dtype=np.int16); arry=np.asarray([b for a,b in v],dtype=np.int32)
        groups.append((arrx,arry)); rows+=len(v); strata+=1
    meta={'eligible_rows':rows,'eligible_strata':strata,'eligible_cores':len(eligible_cores),'contexts':len(ctxmap),'stripset':sorted(stripset),'scope':scope,'representation':rep}
    return groups,meta

def perm_test(groups:list[tuple[np.ndarray,np.ndarray]],seed:int,reps:int=PRIMARY_REPS)->dict:
    obs=cmi(groups); rng=np.random.default_rng(seed); vals=[]
    for _ in range(reps):
        gp=[(rng.permutation(x),y) for x,y in groups]; vals.append(cmi(gp))
    a=np.asarray(vals,float); mu=float(a.mean()); sd=float(a.std(ddof=1)); eff=float(obs-mu); z=eff/sd if sd>0 else float('nan')
    p=float((1+np.sum(np.abs(a-mu)>=abs(eff)))/(len(a)+1))
    return {'observed_cmi_bits':float(obs),'null_mean':mu,'effect_bits':eff,'null_sd':sd,'z':float(z),'empirical_p_2s':p,'reps':reps,'null_min':float(a.min()),'null_max':float(a.max())}

def threshold_scan(groups:list[tuple[np.ndarray,np.ndarray]],seed:int,reps:int=PRIMARY_REPS)->dict:
    obs=[]
    for k in THRESHOLDS:
        gg=[]
        for x,y in groups:
            b=(x>k).astype(np.int8)
            if b.min()==b.max(): continue
            gg.append((b,y))
        obs.append(cmi(gg) if gg else float('nan'))
    rng=np.random.default_rng(seed); null=np.full((reps,len(THRESHOLDS)),np.nan,float)
    for r in range(reps):
        shuffled=[(rng.permutation(x),y) for x,y in groups]
        for j,k in enumerate(THRESHOLDS):
            gg=[]
            for x,y in shuffled:
                b=(x>k).astype(np.int8)
                if b.min()==b.max(): continue
                gg.append((b,y))
            if gg: null[r,j]=cmi(gg)
    rows=[]; zs=[]
    for j,k in enumerate(THRESHOLDS):
        a=null[:,j]; a=a[np.isfinite(a)]
        if not a.size or not math.isfinite(obs[j]):
            rows.append({'k':k,'observed':obs[j],'z':float('nan'),'fwer_p':1.0}); zs.append(float('nan')); continue
        mu=float(a.mean()); sd=float(a.std(ddof=1)); eff=float(obs[j]-mu); z=eff/sd if sd>0 else float('nan')
        rows.append({'k':k,'observed_cmi_bits':float(obs[j]),'null_mean':mu,'effect_bits':eff,'null_sd':sd,'z':float(z)}); zs.append(z)
    Znull=np.zeros_like(null)
    for j,row in enumerate(rows):
        sd=row.get('null_sd',float('nan')); mu=row.get('null_mean',float('nan'))
        if math.isfinite(sd) and sd>0: Znull[:,j]=(null[:,j]-mu)/sd
        else: Znull[:,j]=0
    maxnull=np.nanmax(np.abs(Znull),axis=1)
    for row,z in zip(rows,zs):
        row['fwer_p']=float((1+np.sum(maxnull>=abs(z)))/(reps+1)) if math.isfinite(z) else 1.0
    finite=[r for r in rows if math.isfinite(r.get('z',float('nan')))]
    rank=sorted(finite,key=lambda r:abs(r['z']),reverse=True)
    return {'thresholds':rows,'ranked_k':[r['k'] for r in rank],'reps':reps,'max_abs_z_observed':max((abs(r['z']) for r in finite),default=float('nan'))}

def top_units(occs:list[Occ],n:int=8)->list[str]:
    c=collections.Counter()
    for o in occs: c.update(units(o.token,'eva'))
    return [u for u,_ in c.most_common(n)]

def control_strip_scan(occs:list[Occ],seed:int)->dict:
    tops=top_units(occs,8); sets=[]
    for i in range(len(tops)):
        for j in range(i+1,len(tops)): sets.append(frozenset((tops[i],tops[j])))
    target=frozenset(('e','i'))
    if target not in sets: sets.append(target)
    res=[]
    for q,ss in enumerate(sets):
        gp,meta=prepare_groups(occs,'eva',ss,'full')
        if meta['eligible_rows']<50 or not gp: continue
        d=perm_test(gp,seed+q*17,CONTROL_REPS if ss!=target else PRIMARY_REPS)
        d.update(meta); d['is_ei']=ss==target; res.append(d)
    ranked=sorted(res,key=lambda d:d['effect_bits'],reverse=True)
    for i,d in enumerate(ranked,1): d['effect_rank']=i
    ei=next((d for d in ranked if d['is_ei']),None); med=float(np.median([d['eligible_rows'] for d in ranked if not d['is_ei']])) if len(ranked)>1 else float('nan')
    q25=max(1,math.ceil(len(ranked)*0.25)); passed=bool(ei and ei['effect_rank']<=q25 and (not math.isfinite(med) or ei['eligible_rows']<=2*med))
    return {'top_units':tops,'results':ranked,'ei':ei,'control_median_eligible_rows':med,'top_quartile_cutoff_rank':q25,'specificity_pass':passed,
            'criterion':'e/i must rank in top quartile by bias-corrected CMI effect and eligible rows must be <=2x median non-e/i control eligible rows'}

def lexical_example(occs:list[Occ],rep:str='eva')->dict:
    out={}
    for tok in ('ked','keed'):
        rows=[o for o in occs if o.token==tok]; prev=[]; nxt=[]
        for o in rows:
            if o.prev_token is not None: prev.append(len(units(o.prev_token,rep)))
            if o.next_token is not None: nxt.append(len(units(o.next_token,rep)))
        out[tok]={'n':len(rows),'currier':dict(collections.Counter(o.currier for o in rows)),'section':dict(collections.Counter(o.section for o in rows)),
                  'position':dict(collections.Counter(o.pos for o in rows)),'mean_prev_length':float(np.mean(prev)) if prev else None,'mean_next_length':float(np.mean(nxt)) if nxt else None}
    return out

def getthr(scan:dict,k:int)->dict:
    return next(r for r in scan['thresholds'] if r['k']==k)

def fmt_metric(label:str,d:dict)->str:
    z=d['z']; prefix='the metric does not resolve this — ' if not math.isfinite(z) or abs(z)<2 else ''
    return f"{prefix}{label}: effect={d['effect_bits']:.6f} bits/occurrence; matched-null SD={d['null_sd']:.6f}; z={z:.2f}; observed CMI={d['observed_cmi_bits']:.6f}."

def fmt_thr(label:str,d:dict)->str:
    z=d['z']; prefix='the metric does not resolve this — ' if not math.isfinite(z) or abs(z)<2 else ''
    return f"{prefix}{label}: effect={d['effect_bits']:.6f} bits/occurrence; matched-null SD={d['null_sd']:.6f}; z={z:.2f}; FWER p={d['fwer_p']:.4f}."

def gate_t1(R:dict)->bool:
    vals=[]
    for rep in ('eva','char'):
        d=R['primary'][rep]; full=d['full']['t1']; A=d['A']['t1']; B=d['B']['t1']
        same=np.sign(full['effect_bits'])==np.sign(A['effect_bits'])==np.sign(B['effect_bits'])
        vals.append(abs(full['z'])>=2 and abs(A['z'])>=2 and abs(B['z'])>=2 and same)
    return all(vals)

def gate_t2(R:dict)->tuple[bool,int|None]:
    for k in (3,4):
        ok=True
        for rep in ('eva','char'):
            d=R['primary'][rep]
            full=getthr(d['full']['threshold'],k); A=getthr(d['A']['threshold'],k); B=getthr(d['B']['threshold'],k)
            if not (abs(full['z'])>=2 and abs(A['z'])>=2 and abs(B['z'])>=2 and np.sign(full['effect_bits'])==np.sign(A['effect_bits'])==np.sign(B['effect_bits'])): ok=False
            if k not in d['full']['threshold']['ranked_k'][:2] or full['fwer_p']>0.05: ok=False
        if ok: return True,k
    return False,None

def report(R:dict)->str:
    L=['# JLCD v0.1 — FINAL RESULTS','','# RETRACTED FINDINGS','','None.','','# CURRENT FINDINGS','',f"**Endpoint: {R['gates']['endpoint']}**",'',
       f"Frozen ZL source SHA-256: `{R['audit']['source_sha256']}`; running-text tokens={R['audit']['tokens']}; folios={R['audit']['folios']}." ,'','## T1 — e/i-stripped core: does total length predict external context?','']
    for rep in ('eva','char'):
        L.append(f'### {rep}')
        for scope in ('full','A','B'):
            d=R['primary'][rep][scope]; L.append(fmt_metric(scope,d['t1'])+f" Eligible={d['meta']['eligible_rows']} occurrences / {d['meta']['eligible_cores']} cores.")
        L.append('')
    L += ['## T2 — claimed short/long regime boundary','']
    for rep in ('eva','char'):
        L.append(f'### {rep}')
        for scope in ('full','A','B'):
            sc=R['primary'][rep][scope]['threshold']; top=', '.join(map(str,sc['ranked_k'][:4])); L.append(f"{scope} strongest k: {top}.")
            for k in (3,4): L.append(fmt_thr(f'{scope}, k={k}',getthr(sc,k)))
        L.append('')
    L += ['## T3 — e/i specificity among frequent-unit strip controls','']
    s=R['specificity']; ei=s.get('ei')
    if ei:
        L.append(fmt_metric('e/i strip pair',ei)+f" Rank={ei['effect_rank']}/{len(s['results'])}; eligible={ei['eligible_rows']}; control median eligible={s['control_median_eligible_rows']:.1f}.")
    L.append(f"Specificity gate: {'PASS' if s['specificity_pass'] else 'FAIL'}. Top units: {', '.join(s['top_units'])}.")
    L += ['','Top control effects:']
    for d in s['results'][:8]: L.append(f"- {'/'.join(d['stripset'])}: effect={d['effect_bits']:.6f}; null SD={d['null_sd']:.6f}; z={d['z']:.2f}; eligible={d['eligible_rows']}; rank={d['effect_rank']}.")
    L += ['','## T4 — supplied ked / keed example','',json.dumps(R['lexical_example'],indent=2),'','## Gates','',
          f"- T1 independent within-core length/context effect: {'PASS' if R['gates']['t1'] else 'FAIL'}.",
          f"- T2 preregistered 3/4 change-point: {'PASS at k='+str(R['gates']['t2_k']) if R['gates']['t2'] else 'FAIL'}.",
          f"- T3 e/i specificity: {'PASS' if R['gates']['t3'] else 'FAIL'}.",
          f"- Final endpoint: {R['gates']['endpoint']}.",'','## Audit and interpretation','',
          'Circularity: Joachim’s supplied claims fixed the e/i counter and k=3/4 targets before execution. The wider k=2..8 scan is corrected by a joint max-|z| null.',
          'Leakage: no plaintext labels or proposed readings enter any statistic. External context excludes the token itself.',
          'Confounds: permutation is restricted to identical stripped core × Currier × section × line-position bucket × line-length bin.',
          'Matched nulls: exact observed total lengths are permuted only inside those complete strata.',
          'Control fairness: e/i is compared with every pair among the eight most frequent EVA units, using bias-corrected CMI effect rather than z alone for ranking.',
          'Measurement degeneracy: the primary tests are repeated under greedy EVA-unit length and raw transcription-character length.',
          'Representation dependence: a promoted gate requires both representations.',
          'Decision-rule fragility: k=3/4 was fixed ex ante; the whole k scan is reported rather than selecting the best threshold after the fact.',
          'Audit completeness: source hashes, JSON results and pickle checkpoints are retained.',
          'Interpretation follows these checks.','']
    ep=R['gates']['endpoint']
    if ep=='JLCD-0': L.append('The proposed length-conditioned counter mechanism fails its first necessary condition: after matching the same e/i-stripped core and positional/sectional covariates, total token length does not reproducibly predict external context. The Joachim mechanism is not supported by this programme.')
    elif ep=='JLCD-1': L.append('A reproducible within-core length/context effect exists, but the proposed e/i-specific 3/4 lookup-table mechanism is not resolved. This is structural evidence only; it does not identify a cipher.')
    else: L.append('The e/i-specific within-core effect and the preregistered 3/4 change-point both survive the registered controls. This supports the proposed mechanistic signature but does not validate any plaintext mapping or distinguish cipher from an explicitly length-conditioned non-cipher generator.')
    L += ['','## Hallucination / scope boundary','',
          'The programme tests only claims operationalisable from the supplied post: total length, e/i counters, a 3/4 regime boundary, and the ked/keed example. The “vowel bridge”, “missing 20 percent”, historical construction sequence and plaintext values are not independently specified well enough here to test and are not treated as established.']
    return '\n'.join(L)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--section-map',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    sm=json.loads(args.section_map.read_text(encoding='utf-8'))['mapping']; occs,audit=parse(args.source,sm); audit['source_sha256']=sha256(args.source); audit['section_map_sha256']=sha256(args.section_map)
    if audit['tokens']<10000 or audit['folios']<100: raise RuntimeError('corpus audit failed '+json.dumps(audit))
    R={'programme':'JLCD_v0.1','seed':SEED,'audit':audit,'primary':{}}
    with (args.out/'phase0_audit.pkl').open('wb') as f: pickle.dump(R,f,pickle.HIGHEST_PROTOCOL)
    target=frozenset(('e','i'))
    for ri,rep in enumerate(('eva','char')):
        R['primary'][rep]={}
        for si,scope in enumerate(('full','A','B')):
            gp,meta=prepare_groups(occs,rep,target,scope); t1=perm_test(gp,SEED+ri*10000+si*1000+1,PRIMARY_REPS); th=threshold_scan(gp,SEED+ri*10000+si*1000+2,PRIMARY_REPS)
            R['primary'][rep][scope]={'meta':meta,'t1':t1,'threshold':th}
        with (args.out/f'phase1_{rep}.pkl').open('wb') as f: pickle.dump(R['primary'][rep],f,pickle.HIGHEST_PROTOCOL)
    R['specificity']=control_strip_scan(occs,SEED+50000)
    with (args.out/'phase2_specificity.pkl').open('wb') as f: pickle.dump(R['specificity'],f,pickle.HIGHEST_PROTOCOL)
    R['lexical_example']=lexical_example(occs,'eva')
    t1=gate_t1(R); t2,t2k=gate_t2(R); t3=bool(R['specificity']['specificity_pass']); endpoint='JLCD-0' if not t1 else ('JLCD-2' if t2 and t3 else 'JLCD-1')
    R['gates']={'t1':bool(t1),'t2':bool(t2),'t2_k':t2k,'t3':t3,'endpoint':endpoint}
    with (args.out/'phase3_gates.pkl').open('wb') as f: pickle.dump(R['gates'],f,pickle.HIGHEST_PROTOCOL)
    (args.out/'results.json').write_text(json.dumps(R,indent=2,default=str),encoding='utf-8'); (args.out/'RESULTS_JLCD_v01.md').write_text(report(R),encoding='utf-8'); print(report(R))

if __name__=='__main__': main()

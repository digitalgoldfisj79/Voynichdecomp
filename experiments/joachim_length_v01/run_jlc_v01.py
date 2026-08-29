#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, math, pickle, re
from dataclasses import dataclass
from pathlib import Path
import numpy as np

SEED=20260829
T1_REPS=1000
T2_REPS=2000
T3_TRAIN_REPS=300
T3_TEST_REPS=2000
MULTI=('ckh','cth','cph','cfh','ikh','ith','iph','ifh','ch','sh')

@dataclass(frozen=True)
class Occ:
    folio:str; currier:str; section:str; posbin:str; token:str; length:int; skeleton:str; context:str

def clean_tokens(s:str)->list[str]:
    s=re.sub(r'<!.*?>','',s)
    for x in ('<%>','<$>','<->'): s=s.replace(x,'')
    s=re.sub(r'<[^>]*>','',s)
    out=[]
    for x in re.split(r'[\s\.,]+',s.strip()):
        if not x or any(c in x for c in "[]{}?@'/:;0123456789"): continue
        x=re.sub('[^a-z]','',x.lower())
        if x: out.append(x)
    return out

def units(t:str,rep:str)->list[str]:
    if rep=='char': return list(t)
    out=[]; i=0
    while i<len(t):
        hit=next((u for u in MULTI if t.startswith(u,i)),None)
        if hit: out.append(hit); i+=len(hit)
        else: out.append(t[i]); i+=1
    return out

def posbin(i:int,n:int)->str:
    if i==0:return 'I'
    if i==n-1:return 'F'
    q=min(3,int(4*i/max(1,n-1)))
    return f'M{q}'

def lenbin(n:int)->str: return str(min(n,10))

def parse_lines(src:Path,smap:dict)->list[tuple[str,str,str,list[str]]]:
    out=[]; cur='UNK'
    for raw in src.read_text(encoding='utf-8',errors='replace').splitlines():
        if not raw.startswith('<'): continue
        h=re.match(r'^<([^>]+)>\s*<!\s*(.*?)>\s*$',raw)
        if h and '.' not in h.group(1):
            m=re.search(r'\$L=([^\s>]+)',h.group(2)); cur=m.group(1) if m else 'UNK'; continue
        m=re.match(r'^<([^>]+)>\s*(.*)$',raw)
        if not m or ',' not in m.group(1) or '.' not in m.group(1): continue
        left,code=m.group(1).rsplit(',',1)
        if 'P' not in code: continue
        folio=left.split('.',1)[0]; toks=clean_tokens(m.group(2))
        if len(toks)>=2: out.append((folio,cur,smap.get(folio,'UNK'),toks))
    return out

def build_occ(lines,rep:str)->list[Occ]:
    out=[]
    for folio,cur,sec,toks in lines:
        U=[units(t,rep) for t in toks]; n=len(toks)
        for i,(tok,u) in enumerate(zip(toks,U)):
            prev='BOS' if i==0 else f"{U[i-1][-1]}:{lenbin(len(U[i-1]))}"
            nxt='EOS' if i==n-1 else f"{U[i+1][0]}:{lenbin(len(U[i+1]))}"
            context=prev+'|'+nxt
            sk=''.join(x for x in u if x not in ('e','i')) or '∅'
            out.append(Occ(folio,cur,sec,posbin(i,n),tok,len(u),sk,context))
    return out

def sha256(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def is_test(folio:str)->bool: return hashlib.sha256(('JLC:'+folio).encode()).digest()[0] >=128

def cmi_from_records(records:list[tuple[str,int,str]])->float:
    # records = (stratum, y, x); exact empirical I(Y;X|S)
    if not records:return float('nan')
    n=len(records); ns=collections.Counter(); sy=collections.Counter(); sx=collections.Counter(); syx=collections.Counter()
    for s,y,x in records:
        ns[s]+=1; sy[(s,y)]+=1; sx[(s,x)]+=1; syx[(s,y,x)]+=1
    total=0.0
    for (s,y,x),c in syx.items():
        total += c * math.log2((c*ns[s])/(sy[(s,y)]*sx[(s,x)]))
    return total/n

def usable_strata(records:list[tuple[str,int,str]],min_n=4)->list[tuple[str,int,str]]:
    ys=collections.defaultdict(set); nn=collections.Counter()
    for s,y,x in records: ys[s].add(y); nn[s]+=1
    good={s for s in nn if nn[s]>=min_n and len(ys[s])>=2}
    return [r for r in records if r[0] in good]

def permute_y_within_strata(records,rng):
    rec=[list(r) for r in records]; G=collections.defaultdict(list)
    for i,(s,y,x) in enumerate(records):G[s].append(i)
    for ix in G.values():
        if len(ix)>1:
            vals=np.array([records[i][1] for i in ix],dtype=int); vals=rng.permutation(vals)
            for j,v in zip(ix,vals):rec[j][1]=int(v)
    return [(s,y,x) for s,y,x in rec]

def null_summary(obs,vals):
    a=np.asarray(vals,float); mu=float(a.mean()); sd=float(a.std(ddof=1)); eff=float(obs-mu); z=eff/sd if sd else float('nan')
    p=(1+int(np.sum(np.abs(a-mu)>=abs(eff))))/(len(a)+1)
    return {'observed':float(obs),'null_mean':mu,'effect':eff,'null_sd':sd,'z':float(z),'p_empirical_2s':float(p),'reps':len(vals),'null_min':float(a.min()),'null_max':float(a.max())}

def t1_core(occ:list[Occ],seed:int,reps=T1_REPS):
    bysk=collections.defaultdict(list)
    for o in occ:bysk[o.skeleton].append(o)
    eligible={s for s,v in bysk.items() if len(v)>=12 and len({o.length for o in v})>=2}
    rec=[]
    for o in occ:
        if o.skeleton in eligible:
            s='§'.join((o.skeleton,o.currier,o.section,o.posbin)); rec.append((s,o.length,o.context))
    rec=usable_strata(rec,4); obs=cmi_from_records(rec);rng=np.random.default_rng(seed);null=[]
    for _ in range(reps):null.append(cmi_from_records(permute_y_within_strata(rec,rng)))
    d=null_summary(obs,null);d.update({'eligible_skeletons':len(eligible),'usable_occurrences':len(rec),'conditioning':'skeleton × Currier × section × position_bin','y':'total token length','x':'external neighbour edge+length context'});return d

def enumerate_pairs(occ:list[Occ],rep:str,min_each=6):
    freq=collections.Counter(o.token for o in occ); types=set(freq); pairs={}
    for long in sorted(types):
        u=units(long,rep)
        if len(u)<2:continue
        for j,ins in enumerate(u):
            short=''.join(u[:j]+u[j+1:])
            if short not in types or freq[short]<min_each or freq[long]<min_each:continue
            key=(short,long)
            prev=pairs.get(key)
            if prev is None:pairs[key]=ins
            elif prev!=ins:pairs[key]='AMB'
    out=[]
    for (s,l),ins in pairs.items():
        if ins=='AMB':continue
        out.append({'short':s,'long':l,'inserted':ins,'short_len':len(units(s,rep)),'freq':freq[s]+freq[l],'minfreq':min(freq[s],freq[l])})
    return out,freq

def select_matched_pairs(occ,rep):
    pairs,freq=enumerate_pairs(occ,rep,6)
    ei=[p for p in pairs if p['inserted'] in ('e','i')]
    ot=[p for p in pairs if p['inserted'] not in ('e','i')]
    ei=sorted(ei,key=lambda p:(-p['minfreq'],-p['freq'],p['short'],p['long']))
    available=set(range(len(ot)));used=set();blocks=[]
    for p in ei:
        if p['short'] in used or p['long'] in used:continue
        cand=[]
        for j in available:
            q=ot[j]
            if q['short_len']!=p['short_len']:continue
            if q['short'] in used or q['long'] in used:continue
            score=abs(math.log((p['freq']+1)/(q['freq']+1)))+0.25*abs(math.log((p['minfreq']+1)/(q['minfreq']+1)))
            cand.append((score,-q['minfreq'],j))
        if not cand:continue
        _,_,j=min(cand);q=ot[j];available.remove(j);used.update((p['short'],p['long'],q['short'],q['long']));blocks.append((p,q))
    return blocks

def pair_records(occ,pairs):
    role={}
    for pid,p in enumerate(pairs):role[p['short']]=(pid,0);role[p['long']]=(pid,1)
    rec=[]
    for o in occ:
        r=role.get(o.token)
        if r is None:continue
        pid,y=r;s='§'.join((str(pid),o.currier,o.section,o.posbin));rec.append((s,y,o.context))
    return usable_strata(rec,4)

def t2_specificity(occ,rep,seed,reps=T2_REPS):
    blocks=select_matched_pairs(occ,rep)
    if len(blocks)<4:return {'resolved':False,'reason':'fewer than 4 matched disjoint EI/control blocks','blocks':len(blocks)}
    allpairs=[p for b in blocks for p in b]
    # pair records kept separately so block assignments can swap without changing lexical data
    pair_occ={}
    for idx,p in enumerate(allpairs):pair_occ[idx]=pair_records(occ,[p])
    def group_cmi(assign):
        R0=[];R1=[]
        for bi,(ei,ot) in enumerate(blocks):
            ids=(2*bi,2*bi+1)
            if assign[bi]:ids=ids[::-1]
            for target,pid in enumerate(ids):
                # rewrite stratum with globally unique pair id
                for s,y,x in pair_occ[pid]:
                    ss=f'{pid}§'+s.split('§',1)[1] if '§' in s else f'{pid}§{s}'
                    (R0 if target==0 else R1).append((ss,y,x))
        return cmi_from_records(R0),cmi_from_records(R1),len(R0),len(R1)
    e,o,ne,no=group_cmi([False]*len(blocks));obs=e-o;rng=np.random.default_rng(seed);null=[]
    for _ in range(reps):
        swaps=list(rng.random(len(blocks))<0.5);a,b,_,_=group_cmi(swaps);null.append(a-b)
    d=null_summary(obs,null);d.update({'matched_blocks':len(blocks),'EI_cmi':e,'OTHER_cmi':o,'EI_occurrences':ne,'OTHER_occurrences':no,'null':'swap EI/OTHER identity inside each same-short-length frequency-matched disjoint lexical block'});return d

def select_disjoint_all(occ,rep):
    pairs,freq=enumerate_pairs(occ,rep,5);pairs=sorted(pairs,key=lambda p:(-p['minfreq'],-p['freq'],p['short'],p['long']));used=set();out=[]
    for p in pairs:
        if p['short'] in used or p['long'] in used:continue
        used.update((p['short'],p['long']));out.append(p)
    return out

def cmi_by_short_length(occ,pairs,lengths):
    ans={}
    for L in lengths:
        ps=[p for p in pairs if p['short_len']==L];rec=pair_records(occ,ps);ans[L]={'cmi':cmi_from_records(rec),'n':len(rec),'pairs':len(ps),'records':rec}
    return ans

def corrected_D(occ,pairs,L,seed,reps):
    rec=pair_records(occ,[p for p in pairs if p['short_len']==L])
    if len(rec)<30:return {'effect':float('nan'),'observed':float('nan'),'null_mean':float('nan'),'null_sd':float('nan'),'z':float('nan'),'n':len(rec),'pairs':len([p for p in pairs if p['short_len']==L])}
    obs=cmi_from_records(rec);rng=np.random.default_rng(seed);null=[cmi_from_records(permute_y_within_strata(rec,rng)) for _ in range(reps)];d=null_summary(obs,null);d.update({'n':len(rec),'pairs':len([p for p in pairs if p['short_len']==L])});return d

def choose_k_train(occ,pairs,seed):
    train=[o for o in occ if not is_test(o.folio)];Ds={L:corrected_D(train,pairs,L,seed+L*100,T3_TRAIN_REPS) for L in range(2,7)}
    scores={}
    for k in (3,4):
        vals=[Ds[x]['effect'] for x in (k-1,k,k+1)]
        scores[k]=vals[1]-0.5*(vals[0]+vals[2]) if all(math.isfinite(v) for v in vals) else float('-inf')
    k=max(scores,key=scores.get);return k,Ds,scores

def curvature_test(occ,pairs,k,seed,reps=T3_TEST_REPS):
    groups={L:pair_records(occ,[p for p in pairs if p['short_len']==L]) for L in (k-1,k,k+1)}
    if any(len(groups[L])<30 for L in groups):return {'resolved':False,'reason':'insufficient occurrences','n_by_L':{L:len(v) for L,v in groups.items()}}
    obsD={L:cmi_from_records(groups[L]) for L in groups};obs=obsD[k]-0.5*(obsD[k-1]+obsD[k+1]);rng=np.random.default_rng(seed);null=[]
    for _ in range(reps):
        d={L:cmi_from_records(permute_y_within_strata(groups[L],rng)) for L in groups};null.append(d[k]-0.5*(d[k-1]+d[k+1]))
    r=null_summary(obs,null);r.update({'k':k,'raw_cmi_by_L':obsD,'n_by_L':{L:len(v) for L,v in groups.items()},'null':'short/long labels permuted within pair × Currier × section × position strata independently in each length group'});return r

def run_scope(lines,smap_dummy,rep,scope,selected_k=None):
    occ=build_occ(lines,rep)
    if scope in ('A','B'):occ=[o for o in occ if o.currier==scope]
    t1=t1_core(occ,SEED+(0 if rep=='eva' else 10000)+(0 if scope=='full' else (100 if scope=='A' else 200)))
    t2=t2_specificity(occ,rep,SEED+30000+(0 if rep=='eva' else 10000)+(0 if scope=='full' else (100 if scope=='A' else 200)))
    pairs=select_disjoint_all(occ,rep)
    if selected_k is None:k,trainD,scores=choose_k_train(occ,pairs,SEED+50000);selection={'selected_k':k,'train_D':trainD,'train_curvature':scores}
    else:k=selected_k;selection={'selected_k':k,'frozen_from':'full EVA training split'}
    testocc=[o for o in occ if is_test(o.folio)];t3=curvature_test(testocc,pairs,k,SEED+70000+(0 if rep=='eva' else 10000)+(0 if scope=='full' else (100 if scope=='A' else 200)))
    alt=3 if k==4 else 4;alt3=curvature_test(testocc,pairs,alt,SEED+80000+(0 if rep=='eva' else 10000)+(0 if scope=='full' else (100 if scope=='A' else 200)))
    return {'n_occ':len(occ),'T1_core_length_context':t1,'T2_EI_specificity':t2,'T3_threshold':t3,'T3_sensitivity_other_k':alt3,'threshold_selection':selection,'disjoint_pairs':len(pairs)}

def fmt(name,d):
    if not isinstance(d,dict) or 'z' not in d:return f'{name}: unavailable ({d.get("reason","unknown") if isinstance(d,dict) else "unknown"}).'
    z=d['z'];lead='the metric does not resolve this — ' if not math.isfinite(z) or abs(z)<2 else ''
    return f"{lead}{name}: effect={d['effect']:.6f}; matched-null SD={d['null_sd']:.6f}; z={z:.2f}; observed={d['observed']:.6f}; empirical p={d['p_empirical_2s']:.6f}."

def passed(d,positive=True):return isinstance(d,dict) and math.isfinite(d.get('z',float('nan'))) and d['z']>=2 if positive else abs(d.get('z',0))>=2

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--section-map',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    sm=json.loads(a.section_map.read_text())['mapping'];lines=parse_lines(a.source,sm)
    audit={'source_sha256':sha256(a.source),'section_map_sha256':sha256(a.section_map),'lines':len(lines),'tokens':sum(len(x[3]) for x in lines),'folios':len({x[0] for x in lines}),'currier_tokens':dict(collections.Counter(x[1] for x in lines for _ in x[3]))}
    R={'programme':'JLC_v0.1','audit':audit,'seed':SEED,'results':{}};pickle.dump(R,(a.out/'phase0_audit.pkl').open('wb'))
    full_eva=run_scope(lines,sm,'eva','full',None);k=full_eva['threshold_selection']['selected_k'];R['results']['eva']={'full':full_eva};pickle.dump(full_eva,(a.out/'phase1_eva_full.pkl').open('wb'))
    for scope in ('A','B'):
        R['results']['eva'][scope]=run_scope(lines,sm,'eva',scope,k);pickle.dump(R['results']['eva'][scope],(a.out/f'phase1_eva_{scope}.pkl').open('wb'))
    R['results']['char']={}
    for scope in ('full','A','B'):
        R['results']['char'][scope]=run_scope(lines,sm,'char',scope,k);pickle.dump(R['results']['char'][scope],(a.out/f'phase2_char_{scope}.pkl').open('wb'))
    # Decision: each claim must pass positively in full, A, B and both representations.
    claims={}
    for key in ('T1_core_length_context','T2_EI_specificity','T3_threshold'):
        vals=[R['results'][rep][scope][key] for rep in ('eva','char') for scope in ('full','A','B')]
        claims[key]=bool(all(passed(v,True) for v in vals))
    if not claims['T1_core_length_context']:endpoint='JLC-0: core/length contextual coupling not replicated'
    elif not claims['T2_EI_specificity']:endpoint='JLC-1: length/context coupling exists, but e/i are not shown to be special counters'
    elif not claims['T3_threshold']:endpoint='JLC-2: e/i-specific length effect exists, but proposed 3/4-vs-longer regime is unsupported'
    else:endpoint='JLC-3: all structural predictions pass; cipher interpretation still requires generator competition'
    R['decision']={'frozen_k':k,'claim_pass':claims,'endpoint':endpoint,'cipher_claim_promoted':False}
    (a.out/'results.json').write_text(json.dumps(R,indent=2,default=str),encoding='utf-8');pickle.dump(R,(a.out/'phase3_final.pkl').open('wb'))
    L=['# JLC v0.1 — final results','','# RETRACTED FINDINGS','','None at finalization.','','# CURRENT FINDINGS','',f'**Endpoint: {endpoint}**','',f"Frozen threshold selected on full-EVA training folios: k={k}.",'']
    for rep in ('eva','char'):
        L += [f'## {rep} representation','']
        for scope in ('full','A','B'):
            d=R['results'][rep][scope];L += [f'### {scope}',fmt('T1 core-skeleton length ↔ external-context CMI',d['T1_core_length_context']),fmt('T2 EI-vs-other one-unit specificity',d['T2_EI_specificity']),fmt(f'T3 held-out discontinuity at k={k}',d['T3_threshold']),fmt('T3 unpromoted alternative-k sensitivity',d['T3_sensitivity_other_k']),'']
    L += ['## Decision','','- T1 replicated across full/A/B and both representations: '+('PASS' if claims['T1_core_length_context'] else 'FAIL'),'- T2 e/i specificity replicated across full/A/B and both representations: '+('PASS' if claims['T2_EI_specificity'] else 'FAIL'),'- T3 short/long threshold replicated across full/A/B and both representations: '+('PASS' if claims['T3_threshold'] else 'FAIL'),'- Cipher interpretation promoted: NO.','','## Interpretation','']
    if endpoint.startswith('JLC-0'):L.append('The proposed same-core length/context effect itself does not survive the registered replication gate. The length-dependent cipher claim is not supported by this programme.')
    elif endpoint.startswith('JLC-1'):L.append('Total length is associated with external context inside e/i-stripped core families, but e/i insertions do not show the registered excess contextual separation over matched other one-unit edits. The data support ordinary morphology/length coupling, not the specific claim that e/i act as length counters.')
    elif endpoint.startswith('JLC-2'):L.append('An e/i-specific contextual effect survives, but the claimed discrete short/long lookup regime does not. This supports a special structural role for e/i without supporting the proposed length-indexed codebook architecture.')
    else:L.append('All three registered structural predictions survive. This would justify a second-stage frozen-generator contest, but not plaintext assignments or a cipher conclusion by itself.')
    L += ['','## Hallucination / scope boundary','','No plaintext assignment, vowel-bridge interpretation, historical construction sequence, or claimed “missing 20 percent” was used as evidence. `e/i-stripped skeleton` is an operational test representation, not an assertion that the remaining glyphs are literal consonants. The threshold k was selected only from the deterministic training split and evaluated on untouched test folios.']
    (a.out/'RESULTS.md').write_text('\n'.join(L),encoding='utf-8');print('\n'.join(L))

if __name__=='__main__':main()

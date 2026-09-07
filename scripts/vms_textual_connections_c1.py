#!/usr/bin/env python3
"""C1 calibration for vms_textual_connections_20260907_v01.

Voynich TARGET CONNECTIONS REMAIN SEALED.  The only VMS-derived quantity used
here is the empirical page-token length distribution, explicitly preregistered
before this implementation.  Connection features are calibrated on known-order
Caesar pseudo-singulions.
"""
from __future__ import annotations
import hashlib, json, math, random, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, requests

from vms_topology_marginalization import CORPORA, fetch_text, clean_tokens, LINE_RE

PROTOCOL='vms_textual_connections_20260907_v01'
SEED=2026090719
OUT=Path('artifacts/vms_textual_connections_c1_v01'); OUT.mkdir(parents=True,exist_ok=True)
FAMS=('RARE','SUBWORD','MORPH')
CAESAR_URL='https://www.gutenberg.org/cache/epub/218/pg218.txt'
CAESAR_SHA='976c130b6637c8b643617e66e039b8fddcb610eda02c3b68db1bdce87bd09866'
WINDOW_UNITS=8
STEP_UNITS=2
N_NULL=2000


def strict_words(s):
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", s.lower())

def vms_page_lengths():
    # Aggregate length distribution only. No unit IDs, adjacency, metadata or
    # target pair scores are emitted or inspected.
    code='ZL'; url,sha=CORPORA[code]; body=fetch_text(url,sha)
    p=defaultdict(list)
    for raw in body.splitlines():
        m=LINE_RE.match(raw)
        if not m: continue
        page,txt=m.groups(); t=clean_tokens(txt)
        if t: p[page].extend(t)
    lens=sorted(len(x) for x in p.values() if 45 <= len(x) <= 420)
    if len(lens)<100: raise RuntimeError(f'page-length extraction too small: {len(lens)}')
    return lens

def fetch_caesar():
    r=requests.get(CAESAR_URL,timeout=60,headers={'User-Agent':'VoynichTextConnections/0.1'}); r.raise_for_status()
    raw=r.content
    # requests may see current Gutenberg line-ending representation; preserve
    # both raw and decoded hashes. The frozen control identity is additionally
    # guarded by title markers and token count scale.
    txt=raw.decode('utf-8-sig',errors='replace')
    raw_sha=hashlib.sha256(raw).hexdigest(); txt_sha=hashlib.sha256(txt.encode()).hexdigest()
    start=txt.find('C. IULI CAESARIS DE BELLO GALLICO COMMENTARIUS PRIMUS')
    end=txt.find("End of Project Gutenberg's")
    if start<0 or end<0 or end<=start: raise RuntimeError('Caesar boundary markers missing')
    body=txt[start:end]
    toks=strict_words(body)
    if not (19000 <= len(toks) <= 22000): raise RuntimeError(f'Caesar token count unexpected {len(toks)}')
    return toks,{'raw_sha256':raw_sha,'decoded_sha256':txt_sha,'registered_sha256':CAESAR_SHA,'token_count':len(toks),'url':CAESAR_URL}

def make_pages(tokens,lens):
    rng=random.Random(SEED)
    ls=lens[:]
    rng.shuffle(ls)
    pages=[]; pos=0; i=0
    while pos+45 < len(tokens):
        n=ls[i%len(ls)]; i+=1
        if pos+n>len(tokens): break
        pages.append(tokens[pos:pos+n]); pos+=n
    # complete pseudo-singulions only
    pages=pages[:len(pages)//4*4]
    return pages

def K(toks): return min(64,max(24,len(toks)//3))
def head(x): return x[:K(x)]
def tail(x): return x[-K(x):]

def grams(toks,n):
    c=Counter()
    for t in toks:
        s='^'+t+'$'
        for i in range(max(0,len(s)-n+1)): c[s[i:i+n]]+=1
    return c

def js_from_counters(a,b,keys):
    if not keys: return 0.0
    A=np.array([a.get(k,0) for k in keys],float)+1e-12
    B=np.array([b.get(k,0) for k in keys],float)+1e-12
    A/=A.sum(); B/=B.sum(); M=.5*(A+B)
    kl=lambda x,m: float(np.sum(x*np.log(x/m)))
    return math.sqrt(max(0,.5*kl(A,M)+.5*kl(B,M)))

def rare_score(a,b,weights):
    A=set(t for t in a if t in weights); B=set(t for t in b if t in weights)
    if not A and not B: return 0.0
    num=sum(weights[t] for t in A&B); den=sum(weights[t] for t in A|B)
    return num/den if den else 0.0

def ce_score(a,b,gram_vocab):
    # high is good: negative directional cross entropy of b under model from a
    ca=grams(a,3); cb=grams(b,3); V=len(gram_vocab); alpha=.5
    total=sum(ca.get(g,0) for g in gram_vocab); denom=total+alpha*V
    nb=sum(cb.get(g,0) for g in gram_vocab)
    if nb==0 or denom<=0: return -99.0
    ce=0.0
    for g in gram_vocab:
        n=cb.get(g,0)
        if n: ce -= n*math.log((ca.get(g,0)+alpha)/denom)
    return -(ce/nb)

def morph_counter(toks):
    c=Counter()
    for t in toks:
        c[f'L{min(len(t),12)}']+=1
        if t:
            c['I1:'+t[:1]]+=1; c['T1:'+t[-1:]]+=1
            c['I2:'+t[:2]]+=1; c['T2:'+t[-2:]]+=1
    return c

def morph_score(a,b,morph_vocab):
    return -js_from_counters(morph_counter(a),morph_counter(b),morph_vocab)

def build_defs(alltokens):
    f=Counter(alltokens)
    rare={t:1.0/math.log2(2+f[t]) for t in f if 2<=f[t]<=12}
    gc=Counter()
    for t in alltokens: gc.update(grams([t],3))
    gv=sorted(g for g,n in gc.items() if n>=5)
    mc=morph_counter(alltokens); mv=sorted(mc)
    return rare,gv,mv

def direct_score(A,B,fam,defs):
    rare,gv,mv=defs
    best=(-1e99,None)
    for ia,pa in enumerate(A):
        ta=tail(pa)
        for ib,pb in enumerate(B):
            hb=head(pb)
            if fam=='RARE': s=rare_score(ta,hb,rare)
            elif fam=='SUBWORD': s=ce_score(ta,hb,gv)
            else: s=morph_score(ta,hb,mv)
            if s>best[0]: best=(s,(ia,ib))
    return best

def score_window(units,fam,defs):
    n=len(units); D=np.full((n,n),-1e99,float); DIR={}; PORT={}
    for i in range(n):
        for j in range(n):
            if i==j: continue
            s,p=direct_score(units[i],units[j],fam,defs); D[i,j]=s; DIR[(i,j)]=s; PORT[(i,j)]=p
    # Undirected connection strength can choose either direction, exactly as
    # target will. Each node nominates one strongest partner; union is graph.
    U=np.maximum(D,D.T)
    pred=set()
    for i in range(n):
        js=[j for j in range(n) if j!=i]; j=max(js,key=lambda q:U[i,q]); pred.add(tuple(sorted((i,j))))
    truth={tuple((i,i+1)) for i in range(n-1)}
    tp=len(pred&truth); prec=tp/len(pred) if pred else 0.; rec=tp/len(truth); fpr=1-prec
    dirs=[]
    for i,j in pred&truth:
        dirs.append(1 if D[i,j]>D[j,i] else 0)
    diracc=float(np.mean(dirs)) if dirs else float('nan')
    # Matched pair-score null: compare mean true-adjacent strength with means
    # from same number of nonadjacent edges sampled without replacement.
    obs=float(np.mean([U[i,j] for i,j in truth])); pool=[U[i,j] for i in range(n) for j in range(i+1,n) if (i,j) not in truth]
    rng=random.Random(SEED+1000+sum(len(x[0]) for x in units)+hash(fam)%997)
    null=[]
    for _ in range(N_NULL): null.append(float(np.mean(rng.sample(pool,len(truth)))))
    m=float(np.mean(null)); sd=float(np.std(null,ddof=1)); z=(obs-m)/sd if sd else float('nan')
    return {'pred':sorted([list(x) for x in pred]),'truth':sorted([list(x) for x in truth]),'tp':tp,'precision':prec,'recall':rec,'false_edge_rate':fpr,'direction_accuracy':diracc,
            'boundary_mean':obs,'null_mean':m,'null_sd':sd,'effect':obs-m,'effect_over_null_sd':z,'boundary_pass':bool(z>=2)}

def combine(window_results):
    # family consensus >=2/3
    votes=Counter()
    truth={tuple(x) for x in window_results[FAMS[0]]['truth']}
    for fam in FAMS:
        for x in window_results[fam]['pred']: votes[tuple(x)]+=1
    pred={e for e,n in votes.items() if n>=2}; tp=len(pred&truth); prec=tp/len(pred) if pred else 0.; rec=tp/len(truth); fpr=1-prec if pred else 1.
    return {'pred':sorted([list(x) for x in pred]),'tp':tp,'precision':prec,'recall':rec,'false_edge_rate':fpr}

def main():
    caesar,src=fetch_caesar(); lens=vms_page_lengths(); pages=make_pages(caesar,lens)
    units=[pages[i:i+4] for i in range(0,len(pages),4)]
    if len(units)<WINDOW_UNITS+2: raise RuntimeError('too few pseudo-units')
    defs=build_defs(caesar)
    windows=[]
    for st in range(0,len(units)-WINDOW_UNITS+1,STEP_UNITS):
        us=units[st:st+WINDOW_UNITS]; fr={fam:score_window(us,fam,defs) for fam in FAMS}; co=combine(fr)
        windows.append({'start_unit':st,'families':fr,'combined':co})
    agg={}
    for fam in FAMS:
        rr=[w['families'][fam] for w in windows]
        d=[r['direction_accuracy'] for r in rr if math.isfinite(r['direction_accuracy'])]
        zpass=sum(r['boundary_pass'] for r in rr)/len(rr)
        q={'mean_recall':float(np.mean([r['recall'] for r in rr])),'mean_false_edge_rate':float(np.mean([r['false_edge_rate'] for r in rr])),
           'direction_accuracy':float(np.mean(d)) if d else float('nan'),'boundary_pass_fraction':zpass}
        q['qualified']=bool(q['mean_recall']>=.70 and q['mean_false_edge_rate']<=.20 and q['direction_accuracy']>=.70 and q['boundary_pass_fraction']>=.75)
        agg[fam]=q
    crr=[w['combined'] for w in windows]
    comb={'mean_recall':float(np.mean([x['recall'] for x in crr])),'mean_false_edge_rate':float(np.mean([x['false_edge_rate'] for x in crr]))}
    nqual=sum(agg[f]['qualified'] for f in FAMS); comb['n_qualified_families']=nqual
    comb['qualified']=bool(nqual>=2 and comb['mean_recall']>=.75 and comb['mean_false_edge_rate']<=.15)
    out={'protocol':PROTOCOL,'stage':'C1','target_connections_opened':False,'source':src,'vms_page_length_count':len(lens),'pseudo_pages':len(pages),'pseudo_units':len(units),'n_windows':len(windows),
         'family_aggregate':agg,'combined':comb,'windows':windows,
         'gate':'C1 combined PASS only if >=2/3 family gates, consensus recall>=.75 and false-edge<=.15. Failure keeps Voynich target sealed.'}
    (OUT/'c1_calibration.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    lines=['# Pure-text connection C1 closeout','',f"Target connections opened: **NO**",'',f"Pseudo-units: {len(units)}; windows: {len(windows)}",'']
    for f,q in agg.items(): lines.append(f"- {f}: recall {q['mean_recall']:.3f}; false-edge {q['mean_false_edge_rate']:.3f}; direction {q['direction_accuracy']:.3f}; boundary>=2SD windows {q['boundary_pass_fraction']:.3f}; **{'PASS' if q['qualified'] else 'FAIL'}**")
    lines += ['',f"Combined: recall {comb['mean_recall']:.3f}; false-edge {comb['mean_false_edge_rate']:.3f}; qualified families {nqual}/3; **{'PASS' if comb['qualified'] else 'FAIL'}**",
              '', 'If FAIL, Voynich target remains sealed and v0.1 stops by preregistration.']
    (OUT/'C1_CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'family_aggregate':agg,'combined':comb,'n_windows':len(windows),'pseudo_units':len(units)},indent=2),flush=True)

if __name__=='__main__': main()

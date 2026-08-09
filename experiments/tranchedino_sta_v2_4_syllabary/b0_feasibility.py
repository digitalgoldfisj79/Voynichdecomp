#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd

CSV_SHA='c5eba63cbe8055d3506d099043f5df23fd427df709546df6de70e084fedd3cf6'
ALPH='abcdefghilmnopqrstu'; ASET=set(ALPH)
TRANS=str.maketrans({'j':'i','v':'u','w':'u','y':'i','x':'s','z':'s'})
GEM=('bb','cc','ff','mm','nn','rr','tt','ss')
SYLL=(
'ba','be','bi','bo','bu','ca','ce','ci','co','cu','da','de','di','do','du',
'fa','fe','fi','fo','fu','ga','ge','gi','go','gu','la','le','li','lo','lu',
'ma','me','mi','mo','mu','na','ne','ni','no','nu','pa','pe','pi','po','pu',
'qua','que','qui','quo','ra','re','ri','ro','ru','sa','se','si','so','su',
'ta','te','ti','to','tu')
LEX_RAW=('dinari','galee','nave','grippo','che','perche','como','unde','quando')
P_SYLL=(.25,.50,.75,1.00); P_NULL=(.00,.03,.10)
RF_COVERAGE={
112:.9996566065092144,113:.9996629656479327,114:.9996693247866509,
115:.9996756839253691,116:.9996820430640874,117:.9996884022028056,
118:.9996947613415239,119:.9997011204802422,120:.9997074796189604,
121:.9997138387576786,122:.9997201978963969,123:.9997265570351151,
124:.9997329161738334}

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def seedint(*parts): return int.from_bytes(hashlib.sha256('::'.join(map(str,parts)).encode()).digest()[:8],'big')
def norm_word(x):
    x=str(x).lower().translate(TRANS)
    return ''.join(c for c in x if c in ASET)
def line_words(x): return [w for raw in str(x).split() if (w:=norm_word(raw))]

def build_source(path:Path):
    assert sha(path)==CSV_SHA
    d=pd.read_csv(path).fillna(''); assert len(d)==5735
    d['words']=d.text.astype(str).map(line_words)
    d['letters19']=d.words.map(lambda ws:''.join(ws))
    pages=sorted(d.loc[d.letters19.str.len()>0,'page'].unique())
    cut=pages[int(len(pages)*.72)]
    tr=d.loc[(d.page<cut)&(d.letters19.str.len()>0)].copy()
    te=d.loc[(d.page>=cut)&(d.letters19.str.len()>0)].copy()
    assert cut==183 and len(tr)==4119 and len(te)==1423
    assert int(tr.letters19.str.len().sum())==172347
    assert int(te.letters19.str.len().sum())==54750
    return tr,te

def select_windows(te,target=12000):
    pages=sorted(te.page.unique())
    starts=sorted(pages,key=lambda p:hashlib.sha256(f'TRANCHSTA24B0window::{int(p)}'.encode()).hexdigest())[:4]
    lines=[(int(r.page),r.words) for r in te.itertuples() if r.words]
    out=[]
    for sp in starts:
        start=next(i for i,(p,_ws) in enumerate(lines) if p==int(sp))
        words=[]; chars=0; i=0
        while chars<target:
            _p,ws=lines[(start+i)%len(lines)]
            for w in ws:
                if chars+len(w)>target and chars>0: break
                words.append(w); chars+=len(w)
                if chars>=target: break
            i+=1
        assert chars==target
        out.append((int(sp),words))
    return out

def inventory():
    mult={c:(3 if c in 'aeiou' else 2) for c in ALPH}
    alpha={c:[f'A:{c}:{j}' for j in range(mult[c])] for c in ALPH}
    gem={u:f'G:{u}' for u in GEM}; syll={u:f'S:{u}' for u in SYLL}
    null=[f'N:{j}' for j in range(7)]
    lex={norm_word(u):f'L:{norm_word(u)}' for u in LEX_RAW}
    latent=[f'L:LATENT:{j}' for j in range(35)]
    ids=[x for v in alpha.values() for x in v]+list(gem.values())+list(syll.values())+null+list(lex.values())+latent
    assert len(ids)==len(set(ids))==166
    return alpha,gem,syll,null,lex

def encode(words,p_syll,p_null,rep,inv):
    alpha,gem,syll,null,lex=inv
    rng=random.Random(seedint('TRANCHSTA24B0',rep,f'{p_syll:.2f}',f'{p_null:.2f}'))
    order=sorted(SYLL,key=lambda s:(-len(s),s)); events=[]; plain=0
    for w in words:
        plain+=len(w)
        if w in lex:
            events.append(lex[w])
            if p_null and rng.random()<p_null: events.append(rng.choice(null))
            continue
        i=0
        while i<len(w):
            match=next((u for u in order if w.startswith(u,i)),None)
            if match is not None and rng.random()<p_syll:
                events.append(syll[match]); i+=len(match)
            elif i+1<len(w) and w[i:i+2] in gem:
                u=w[i:i+2]; events.append(gem[u]); i+=2
            else:
                events.append(rng.choice(alpha[w[i]])); i+=1
            if p_null and rng.random()<p_null: events.append(rng.choice(null))
    c=collections.Counter(events); occ=collections.Counter(); distinct=collections.Counter()
    names={'A':'alphabet','G':'geminate','S':'syllable','N':'null','L':'lexical'}
    for event,n in c.items():
        cls=names[event[0]]; occ[cls]+=n; distinct[cls]+=1
    probs=np.asarray(list(c.values()),dtype=float)/len(events)
    K=len(c); assert K in RF_COVERAGE
    return {
      'plaintext_letters':plain,'cipher_events':len(events),'expansion_ratio':plain/len(events),
      'K_active':K,'distinct_by_class':dict(distinct),'occurrences_by_class':dict(occ),
      'surface_entropy_bits':float(-(probs*np.log2(probs)).sum()),
      'rf_top_K_active_coverage':RF_COVERAGE[K],
      'full_sta_gate':bool(K<=166 and RF_COVERAGE[K]>=.995),
      'aaa_cardinality_diagnostic':bool(K<=150)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('paduan_lines_csv'); ap.add_argument('--output',type=Path); a=ap.parse_args()
    tr,te=build_source(Path(a.paduan_lines_csv)); windows=select_windows(te); inv=inventory(); rows=[]
    for rep,(page,words) in enumerate(windows):
        for ps in P_SYLL:
            for pn in P_NULL:
                r=encode(words,ps,pn,rep,inv); r.update(window=rep,start_page=page,p_syll=ps,p_null=pn); rows.append(r)
    assert len(rows)==48
    vals=lambda key:[r[key] for r in rows]
    aggregate={k:{'min':float(min(vals(k))),'median':float(np.median(vals(k))),'max':float(max(vals(k)))} for k in ('K_active','expansion_ratio','cipher_events','surface_entropy_bits')}
    payload={
      'namespace':'TRANCHSTA24B0','date':'2026-08-09','verdict':'STAGE B0 PASS' if all(r['full_sta_gate'] for r in rows) else 'STAGE B0 FAIL',
      'voynich_fit_authorised':False,'source':{'cut_page':183,'train_lines':4119,'train_letters19':172347,'heldout_lines':1423,'heldout_letters19':54750,'window_start_pages':[p for p,_ in windows]},
      'grid':{'p_syll':P_SYLL,'p_null':P_NULL,'controls':48},'aggregate':aggregate,
      'representation':{'RF1b_observed_types':166,'K_active_range':[min(vals('K_active')),max(vals('K_active'))],'rf_min_top_K_active_coverage':min(r['rf_top_K_active_coverage'] for r in rows),'full_sta_all_48_gate':all(r['full_sta_gate'] for r in rows),'aaa_signature_types':150,'aaa_all_48_cardinality_diagnostic':all(r['aaa_cardinality_diagnostic'] for r in rows),'aaa_target_mapping_authorised':False,'rf_coverage_hf_job':'6a78a1afda2af92a634f04a8'},
      'rows':rows}
    scientific=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(); payload['scientific_sha256']=scientific
    text=json.dumps(payload,indent=2,sort_keys=True)
    if a.output: a.output.write_text(text,encoding='utf-8')
    print('TRANCHSTA24B0='+json.dumps(payload,separators=(',',':')),flush=True)

if __name__=='__main__': main()

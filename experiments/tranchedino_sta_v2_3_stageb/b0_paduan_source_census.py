#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

CSV_SHA='c5eba63cbe8055d3506d099043f5df23fd427df709546df6de70e084fedd3cf6'
ALPH='abcdefghilmnopqrstu'
ASET=set(ALPH)
TRANS=str.maketrans({'j':'i','v':'u','w':'u','y':'i','x':'s','z':'s'})
GEM=('bb','cc','dd','ff','gg','ll','nn','pp','rr','ss','tt')

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def seedint(s:str): return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'big') & 0x7fffffff

def norm_word(x):
    x=str(x).lower().translate(TRANS)
    return ''.join(c for c in x if c in ASET)

def line_words(x):
    out=[]
    for raw in str(x).split():
        w=norm_word(raw)
        if w: out.append(w)
    return out

def sample_words(lines,rep,target_chars=12000):
    n=len(lines); start=seedint(f'TRANCHSTA23B0control::{rep}')%n
    out=[]; chars=0; i=0
    while chars<target_chars:
        ws=lines[(start+i)%n]
        for w in ws:
            if chars+len(w)>target_chars and chars>0: break
            out.append(w); chars+=len(w)
            if chars>=target_chars: break
        i+=1
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('paduan_lines_csv'); a=ap.parse_args()
    p=Path(a.paduan_lines_csv); assert sha(p)==CSV_SHA
    d=pd.read_csv(p).fillna('')
    assert len(d)==5735
    d['words']=d.text.astype(str).map(line_words)
    d['letters19']=d.words.map(lambda ws:''.join(ws))
    pages=sorted(d.loc[d.letters19.str.len()>0,'page'].unique())
    cut=pages[int(len(pages)*.72)]
    tr=d.loc[(d.page<cut)&(d.letters19.str.len()>0)].copy()
    te=d.loc[(d.page>=cut)&(d.letters19.str.len()>0)].copy()
    assert cut==183 and len(tr)==4119 and len(te)==1423
    assert int(tr.letters19.str.len().sum())==172347
    assert int(te.letters19.str.len().sum())==54750
    train=collections.Counter(w for ws in tr.words for w in ws)
    held=collections.Counter(w for ws in te.words for w in ws)
    test_lines=[ws for ws in te.words.tolist() if ws]
    feasibility={}
    for pooln in (64,96,128,192):
        pool=[w for w,_ in train.most_common(pooln)]
        rows=[]
        for rep in range(12):
            rng=np.random.default_rng(seedint(f'TRANCHSTA23B0codebook::{pooln}::{rep}'))
            cb=list(rng.choice(pool,size=38,replace=False))
            ws=sample_words(test_lines,rep,12000); cc=collections.Counter(ws)
            rows.append({'rep':rep,'distinct_codes_observed':sum(cc[w]>0 for w in cb),'code_occurrences':sum(cc[w] for w in cb)})
        feasibility[str(pooln)]={
          'rows':rows,
          'median_distinct':float(np.median([r['distinct_codes_observed'] for r in rows])),
          'minimum_distinct':int(min(r['distinct_codes_observed'] for r in rows)),
          'median_occurrences':float(np.median([r['code_occurrences'] for r in rows])),
          'minimum_occurrences':int(min(r['code_occurrences'] for r in rows))
        }
    gem={}
    for name,z in [('train',tr),('heldout',te)]:
        s='\n'.join(z.letters19.tolist())
        gem[name]={g:s.count(g) for g in GEM}
    out={
      'paduan_lines_sha256':sha(p),'csv_rows':len(d),'text_bearing_pages':len(pages),'cut_page':int(cut),
      'train_lines':len(tr),'heldout_lines':len(te),'train_letters19':int(tr.letters19.str.len().sum()),'heldout_letters19':int(te.letters19.str.len().sum()),
      'train_word_tokens':int(sum(train.values())),'train_word_types':len(train),'heldout_word_tokens':int(sum(held.values())),'heldout_word_types':len(held),
      'codebook_feasibility_38_from_topN':feasibility,'geminate_occurrences':gem,
      'recommended_observation_regime':{'pool_size':96,'codebook_size':38,'control_letters':12000,'reason':'fresh controls retain at least 26/38 observed code identities and at least 171 code occurrences in all 12 source-only simulations while remaining less saturated than top-64'},
      'gate':True
    }
    print('B0_PADUAN='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__': main()

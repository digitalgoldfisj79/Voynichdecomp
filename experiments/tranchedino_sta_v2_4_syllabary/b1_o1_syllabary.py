#!/usr/bin/env python3
"""Frozen B1-O1 component-oracle runner for Tranchedino × STA v2.4.

Requires: pandas, numpy, rapidfuzz.
Prints numerical truth-based metrics only; never prints decoded Q1 text.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

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


def file_sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,parts)).encode()).digest()[:8],'big') & 0x7fffffff

def norm_word(x):
    x=str(x).lower().translate(TRANS)
    return ''.join(c for c in x if c in ASET)
def line_words(x): return [w for raw in str(x).split() if (w:=norm_word(raw))]
LEX=tuple(norm_word(x) for x in LEX_RAW)
SEM=list(ALPH)+list(GEM)+list(SYLL)+list(LEX)
SEM2ID={s:i for i,s in enumerate(SEM)}
SYLL_ORDER=sorted(SYLL,key=lambda s:(-len(s),s))


def load_source(path:Path):
    assert file_sha(path)==CSV_SHA
    d=pd.read_csv(path).fillna(''); assert len(d)==5735
    d['words']=d.text.astype(str).map(line_words)
    d['letters19']=d.words.map(lambda ws:''.join(ws))
    pages=sorted(d.loc[d.letters19.str.len()>0,'page'].unique())
    cut=pages[int(len(pages)*.72)]
    tr=d.loc[(d.page<cut)&(d.letters19.str.len()>0)].copy()
    te=d.loc[(d.page>=cut)&(d.letters19.str.len()>0)].copy()
    assert cut==183 and len(tr)==4119 and int(tr.letters19.str.len().sum())==172347
    assert len(te)==1423 and int(te.letters19.str.len().sum())==54750
    # Reproduce the D1 contamination boundary: four sequential chunks, each
    # whole-line accumulated until >=12,000 letters, consumed 1,248 records.
    rows=list(te.itertuples()); idx=0
    for _ in range(4):
        chars=0
        while chars<12000:
            chars += sum(map(len,rows[idx].words)); idx+=1
    assert idx==1248
    q=rows[idx:]
    assert len(q)==175 and sum(sum(map(len,r.words)) for r in q)==6619
    assert min(int(r.page) for r in q)==243 and max(int(r.page) for r in q)==251
    return tr,[r.words for r in q if r.words]


def encode_semantic_words(words,p_syll,rng):
    out=[]
    for w in words:
        if w in LEX:
            out.append(w); continue
        i=0
        while i<len(w):
            match=next((u for u in SYLL_ORDER if w.startswith(u,i)),None)
            if match is not None and rng.random()<p_syll:
                out.append(match); i+=len(match)
            elif i+1<len(w) and w[i:i+2] in GEM:
                out.append(w[i:i+2]); i+=2
            else:
                out.append(w[i]); i+=1
    return out


def build_model(tr,p_syll,passes=4,alpha=.25):
    V=len(SEM); tri=np.full((V,V,V),alpha,dtype=np.float64)
    for pa in range(passes):
        rng=random.Random(100000+pa+int(p_syll*1000))
        for ws in tr.words:
            ids=[SEM2ID[x] for x in encode_semantic_words(ws,p_syll,rng)]
            for a,b,c in zip(ids,ids[1:],ids[2:]): tri[a,b,c]+=1
    return np.log(tri/tri.sum(axis=2,keepdims=True))


def training_syllable_frequency(tr,p_syll,passes=2):
    cnt=collections.Counter()
    for pa in range(passes):
        rng=random.Random(200000+pa+int(p_syll*1000))
        for ws in tr.words: cnt.update(encode_semantic_words(ws,p_syll,rng))
    return cnt


def inventory():
    mult={c:(3 if c in 'aeiou' else 2) for c in ALPH}
    alpha={c:[] for c in ALPH}; true={}; cls={}; label=0
    for c in ALPH:
        for _ in range(mult[c]):
            alpha[c].append(label); true[label]=c; cls[label]='A'; label+=1
    gem={}
    for u in GEM: gem[u]=label; true[label]=u; cls[label]='G'; label+=1
    syll={}
    for u in SYLL: syll[u]=label; true[label]=u; cls[label]='S'; label+=1
    null=[]
    for _ in range(7): null.append(label); true[label]=''; cls[label]='N'; label+=1
    lex={}
    for u in LEX: lex[u]=label; true[label]=u; cls[label]='L'; label+=1
    for _ in range(35): true[label]=None; cls[label]='L'; label+=1
    assert label==166
    return alpha,gem,syll,null,lex,true,cls


def generate_control(q_lines,p_syll,p_null):
    alpha,gem,syll,null,lex,true,cls=inventory()
    seed=stable_seed('TRANCHSTA24B1O1-control',f'{p_syll:.2f}',f'{p_null:.2f}')
    rng=random.Random(seed); perm=list(range(166)); rng.shuffle(perm)
    true_map={perm[b]:true[b] for b in range(166)}
    class_map={perm[b]:cls[b] for b in range(166)}
    syll_true={perm[b]:u for u,b in syll.items()}
    lines=[]
    for words in q_lines:
        events=[]
        for w in words:
            if w in lex:
                events.append(perm[lex[w]])
                if p_null and rng.random()<p_null: events.append(perm[rng.choice(null)])
                continue
            i=0
            while i<len(w):
                match=next((u for u in SYLL_ORDER if w.startswith(u,i)),None)
                if match is not None and rng.random()<p_syll:
                    events.append(perm[syll[match]]); i+=len(match)
                elif i+1<len(w) and w[i:i+2] in gem:
                    u=w[i:i+2]; events.append(perm[gem[u]]); i+=2
                else:
                    c=w[i]; events.append(perm[rng.choice(alpha[c])]); i+=1
                if p_null and rng.random()<p_null: events.append(perm[rng.choice(null)])
        if events: lines.append(events)
    return lines,true_map,class_map,syll_true


def frequency_initialisation(lines,class_map,syll_true,train_freq):
    obs=collections.Counter(x for line in lines for x in line if class_map[x]=='S')
    signs=sorted(syll_true,key=lambda x:(-obs[x],x))
    candidates=sorted(SYLL,key=lambda u:(-train_freq[u],u))
    return {s:u for s,u in zip(signs,candidates)}


def prepare(lines,true_map,class_map,syll_true,mapping):
    seq=[]; sign_at=[]; ranges=[]
    for line in lines:
        start=len(seq)
        for op in line:
            c=class_map[op]
            if c=='N': continue
            if c=='S': seq.append(SEM2ID[mapping[op]]); sign_at.append(op)
            else:
                sem=true_map[op]
                if sem is not None: seq.append(SEM2ID[sem]); sign_at.append(-1)
        if len(seq)>start: ranges.append((start,len(seq)))
    seq=np.asarray(seq,dtype=np.int32); sign_at=np.asarray(sign_at,dtype=np.int32)
    valid=np.zeros(max(0,len(seq)-2),dtype=bool)
    for s,e in ranges:
        if e-s>=3: valid[s:e-2]=True
    starts=np.nonzero(valid)[0]
    pos_by={op:np.nonzero(sign_at==op)[0] for op in syll_true}
    aff_by={}
    for op,pos in pos_by.items():
        affected=[]
        for pp in pos:
            for a in (int(pp)-2,int(pp)-1,int(pp)):
                if 0<=a<len(valid) and valid[a]: affected.append(a)
        aff_by[op]=np.unique(affected).astype(np.int32)
    return seq,starts,pos_by,aff_by


def anneal(lines,true_map,class_map,syll_true,p_syll,trigram,train_freq,ensemble):
    mapping=frequency_initialisation(lines,class_map,syll_true,train_freq)
    seq,starts,pos_by,aff_by=prepare(lines,true_map,class_map,syll_true,mapping)
    score=float(trigram[seq[starts],seq[starts+1],seq[starts+2]].sum())
    best_score=score; best_map=dict(mapping)
    rng=random.Random(stable_seed('TRANCHSTA24B1O1-anneal',ensemble,f'{p_syll:.2f}',f'{CURRENT_P_NULL:.2f}'))
    signs=list(syll_true)
    for it in range(50000):
        a,b=rng.sample(signs,2); ua,ub=mapping[a],mapping[b]
        affected=np.union1d(aff_by[a],aff_by[b])
        old=float(trigram[seq[affected],seq[affected+1],seq[affected+2]].sum()) if len(affected) else 0.0
        va,vb=SEM2ID[ua],SEM2ID[ub]; pa,pb=pos_by[a],pos_by[b]
        seq[pa]=vb; seq[pb]=va
        new=float(trigram[seq[affected],seq[affected+1],seq[affected+2]].sum()) if len(affected) else 0.0
        delta=new-old; temp=3.0*(.01/3.0)**(it/49999)
        if delta>=0 or rng.random()<math.exp(delta/max(temp,1e-12)):
            score+=delta; mapping[a],mapping[b]=ub,ua
            if score>best_score: best_score=score; best_map=dict(mapping)
        else:
            seq[pa]=va; seq[pb]=vb
    return best_score,best_map


def expanded_text(lines,true_map,class_map,syll_mapping):
    out=[]
    for line in lines:
        for op in line:
            c=class_map[op]
            if c=='N': continue
            sem=syll_mapping[op] if c=='S' else true_map[op]
            if sem is not None: out.append(sem)
    return ''.join(out)

CURRENT_P_NULL=0.0

def main():
    global CURRENT_P_NULL
    ap=argparse.ArgumentParser(); ap.add_argument('paduan_lines_csv'); ap.add_argument('--output',type=Path); args=ap.parse_args()
    tr,q_lines=load_source(Path(args.paduan_lines_csv))
    models={p:build_model(tr,p) for p in P_SYLL}
    freqs={p:training_syllable_frequency(tr,p) for p in P_SYLL}
    rows=[]
    for p in P_SYLL:
        for pn in P_NULL:
            CURRENT_P_NULL=pn
            lines,true_map,class_map,syll_true=generate_control(q_lines,p,pn)
            obs=collections.Counter(x for line in lines for x in line if class_map[x]=='S')
            true_text=expanded_text(lines,true_map,class_map,syll_true)
            erows=[]
            for ensemble in ('A','B'):
                score,mapping=anneal(lines,true_map,class_map,syll_true,p,models[p],freqs[p],ensemble)
                wacc=sum(n for x,n in obs.items() if mapping[x]==syll_true[x])/sum(obs.values())
                idacc=sum(mapping[x]==syll_true[x] for x in obs)/len(obs)
                pred=expanded_text(lines,true_map,class_map,mapping)
                edit=1-Levenshtein.distance(true_text,pred)/max(len(true_text),len(pred),1)
                nonnull=sum(1 for line in lines for x in line if class_map[x]!='N')
                erows.append({'ensemble':ensemble,'score':score,'score_per_event':score/nonnull,'occurrence_weighted_recovery':wacc,'observed_identity_recovery':idacc,'expanded_plaintext_edit_accuracy':edit,'_map':mapping})
            A,B=erows
            agree=sum(n for x,n in obs.items() if A['_map'][x]==B['_map'][x])/sum(obs.values())
            for e in erows: e.pop('_map')
            rows.append({'p_syll':p,'p_null':pn,'observed_syllable_identities':len(obs),'syllable_occurrences':sum(obs.values()),'A':A,'B':B,'AB_occurrence_weighted_map_agreement':agree,'AB_score_diff_per_event':abs(A['score_per_event']-B['score_per_event'])})
    w=[r[e]['occurrence_weighted_recovery'] for r in rows for e in ('A','B')]
    ids=[r[e]['observed_identity_recovery'] for r in rows for e in ('A','B')]
    edits=[r[e]['expanded_plaintext_edit_accuracy'] for r in rows for e in ('A','B')]
    agree=[r['AB_occurrence_weighted_map_agreement'] for r in rows]
    source_ok=all(r['observed_syllable_identities']>=45 and r['syllable_occurrences']>=400 for r in rows)
    summary={'source_sufficient':source_ok,'minimum_observed_syllable_identities':min(r['observed_syllable_identities'] for r in rows),'minimum_syllable_occurrences':min(r['syllable_occurrences'] for r in rows),'median_occurrence_weighted_recovery':float(np.median(w)),'minimum_occurrence_weighted_recovery':min(w),'median_observed_identity_recovery':float(np.median(ids)),'minimum_observed_identity_recovery':min(ids),'median_expanded_plaintext_edit_accuracy':float(np.median(edits)),'minimum_expanded_plaintext_edit_accuracy':min(edits),'median_AB_map_agreement':float(np.median(agree)),'minimum_AB_map_agreement':min(agree)}
    gates={'occurrence_median':summary['median_occurrence_weighted_recovery']>=.95,'occurrence_minimum':summary['minimum_occurrence_weighted_recovery']>=.85,'identity_median':summary['median_observed_identity_recovery']>=.90,'identity_minimum':summary['minimum_observed_identity_recovery']>=.75,'plaintext_median':summary['median_expanded_plaintext_edit_accuracy']>=.97,'plaintext_minimum':summary['minimum_expanded_plaintext_edit_accuracy']>=.93,'agreement_median':summary['median_AB_map_agreement']>=.95,'agreement_minimum':summary['minimum_AB_map_agreement']>=.85}
    verdict='B1-O1 SOURCE INSUFFICIENT' if not source_ok else ('B1-O1 SYLLABARY COMPONENT QUALIFIED' if all(gates.values()) else 'B1-O1 SYLLABARY COMPONENT NOT QUALIFIED')
    payload={'namespace':'TRANCHSTA24B1O1','date':'2026-08-09','verdict':verdict,'voynich_fit_authorised':False,'summary':summary,'gates':gates,'rows':rows}
    payload['scientific_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    text=json.dumps(payload,indent=2,sort_keys=True)
    if args.output: args.output.write_text(text,encoding='utf-8')
    print('TRANCHSTA24B1O1='+json.dumps(payload,separators=(',',':')),flush=True)

if __name__=='__main__': main()

#!/usr/bin/env python3
"""Stage M for vms_textual_connections_20260907_v02.

Topology-safe only: learns a generic adjacency kernel from the known internal
order of two-line blocks inside ordinary Voynich paragraphs. No inter-page or
inter-bifolium target edges are computed.
"""
from __future__ import annotations
import json, math, random, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from vms_topology_marginalization import CORPORA, fetch_text, clean_tokens

PROTOCOL='vms_textual_connections_20260907_v02'
OUT=Path('artifacts/vms_textual_connections_v02_m'); OUT.mkdir(parents=True,exist_ok=True)
SEED=2026090721
N_NULL=200
LINE=re.compile(r'^<(f\d+[rv]\d*)\.(\d+),([^>]*)>\s*(.*)$')
FEATURES=['WORD_COS','RARE_JACC','CHAR3_JS','MORPH_JS','BOUNDARY_WORD','BOUNDARY_CHAR','EXACT_BRIDGE']


def cos_counter(a,b):
    if not a or not b:return 0.0
    keys=set(a)|set(b); dot=sum(a.get(k,0)*b.get(k,0) for k in keys)
    na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
    return dot/(na*nb) if na and nb else 0.0

def word_cos(a,b):return cos_counter(Counter(a),Counter(b))
def grams(tokens,n=3):
    c=Counter()
    for t in tokens:
        s='^'+t+'$'
        for i in range(len(s)-n+1):c[s[i:i+n]]+=1
    return c

def js_sim(a,b):
    if not a or not b:return 0.0
    keys=list(set(a)|set(b)); A=np.array([a.get(k,0) for k in keys],float)+1e-9; B=np.array([b.get(k,0) for k in keys],float)+1e-9
    A/=A.sum();B/=B.sum();M=(A+B)/2
    js=.5*np.sum(A*np.log(A/M))+.5*np.sum(B*np.log(B/M))
    return -math.sqrt(max(0,float(js)))
def morph(tokens):
    c=Counter()
    for t in tokens:
        c['L'+str(min(12,len(t)))]+=1
        if t:
            c['I1:'+t[:1]]+=1;c['T1:'+t[-1:]]+=1;c['I2:'+t[:2]]+=1;c['T2:'+t[-2:]]+=1
    return c
def rare_jacc(a,b,w):
    A=set(a)&w.keys();B=set(b)&w.keys()
    if not A and not B:return 0.0
    den=sum(w[x] for x in A|B);return sum(w[x] for x in A&B)/den if den else 0.0
def exact_overlap(a,b):
    A=set(a);B=set(b);return len(A&B)/max(1,len(A|B))

def parse_paragraphs(body):
    paragraphs=[]; current=[]; cur_page=None
    for raw in body.splitlines():
        m=LINE.match(raw)
        if not m:continue
        page,num,flag,txt=m.groups()
        # Ordinary paragraph loci only. Labels/circular/other units do not carry uppercase P.
        if 'P' not in flag:
            if current: paragraphs.append((cur_page,current));current=[];cur_page=None
            continue
        toks=clean_tokens(txt)
        if not toks:continue
        is_start=flag.startswith('@P') or flag.startswith('*P')
        if is_start and current:
            paragraphs.append((cur_page,current));current=[]
        if cur_page is not None and page!=cur_page and current:
            paragraphs.append((cur_page,current));current=[]
        cur_page=page; current.append(toks)
        if flag.startswith('=P'):
            paragraphs.append((cur_page,current));current=[];cur_page=None
    if current:paragraphs.append((cur_page,current))
    return [(p,ls) for p,ls in paragraphs if p and len(ls)>=8]

def blockify(pars):
    out=[]
    for pi,(page,lines) in enumerate(pars):
        blocks=[lines[i]+lines[i+1] for i in range(0,len(lines)-1,2)]
        blocklines=[(lines[i],lines[i+1]) for i in range(0,len(lines)-1,2)]
        if len(blocks)>=4:out.append((pi,page,blocks,blocklines))
    return out

def feature(a,b,alines,blines,rare):
    af,al=alines;bf,bl=blines
    base_wc=np.mean([word_cos(af,bf),word_cos(al,bl),word_cos(af,bl)])
    base_ch=np.mean([js_sim(grams(af),grams(bf)),js_sim(grams(al),grams(bl)),js_sim(grams(af),grams(bl))])
    bridge=exact_overlap(al,bf)-np.mean([exact_overlap(af,bf),exact_overlap(al,bl),exact_overlap(af,bl)])
    return [word_cos(a,b),rare_jacc(a,b,rare),js_sim(grams(a),grams(b)),js_sim(morph(a),morph(b)),word_cos(al,bf)-base_wc,js_sim(grams(al),grams(bf))-base_ch,float(bridge)]

def dataset(body):
    pars=parse_paragraphs(body); blocks=blockify(pars)
    alltok=[t for _,ls in pars for line in ls for t in line];freq=Counter(alltok);rare={t:1/math.log2(2+freq[t]) for t in freq if 2<=freq[t]<=20}
    X=[];y=[];groups=[];para=[]
    for pi,page,bs,bls in blocks:
        for i in range(len(bs)):
            for j in range(i+1,len(bs)):
                X.append(feature(bs[i],bs[j],bls[i],bls[j],rare));y.append(1 if j==i+1 else 0);groups.append(page);para.append(pi)
    return np.asarray(X,float),np.asarray(y,int),np.asarray(groups),np.asarray(para),{'paragraphs_raw':len(pars),'paragraphs_admitted':len(blocks),'pages':len(set(groups)),'pairs':len(y),'positives':int(y.sum()),'base_rate':float(y.mean()) if len(y) else 0.0}

def oof_scores(X,y,g):
    ug=len(set(g));nfold=min(5,ug)
    cv=GroupKFold(n_splits=nfold);pred=np.full(len(y),np.nan)
    for tr,te in cv.split(X,y,g):
        if len(set(y[tr]))<2:continue
        model=Pipeline([('s',StandardScaler()),('m',LogisticRegression(C=1.0,penalty='l2',solver='liblinear',max_iter=2000,random_state=SEED))])
        model.fit(X[tr],y[tr]);pred[te]=model.predict_proba(X[te])[:,1]
    ok=np.isfinite(pred);return pred,ok

def permute_within_para(y,para,rng):
    z=y.copy()
    for p in np.unique(para):
        ix=np.where(para==p)[0];v=z[ix].copy();rng.shuffle(v);z[ix]=v
    return z

def run(code,url,sha):
    body=fetch_text(url,sha);X,y,g,para,meta=dataset(body)
    pred,ok=oof_scores(X,y,g); yy=y[ok];pp=pred[ok]
    auc=float(roc_auc_score(yy,pp));ap=float(average_precision_score(yy,pp));base=float(yy.mean());obs=float(pp[yy==1].mean()-pp[yy==0].mean())
    rng=np.random.default_rng(SEED+sum(map(ord,code)));null=[]
    for k in range(N_NULL):
        yp=permute_within_para(y,para,rng);pr,okp=oof_scores(X,yp,g)
        if okp.sum() and len(set(yp[okp]))==2:null.append(float(pr[okp][yp[okp]==1].mean()-pr[okp][yp[okp]==0].mean()))
    nm=float(np.mean(null));ns=float(np.std(null,ddof=1));z=(obs-nm)/ns if ns else float('nan')
    qual=bool(auc>=.70 and ap>=2*base and z>=2)
    # descriptive full-model coefficients only after CV result frozen, not used for qualification
    mod=Pipeline([('s',StandardScaler()),('m',LogisticRegression(C=1.0,penalty='l2',solver='liblinear',max_iter=2000,random_state=SEED))]);mod.fit(X,y)
    coefs=dict(zip(FEATURES,[float(x) for x in mod.named_steps['m'].coef_[0]]))
    return {'code':code,'meta':meta,'auc':auc,'average_precision':ap,'base_rate':base,'ap_over_base':ap/base if base else float('nan'),'score_gap':obs,'null_mean':nm,'null_sd':ns,'effect':obs-nm,'effect_over_null_sd':z,'qualified':qual,'full_model_standardized_coefficients':coefs}

def main():
    rs=[]
    for code,(url,sha) in CORPORA.items():
        r=run(code,url,sha);rs.append(r);print(code,r,flush=True)
    n=sum(r['qualified'] for r in rs);passed=n>=3
    out={'protocol':PROTOCOL,'stage':'M','target_page_edges_opened':False,'n_null':N_NULL,'features':FEATURES,'results':rs,'n_qualified_transcriptions':n,'stage_pass':passed,'gate':'AUC>=.70, AP>=2x base, score-gap >=2 within-paragraph permutation-null SD; >=3/4 transcriptions'}
    (OUT/'m_summary.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    lines=['# v0.2 Stage M closeout','', 'Inter-page / inter-bifolium target edges opened: **NO**','']
    for r in rs:lines.append(f"- {r['code']}: AUC {r['auc']:.3f}; AP {r['average_precision']:.3f} vs base {r['base_rate']:.3f} ({r['ap_over_base']:.2f}x); score-gap effect {r['effect']:+.5f} vs null SD {r['null_sd']:.5f} = {r['effect_over_null_sd']:.3f} SD; **{'PASS' if r['qualified'] else 'FAIL'}**")
    lines+=['',f"Qualified: {n}/4. **STAGE M {'PASS' if passed else 'FAIL'}**",'', 'FAIL => v0.2 stops before Caesar page-scale / Aberdeen / Voynich target by preregistration.']
    (OUT/'M_CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()

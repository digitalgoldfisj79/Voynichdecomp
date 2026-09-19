#!/usr/bin/env python3
import hashlib, io, json, math, re, unicodedata, urllib.request
from collections import Counter
import numpy as np
import pandas as pd

ALPHAS=(1.0,4.0,16.0,64.0,256.0)
NNULL=100
MIN_TEST_TOKENS=100
WORKS={
 'dietsche_catoen':{
   'url':'https://raw.githubusercontent.com/SofieMoors/controlcorpus/d43bed36a7cbcf7eb6b1caea8b07ac72bd4454a0/dietschecatoen/data/xlsx/synoptic.html',
   'sha':'0f85ffecf57cabb65eb43abdbef8162cadd3a16f73e86ea96fdc885661145b1d'},
 'karel_ende_elegast':{
   'url':'https://raw.githubusercontent.com/SofieMoors/controlcorpus/d43bed36a7cbcf7eb6b1caea8b07ac72bd4454a0/karelendeelegast/data/xlsx/synoptic.html',
   'sha':'5e0dfe694eed3acc16f0ade5f214546c0349958d5f331c98d607095c8585bb0b'},
 'scolastica':{
   'url':'https://raw.githubusercontent.com/SofieMoors/controlcorpus/d43bed36a7cbcf7eb6b1caea8b07ac72bd4454a0/scolastica/data/xlsx/synoptic.html',
   'sha':'24563416385a4598fd880f7e3e6cfdbeecf0e1f91bab0cd3534c5c652e69db98'},
 'martijn':{
   'url':'https://raw.githubusercontent.com/SofieMoors/martijnmanuscripts/7b629e2684aad6294edc247796dbb5e262bf945b/data/xlsx/synoptic.html',
   'sha':'e6501dda03bd803d700d4007d3b5d1af8803bada0aae29724fb5a10f7c32d332'}
}

def fetch_table(meta):
    b=urllib.request.urlopen(meta['url']).read()
    got=hashlib.sha256(b).hexdigest()
    if got!=meta['sha']:raise RuntimeError(f"sha mismatch {got} != {meta['sha']}")
    return pd.read_html(io.BytesIO(b))[0]

def norm_tokens(x):
    if x is None or (isinstance(x,float) and np.isnan(x)):return []
    s=unicodedata.normalize('NFKD',str(x).lower())
    out=[];cur=[]
    for ch in s:
        if unicodedata.category(ch).startswith('L'):
            cur.append(ch)
        else:
            if cur:out.append(''.join(cur));cur=[]
    if cur:out.append(''.join(cur))
    return out

def render(tok,rep):
    if rep=='lexical':return tok
    if rep=='opaque256':
        return 'h'+str(int(hashlib.sha256(('OPAQUE256|'+tok).encode()).hexdigest()[:16],16)%256)
    raise KeyError(rep)

def prepare(t,rep):
    ids=[str(x) for x in t.iloc[:,0].tolist()]
    wits=[str(x) for x in t.columns[1:]]
    cells={}
    for ri in range(len(t)):
        for wi,w in enumerate(wits,1):
            cells[(ri,w)]=[render(z,rep) for z in norm_tokens(t.iloc[ri,wi])]
    return ids,wits,cells

def train_model(rows,wits,cells):
    glob=Counter(); rowc={}; rown={}
    for ri in rows:
        c=Counter()
        for w in wits:c.update(cells[(ri,w)])
        rowc[ri]=c;rown[ri]=sum(c.values());glob.update(c)
    # map all outer-unseen lexical types to UNK; opaque has closed vocabulary naturally
    vocab=set(glob);vocab.add('<UNK>')
    V=len(vocab);N=sum(glob.values())
    p0={x:(glob.get(x,0)+0.5)/(N+0.5*V) for x in vocab}
    return p0,rowc,rown,vocab

def score_rows(test_wit,rows,cells,p0,rowc,rown,vocab,alpha,row_map=None):
    b=0.0;s=0.0;n=0
    punk=p0['<UNK>']
    for ri in rows:
        toks=cells[(ri,test_wit)]
        if not toks:continue
        src=ri if row_map is None else row_map.get(ri,ri)
        c=rowc.get(src,Counter());nn=rown.get(src,0)
        for t in toks:
            tt=t if t in vocab else '<UNK>'
            q=p0.get(tt,punk)
            pr=(c.get(tt,0)+alpha*q)/(nn+alpha) if nn+alpha>0 else q
            b += -math.log2(max(q,1e-300))
            s += -math.log2(max(pr,1e-300))
            n+=1
    return (b-s)/n if n else float('nan'),n

def eligible_rows(witness,train_wits,cells,nrows,min_other=2):
    out=[]
    for ri in range(nrows):
        if not cells[(ri,witness)]:continue
        cov=sum(bool(cells[(ri,w)]) for w in train_wits)
        if cov>=min_other:out.append(ri)
    return out

def choose_alpha(outer_train,nrows,cells):
    total={a:[0.0,0] for a in ALPHAS}
    allrows=list(range(nrows))
    for iw in outer_train:
        inn=[w for w in outer_train if w!=iw]
        if len(inn)<2:continue
        er=eligible_rows(iw,inn,cells,nrows,2)
        if not er:continue
        p0,rowc,rown,vocab=train_model(allrows,inn,cells)
        for a in ALPHAS:
            gain,n=score_rows(iw,er,cells,p0,rowc,rown,vocab,a)
            if n and np.isfinite(gain):
                # maximize gain == minimize source codelength because baseline fixed per split
                total[a][0]+=gain*n;total[a][1]+=n
    vals={a:(x/n if n else -1e99) for a,(x,n) in total.items()}
    # deterministic tie break toward stronger shrinkage
    return max(ALPHAS,key=lambda a:(vals[a],a)),vals

def length_quintiles(rows,wit,cells):
    vals=np.array([len(cells[(ri,wit)]) for ri in rows],float)
    if len(vals)==0:return {}
    qs=np.quantile(vals,[.2,.4,.6,.8])
    return {ri:int(np.searchsorted(qs,len(cells[(ri,wit)]),side='right')) for ri in rows}

def null_row_map(rows,wit,cells,seed):
    rng=np.random.default_rng(seed);q=length_quintiles(rows,wit,cells);groups={}
    for ri in rows:
        groups.setdefault((ri//21,q[ri]),[]).append(ri)
    mp={}
    for g,rr in groups.items():
        rr=list(rr)
        if len(rr)<=1:
            mp[rr[0]]=rr[0];continue
        pp=rr.copy();rng.shuffle(pp)
        # derange fixed points when possible by rotating any residual identity pattern
        if any(a==b for a,b in zip(rr,pp)):
            pp=pp[1:]+pp[:1]
        for a,b in zip(rr,pp):mp[a]=b
    return mp

def run_work(name,meta,rep):
    t=fetch_table(meta);ids,wits,cells=prepare(t,rep);nrows=len(t);allrows=list(range(nrows))
    witness_results=[]
    null_by_wit={}
    for oi,w in enumerate(wits):
        tr=[x for x in wits if x!=w]
        er=eligible_rows(w,tr,cells,nrows,2)
        if not er:continue
        alpha,cv=choose_alpha(tr,nrows,cells)
        p0,rowc,rown,vocab=train_model(allrows,tr,cells)
        eff,n=score_rows(w,er,cells,p0,rowc,rown,vocab,alpha)
        if n<MIN_TEST_TOKENS:continue
        null=[]
        for b in range(NNULL):
            mp=null_row_map(er,w,cells,202610200000 + (int(hashlib.sha256(name.encode()).hexdigest()[:6],16)%100000)*1000 + oi*100 + b)
            x,_=score_rows(w,er,cells,p0,rowc,rown,vocab,alpha,mp);null.append(x)
        witness_results.append({'witness':w,'n':n,'alpha':alpha,'effect':eff,'null_mean':float(np.mean(null)),'null_sd':float(np.std(null,ddof=1))})
        null_by_wit[w]=null
    if not witness_results:return {'work':name,'representation':rep,'eligible_witnesses':0}
    actual=float(np.mean([x['effect'] for x in witness_results]))
    # same null replicate index aggregated across witnesses
    null_work=np.array([[null_by_wit[x['witness']][b] for x in witness_results] for b in range(NNULL)],float).mean(axis=1)
    nsd=float(np.std(null_work,ddof=1));z=(actual-float(np.mean(null_work)))/nsd if nsd>0 else float('inf')
    p=(1+int(np.sum(null_work>=actual)))/(NNULL+1)
    wins=sum(x['effect']>0 for x in witness_results)
    need=math.ceil(2*len(witness_results)/3)
    resolved=bool(z>=2 and p<=.05 and wins>=need)
    return {'work':name,'representation':rep,'eligible_witnesses':len(witness_results),'effect_bits_per_token':actual,
            'null_mean':float(np.mean(null_work)),'null_sd':nsd,'z':z,'p':p,'positive_witnesses':wins,'required_positive':need,
            'resolved':resolved,'witnesses':witness_results}

def main():
    allres={}
    for rep in ('lexical','opaque256'):
        allres[rep]={}
        for name,meta in WORKS.items():
            print('START',rep,name,flush=True)
            allres[rep][name]=run_work(name,meta,rep)
            print('DONE',rep,name,json.dumps({k:v for k,v in allres[rep][name].items() if k!='witnesses'},sort_keys=True),flush=True)
    prim=allres['lexical'];sens=allres['opaque256']
    prim_res=sum(bool(x.get('resolved')) for x in prim.values())
    sign_ok=all((prim[w].get('effect_bits_per_token',0)>=0 and sens[w].get('effect_bits_per_token',0)>=0) for w in WORKS)
    loo=[]
    vals=[prim[w].get('effect_bits_per_token',float('nan')) for w in WORKS]
    names=list(WORKS)
    for i,n in enumerate(names):
        vv=[vals[j] for j in range(len(vals)) if j!=i and np.isfinite(vals[j])]
        loo.append({'heldout_work':n,'mean_other_effect':float(np.mean(vv)) if vv else float('nan')})
    passed=bool(prim_res>=3 and sign_ok and all(x['mean_other_effect']>0 for x in loo))
    out={'source_control_pass':passed,'primary_resolved_works':prim_res,'opacity_no_sign_reversal':sign_ok,'leave_one_work_out':loo,'results':allres}
    print('PHASE1B_SOURCE_CONTROL_RESULT='+json.dumps(out,separators=(',',':'),sort_keys=True),flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
import hashlib, io, json, math, unicodedata, urllib.request
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

SOURCE_ALPHAS=(1.0,4.0,16.0,64.0,256.0)
BETAS=(1.0,4.0,16.0,64.0,256.0)
NNULL=100
MIN_TEST_TOKENS=100
BLOCK=20
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
    if got!=meta['sha']: raise RuntimeError(f"sha mismatch {got}")
    return pd.read_html(io.BytesIO(b))[0]

def norm_tokens(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return []
    s=unicodedata.normalize('NFKD',str(x).lower())
    out=[];cur=[]
    for ch in s:
        if unicodedata.category(ch).startswith('L'):
            cur.append(ch)
        else:
            if cur: out.append(''.join(cur));cur=[]
    if cur: out.append(''.join(cur))
    return out

def lenbin(n):
    return '12' if n<=2 else ('34' if n<=4 else ('56' if n<=6 else '7p'))

def render(tok,rep):
    if rep=='shape':
        return f'{tok[0]}|{tok[-1]}|{lenbin(len(tok))}' if tok else '<UNK>'
    if rep=='opaque256':
        return 'h'+str(int(hashlib.sha256(('OPAQUE256|'+tok).encode()).hexdigest()[:16],16)%256)
    raise KeyError(rep)

def prepare(t,rep):
    wits=[str(x) for x in t.columns[1:]]
    cells={}
    for ri in range(len(t)):
        for wi,w in enumerate(wits,1):
            cells[(ri,w)]=[render(z,rep) for z in norm_tokens(t.iloc[ri,wi])]
    return wits,cells

def eligible_rows(w,others,cells,nrows):
    out=[]
    for ri in range(nrows):
        if not cells[(ri,w)]: continue
        if sum(bool(cells[(ri,x)]) for x in others)>=2: out.append(ri)
    return out

def train_source(rows,wits,cells):
    glob=Counter(); rowc={};rown={}
    for ri in rows:
        c=Counter()
        for w in wits: c.update(cells[(ri,w)])
        rowc[ri]=c;rown[ri]=sum(c.values());glob.update(c)
    vocab=set(glob);vocab.add('<UNK>')
    V=len(vocab);N=sum(glob.values())
    p0={x:(glob.get(x,0)+0.5)/(N+0.5*V) for x in vocab}
    return p0,rowc,rown,vocab

def source_prob(tok,ri,p0,rowc,rown,vocab,alpha):
    tt=tok if tok in vocab else '<UNK>'
    q=p0.get(tt,p0['<UNK>'])
    c=rowc.get(ri,Counter()); n=rown.get(ri,0)
    return (c.get(tt,0)+alpha*q)/(n+alpha) if n+alpha>0 else q

def choose_source_alpha(outer_train,nrows,cells):
    rows=list(range(nrows)); score={a:[0.0,0] for a in SOURCE_ALPHAS}
    for iw in outer_train:
        inn=[x for x in outer_train if x!=iw]
        if len(inn)<2: continue
        er=eligible_rows(iw,inn,cells,nrows)
        if not er: continue
        p0,rowc,rown,vocab=train_source(rows,inn,cells)
        for a in SOURCE_ALPHAS:
            ll=0.0;n=0
            for ri in er:
                for t in cells[(ri,iw)]:
                    ll += -math.log2(max(source_prob(t,ri,p0,rowc,rown,vocab,a),1e-300)); n+=1
            score[a][0]+=ll;score[a][1]+=n
    vals={a:(x/n if n else 1e99) for a,(x,n) in score.items()}
    return min(SOURCE_ALPHAS,key=lambda a:(vals[a],-a)),vals

def blocks(rows):
    d=defaultdict(list)
    for ri in rows:d[ri//BLOCK].append(ri)
    return dict(d)

def expected_counts(rows,w,cells,p0,rowc,rown,vocab,alpha):
    E=Counter(); O=Counter(); total=0
    for ri in rows:
        toks=cells[(ri,w)]
        if not toks: continue
        # expected one family draw per observed token, from source distribution over known vocabulary
        # restrict support to vocabulary and normalize exactly
        probs={}
        z=0.0
        for f in vocab:
            p=source_prob(f,ri,p0,rowc,rown,vocab,alpha)
            probs[f]=p;z+=p
        mult=len(toks)
        for f,p in probs.items(): E[f]+=mult*(p/z)
        for t in toks:
            tt=t if t in vocab else '<UNK>'; O[tt]+=1; total+=1
    return E,O,total

def learn_ratio(train_rows,w,cells,p0,rowc,rown,vocab,alpha,beta):
    E,O,n=expected_counts(train_rows,w,cells,p0,rowc,rown,vocab,alpha)
    et=sum(E.values())
    share={f:E[f]/et for f in vocab} if et else {f:1/len(vocab) for f in vocab}
    R={}
    for f in vocab:
        R[f]=(O[f]+beta*share[f])/(E[f]+beta*share[f]+1e-15)
    return R,n

def score_gain(test_rows,w,cells,p0,rowc,rown,vocab,alpha,R):
    base=0.0;aug=0.0;n=0
    # cache source distributions per row
    for ri in test_rows:
        toks=cells[(ri,w)]
        if not toks: continue
        probs={};z0=0.0
        for f in vocab:
            p=source_prob(f,ri,p0,rowc,rown,vocab,alpha);probs[f]=p;z0+=p
        z=sum((p/z0)*R.get(f,1.0) for f,p in probs.items())
        for t in toks:
            tt=t if t in vocab else '<UNK>'
            q=probs.get(tt,source_prob('<UNK>',ri,p0,rowc,rown,vocab,alpha))/z0
            pa=q*R.get(tt,1.0)/max(z,1e-300)
            base += -math.log2(max(q,1e-300));aug += -math.log2(max(pa,1e-300));n+=1
    return ((base-aug)/n if n else float('nan')),n

def choose_beta(train_rows,w,cells,p0,rowc,rown,vocab,alpha):
    bl=blocks(train_rows); bids=sorted(bl)
    if len(bids)<2:return 64.0
    score={b:[0.0,0] for b in BETAS}
    for fold in (0,1):
        tr=[ri for bid in bids if bid%2!=fold for ri in bl[bid]]
        te=[ri for bid in bids if bid%2==fold for ri in bl[bid]]
        if not tr or not te:continue
        for beta in BETAS:
            R,_=learn_ratio(tr,w,cells,p0,rowc,rown,vocab,alpha,beta)
            g,n=score_gain(te,w,cells,p0,rowc,rown,vocab,alpha,R)
            if n and np.isfinite(g):score[beta][0]+=g*n;score[beta][1]+=n
    vals={b:(x/n if n else -1e99) for b,(x,n) in score.items()}
    return max(BETAS,key=lambda b:(vals[b],b))

def block_mean_len(rows,w,cells):
    lens=[len(cells[(ri,w)]) for ri in rows if cells[(ri,w)]]
    return float(np.mean(lens)) if lens else 0.0

def donor_map(train_rows,target,others,cells,seed):
    rng=np.random.default_rng(seed); bl=blocks(train_rows)
    # global quintiles from all candidate witness-block means in this training set
    vals=[]; rec=[]
    for bid,rr in bl.items():
        for w in others:
            m=block_mean_len(rr,w,cells)
            if m>0: vals.append(m);rec.append((bid,w,m))
    qs=np.quantile(vals,[.2,.4,.6,.8]) if vals else np.array([0,0,0,0])
    def qv(x):return int(np.searchsorted(qs,x,side='right'))
    target_q={bid:qv(block_mean_len(rr,target,cells)) for bid,rr in bl.items()}
    donors={}
    for bid,rr in bl.items():
        cand=[w for w in others if block_mean_len(rr,w,cells)>0 and qv(block_mean_len(rr,w,cells))==target_q[bid]]
        if not cand:cand=[w for w in others if block_mean_len(rr,w,cells)>0]
        donors[bid]=cand[int(rng.integers(len(cand)))] if cand else None
    return donors

def learn_ratio_donors(train_rows,donors,cells,p0,rowc,rown,vocab,alpha,beta):
    E=Counter();O=Counter()
    bl=blocks(train_rows)
    for bid,rr in bl.items():
        w=donors.get(bid)
        if w is None:continue
        e,o,_=expected_counts(rr,w,cells,p0,rowc,rown,vocab,alpha)
        E.update(e);O.update(o)
    et=sum(E.values());share={f:E[f]/et for f in vocab} if et else {f:1/len(vocab) for f in vocab}
    return {f:(O[f]+beta*share[f])/(E[f]+beta*share[f]+1e-15) for f in vocab}

def run_witness(name,w,wits,cells,nrows,rep,wi):
    others=[x for x in wits if x!=w]
    er=eligible_rows(w,others,cells,nrows)
    if not er:return None
    alpha,_=choose_source_alpha(others,nrows,cells)
    p0,rowc,rown,vocab=train_source(list(range(nrows)),others,cells)
    bl=blocks(er); bids=sorted(bl)
    if len(bids)<4:return None
    actual_num=0.0;actual_den=0; null_num=np.zeros(NNULL);null_den=np.zeros(NNULL,int)
    positive_halves=0;half_results=[]
    for test_parity in (0,1):
        train_rows=[ri for bid in bids if bid%2!=test_parity for ri in bl[bid]]
        test_rows=[ri for bid in bids if bid%2==test_parity for ri in bl[bid]]
        if not train_rows or not test_rows:continue
        beta=choose_beta(train_rows,w,cells,p0,rowc,rown,vocab,alpha)
        R,_=learn_ratio(train_rows,w,cells,p0,rowc,rown,vocab,alpha,beta)
        g,n=score_gain(test_rows,w,cells,p0,rowc,rown,vocab,alpha,R)
        if n<MIN_TEST_TOKENS:continue
        actual_num+=g*n;actual_den+=n;positive_halves+=int(g>0)
        half_results.append({'test_parity':test_parity,'beta':beta,'gain':g,'n':n})
        for b in range(NNULL):
            seed=202610300000+(int(hashlib.sha256(name.encode()).hexdigest()[:6],16)%100000)*10000+wi*200+b*2+test_parity
            dm=donor_map(train_rows,w,others,cells,seed)
            Rd=learn_ratio_donors(train_rows,dm,cells,p0,rowc,rown,vocab,alpha,beta)
            gn,nn=score_gain(test_rows,w,cells,p0,rowc,rown,vocab,alpha,Rd)
            null_num[b]+=gn*nn;null_den[b]+=nn
    if actual_den<MIN_TEST_TOKENS:return None
    actual=actual_num/actual_den
    null=np.divide(null_num,null_den,out=np.full(NNULL,np.nan),where=null_den>0)
    return {'witness':w,'alpha':alpha,'effect':actual,'n':actual_den,'positive_halves':positive_halves,
            'null':null.tolist(),'halves':half_results}

def run_work(name,meta,rep):
    t=fetch_table(meta);wits,cells=prepare(t,rep); wr=[]
    for wi,w in enumerate(wits):
        x=run_witness(name,w,wits,cells,len(t),rep,wi)
        if x is not None:wr.append(x)
    if not wr:return {'work':name,'representation':rep,'eligible_witnesses':0}
    actual=float(np.mean([x['effect'] for x in wr]))
    null=np.array([[x['null'][b] for x in wr] for b in range(NNULL)],float)
    null_work=np.nanmean(null,axis=1)
    nm=float(np.mean(null_work));nsd=float(np.std(null_work,ddof=1));z=(actual-nm)/nsd if nsd>0 else float('inf')
    p=(1+int(np.sum(null_work>=actual)))/(NNULL+1)
    wins=sum(x['effect']>0 for x in wr);need=math.ceil(2*len(wr)/3)
    resolved=bool(z>=2 and p<=.05 and wins>=need)
    return {'work':name,'representation':rep,'eligible_witnesses':len(wr),'effect_bits_per_token':actual,
            'null_mean':nm,'null_sd':nsd,'z':z,'p':p,'positive_witnesses':wins,'required_positive':need,
            'resolved':resolved,'witnesses':[dict(x, null=None) for x in wr]}

def main():
    allres={}
    for rep in ('shape','opaque256'):
        allres[rep]={}
        for name,meta in WORKS.items():
            print('START',rep,name,flush=True)
            allres[rep][name]=run_work(name,meta,rep)
            print('DONE',rep,name,json.dumps({k:v for k,v in allres[rep][name].items() if k!='witnesses'},sort_keys=True),flush=True)
    primary=allres['shape'];sens=allres['opaque256']
    nres=sum(bool(x.get('resolved')) for x in primary.values())
    sign_ok=all((primary[w].get('effect_bits_per_token',0)>=0 and sens[w].get('effect_bits_per_token',0)>=0) for w in WORKS)
    passed=bool(nres>=3 and sign_ok)
    out={'production_control_pass':passed,'primary_resolved_works':nres,'sensitivity_no_sign_reversal':sign_ok,'results':allres}
    print('PHASE1C_PRODUCTION_CONTROL_RESULT='+json.dumps(out,separators=(',',':'),sort_keys=True),flush=True)

if __name__=='__main__':main()

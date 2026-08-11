# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import collections, hashlib, json, math, re, urllib.request
import numpy as np
from datasets import load_dataset
from unidecode import unidecode

NS='VBMBGCONDV5'
PLAIN='abcdefghilmnopqrstu'
V=set('aeiou')

URL_GSD_TRAIN='https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu'
URL_GSD_DEV='https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-dev.conllu'
URL_GSD_TEST='https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-test.conllu'
URL_MAIBAAM='https://raw.githubusercontent.com/UniversalDependencies/UD_Bavarian-MaiBaam/master/bar_maibaam-ud-test.conllu'
URL_PUD='https://raw.githubusercontent.com/UniversalDependencies/UD_German-PUD/master/de_pud-ud-test.conllu'

def seed(*x):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,x)).encode()).digest()[:8],'big') & 0x7fffffff

def norm(s):
    s=unidecode(s).lower().replace('j','i').replace('v','u').replace('w','u').replace('y','i').replace('x','s').replace('z','s')
    return ''.join(c for c in s if c in PLAIN)

def cv(s):
    return ''.join('V' if c in V else 'C' for c in norm(s))

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    return urllib.request.urlopen(req,timeout=120).read()

def parse_conllu(data, with_meta=False):
    out=[]; cur=[]; meta={}
    def flush():
        nonlocal cur,meta
        if cur:
            text=''.join(cur)
            if text: out.append((text,dict(meta)) if with_meta else text)
        cur=[];meta={}
    for ln in data.decode('utf-8','replace').splitlines():
        if not ln:
            flush(); continue
        if ln.startswith('#'):
            if with_meta and '=' in ln:
                a,b=ln[1:].split('=',1);meta[a.strip()]=b.strip()
            continue
        cols=ln.split('\t')
        if len(cols)>=2 and cols[0].isdigit():
            z=norm(cols[1])
            if z: cur.append(z)
    flush()
    return out

def load_discovery():
    ds=load_dataset('bavarian-nlp/barwiki-20250720',split='train')
    btr=[];bct=[];tc=cc=0
    for row in ds:
        rid=str(row.get('id',''))
        try:r=int(rid)
        except:r=int.from_bytes(hashlib.sha256(rid.encode()).digest()[:4],'big')
        dest=btr if r%10<6 else bct
        for z in re.split(r'[.!?\n]+',row.get('text','')):
            q=norm(z)
            if len(q)>=20:
                dest.append(q)
                if dest is btr:tc+=len(q)
                else:cc+=len(q)
        if tc>=1800000 and cc>=700000: break
    gtr=parse_conllu(get(URL_GSD_TRAIN));gct=parse_conllu(get(URL_GSD_DEV))+parse_conllu(get(URL_GSD_TEST))
    return {'bavarian':(btr,bct),'german':(gtr,gct)}

def load_transfer():
    mb=parse_conllu(get(URL_MAIBAAM),with_meta=True)
    b=[];bmeta=[]
    for text,meta in mb:
        genre=meta.get('genre','').strip().lower()
        if genre in {'wiki','social'}: continue
        if len(text)>=10:
            b.append(text);bmeta.append(meta)
    g=parse_conllu(get(URL_PUD))
    return b,bmeta,g

def split_controls(seqs,label):
    A=[];B=[]
    for i,s in enumerate(seqs):
        h=seed(NS,'split',label,i,s[:32])&1
        (A if h==0 else B).append(s)
    return A,B

def windows(seqs,n,tag,cap=48):
    flat=''.join(cv(x) for x in seqs if x)
    if len(flat)<n:return []
    off=seed(NS,'window',tag)%n
    z=[]
    for st in range(off,len(flat)-n+1,n):
        z.append(flat[st:st+n])
        if len(z)>=cap:break
    return z

def fit_markov(seqs,order):
    # binary C/V, no boundary token; Jeffreys smoothing
    nctx=1<<order
    C=np.full((nctx,2),0.5,float)
    for raw in seqs:
        s=cv(raw)
        if len(s)<=order:continue
        bits=[1 if x=='V' else 0 for x in s]
        ctx=0
        for b in bits[:order]:ctx=(ctx<<1)|b
        mask=nctx-1
        for y in bits[order:]:
            C[ctx,y]+=1
            ctx=((ctx<<1)&mask)|y
    C/=C.sum(1,keepdims=True)
    return np.log(C)

def score_markov(s,logp,order):
    if len(s)<=order:return 0.0
    bits=[1 if x=='V' else 0 for x in s];nctx=1<<order;mask=nctx-1;ctx=0
    for b in bits[:order]:ctx=(ctx<<1)|b
    ll=0.;n=0
    for y in bits[order:]:
        ll+=float(logp[ctx,y]);n+=1;ctx=((ctx<<1)&mask)|y
    return ll/max(1,n)

def run_counts(seqs,typ):
    C=np.full(8,0.5,float)
    for raw in seqs:
        s=cv(raw);i=0
        while i<len(s):
            t=s[i];j=i+1
            while j<len(s) and s[j]==t:j+=1
            if t==typ:C[min(8,j-i)-1]+=1
            i=j
    C/=C.sum();return np.log(C)

def score_runs(s,typ,logp):
    ll=0.;n=0;i=0
    while i<len(s):
        t=s[i];j=i+1
        while j<len(s) and s[j]==t:j+=1
        if t==typ:
            ll+=float(logp[min(8,j-i)-1]);n+=1
        i=j
    return ll/max(1,n)

def fit_feature_models(train):
    m={}
    for la in ['bavarian','german']:
        m[la]={'markov':{o:fit_markov(train[la],o) for o in range(1,7)},
               'runC':run_counts(train[la],'C'),'runV':run_counts(train[la],'V')}
    return m

def feat(s,m):
    out=[]
    for o in range(1,7):
        out.append(score_markov(s,m['bavarian']['markov'][o],o)-score_markov(s,m['german']['markov'][o],o))
    out.append(score_runs(s,'C',m['bavarian']['runC'])-score_runs(s,'C',m['german']['runC']))
    out.append(score_runs(s,'V',m['bavarian']['runV'])-score_runs(s,'V',m['german']['runV']))
    return np.asarray(out,float)

def ridge_logistic(X,y,lam=1.0,maxit=80):
    X=np.asarray(X,float);y=np.asarray(y,float)
    mu=X.mean(0);sd=X.std(0);sd=np.where(sd<1e-9,1.0,sd);Z=(X-mu)/sd
    D=np.column_stack([np.ones(len(Z)),Z]);w=np.zeros(D.shape[1]);pen=np.eye(D.shape[1])*lam;pen[0,0]=0
    for _ in range(maxit):
        eta=np.clip(D@w,-30,30);p=1/(1+np.exp(-eta));g=D.T@(p-y)+pen@w
        ww=np.maximum(p*(1-p),1e-6);H=D.T@(D*ww[:,None])+pen
        step=np.linalg.solve(H,g);w-=step
        if float(np.max(np.abs(step)))<1e-8:break
    return {'mu':mu,'sd':sd,'w':w}

def logits(X,clf):
    X=np.asarray(X,float);Z=(X-clf['mu'])/clf['sd'];D=np.column_stack([np.ones(len(Z)),Z]);return D@clf['w']

def metrics(lb,lg):
    lb=np.asarray(lb,float);lg=np.asarray(lg,float)
    rb=float(np.mean(lb>0)) if len(lb) else 0.;rg=float(np.mean(lg<0)) if len(lg) else 0.
    return {'bavarian_n':int(len(lb)),'german_n':int(len(lg)),'bavarian_recall':rb,'german_recall':rg,'balanced_accuracy':(rb+rg)/2,
            'bavarian_median_logit':float(np.median(lb)) if len(lb) else None,'german_median_logit':float(np.median(lg)) if len(lg) else None}

def dialect_diagnostic(texts,meta,m,clf):
    by=collections.defaultdict(list)
    for s,md in zip(texts,meta):
        dg=md.get('dialect_group','unk') or 'unk';by[dg].append(s)
    out={}
    for dg,seqs in sorted(by.items()):
        ws=windows(seqs,600,'dialect:'+dg,cap=24)
        if not ws:continue
        ls=logits([feat(x,m) for x in ws],clf)
        out[dg]={'windows':len(ws),'median_logit':float(np.median(ls)),'bavarian_fraction':float(np.mean(ls>0))}
    return out

def main():
    corp=load_discovery();train={la:corp[la][0] for la in corp};models=fit_feature_models(train)
    cal={};val={}
    for la in ['bavarian','german']:
        a,b=split_controls(corp[la][1],la);cal[la]=windows(a,1800,'cal:'+la,48);val[la]=windows(b,1800,'val:'+la,48)
    nfit=min(len(cal['bavarian']),len(cal['german']),48)
    X=[];y=[]
    for s in cal['bavarian'][:nfit]:X.append(feat(s,models));y.append(1)
    for s in cal['german'][:nfit]:X.append(feat(s,models));y.append(0)
    if nfit<8:raise RuntimeError(('insufficient classifier-fit windows',len(cal['bavarian']),len(cal['german'])))
    clf=ridge_logistic(X,y,1.0)
    dvb=logits([feat(s,models) for s in val['bavarian']],clf);dvg=logits([feat(s,models) for s in val['german']],clf);disc=metrics(dvb,dvg)
    bt,bmeta,gt=load_transfer();tbw=windows(bt,1200,'transfer:bavarian',48);tgw=windows(gt,1200,'transfer:german',48)
    tlb=logits([feat(s,models) for s in tbw],clf);tlg=logits([feat(s,models) for s in tgw],clf);trans=metrics(tlb,tlg)
    q0=bool(disc['balanced_accuracy']>=.90 and disc['bavarian_recall']>=.85 and disc['german_recall']>=.85 and trans['balanced_accuracy']>=.80 and trans['bavarian_recall']>=.75 and trans['german_recall']>=.75 and trans['bavarian_n']>=8 and trans['german_n']>=8)
    corr=tlb[tlb>0]
    tau_bg=float(max(0.0,np.quantile(corr,.10,method='linear'))) if len(corr) else None
    out={'namespace':NS,'pass':q0,'discovery':disc,'transfer':trans,'TAU_BG':tau_bg,
         'classifier_fit_windows_per_class':nfit,'feature_count':8,'coefficients':clf['w'].tolist(),'feature_mu':clf['mu'].tolist(),'feature_sd':clf['sd'].tolist(),
         'corpus_chars':{'bavarian_train':sum(map(len,train['bavarian'])),'german_train':sum(map(len,train['german'])),'bavarian_control':sum(map(len,corp['bavarian'][1])),'german_control':sum(map(len,corp['german'][1])),'maibaam_nonwiki':sum(map(len,bt)),'german_pud':sum(map(len,gt))},
         'maibaam_nonwiki_sentences':len(bt),'pud_sentences':len(gt),'dialect_diagnostic':dialect_diagnostic(bt,bmeta,models,clf),
         'transfer_bavarian_logits':tlb.tolist(),'transfer_german_logits':tlg.tolist(),'discovery_bavarian_logits':dvb.tolist(),'discovery_german_logits':dvg.tolist()}
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()

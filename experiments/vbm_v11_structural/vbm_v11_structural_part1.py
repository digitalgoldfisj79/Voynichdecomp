# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "scipy>=1.13,<2", "scikit-learn>=1.5,<2", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import collections, hashlib, json, math, re, urllib.request
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

NS='VBMV11STRUCT20260902'
BRANCH='experiment/vbm-structural-constraints-v11-20260902'
DATA_URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/gpt56/vbm-bridge-factor-v0.2-20260821/voynich_transcriptions_slim.json'
V10_URL=f'https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/{BRANCH}/experiments/vbm_v10_terminal/vbm_v10_stage_a.py'
ATOMS=('ckh','cth','cph','cfh','ch','sh','qo')
H1={'f28v','f31v','f88r','f5r','f34r','f81v'}
C1={'f85r1','f53v','f33r','f10r','f23r','f111r'}
ALPHA=.5
UA={'User-Agent':'VBMV11Structural/2026-09-02'}

def seed(*xs):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,xs)).encode()).digest()[:8],'big') & 0x7fffffff

def get_text(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=120) as r:return r.read().decode('utf-8')

def get_json(url): return json.loads(get_text(url))

def left_half(w):
    for a in ATOMS:
        if len(w)>=len(a)+1 and w.startswith(a): return a
    return w[0]

def parse_token(w):
    if not re.fullmatch(r'[a-z]+',w): return None
    if len(w)==1:return (w,'',w)
    L=left_half(w);R=w[-1]
    if len(w)<len(L)+1:return None
    return (L,w[len(L):-1],R)

def split_folio(fid):
    h=hashlib.sha256(f'VBMJOACHIMEXACTV9Q0::{fid}'.encode()).hexdigest()[:8]
    return 'HOLD' if int(h,16)%5==0 else 'TRAIN'

def half_train(fid):
    h=hashlib.sha256(f'{NS}::HALF::{fid}'.encode()).hexdigest()[:8]
    return 'A' if int(h,16)%2==0 else 'B'

def line_key(x):
    try:return (0,int(x))
    except:return (1,str(x))

def build_corpus(data):
    segments=[]; lines=[]
    for fid,ld in sorted(data['pages'].items()):
        if fid in H1 or fid in C1: continue
        sp=split_folio(fid); hh=half_train(fid) if sp=='TRAIN' else None
        line_segments=[]
        for lno in sorted(ld,key=line_key):
            txt=ld[lno].get('t',{}).get('ZLZI','')
            if not txt:continue
            seg=[]; segs=[]
            def flush():
                nonlocal seg
                if len(seg)>=1:
                    tr=[parse_token(w) for w in seg]
                    br=[tr[i][2]+'|'+tr[i+1][0] for i in range(len(tr)-1)]
                    rec={'folio':fid,'line':str(lno),'split':sp,'half':hh,'words':list(seg),'triples':tr,
                         'nuclei':[t[1] for t in tr],'bridges':br,'start_left':tr[0][0],'end_right':tr[-1][2]}
                    segments.append(rec);segs.append(rec)
                seg=[]
            for w in txt.split():
                if parse_token(w) is None:flush()
                else:seg.append(w)
            flush()
            if segs:
                lines.append({'folio':fid,'line':str(lno),'split':sp,'half':hh,
                              'start_left':segs[0]['start_left'],'end_right':segs[-1]['end_right'],
                              'segments':segs})
    return segments,lines

def bridge_occurrences(segments, which='TRAIN', half=None):
    out=[]
    for s in segments:
        if s['split']!=which:continue
        if half is not None and s['half']!=half:continue
        m=len(s['bridges'])
        for j,b in enumerate(s['bridges']):
            if m==1:pos='SINGLE'
            elif j==0:pos='FIRST'
            elif j==m-1:pos='LAST'
            else:pos='MIDDLE'
            out.append({'folio':s['folio'],'bridge':b,'leftN':s['nuclei'][j] or 'EMPTY',
                        'rightN':s['nuclei'][j+1] or 'EMPTY','pos':pos})
    return out

def synthetic_bridge_occ(lines):
    out=[]
    for li,L in enumerate(lines):
        m=len(L['b'])
        for j,b in enumerate(L['b']):
            if m==1:pos='SINGLE'
            elif j==0:pos='FIRST'
            elif j==m-1:pos='LAST'
            else:pos='MIDDLE'
            ln=L['n'][j];rn=L['n'][j+1]
            out.append({'folio':f'S{li//20}','bridge':int(b),'leftN':'EMPTY' if ln<0 else int(ln),
                        'rightN':'EMPTY' if rn<0 else int(rn),'pos':pos})
    return out

def top_values(occ,key,k,exclude=()):
    c=collections.Counter(o[key] for o in occ if o[key] not in exclude)
    return [x for x,_ in c.most_common(k)]

def context_matrix_bridge(occ, eligible, topn):
    idx={x:i for i,x in enumerate(eligible)}; nb={x:i for i,x in enumerate(topn)}
    K=len(topn)+2
    P={'FIRST':0,'MIDDLE':1,'LAST':2,'SINGLE':3}
    X=np.full((len(eligible),2*K+4),ALPHA,float)
    def binN(x):
        if x=='EMPTY':return len(topn)+1
        return nb.get(x,len(topn))
    for o in occ:
        i=idx.get(o['bridge'])
        if i is None:continue
        X[i,binN(o['leftN'])]+=1
        X[i,K+binN(o['rightN'])]+=1
        X[i,2*K+P[o['pos']]]+=1
    X/=X.sum(1,keepdims=True)
    return X

def js_matrix(X):
    if len(X)<2:return np.zeros((len(X),len(X)))
    return squareform(pdist(X,metric='jensenshannon'))

def cluster_precomputed(D,k):
    if len(D)<=k:return np.arange(len(D))
    return AgglomerativeClustering(n_clusters=k,metric='precomputed',linkage='average').fit_predict(D)

def sil(D,lab):
    if len(set(lab))<2 or len(lab)<=len(set(lab)):return float('nan')
    return float(silhouette_score(D,lab,metric='precomputed'))

def ari_common(types1,lab1,types2,lab2):
    a={t:int(l) for t,l in zip(types1,lab1)};b={t:int(l) for t,l in zip(types2,lab2)}
    common=sorted(set(a)&set(b),key=str)
    if len(common)<5:return float('nan'),len(common)
    return float(adjusted_rand_score([a[t] for t in common],[b[t] for t in common])),len(common)

def shuffled_contexts(occ,tag):
    out=[dict(x) for x in occ]
    groups=collections.defaultdict(list)
    for i,o in enumerate(out):groups[(o['folio'],o['pos'])].append(i)
    rng=np.random.default_rng(seed(NS,'A_NULL',tag))
    for g,ii in groups.items():
        ctx=[(out[i]['leftN'],out[i]['rightN']) for i in ii]
        perm=rng.permutation(len(ii))
        for dst,p in zip(ii,perm):out[dst]['leftN'],out[dst]['rightN']=ctx[int(p)]
    return out

def load_v10():
    src=get_text(V10_URL);ns={'__name__':'v10base'};exec(compile(src,V10_URL,'exec'),ns);return ns

def branch_A(segments,v10):
    aris=[]
    for lang in ['DE','IT']:
        A=v10['assets'](lang)
        for rep in range(3):
            lines,key=v10['make_positive'](lang,rep,A);occ=synthetic_bridge_occ(lines[:2000])
            cnt=collections.Counter(o['bridge'] for o in occ);elig=sorted([b for b,n in cnt.items() if n>=20])
            c=collections.Counter(x for o in occ for x in (o['leftN'],o['rightN']) if x!='EMPTY')
            topn=[x for x,_ in c.most_common(64)]
            X=context_matrix_bridge(occ,elig,topn);lab=cluster_precomputed(js_matrix(X),5)
            true=[int(key['bmap'][int(b)]) for b in elig]
            aris.append(float(adjusted_rand_score(true,lab)))
    qual=(float(np.median(aris))>=.70 and sum(x>=.50 for x in aris)>=4)
    occ=bridge_occurrences(segments,'TRAIN');cnt=collections.Counter(o['bridge'] for o in occ)
    elig=sorted([b for b,n in cnt.items() if n>=20]); c=collections.Counter(x for o in occ for x in (o['leftN'],o['rightN']) if x!='EMPTY')
    topn=[x for x,_ in c.most_common(64)]
    X=context_matrix_bridge(occ,elig,topn);D=js_matrix(X)
    curves={}
    for k in range(2,11):
        if len(elig)>k:curves[k]=sil(D,cluster_precomputed(D,k))
    obs_sil=curves.get(5,float('nan'))
    occA=bridge_occurrences(segments,'TRAIN','A');occB=bridge_occurrences(segments,'TRAIN','B')
    cntA=collections.Counter(o['bridge'] for o in occA);cntB=collections.Counter(o['bridge'] for o in occB)
    eA=sorted([b for b,n in cntA.items() if n>=20]);eB=sorted([b for b,n in cntB.items() if n>=20])
    lA=cluster_precomputed(js_matrix(context_matrix_bridge(occA,eA,topn)),5) if len(eA)>5 else np.arange(len(eA))
    lB=cluster_precomputed(js_matrix(context_matrix_bridge(occB,eB,topn)),5) if len(eB)>5 else np.arange(len(eB))
    obs_ari,ncommon=ari_common(eA,lA,eB,lB)
    nsil=[];nari=[]
    for r in range(100):
        q=shuffled_contexts(occ,r);Xd=context_matrix_bridge(q,elig,topn);Dd=js_matrix(Xd);ld=cluster_precomputed(Dd,5);nsil.append(sil(Dd,ld))
        qA=shuffled_contexts(occA,f'{r}:A');qB=shuffled_contexts(occB,f'{r}:B')
        la=cluster_precomputed(js_matrix(context_matrix_bridge(qA,eA,topn)),5) if len(eA)>5 else np.arange(len(eA))
        lb=cluster_precomputed(js_matrix(context_matrix_bridge(qB,eB,topn)),5) if len(eB)>5 else np.arange(len(eB))
        aa,_=ari_common(eA,la,eB,lb);nari.append(aa)
    p_s=(1+sum(x>=obs_sil for x in nsil))/(len(nsil)+1) if np.isfinite(obs_sil) else 1
    finite_ari=[x for x in nari if np.isfinite(x)]
    p_a=(1+sum(x>=obs_ari for x in finite_ari))/(len(finite_ari)+1) if np.isfinite(obs_ari) else 1
    best=max([v for v in curves.values() if np.isfinite(v)],default=float('-inf'))
    gate=bool(qual and p_s<=.01 and p_a<=.01 and np.isfinite(obs_sil) and obs_sil>=best-.05)
    return {'synthetic_ARI':aris,'synthetic_median_ARI':float(np.median(aris)),'synthetic_qualifies':qual,
            'eligible_bridge_types':len(elig),'silhouette_k2_k10':curves,'observed_k5_silhouette':obs_sil,
            'split_half_ARI':obs_ari,'split_common_types':ncommon,'null_sil_p':p_s,'null_ari_p':p_a,
            'null_sil_p99':float(np.quantile(nsil,.99)),'null_ari_p99':float(np.nanquantile(nari,.99)),
            'gate':gate,'verdict':'A_SUPPORT_FIVE_CONTEXT_CLASSES' if gate else 'A_NO_FIVE_CLASS_EVIDENCE'}

def nucleus_occurrences(segments,half=None):
    out=[]
    for s in segments:
        if s['split']!='TRAIN':continue
        if half is not None and s['half']!=half:continue
        for i,n in enumerate(s['nuclei']):
            if not n:continue
            prev=s['bridges'][i-1] if i>0 else 'EDGE'; nxt=s['bridges'][i] if i<len(s['bridges']) else 'EDGE'
            out.append({'folio':s['folio'],'nucleus':n,'prev':prev,'next':nxt})
    return out

def nucleus_context_matrix(occ,eligible,topb):
    idx={x:i for i,x in enumerate(eligible)};bb={x:i for i,x in enumerate(topb)};K=len(topb)+2
    X=np.full((len(eligible),2*K),ALPHA,float)
    def bn(x):
        if x=='EDGE':return len(topb)+1
        return bb.get(x,len(topb))
    for o in occ:
        i=idx.get(o['nucleus'])
        if i is None:continue
        X[i,bn(o['prev'])]+=1;X[i,K+bn(o['next'])]+=1
    X/=X.sum(1,keepdims=True);return X

def ecount(s):return s.count('e')
def eskel(s):return re.sub(r'e+','E',s)

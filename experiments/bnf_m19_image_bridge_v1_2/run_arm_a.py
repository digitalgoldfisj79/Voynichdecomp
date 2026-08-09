#!/usr/bin/env python3
import os,re,json,math,hashlib
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from unidecode import unidecode
import urllib.request, urllib.parse

SEED0=20260809
ALPH='abcdefghiklmnopqrstuxyz'; N=len(ALPH); A2I={c:i for i,c in enumerate(ALPH)}
TABLES={
'F':[1,2,3,4,5,6,7,8,9,10,10,2,12,22,4,12,24,6,16,4,20,8,24],
'M':[1,2,3,4,5,28,10,12,1,16,2,12,23,6,2,20,3,30,9,1,20,0,4],
'G':[1,2,6,4,5,8,1,6,7,1,8,8,5,6,5,2,2,1,4,1,1,3,3],
'L':[1,2,6,4,1,8,4,3,10,2,3,8,5,6,8,7,2,6,1,6,5,0,7],
'H':[1,2,6,4,5,6,3,1,3,6,2,4,1,6,7,2,8,6,1,6,1,0,7]}
VALUES=sorted(set(sum(TABLES.values(),[]))); NV=len(VALUES); V2I={v:i for i,v in enumerate(VALUES)}
LETTER_VALS=[sorted(set(TABLES[t][i] for t in TABLES)) for i in range(N)]
EMIT=np.zeros((N,NV),float)
for l,vs in enumerate(LETTER_VALS):
    for v in vs: EMIT[l,V2I[v]]=1/len(vs)
LANGS=['latin','italian','german','french','greek','hebrew','arabic','spanish']
QUAL=['latin','italian','german','french','arabic','spanish']
LM_URLS={
'latin':'https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-train.conllu',
'italian':'https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu',
'german':'https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu',
'french':'https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master/fr_gsd-ud-train.conllu',
'greek':'https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-train.conllu',
'hebrew':'https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/master/he_htb-ud-train.conllu',
'arabic':'https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-train.conllu',
'spanish':'https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-train.conllu'}
KLIST=[19,25,31,38]
TRAIN_RES={3,4,8,9}; QUAL_RES={2,7}
QTRAIN=45000; QHOLD=39000
CONTROL_STEPS=8000; CONTROL_RESTARTS=3
VMS_STEPS=16000; VMS_RESTARTS=6
DATA='/data'


def seed(*p):
    h=hashlib.sha256(('::'.join(map(str,p))).encode()).digest()
    return (SEED0+int.from_bytes(h[:8],'big')) & 0xffffffff

def fetch(u):
    req=urllib.request.Request(urllib.parse.quote(u,safe=':/?=&%'),headers={'User-Agent':'M19-image-v12'})
    with urllib.request.urlopen(req,timeout=120) as r:return r.read().decode('utf-8','replace')

def conllu(txt):
    out=[];cur=[]
    for ln in txt.splitlines():
        if not ln:
            if cur: out.append(' '.join(cur)); cur=[]
            continue
        if ln.startswith('#'): continue
        c=ln.split('\t')
        if len(c)>=2 and c[0].isdigit(): cur.append(c[1])
    if cur:out.append(' '.join(cur))
    return out

def norm_words(s):
    s=unidecode(s).lower().replace('j','i').replace('v','u').replace('w','u')
    out=[]
    for w in re.findall(r'[a-z]+',s):
        z=''.join(c for c in w if c in A2I)
        if z: out.append(z)
    return out

def pool_text(ss):return ' '.join(w for s in ss for w in norm_words(s))

def build_lm(ss):
    a=.25;T=np.ones((N,N))*a;st=np.ones(N)*a;en=np.ones(N)*a;uni=np.ones(N)*a;letters=0
    for s in ss:
        for w in norm_words(s):
            q=[A2I[c] for c in w]; letters+=len(q)
            if not q:continue
            st[q[0]]+=1;en[q[-1]]+=1
            for x in q:uni[x]+=1
            for x,y in zip(q,q[1:]):T[x,y]+=1
    T/=T.sum(1,keepdims=True);st/=st.sum();en/=en.sum();uni/=uni.sum()
    return {'T':T,'st':st,'en':en,'uni':uni,'letters':letters}

def load_lms():
    lms={};pools={};meta={}
    for la,u in LM_URLS.items():
        ss=conllu(fetch(u));tr=[s for i,s in enumerate(ss) if i%10 in TRAIN_RES]; qo=[s for i,s in enumerate(ss) if i%10 in QUAL_RES]
        lms[la]=build_lm(tr);pools[la]=pool_text(qo);meta[la]={'train_sentences':len(tr),'qual_sentences':len(qo),'lm_letters':lms[la]['letters'],'qual_letters':sum(c!=' ' for c in pools[la])}
        print('LM',la,json.dumps(meta[la],separators=(',',':')),flush=True)
    return lms,pools,meta

def induced(lm):
    uni,T,st,en=lm['uni'],lm['T'],lm['st'],lm['en']
    start=st@EMIT;start=np.maximum(start,1e-15);start/=start.sum()
    post=uni[:,None]*EMIT; post/=np.maximum(post.sum(0),1e-15)[None,:]
    trans=np.empty((NV,NV))
    for v in range(NV): trans[v]=post[:,v]@T@EMIT
    trans=np.maximum(trans,1e-15);trans/=trans.sum(1,keepdims=True)
    end=np.maximum(post.T@en,1e-15)
    return np.log(trans),np.log(start),np.log(end)

def forward_word_values(obs,lm):
    if not obs:return (0.,0)
    a=lm['st']*EMIT[:,obs[0]];z=float(a.sum())
    if z<=0:return(-1e100,0)
    ll=math.log(z);a/=z
    for v in obs[1:]:
        a=(a@lm['T'])*EMIT[:,v];z=float(a.sum())
        if z<=0:return(-1e100,0)
        ll+=math.log(z);a/=z
    z=float(np.dot(a,lm['en']))
    if z>0:ll+=math.log(z)
    return ll,len(obs)

def forward_sequences(words,m,lm):
    ll=0.;n=0
    for w in words:
        obs=[int(m[x]) for x in w]
        x,k=forward_word_values(obs,lm);ll+=x;n+=k
    return ll/max(1,n),n

def split_text_letters(text,n):
    k=0
    for i,c in enumerate(text):
        if c!=' ':k+=1
        if k==n:return text[:i+1].strip(),text[i+1:].strip()
    raise RuntimeError('short split')
def choose_span(pool,n,tag):
    pos=[i for i,c in enumerate(pool) if c!=' ']
    if len(pos)<n:raise RuntimeError(('pool short',tag,len(pos),n))
    st=seed('span',tag)%(len(pos)-n+1);a=pos[st];b=pos[st+n-1]+1;return pool[a:b].strip()

def legal_map(m,K):
    c=np.bincount(m,minlength=NV)
    return len(m)==K and np.all(c>=1) and np.all(c<=2) and int(np.sum(c==2))==K-NV

def init_map(K,rng):
    d=K-NV; dup=[] if d==0 else list(map(int,rng.choice(NV,d,replace=False)))
    a=np.array(list(range(NV))+dup,dtype=np.int16);rng.shuffle(a);return a

def sym_stats(words,K):
    B=np.zeros((K,K),np.int64);st=np.zeros(K,np.int64);en=np.zeros(K,np.int64);freq=np.zeros(K,np.int64)
    for w in words:
        if not w:continue
        st[w[0]]+=1;en[w[-1]]+=1
        for x in w:freq[x]+=1
        for x,y in zip(w,w[1:]):B[x,y]+=1
    return {'B':B,'st':st,'en':en,'freq':freq,'denom':max(1,int(B.sum()+st.sum()+en.sum()+freq.sum()))}

def approx_score(S,m,comp):
    lt,ls,le=comp;cnt=np.bincount(m,minlength=NV)
    z=float(np.sum(S['B']*lt[np.ix_(m,m)])+np.dot(S['st'],ls[m])+np.dot(S['en'],le[m]))
    z-=float(np.dot(S['freq'],np.log(cnt[m])))
    return z/S['denom']

def optimize(S,comp,K,tag,steps,restarts):
    best=(-1e100,None)
    for rr in range(restarts):
        rng=np.random.default_rng(seed('opt',tag,rr));m=init_map(K,rng);s=approx_score(S,m,comp)
        ds=[]
        for _ in range(40):
            a,b=rng.choice(K,2,replace=False);x=m.copy();x[a],x[b]=x[b],x[a];ds.append(abs(approx_score(S,x,comp)-s))
        t0=max(1e-5,float(np.median(ds))*4)
        for k in range(steps):
            frac=k/max(1,steps-1);temp=max(1e-6,t0*(.01**frac));x=m.copy()
            if rng.random()<.78 or K==NV:
                a,b=rng.choice(K,2,replace=False);x[a],x[b]=x[b],x[a]
            else:
                cnt=np.bincount(m,minlength=NV);src=np.flatnonzero(cnt==2);dst=np.flatnonzero(cnt==1)
                if not len(src) or not len(dst):continue
                sv=int(rng.choice(src));dv=int(rng.choice(dst));ii=int(rng.choice(np.flatnonzero(m==sv)));x[ii]=dv
            s2=approx_score(S,x,comp);d=s2-s
            if d>=0 or rng.random()<math.exp(max(-50,d/temp)):m,s=x,s2
            if s>best[0]:best=(s,m.copy())
        # small deterministic polish
        m=best[1].copy();s=approx_score(S,m,comp)
        for _ in range(5):
            bd=0.;bx=None
            for a in range(K):
                for bb in range(a+1,K):
                    if m[a]==m[bb]:continue
                    x=m.copy();x[a],x[bb]=x[bb],x[a];d=approx_score(S,x,comp)-s
                    if d>bd+1e-12:bd=d;bx=x
            if K>NV:
                cnt=np.bincount(m,minlength=NV)
                for sv in np.flatnonzero(cnt==2):
                    for dv in np.flatnonzero(cnt==1):
                        for ii in np.flatnonzero(m==sv):
                            x=m.copy();x[ii]=dv;d=approx_score(S,x,comp)-s
                            if d>bd+1e-12:bd=d;bx=x
            if bx is None:break
            m=bx;s+=bd
            if s>best[0]:best=(s,m.copy())
    assert legal_map(best[1],K)
    return best

def agreement(freq,m1,m2):return float(np.dot(freq,m1==m2)/max(1,freq.sum()))
def weighted_acc(freq,m,true):return float(np.dot(freq,m==true)/max(1,freq.sum()))

def generate_control(plain,lang,K):
    # exact M19 values; duplicated surfaces go to most frequent training values, then opaque permutation
    for attempt in range(200):
        rng=np.random.default_rng(seed('control-values',lang,K,attempt));vals=[];traincnt=Counter();n=0
        for c in plain:
            if c==' ':vals.append(None);continue
            vi=V2I[int(rng.choice(LETTER_VALS[A2I[c]]))];vals.append(vi)
            if n<QTRAIN:traincnt[vi]+=1
            n+=1
        if len(traincnt)<NV:continue
        dup=[v for v,_ in sorted(traincnt.items(),key=lambda kv:(-kv[1],kv[0]))[:K-NV]]
        raw={v:[v] for v in range(NV)}
        for j,v in enumerate(dup):raw[v].append(NV+j)
        perm=np.arange(K);rng2=np.random.default_rng(seed('control-opaque',lang,K,attempt));rng2.shuffle(perm)
        true=np.full(K,-1,np.int16)
        for v,forms in raw.items():
            for r in forms:true[int(perm[r])]=v
        out=[];used=set();n=0;rng3=np.random.default_rng(seed('control-surface',lang,K,attempt))
        for v in vals:
            if v is None:out.append(' ');continue
            sid=int(perm[int(rng3.choice(raw[v]))]);out.append(chr(0x1000+sid))
            if n<QTRAIN:used.add(sid)
            n+=1
        if len(used)==K:return ''.join(out),true,attempt
    raise RuntimeError(('control generation',lang,K))
def control_words(s,K):
    base=0x1000;return [[ord(c)-base for c in w] for w in s.split() if w]

def qualify(K,lms,pools,comps):
    rows=[]
    for la in QUAL:
        span=choose_span(pools[la],QTRAIN+QHOLD,('image-v12-qual',la,K));ct,true,att=generate_control(span,la,K);tr,ho=split_text_letters(ct,QTRAIN);tw=control_words(tr,K);hw=control_words(ho,K);S=sym_stats(tw,K);SH=sym_stats(hw,K)
        fits={};rank=[]
        for cand in LANGS:
            sc,m=optimize(S,comps[cand],K,('qual',K,la,cand),CONTROL_STEPS,CONTROL_RESTARTS);fw,_=forward_sequences(hw,m,lms[cand]);fits[cand]=m;rank.append((cand,fw))
        rank.sort(key=lambda x:x[1],reverse=True);m=fits[la];_,m2=optimize(S,comps[la],K,('qual2',K,la),CONTROL_STEPS,CONTROL_RESTARTS);acc=weighted_acc(SH['freq'],m,true);agr=agreement(S['freq'],m,m2);margin=rank[0][1]-rank[1][1]
        r={'lang':la,'top':rank[0][0],'margin':margin,'rank':1+next(i for i,x in enumerate(rank) if x[0]==la),'mapping_acc':acc,'agreement':agr,'attempt':att};rows.append(r);print('QUAL',json.dumps(r,separators=(',',':')),flush=True)
    g={'correct':sum(r['top']==r['lang'] for r in rows),'min_margin':min(r['margin'] for r in rows),'median_acc':float(np.median([r['mapping_acc'] for r in rows])),'min_acc':min(r['mapping_acc'] for r in rows),'min_agreement':min(r['agreement'] for r in rows)}
    g['pass']=g['correct']==6 and g['min_margin']>=.05 and g['median_acc']>=.95 and g['min_acc']>=.85 and g['min_agreement']>=.90
    print('QUAL_GATE',json.dumps(g,separators=(',',':')),flush=True);return rows,g

def load_image_data():
    meta=[];sel=[];folios=set();path=os.path.join(DATA,'corpus_crop_manifest.jsonl')
    with open(path) as h:
        for rowi,line in enumerate(h):
            r=json.loads(line)
            if r.get('kind')=='ccmerge' and r.get('view')=='norm' and not r.get('low_conf',False):
                # Explicitly project only permitted fields. EVA/text fields are not retained.
                q=(rowi,r['id'],r['folio'],int(r['word_index']),int(r['slot']),int(r['n_slots']))
                sel.append(q);folios.add(r['folio'])
    idx=np.array([q[0] for q in sel],dtype=np.int64)
    z=np.load(os.path.join(DATA,'corpus_embeddings_full.npz'),allow_pickle=False);ids=z['ids'];checks=np.linspace(0,len(sel)-1,min(1000,len(sel)),dtype=int)
    for j in checks:
        q=sel[j]
        if ids[q[0]]!=q[1]+'::norm':raise RuntimeError(('manifest/vector order mismatch',j,ids[q[0]],q[1]))
    print('VECTOR_ARCHIVE',len(ids),z['vectors'].shape,flush=True)
    V=z['vectors'];X=np.asarray(V[idx],dtype=np.float32);del V,ids,z
    X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    rec={'folio':np.array([q[2] for q in sel],dtype=object),'word':np.array([q[3] for q in sel],np.int32),'slot':np.array([q[4] for q in sel],np.int16),'nslots':np.array([q[5] for q in sel],np.int16)}
    folios=sorted(folios,key=lambda f:hashlib.sha256(('M19IMAGEv12split::'+f).encode()).digest());nt=round(.5*len(folios));nh=round(.2*len(folios));T=folios[:nt];H=folios[nt:nt+nh];C=folios[nt+nh:]
    tv=sorted(T,key=lambda f:hashlib.sha256(('M19IMAGEv12vis::'+f).encode()).digest());cut=round(.8*len(tv));Tf=set(tv[:cut]);Tv=set(tv[cut:])
    split={'T':set(T),'H':set(H),'C':set(C),'Tf':Tf,'Tv':Tv}
    print('IMAGE_CENSUS',json.dumps({'rows':len(sel),'folios':len(folios),'T':len(T),'H':len(H),'C':len(C),'Tfit':len(Tf),'Tvis':len(Tv)},separators=(',',':')),flush=True)
    return X,rec,split

def folio_center(X,rec):
    Y=np.empty_like(X);by=defaultdict(list)
    for i,f in enumerate(rec['folio']):by[f].append(i)
    for f,ii in by.items():
        a=np.array(ii);v=X[a]-X[a].mean(0,keepdims=True);v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12);Y[a]=v
    return Y

def indices_for(rec,F):return np.array([i for i,f in enumerate(rec['folio']) if f in F],dtype=np.int64)
def stable_sample(idx,n,tag):
    if len(idx)<=n:return idx
    rng=np.random.default_rng(seed('sample',tag));return np.sort(rng.choice(idx,n,replace=False))

def assign(A,cent):
    sim=A@cent.T;lab=sim.argmax(1);mx=sim[np.arange(len(A)),lab];return lab,mx

def visual_candidate(X,rec,split,Rname,K):
    tf=indices_for(rec,split['Tf']);tv=indices_for(rec,split['Tv']);fit=stable_sample(tf,80000,(Rname,K,'fit'));vis=stable_sample(tv,10000,(Rname,K,'vis'));silidx=stable_sample(tv,3000,(Rname,K,'sil'))
    cents=[]
    for rs in [408,409]:
        km=MiniBatchKMeans(n_clusters=K,random_state=rs,batch_size=4096,n_init=3,max_iter=180,reassignment_ratio=.005).fit(X[fit]);c=km.cluster_centers_.astype(np.float32);c/=np.maximum(np.linalg.norm(c,axis=1,keepdims=True),1e-12);cents.append(c)
    c0,c1=cents;row,col=linear_sum_assignment(-(c0@c1.T));map1=np.zeros(K,dtype=int);map1[col]=row
    l0,_=assign(X[vis],c0);l1,_=assign(X[vis],c1);stab=float(np.mean(l0==map1[l1]))
    lt,st=assign(X[fit],c0);thr=np.array([np.quantile(st[lt==k],.05) if np.any(lt==k) else 1. for k in range(K)])
    lv,sv=assign(X[tv],c0);accept=sv>=thr[lv];cov=float(accept.mean())
    # recurrence/counts on accepted Tvis
    counts=np.bincount(lv[accept],minlength=K);fsets=[set() for _ in range(K)]
    for lab,ok,i in zip(lv,accept,tv):
        if ok:fsets[int(lab)].add(rec['folio'][i])
    recmin=min(map(len,fsets));cntmin=int(counts.min())
    ls,_=assign(X[silidx],c0);sil=float(silhouette_score(X[silidx],ls,metric='cosine')) if len(set(ls))>1 else -1.
    passed=stab>=.75 and cov>=.75 and recmin>=3 and cntmin>=25
    r={'R':Rname,'K':K,'stability':stab,'coverage':cov,'min_cluster_folios':recmin,'min_cluster_count':cntmin,'silhouette':sil,'pass':passed};print('VISUAL',json.dumps(r,separators=(',',':')),flush=True);return r

def choose_visual(cands):
    good=[r for r in cands if r['pass']]
    pool=good or cands;mx=max(r['silhouette'] for r in pool);near=[r for r in pool if mx-r['silhouette']<=.005];near.sort(key=lambda r:(r['K'],0 if r['R']=='R1' else 1));ch=near[0].copy();ch['image_gate_pass']=bool(good);print('VISUAL_CHOICE',json.dumps(ch,separators=(',',':')),flush=True);return ch

def refit_centroids(X,rec,split,K,R):
    ti=indices_for(rec,split['T']);fit=stable_sample(ti,160000,(R,K,'refit'));km=MiniBatchKMeans(n_clusters=K,random_state=408,batch_size=4096,n_init=5,max_iter=220,reassignment_ratio=.005).fit(X[fit]);c=km.cluster_centers_.astype(np.float32);c/=np.maximum(np.linalg.norm(c,axis=1,keepdims=True),1e-12);l,s=assign(X[ti],c);thr=np.array([np.quantile(s[l==k],.05) if np.any(l==k) else 1. for k in range(K)]);return c,thr

def make_words(rec,labels,sims,thr,F):
    by=defaultdict(list);total=acc=0
    for i,(f,w,sl) in enumerate(zip(rec['folio'],rec['word'],rec['slot'])):
        if f not in F:continue
        total+=1;lab=int(labels[i]);ok=bool(sims[i]>=thr[lab]);by[(f,int(w))].append((int(sl),lab if ok else -1));acc+=int(ok)
    words=[]
    for key,a in sorted(by.items()):
        a.sort();run=[]
        for _,x in a:
            if x<0:
                if run:words.append(run);run=[]
            else:run.append(x)
        if run:words.append(run)
    return words,acc/max(1,total),total

def fit_voynich(Tw,Hw,K,lms,comps):
    S=sym_stats(Tw,K);rows=[];maps={}
    for la in LANGS:
        s1,m1=optimize(S,comps[la],K,('VMS',la,'fit1'),VMS_STEPS,VMS_RESTARTS);s2,m2=optimize(S,comps[la],K,('VMS',la,'fit2'),VMS_STEPS,VMS_RESTARTS);m=m1 if s1>=s2 else m2;fw,n=forward_sequences(Hw,m,lms[la]);agr=agreement(S['freq'],m1,m2);r={'lang':la,'Hscore':fw,'agreement':agr,'train_score':max(s1,s2)};rows.append(r);maps[la]=m;print('H12_LANG',json.dumps(r,separators=(',',':')),flush=True)
    rank=sorted(rows,key=lambda r:r['Hscore'],reverse=True);margin=rank[0]['Hscore']-rank[1]['Hscore'];return rows,rank,maps,margin

def c_buckets(rec,C):
    out=[set() for _ in range(4)]
    for f in C:out[hashlib.sha256(('M19IMAGEv12bucket::'+f).encode()).digest()[0]%4].add(f)
    return out

def main():
    lms,pools,lmmeta=load_lms();comps={la:induced(lms[la]) for la in LANGS}
    X,rec,split=load_image_data();Xc=folio_center(X,rec)
    cands=[]
    for R,A in [('R0',X),('R1',Xc)]:
        for K in KLIST:cands.append(visual_candidate(A,rec,split,R,K))
    choice=choose_visual(cands);K=choice['K'];R=choice['R'];A=Xc if R=='R1' else X
    out={'protocol':'v1.2-armA','lm_meta':lmmeta,'visual_candidates':cands,'visual_choice':choice}
    if not choice['image_gate_pass']:
        out['verdict']='ARM A IMAGE-UNDERPOWERED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    qrows,qgate=qualify(K,lms,pools,comps);out['qualification']=qrows;out['qualification_gate']=qgate
    if not qgate['pass']:
        out['verdict']='IMAGE INSTRUMENT NOT QUALIFIED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    cent,thr=refit_centroids(A,rec,split,K,R);lab,sim=assign(A,cent);Tw,Tcov,Tn=make_words(rec,lab,sim,thr,split['T']);Hw,Hcov,Hn=make_words(rec,lab,sim,thr,split['H']);out['stream']={'T_words':len(Tw),'T_units':sum(map(len,Tw)),'T_coverage':Tcov,'H_words':len(Hw),'H_units':sum(map(len,Hw)),'H_coverage':Hcov,'K':K,'R':R}
    print('STREAM',json.dumps(out['stream'],separators=(',',':')),flush=True)
    rows,rank,maps,margin=fit_voynich(Tw,Hw,K,lms,comps);top=rank[0];primary=top['agreement']>=.90 and margin>=.05 and Hcov>=.90
    signal={'top':top['lang'],'top_score':top['Hscore'],'second':rank[1]['lang'],'second_score':rank[1]['Hscore'],'margin':margin,'agreement':top['agreement'],'Hcoverage':Hcov,'primary':primary};out['H12']=rows;out['signal']=signal;print('H12_SIGNAL',json.dumps(signal,separators=(',',':')),flush=True)
    if not primary:
        out['verdict']='NO IMAGE-M19 SIGNAL';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    # C12 unlocked only here. Fixed centroids and fixed T12 map; no refitting.
    Cw,Ccov,Cn=make_words(rec,lab,sim,thr,split['C']);cand=top['lang'];m=maps[cand];cr=[]
    for la in LANGS:
        sc,n=forward_sequences(Cw,m,lms[la]);cr.append((la,sc))
    cr.sort(key=lambda x:x[1],reverse=True);cm=cr[0][1]-cr[1][1] if cr[0][0]==cand else None
    buckets=[]
    for bi,B in enumerate(c_buckets(rec,split['C'])):
        Bw,bc,_=make_words(rec,lab,sim,thr,B);rr=[]
        for la in LANGS:rr.append((la,forward_sequences(Bw,m,lms[la])[0]))
        rr.sort(key=lambda x:x[1],reverse=True);marg=rr[0][1]-rr[1][1] if rr[0][0]==cand else -(next(x[1] for x in rr if x[0]==rr[0][0])-next(x[1] for x in rr if x[0]==cand));buckets.append({'bucket':bi,'folios':len(B),'units':sum(map(len,Bw)),'coverage':bc,'ranking':rr,'candidate_margin':marg})
    confirmed=cr[0][0]==cand and cm is not None and cm>=.05 and Ccov>=.90 and all(b['candidate_margin']>0 for b in buckets)
    out['C12']={'coverage':Ccov,'words':len(Cw),'units':sum(map(len,Cw)),'ranking':cr,'candidate':cand,'margin':cm,'buckets':buckets,'confirmed':confirmed}
    print('C12',json.dumps(out['C12'],separators=(',',':')),flush=True)
    out['winning_map']={str(i):VALUES[int(m[i])] for i in range(K)}
    out['verdict']=('CONFIRMED IMAGE-M19 SIGNAL '+cand) if confirmed else 'H12 IMAGE-M19 CANDIDATE / C12 FAILED'
    print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__':main()

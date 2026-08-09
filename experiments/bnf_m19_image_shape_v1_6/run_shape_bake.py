#!/usr/bin/env python3
import os,json,hashlib,math
import cv2
import numpy as np
from skimage.feature import hog
from skimage.morphology import skeletonize
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment

BASE=os.environ.get('VDINO_DATA','/data/crops/crop_shard_000')
MAN=os.path.join(BASE,'crop_manifest.jsonl')
KS=[19,25,31,38]
REPS=['T','H','R','HT','RT','HRT']

def hkey(s):return hashlib.sha256(s.encode()).digest()

def canvas48(img):
    if img is None: return None,None,None
    if img.ndim==3: img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # choose darker-than-background as ink unless image already dark-background
    if float(img.mean())>127: ink=(img<128).astype(np.uint8)
    else: ink=(img>127).astype(np.uint8)
    ys,xs=np.where(ink>0)
    if len(xs)==0:return np.zeros((48,48),np.uint8),1,1
    crop=ink[ys.min():ys.max()+1,xs.min():xs.max()+1]
    h,w=crop.shape
    sc=min(44/max(w,1),44/max(h,1))
    nw=max(1,int(round(w*sc)));nh=max(1,int(round(h*sc)))
    z=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_AREA)
    z=(z>0.35).astype(np.uint8)
    can=np.zeros((48,48),np.uint8)
    y=(48-nh)//2;x=(48-nw)//2;can[y:y+nh,x:x+nw]=z
    return can,h,w

def topo(can,orig_h,orig_w):
    b=can.astype(np.uint8)
    area=float(b.sum()); eps=1e-6
    ys,xs=np.where(b>0)
    if area<=0:
        return np.zeros(16,np.float32)
    # components
    ncc,_=cv2.connectedComponents(b,connectivity=8); ncc=max(0,ncc-1)
    # contours, holes, perimeter, solidity
    cs,hier=cv2.findContours((b*255),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    per=sum(cv2.arcLength(c,True) for c in cs)
    holes=0
    if hier is not None:
        holes=int(sum(1 for x in hier[0] if x[3]>=0))
    pts=np.column_stack([xs,ys]).astype(np.int32)
    if len(pts)>=3:
        hull=cv2.convexHull(pts.reshape(-1,1,2)); hull_area=max(float(cv2.contourArea(hull)),eps)
    else:hull_area=max(area,eps)
    solidity=min(1.0,area/hull_area)
    # eccentricity from coordinate covariance
    if len(xs)>2:
        C=np.cov(np.vstack([xs,ys])); vals=np.linalg.eigvalsh(C); vals=np.maximum(vals,0)
        ecc=math.sqrt(max(0.0,1.0-(vals[0]+eps)/(vals[1]+eps))) if vals[1]>0 else 0.0
    else:ecc=0.0
    sk=skeletonize(b>0)
    sklen=float(sk.sum())
    # neighbour degree in 8-neighbourhood
    nb=cv2.filter2D(sk.astype(np.uint8),-1,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT)-sk.astype(np.uint8)
    endpoints=float(np.logical_and(sk,nb==1).sum())
    junctions=float(np.logical_and(sk,nb>=3).sum())
    cx=float(xs.mean()/47.0);cy=float(ys.mean()/47.0)
    mx2=float(np.mean(((xs-xs.mean())/47.0)**2));my2=float(np.mean(((ys-ys.mean())/47.0)**2))
    extent=area/(max(1,(xs.max()-xs.min()+1))*max(1,(ys.max()-ys.min()+1)))
    euler=float(ncc-holes)
    return np.array([
        math.log((orig_w+1)/(orig_h+1)), area/(48*48), extent,
        per/(area+eps), solidity, ecc, euler, float(holes), float(ncc),
        sklen/(area+eps), endpoints, junctions, cx,cy,mx2,my2
    ],np.float32)

def load_rows():
    rows=[]
    with open(MAN,encoding='utf-8') as h:
        for line in h:
            r=json.loads(line)
            if r.get('kind')=='ccmerge' and r.get('view')=='norm' and not r.get('low_conf',False):rows.append(r)
    fols=sorted({r['folio'] for r in rows},key=lambda f:hkey('M19IMAGEv16shape::'+f))
    train=set(fols[:4]);test=set(fols[4:])
    keep=[]
    for f in fols:
        rr=[r for r in rows if r['folio']==f]
        rr=sorted(rr,key=lambda r:hkey('M19IMAGEv16unit::'+r['id']))[:750]
        keep.extend(rr)
    print('S0_SPLIT',json.dumps({'folios':fols,'train':sorted(train),'test':sorted(test),'n_total':len(rows),'n_keep':len(keep)},separators=(',',':')),flush=True)
    return keep,train,test

def fit_blocks(rows):
    T=[];H=[];R=[];F=[];I=[]
    missing=0
    for j,r in enumerate(rows):
        p=os.path.join(BASE,r['path']);img=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
        can,oh,ow=canvas48(img)
        if can is None:
            missing+=1;continue
        T.append(topo(can,oh,ow))
        H.append(hog(can.astype(np.float32),orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',feature_vector=True).astype(np.float32))
        sm=cv2.resize(can,(24,24),interpolation=cv2.INTER_AREA).astype(np.float32)
        R.append(sm.ravel())
        F.append(r['folio']);I.append(r['id'])
        if (j+1)%1000==0:print('FEATURES',j+1,flush=True)
    print('FEATURE_CENSUS',json.dumps({'n':len(F),'missing':missing,'Tdim':len(T[0]),'Hdim':len(H[0]),'Rdim':len(R[0])},separators=(',',':')),flush=True)
    return np.asarray(T,np.float32),np.asarray(H,np.float32),np.asarray(R,np.float32),np.asarray(F,object),I

def rep_mats(T,H,R,F,train):
    tr=np.flatnonzero(np.isin(F,list(train))); te=np.flatnonzero(~np.isin(F,list(train)))
    st=StandardScaler().fit(T[tr]); Tz=st.transform(T).astype(np.float32); Tn=normalize(Tz).astype(np.float32)
    sh=StandardScaler().fit(H[tr]);Hz=sh.transform(H).astype(np.float32)
    dh=min(32,Hz.shape[1],len(tr)-1);ph=PCA(dh,whiten=True,random_state=408,svd_solver='randomized').fit(Hz[tr]);Hp=ph.transform(Hz).astype(np.float32);Hn=normalize(Hp).astype(np.float32)
    sr=StandardScaler().fit(R[tr]);Rz=sr.transform(R).astype(np.float32)
    dr=min(32,Rz.shape[1],len(tr)-1);pr=PCA(dr,whiten=True,random_state=408,svd_solver='randomized').fit(Rz[tr]);Rp=pr.transform(Rz).astype(np.float32);Rn=normalize(Rp).astype(np.float32)
    HT=normalize(np.c_[Hp,Tz]).astype(np.float32);RT=normalize(np.c_[Rp,Tz]).astype(np.float32)
    raw=np.c_[Hp,Rp,Tz].astype(np.float32);d=min(48,raw.shape[1],len(tr)-1);p=PCA(d,whiten=True,random_state=408,svd_solver='randomized').fit(raw[tr]);HRT=normalize(p.transform(raw)).astype(np.float32)
    return {'T':Tn,'H':Hn,'R':Rn,'HT':HT,'RT':RT,'HRT':HRT},tr,te

def eval_rep(name,X,tr,te,F):
    out=[]
    for K in KS:
        cents=[]
        labs=[]
        for seed in [408,409]:
            km=MiniBatchKMeans(n_clusters=K,random_state=seed,batch_size=1024,n_init=5,max_iter=200).fit(X[tr])
            c=normalize(km.cluster_centers_).astype(np.float32);cents.append(c);labs.append((X[te]@c.T).argmax(1))
        c0,c1=cents;rr,cc=linear_sum_assignment(-(c0@c1.T));mp=np.empty(K,np.int32);mp[cc]=rr
        l0,l1=labs;stable=float(np.mean(l0==mp[l1]))
        counts=np.bincount(l0,minlength=K);min_count=int(counts.min())
        min_fol=999
        for k in range(K):min_fol=min(min_fol,len(set(F[te][l0==k].tolist())))
        rng=np.random.default_rng(408);samp=te if len(te)<=3000 else rng.choice(te,3000,replace=False);ls=(X[samp]@c0.T).argmax(1)
        try:sil=float(silhouette_score(X[samp],ls,metric='cosine'))
        except Exception:sil=float('nan')
        passed=stable>=.75 and min_count>=20 and min_fol>=2 and sil>=.08
        r={'rep':name,'K':K,'stability':stable,'min_cluster_count':min_count,'min_cluster_folios':min_fol,'silhouette':sil,'pass':bool(passed)};out.append(r)
        print('S0',json.dumps(r,separators=(',',':')),flush=True)
    return out

def main():
    rows,train,test=load_rows();T,H,R,F,I=fit_blocks(rows);mats,tr,te=rep_mats(T,H,R,F,train)
    allr=[]
    for name in REPS:allr+=eval_rep(name,mats[name],tr,te,F)
    passing=[r for r in allr if r['pass']]
    order={x:i for i,x in enumerate(REPS)}
    if passing:
        # maximum stability; near-tie handling is reported but deterministic sorting approximates frozen rule
        best=sorted(passing,key=lambda r:(-r['stability'],-r['silhouette'],r['K'],order[r['rep']]))[0]
        verdict='S0 PASS — FULL-CORPUS REGENERATION AUTHORIZED'
    else:
        best=max(allr,key=lambda r:(r['stability'],r['silhouette']));verdict='SHAPE FEATURES DO NOT STABILIZE M19 SURFACE STATES'
    result={'protocol':'v1.6-S0','best':best,'n_pass':len(passing),'verdict':verdict,'all':allr}
    print('RESULT_JSON='+json.dumps(result,separators=(',',':')),flush=True)
if __name__=='__main__':main()

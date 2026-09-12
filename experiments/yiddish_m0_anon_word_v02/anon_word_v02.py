import cv2, numpy as np, re
from pathlib import Path
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from skimage.feature import hog

# Frozen v0.2 segmentation constants.
X0,X1=500,3150
ROW_SMOOTH=15
ROW_THR=220
LINE_H_MIN=60
LINE_H_MAX=120
WORD_GAP=25
PUNCT_H=55
Y_LO=0.03
Y_HI=0.97
TAU=0.410


def read_bw(path):
    im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if im is None: raise FileNotFoundError(path)
    _,bw=cv2.threshold(im,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return im,bw


def segment_words(path_or_im):
    if isinstance(path_or_im,(str,Path)):
        im,bw=read_bw(path_or_im)
    else:
        im=path_or_im
        _,bw=cv2.threshold(im,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    prof=bw[:,X0:X1].sum(axis=1).astype(float)
    sm=np.convolve(prof,np.ones(ROW_SMOOTH)/ROW_SMOOTH,mode='same')
    mask=sm>ROW_THR
    lines=[]; st=None
    for i,v in enumerate(mask):
        if v and st is None: st=i
        if st is not None and (not v or i==len(mask)-1):
            en=i if not v else i+1
            h=en-st
            if LINE_H_MIN<=h<=LINE_H_MAX and int(Y_LO*bw.shape[0])<st<int(Y_HI*bw.shape[0]):
                crop=bw[max(st-5,0):min(en+5,bw.shape[0]),X0:X1]
                idx=np.where(crop.sum(axis=0)>1)[0]
                if len(idx):
                    lines.append((X0+int(idx[0]),max(st-5,0),X0+int(idx[-1])+1,min(en+5,bw.shape[0])))
            st=None
    words=[]
    for li,(x1,y1,x2,y2) in enumerate(lines):
        crop=bw[y1:y2,x1:x2]
        ink=crop.sum(axis=0)>1
        gaps=[]; s=None
        for j,v in enumerate(~ink):
            if v and s is None: s=j
            if s is not None and (not v or j==len(ink)-1):
                e=j if not v else j+1
                if e-s>WORD_GAP: gaps.append((s,e))
                s=None
        cuts=[0]+[(a+b)//2 for a,b in gaps]+[crop.shape[1]]
        line_words=[]
        for a,b in zip(cuts[:-1],cuts[1:]):
            sub=crop[:,a:b]
            yy,xx=np.where(sub>0)
            if len(xx)<5: continue
            xa,xb=int(xx.min()),int(xx.max()+1)
            ya,yb=int(yy.min()),int(yy.max()+1)
            if yb-ya<PUNCT_H: continue
            line_words.append((x1+a+xa,y1+ya,x1+a+xb,y1+yb))
        line_words.sort(key=lambda z:z[0],reverse=True)
        for wi,b in enumerate(line_words): words.append((li,wi,b))
    return im,bw,words


def raw_descriptor(im,bw,box):
    x1,y1,x2,y2=box
    crop=bw[y1:y2,x1:x2].astype(np.uint8)
    h,w=crop.shape
    sc=48/max(h,1); nw=max(1,int(round(w*sc))); nh=48
    if nw>240:
        sc=240/max(w,1); nw=240; nh=max(1,int(round(h*sc)))
    r=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_AREA if sc<1 else cv2.INTER_NEAREST)
    canvas=np.zeros((64,256),np.float32)
    yy=(64-nh)//2; xx=(256-nw)//2
    canvas[yy:yy+nh,xx:xx+nw]=r
    canvas=cv2.GaussianBlur(canvas,(3,3),0)
    hv=hog(canvas,orientations=8,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',feature_vector=True)
    aspect=np.log(max(w,1)/max(h,1))
    ink=float(crop.mean())
    return hv.astype(np.float32),np.array([aspect,ink],np.float32)


def page_raw(path_or_im):
    im,bw,words=segment_words(path_or_im)
    H=[]; S=[]
    for _,_,b in words:
        h,s=raw_descriptor(im,bw,b); H.append(h); S.append(s)
    return np.array(H),np.array(S),words


def fit_scaler(S):
    mu=S.mean(0); sd=S.std(0); sd[sd==0]=1
    return mu,sd


def combine(H,S,mu,sd):
    Sz=(S-mu)/sd
    X=np.hstack([H,2.0*Sz])
    n=np.linalg.norm(X,axis=1,keepdims=True); n[n==0]=1
    return X/n


def cluster_labels(X,tau=TAU):
    if len(X)==0:return np.array([],int)
    if len(X)==1:return np.array([0],int)
    model=AgglomerativeClustering(n_clusters=None,distance_threshold=tau,metric='cosine',linkage='complete',compute_full_tree=True)
    return model.fit_predict(X)


def stats_from_labels(labels):
    N=len(labels); c=Counter(labels); freqs=list(c.values()); T=len(freqs); V1=sum(v==1 for v in freqs)
    ep=sum(v*(v-1)//2 for v in freqs)/(N*(N-1)/2) if N>1 else 0
    return {'N':N,'T':T,'ttr':T/N if N else 0,'hapax_frac':V1/N if N else 0,'repeat_frac':(N-V1)/N if N else 0,'max_frac':max(freqs)/N if N else 0,'equal_pair_density':ep,'freqs':sorted(freqs,reverse=True)}


def penn_stanzas(path,wanted):
    text=open(path,encoding='utf-8').read(); blocks=re.split(r'\n\s*\n',text)
    out=[]
    for b in blocks:
        m=re.search(r'\(ID 1507W-BOVO,([^.)]+)\.(\d+)\)',b)
        if not m or m.group(1) not in {str(x) for x in wanted}: continue
        pairs=re.findall(r'\(([A-Z][A-Z0-9$+\-_=]*)\s+([^()\s]+)\)',b)
        toks=[]
        for tag,tok in pairs:
            if tag in ('ID','CODE') or tok.startswith('*') or tok=='0' or tok.startswith('{'): continue
            toks.append(tok)
        cur=''
        for t in toks:
            if cur: cur+=t.lstrip('@')
            else: cur=t.lstrip('@')
            if t.endswith('@'):
                cur=cur.rstrip('@'); continue
            out.append(cur.rstrip('@')); cur=''
        if cur: out.append(cur)
    return out


def stats_words(words):
    c=Counter(words); N=len(words); freqs=list(c.values()); T=len(c); V1=sum(v==1 for v in freqs)
    ep=sum(v*(v-1)//2 for v in freqs)/(N*(N-1)/2) if N>1 else 0
    return {'N':N,'T':T,'ttr':T/N,'hapax_frac':V1/N,'repeat_frac':(N-V1)/N,'max_frac':max(freqs)/N,'equal_pair_density':ep,'freqs':sorted(freqs,reverse=True)}

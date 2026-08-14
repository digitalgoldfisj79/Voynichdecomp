#!/usr/bin/env python3
import hashlib
import numpy as np

ARCHITECTURES=("OCCURRENCE","TYPE_LOCKED_RANDOM","TYPE_HASH","LINE_LOCKED","RUN4_LOCKED","DOC_LOCKED")
ENTROPIES=(2,3,4,"ALL")

def _u01(*parts):
    s="|".join(map(str,parts)).encode("utf-8")
    x=int(hashlib.sha256(s).hexdigest()[:14],16)
    return x / float(16**14)

def _offset(u,n,K):
    if n<2:return 0
    if K=="ALL":return min(n-1,int(u*n))
    K=int(K);j=min(K-1,int(u*K))
    return min(n-1,(j*n)//K)

def _rotate(w,k):
    if len(w)<2 or k%len(w)==0:return w
    k%=len(w);return w[k:]+w[:k]

def transform_origin(cipher_lines,plain_lines,architecture,K,seed):
    assert len(cipher_lines)==len(plain_lines)
    out=[];global_pos=0
    doc_u=_u01("P3",architecture,seed,"DOC") if architecture=="DOC_LOCKED" else None
    for li,(cline,pline) in enumerate(zip(cipher_lines,plain_lines)):
        assert len(cline)==len(pline);q=[]
        line_u=_u01("P3",architecture,seed,"LINE",li) if architecture=="LINE_LOCKED" else None
        for wi,(cw,pw) in enumerate(zip(cline,pline)):
            if architecture=="OCCURRENCE":u=_u01("P3",architecture,seed,"TOK",li,wi)
            elif architecture=="TYPE_LOCKED_RANDOM":u=_u01("P3",architecture,seed,"TYPE",pw)
            elif architecture=="TYPE_HASH":u=_u01("P3-TYPEHASH",pw)
            elif architecture=="LINE_LOCKED":u=line_u
            elif architecture=="RUN4_LOCKED":u=_u01("P3",architecture,seed,"RUN4",global_pos//4)
            elif architecture=="DOC_LOCKED":u=doc_u
            else:raise ValueError(architecture)
            q.append(_rotate(cw,_offset(u,len(cw),K)));global_pos+=1
        out.append(q)
    return out

def full_shuffle(lines,seed):
    rng=np.random.default_rng(seed);out=[]
    for line in lines:
        q=[]
        for w in line:
            if len(w)<2:q.append(w);continue
            idx=rng.permutation(len(w));q.append("".join(w[i] for i in idx))
        out.append(q)
    return out

def self_tests():
    p=[["alpha","beta","alpha"],["gamma","beta"]];c=[["ABCDE","WXYZ","ABCDE"],["QRSTU","WXYZ"]]
    for a in ARCHITECTURES:
        for K in ENTROPIES:
            x=transform_origin(c,p,a,K,123)
            assert [len(w) for line in x for w in line]==[len(w) for line in c for w in line]
            for old,new in zip((w for l in c for w in l),(w for l in x for w in l)):assert sorted(old)==sorted(new)
    assert transform_origin(c,p,"TYPE_HASH",4,1)==transform_origin(c,p,"TYPE_HASH",4,999)
    x=transform_origin(c,p,"TYPE_LOCKED_RANDOM","ALL",123);assert x[0][0]==x[0][2]
    return True

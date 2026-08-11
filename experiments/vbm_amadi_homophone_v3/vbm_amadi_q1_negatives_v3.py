# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import collections,concurrent.futures,json,sys
import numpy as np
sys.path.insert(0,'experiments/amadi_residuals_v1');sys.path.insert(0,'experiments/amadi_expanded_vbm_v1');sys.path.insert(0,'experiments/vbm_hmm_v2')
import amadi_residuals_v1 as ar
import vbm_typed_v1 as vt
import vbm_hmm_moment_v2 as m
b=m.b
NS='VBMAMADIV3NEG'; b.NS=NS
FLOORS={'bavarian':0.03265678670240173,'german':0.08360344975587261,'italian':0.15600209198750553}
KINDS=['iid','markov','motif','copy','slot']

def seed(*x):return b.seed(NS,*x)

def target_profile():
    pages,_=ar.parse_rf();T,H,_,_,_=ar.target_split(pages);FIT=T+H;lines,_,core,bridges,geom=vt.target_geometry();seqs,meta=vt.target_sequences(lines,FIT,core,bridges);flat,_=b.flatten(seqs);freq=np.bincount(flat,minlength=b.NOBS).astype(float);freq+=0.05;freq/=freq.sum();lengths=np.array([len(q) for q in seqs if len(q)>=4],int)
    C=np.full((b.NOBS,b.NOBS),0.05,float)
    for q in seqs:
        for x,y in zip(q,q[1:]):C[int(x),int(y)]+=1
    C/=C.sum(1,keepdims=True)
    pc=freq[:b.KCORE].copy();pc/=pc.sum();pv=freq[b.KCORE:].copy();pv/=pv.sum()
    return freq,lengths,C,pc,pv,meta

def split_lengths(rng,lengths,total):
    out=[];n=0
    while n<total:
        L=int(lengths[int(rng.integers(0,len(lengths)))]);L=max(4,min(120,L));out.append(L);n+=L
    return out

def gen_one(kind,rep,total,freq,lengths,C,pc,pv,phase):
    rng=np.random.default_rng(seed('gen',kind,rep,phase));lens=split_lengths(rng,lengths,total);seqs=[]
    rowperm=np.arange(b.NOBS)
    if kind=='markov':
        q=np.arange(b.KCORE);rng.shuffle(q);rowperm[:b.KCORE]=q
        q=np.arange(b.KCORE,b.NOBS);rng.shuffle(q);rowperm[b.KCORE:]=q
    motif=None;copyblock=None
    if kind=='motif':
        M=int(rng.integers(7,19));motif=rng.choice(b.NOBS,size=M,p=freq).astype(int)
    if kind=='copy':copyblock=rng.choice(b.NOBS,size=192,p=freq).astype(int)
    for si,L in enumerate(lens):
        out=[]
        if kind=='iid':out=list(map(int,rng.choice(b.NOBS,size=L,p=freq)))
        elif kind=='markov':
            x=int(rng.choice(b.NOBS,p=freq));out=[x]
            for j in range(1,L):
                p=C[int(rowperm[x])];x=int(rng.choice(b.NOBS,p=p));out.append(x)
        elif kind=='motif':
            for j in range(L):
                x=int(motif[j%len(motif)])
                if rng.random()<.18:x=int(rng.choice(b.NOBS,p=freq))
                out.append(x)
        elif kind=='copy':
            st=int(rng.integers(0,len(copyblock)))
            for j in range(L):
                x=int(copyblock[(st+j)%len(copyblock)])
                if rng.random()<.14:x=int(rng.choice(b.NOBS,p=freq))
                out.append(x)
        elif kind=='slot':
            ph=int(rng.integers(0,4))
            for j in range(L):
                if (j+ph)%4==3:x=b.KCORE+int(rng.choice(b.KBR,p=pv))
                else:x=int(rng.choice(b.KCORE,p=pc))
                out.append(x)
        seqs.append(np.array(out,np.int32))
    return seqs

def eval_negative(lms,profile,kind,rep):
    freq,lengths,C,pc,pv,_=profile;fit=gen_one(kind,rep,18000,freq,lengths,C,pc,pv,'F');hold=gen_one(kind,rep,7000,freq,lengths,C,pc,pv,'H');cand=[]
    for la,lm in lms.items():
        r=m.paired_fit_moment(fit,hold,lm,f'NEG:{kind}:{rep}:{la}',None,40);cand.append({'language':la,'score_A':r['A_eval']['score'],'score_B':r['B_eval']['score'],'score_mean':r['score'],'score_gap':r['score_gap']})
    meanrank=sorted(cand,key=lambda x:(-x['score_mean'],x['language']));Arank=sorted(cand,key=lambda x:(-x['score_A'],x['language']));Brank=sorted(cand,key=lambda x:(-x['score_B'],x['language']));win=meanrank[0]['language'];cw=next(x for x in cand if x['language']==win);margin=meanrank[0]['score_mean']-meanrank[1]['score_mean'];fp=bool(Arank[0]['language']==win and Brank[0]['language']==win and margin>=max(.02,FLOORS[win]) and cw['score_gap']<=.10)
    return {'kind':kind,'rep':rep,'winner_A':Arank[0]['language'],'winner_B':Brank[0]['language'],'winner_mean':win,'margin_mean':margin,'winner_floor':FLOORS[win],'winner_score_gap':cw['score_gap'],'false_positive':fp,'candidates':cand}

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();lms=b.load_lms();profile=target_profile();jobs=[(k,r) for k in KINDS for r in range(8)];rows=[]
    def one(j):return eval_negative(lms,profile,*j)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for z in ex.map(one,jobs):rows.append(z);print('Q1NEG',json.dumps({k:z[k] for k in ['kind','rep','winner_A','winner_B','winner_mean','margin_mean','winner_floor','winner_score_gap','false_positive']},sort_keys=True),flush=True)
    by={k:sum(x['false_positive'] for x in rows if x['kind']==k) for k in KINDS};fp=sum(by.values());passed=bool(fp<=1 and all(v<=1 for v in by.values()));out={'namespace':NS,'trials':40,'false_positives':fp,'by_generator':by,'pass':passed,'margin_floors':FLOORS,'rows':rows}
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()

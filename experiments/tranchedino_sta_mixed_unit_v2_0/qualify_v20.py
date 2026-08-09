import pandas as pd, numpy as np, hashlib, json, math
from numba import njit
from pathlib import Path

ALPH='abcdefghilmnopqrstu'
MULT=np.array([1,2,1]+[2]*16,dtype=np.int32)
assert len(ALPH)==19 and MULT.sum()==36
AIDX={c:i for i,c in enumerate(ALPH)}
BASE=Path('/mnt/data/tranchedino_sta_v20/old/tranchedino_paduan_payload_program/data')

def norm(s):
    s=str(s).lower().translate(str.maketrans({'j':'i','v':'u','w':'u','y':'i','x':'s','z':'s'}))
    return ''.join(c for c in s if c in AIDX)

def load():
    d=pd.read_csv(BASE/'paduan_lines.csv').fillna('')
    d['letters']=d.text.astype(str).map(norm)
    pages=sorted(d.loc[d.letters.str.len()>0,'page'].unique())
    cut=pages[int(len(pages)*.72)]
    tr=d.loc[(d.page<cut)&(d.letters.str.len()>0),'letters'].tolist()
    te=d.loc[(d.page>=cut)&(d.letters.str.len()>0),'letters'].tolist()
    return tr,te,int(cut)

def fit_model(lines,alpha=.5):
    V=19
    uni=np.full(V,alpha,dtype=np.float64)
    cnt=np.full(V**4,alpha,dtype=np.float64)
    for s in lines:
        a=np.array([AIDX[c] for c in s],dtype=np.int32)
        if len(a): uni += np.bincount(a,minlength=V)
        for i in range(3,len(a)):
            idx=((a[i-3]*V+a[i-2])*V+a[i-1])*V+a[i]
            cnt[idx]+=1
    tab=cnt.reshape(V**3,V); tab/=tab.sum(axis=1,keepdims=True)
    uni/=uni.sum()
    return np.log(tab.ravel()),np.log(uni)

@njit(cache=True)
def score_key(cipher,key,quad,uni):
    V=19; total=0.0; n=0
    hist0=-1;hist1=-1;hist2=-1
    for i in range(cipher.size):
        c=cipher[i]
        if c<0:
            hist0=hist1=hist2=-1; continue
        p=key[c]
        if hist0<0 or hist1<0 or hist2<0:
            total += uni[p]; n+=1
        else:
            idx=((hist0*V+hist1)*V+hist2)*V+p
            total += quad[idx]; n+=1
        hist0,hist1,hist2=hist1,hist2,p
    return total/max(n,1)

@njit(cache=True)
def weighted_agreement(k1,k2,freq):
    den=0.;num=0.
    for i in range(k1.size):
        den+=freq[i]
        if k1[i]==k2[i]: num+=freq[i]
    return num/den if den else 0.

@njit(cache=True)
def rng_step(state):
    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    return state * np.uint64(2685821657736338717)

@njit(cache=True)
def rng_int(state,upper):
    state=rng_step(state)
    return state,int(state % np.uint64(upper))

@njit(cache=True)
def polish(cipher,key,quad,uni,mult,max_sweeps,seed):
    cur=key.copy(); best=score_key(cipher,cur,quad,uni)
    state=np.uint64(seed if seed>0 else 1)
    inds=np.empty(36,dtype=np.int32)
    for sw in range(max_sweeps):
        order=np.arange(19,dtype=np.int32)
        for z in range(18,0,-1):
            state,j=rng_int(state,z+1); tmp=order[z];order[z]=order[j];order[j]=tmp
        imp=0
        for ii in range(18):
            a=order[ii]
            for jj in range(ii+1,19):
                b=order[jj]; m=0
                for s in range(36):
                    if cur[s]==a or cur[s]==b:
                        inds[m]=s;m+=1
                ra=mult[a]
                bestmask=-1; bs=best
                for mask in range(1<<m):
                    ca=0
                    for q in range(m): ca += (mask>>q)&1
                    if ca!=ra: continue
                    old=np.empty(m,dtype=np.int32)
                    for q in range(m):
                        old[q]=cur[inds[q]]; cur[inds[q]]=a if ((mask>>q)&1) else b
                    sc=score_key(cipher,cur,quad,uni)
                    if sc>bs+1e-12: bs=sc;bestmask=mask
                    for q in range(m): cur[inds[q]]=old[q]
                if bestmask>=0:
                    for q in range(m): cur[inds[q]]=a if ((bestmask>>q)&1) else b
                    best=bs;imp+=1
        if imp==0: break
    return cur,best

def seedint(s): return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'big') & 0x7fffffff

def freq_init(cipher,train_uni):
    f=np.bincount(cipher[cipher>=0],minlength=36).astype(float); f/=f.sum()
    slots=[]
    for l,m in enumerate(MULT):
        for _ in range(int(m)): slots.append((train_uni[l]/m,l))
    slots=sorted(slots,key=lambda x:(-x[0],x[1]))
    syms=sorted(range(36),key=lambda s:(-f[s],s))
    key=np.empty(36,dtype=np.int32)
    for s,(_,l) in zip(syms,slots):key[s]=l
    return key

def optimize(cipher,quad,uni,train_uni,ns,ensemble,max_restarts=36):
    freq=np.bincount(cipher[cipher>=0],minlength=36).astype(np.float64)
    init=freq_init(cipher,train_uni)
    slot=np.concatenate([np.full(int(m),l,dtype=np.int32) for l,m in enumerate(MULT)])
    bestk=None;bestsc=-1e100;history=[]
    for r in range(max_restarts):
        sd=seedint(f'{ns}|{ensemble}|{r}'); rng=np.random.default_rng(sd)
        if r==0: k=init.copy()
        elif r%3==1 and bestk is not None:
            k=bestk.copy()
            for _ in range(2+r%7):
                i,j=rng.integers(0,36,2);k[i],k[j]=k[j],k[i]
        else: k=rng.permutation(slot).astype(np.int32)
        k,sc=polish(cipher,k,quad,uni,MULT,12,sd)
        if sc>bestsc:bestk,bestsc=k.copy(),float(sc)
        if (r+1)%6==0: history.append({'restarts':r+1,'score':bestsc})
    return bestk,bestsc,history,freq

def fresh_key(rep):
    slot=np.concatenate([np.full(int(m),l,dtype=np.int32) for l,m in enumerate(MULT)])
    rng=np.random.default_rng(seedint(f'TRANCHSTA20controlkey::{rep}'))
    return rng.permutation(slot).astype(np.int32)

def sample_payload(test_lines,rep,target=12000):
    n=len(test_lines); start=seedint(f'TRANCHSTA20control::{rep}')%n
    out=[];chars=0;i=0
    while chars<target:
        s=test_lines[(start+i)%n]
        if s:
            remain=target-chars; q=s[:remain]; out.append(q); chars+=len(q)
        i+=1
    return out

def encrypt(lines,true_key,rep):
    pools=[np.where(true_key==l)[0] for l in range(19)]
    rng=np.random.default_rng(seedint(f'TRANCHSTA20encrypt::{rep}'))
    out=[]
    for s in lines:
        for ch in s:
            l=AIDX[ch]; p=pools[l]; out.append(int(p[rng.integers(0,len(p))]))
        out.append(-1)
    return np.array(out[:-1],dtype=np.int32)

def plaintext_accuracy(cipher,key,true):
    good=tot=0
    for c in cipher:
        if c>=0:good+=int(key[c]==true[c]);tot+=1
    return good/tot

def main():
    tr,te,cut=load(); quad,uni=fit_model(tr,.5)
    counts=np.zeros(19)
    for s in tr:
        for c in s:counts[AIDX[c]]+=1
    train_uni=(counts+.5)/(counts.sum()+.5*19)
    print('SOURCE',json.dumps({'train_lines':len(tr),'test_lines':len(te),'train_letters':sum(map(len,tr)),'test_letters':sum(map(len,te)),'cut':cut}))
    rows=[]
    for rep in range(12):
        lines=sample_payload(te,rep,12000); true=fresh_key(rep); cipher=encrypt(lines,true,rep)
        ka=kb=None; sa=sb=-1e100; ha=hb=[]; fr=None
        for nr in (6,12,18,24,30,36):
            ka,sa,ha,fr=optimize(cipher,quad,uni,train_uni,f'TRANCHSTA20CTRL{rep}','A',nr)
            kb,sb,hb,_=optimize(cipher,quad,uni,train_uni,f'TRANCHSTA20CTRL{rep}','B',nr)
            if abs(sa-sb)<=1e-7 and weighted_agreement(ka,kb,fr)>=.90: break
        bestk,bestsc=(ka,sa) if sa>=sb else (kb,sb)
        true_sc=float(score_key(cipher,true,quad,uni))
        maprec=float(weighted_agreement(bestk,true,fr))
        row={'rep':rep,'scoreA':sa,'scoreB':sb,'score_gap':abs(sa-sb),'AB_agreement':float(weighted_agreement(ka,kb,fr)),'plaintext_recovery':maprec,'map_recovery':maprec,'true_score':true_sc,'best_minus_true':bestsc-true_sc,'histA':ha,'histB':hb}
        rows.append(row); print('CONTROL',json.dumps(row),flush=True)
    vals=lambda k:[r[k] for r in rows]
    summary={'n':12,'converged':sum(r['score_gap']<=1e-7 and r['AB_agreement']>=.90 for r in rows),'median_plain':float(np.median(vals('plaintext_recovery'))),'min_plain':min(vals('plaintext_recovery')),'median_map':float(np.median(vals('map_recovery'))),'min_map':min(vals('map_recovery')),'min_AB':min(vals('AB_agreement')),'min_best_minus_true':min(vals('best_minus_true')),'max_true_advantage':max(-r['best_minus_true'] for r in rows)}
    gate=(summary['converged']==12 and summary['median_plain']>=.95 and summary['min_plain']>=.85 and summary['median_map']>=.95 and summary['min_map']>=.85 and summary['min_AB']>=.90 and summary['max_true_advantage']<=1e-5)
    summary['gate']=gate
    print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()

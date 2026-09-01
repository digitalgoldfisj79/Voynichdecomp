# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, collections, concurrent.futures, hashlib, json, math, multiprocessing as mp
import numpy as np
from unidecode import unidecode
from wordfreq import top_n_list, zipf_frequency

NS='VBMV10TERMINAL'
VOW='aeiou'; NV=5; KB=30; KR=32; KN=96
ALPHA=.05; LM_CHARS=650_000; PT_CHARS=650_000
TEMPS=np.geomspace(.35,.01,12)
_WORK={}

def seed(*xs):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,xs)).encode()).digest()[:8],'big') & 0x7fffffff

def norm(s): return ''.join(c for c in unidecode(s).lower() if 'a'<=c<='z')

def bank(lang,tag,nchars):
    ws=[];wt=[]
    for w in top_n_list(lang,30000):
        q=norm(w)
        if not q or len(q)>24: continue
        z=zipf_frequency(w,lang)
        if not np.isfinite(z): continue
        ws.append(q);wt.append(10.0**(.45*z))
    p=np.asarray(wt,float);p/=p.sum();rng=np.random.default_rng(seed(NS,'BANK',lang,tag));out=[];n=0
    while n<nchars:
        ix=rng.choice(len(ws),4096,p=p);s=''.join(ws[int(i)] for i in ix);out.append(s);n+=len(s)
    return ''.join(out)[:nchars]

def decomp(s):
    runs=[];vs=[];cur=[]
    for ch in s:
        if ch in VOW:runs.append(''.join(cur));cur=[];vs.append(ch)
        else:cur.append(ch)
    runs.append(''.join(cur));return runs,vs

class LM:
    def __init__(self,s):
        self.ctx=collections.defaultdict(collections.Counter);self.tot=collections.Counter()
        for i in range(4,len(s)):
            c=s[i-4:i];x=s[i];self.ctx[c][x]+=1;self.tot[c]+=1
    def score(self,s):
        if len(s)<5:return (-25.0,1)
        ll=0.;n=0
        for i in range(4,len(s)):
            c=s[i-4:i];x=s[i];ll+=math.log((self.ctx[c][x]+ALPHA)/(self.tot[c]+26*ALPHA));n+=1
        return ll,n

def asset(lang):
    la={'DE':'de','IT':'it'}[lang]
    lms=bank(la,'LM',LM_CHARS);pts=bank(la,'PT',PT_CHARS);rr,_=decomp(lms);cnt=collections.Counter(r for r in rr if r and len(r)<=5);runs=[r for r,_ in cnt.most_common(KR)]
    if len(runs)!=KR:raise RuntimeError(('run inventory',lang,len(runs)))
    rf=np.asarray([cnt[r] for r in runs],float);rf/=rf.sum();_,vv=decomp(lms);vc=collections.Counter(vv);vf=np.asarray([vc[v] for v in VOW],float);vf/=vf.sum()
    return {'lm':LM(lms),'pt':pts,'runs':runs,'run_freq':rf,'vowel_freq':vf}

def plaintext_lines(a,lang,rep,nlines=2000):
    runs,vs=decomp(a['pt']);allow=set(a['runs']);rng=np.random.default_rng(seed(NS,'PLAIN',lang,rep));out=[];tries=0
    while len(out)<nlines and tries<4_000_000:
        tries+=1;B=int(rng.integers(8,15));st=int(rng.integers(0,len(vs)-B-1));rs=runs[st:st+B+1];vv=vs[st:st+B]
        if any((r and (len(r)>5 or r not in allow)) for r in rs):continue
        out.append({'runs':list(rs),'vowels':list(vv)})
    if len(out)<nlines:raise RuntimeError(('plaintext shortage',lang,rep,len(out),tries))
    return out

def codebook(a,lang,rep):
    rng=np.random.default_rng(seed(NS,'KEY',lang,rep));bm=np.repeat(np.arange(NV,dtype=np.int16),KB//NV);rng.shuffle(bm);nm=np.repeat(np.arange(KR,dtype=np.int16),KN//KR);rng.shuffle(nm)
    bp={v:np.flatnonzero(bm==v) for v in range(NV)};npool={r:np.flatnonzero(nm==r) for r in range(KR)}
    bw={v:rng.dirichlet(np.full(len(bp[v]),.35)) for v in bp};nw={r:rng.dirichlet(np.full(len(npool[r]),.35)) for r in npool}
    return {'bmap':bm,'nmap':nm,'bp':bp,'np':npool,'bw':bw,'nw':nw}

def encode(pl,a,key,lang,rep):
    ridx={r:i for i,r in enumerate(a['runs'])};out=[]
    for li,x in enumerate(pl):
        rng=np.random.default_rng(seed(NS,'EMIT',lang,rep,li));ns=[];bs=[]
        for i,r in enumerate(x['runs']):
            if r=='':ns.append(-1)
            else:
                z=ridx[r];pool=key['np'][z];ns.append(int(rng.choice(pool,p=key['nw'][z])))
            if i<len(x['vowels']):
                v=VOW.index(x['vowels'][i]);pool=key['bp'][v];bs.append(int(rng.choice(pool,p=key['bw'][v])))
        out.append({'n':ns,'b':bs})
    return out

def decode_line(L,bm,nm,runs):
    s=[]
    for i,n in enumerate(L['n']):
        if n>=0:s.append(runs[int(nm[n])])
        if i<len(L['b']):s.append(VOW[int(bm[L['b'][i]])])
    return ''.join(s)

def score_lines(lines,a,bm,nm):
    ll=0.;nn=0
    for L in lines:
        x,n=a['lm'].score(decode_line(L,bm,nm,a['runs']));ll+=x;nn+=n
    return ll/max(1,nn),ll,nn

def counts_indexes(lines):
    cb=np.zeros(KB,dtype=np.int64);cn=np.zeros(KN,dtype=np.int64);ib=[set() for _ in range(KB)];inn=[set() for _ in range(KN)]
    for j,L in enumerate(lines):
        for x in L['b']:cb[x]+=1;ib[x].add(j)
        for x in L['n']:
            if x>=0:cn[x]+=1;inn[x].add(j)
    return cb,cn,[sorted(x) for x in ib],[sorted(x) for x in inn]

def init_map(a,tag,fixed_b=None,fixed_n=None):
    rng=np.random.default_rng(seed(NS,'INIT',tag));bm=rng.choice(NV,KB,p=a['vowel_freq']).astype(np.int16);nm=rng.choice(KR,KN,p=a['run_freq']).astype(np.int16)
    if fixed_b:
        for k,v in fixed_b.items():bm[k]=v
    if fixed_n:
        for k,v in fixed_n.items():nm[k]=v
    return bm,nm

def weighted_other(rng,p,current):
    q=np.array(p,float);q[current]=0.;q/=q.sum();return int(rng.choice(len(q),p=q))

def chain_fit(chain_id):
    W=_WORK;lines=W['lines'];a=W['asset'];fixed_b=W['fixed_b'];fixed_n=W['fixed_n'];tag=W['tag'];cb=W['cb'];cn=W['cn'];ib=W['ib'];inn=W['inn'];smoke=W['smoke']
    bm,nm=init_map(a,f'{tag}:C{chain_id}',fixed_b,fixed_n);cache=[];totll=0.;totn=0
    for L in lines:
        ll,n=a['lm'].score(decode_line(L,bm,nm,a['runs']));cache.append((ll,n));totll+=ll;totn+=n
    def eval_affected(affected):
        dll=0.;dn=0;pack=[]
        for j in affected:
            ll0,n0=cache[j];ll1,n1=a['lm'].score(decode_line(lines[j],bm,nm,a['runs']));dll+=ll1-ll0;dn+=n1-n0;pack.append((j,ll1,n1))
        return dll,dn,pack
    def commit(pack,dll,dn):
        nonlocal totll,totn
        totll+=dll;totn+=dn
        for j,ll,n in pack:cache[j]=(ll,n)
    obsb=[i for i in range(KB) if cb[i]>0 and i not in fixed_b];obsn=[i for i in range(KN) if cn[i]>0 and i not in fixed_n]
    rng=np.random.default_rng(seed(NS,'CHAIN',tag,chain_id));
    def greedy(passno):
        items=[('b',i) for i in obsb]+[('n',i) for i in obsn];rng2=np.random.default_rng(seed(NS,'GREEDY',tag,chain_id,passno));rng2.shuffle(items)
        for typ,t in items:
            arr=bm if typ=='b' else nm;idx=ib if typ=='b' else inn;K=NV if typ=='b' else KR;old=int(arr[t]);best=old;best_ratio=totll/max(1,totn);bestpack=None
            for v in range(K):
                if v==old:continue
                arr[t]=v;dll,dn,pack=eval_affected(idx[t]);rat=(totll+dll)/max(1,totn+dn)
                if rat>best_ratio+1e-12:best_ratio=rat;best=v;bestpack=(dll,dn,pack)
            arr[t]=old
            if best!=old and bestpack is not None:
                arr[t]=best;commit(bestpack[2],bestpack[0],bestpack[1])
    for p in range(1 if smoke else 2):greedy(p)
    temps=TEMPS[:2] if smoke else TEMPS
    wb=np.asarray([cb[i] for i in obsb],float);wn=np.asarray([cn[i] for i in obsn],float)
    if len(wb):wb/=wb.sum()
    if len(wn):wn/=wn.sum()
    nprop=max(1,len(obsb)+len(obsn))
    for si,T in enumerate(temps):
        for _ in range(nprop):
            kinds=[]
            if obsb:kinds+=['br','bs'] if len(obsb)>1 else ['br']
            if obsn:kinds+=['nr','ns'] if len(obsn)>1 else ['nr']
            kind=str(rng.choice(kinds));old_ratio=totll/max(1,totn)
            if kind=='br':
                t=int(rng.choice(obsb,p=wb));old=int(bm[t]);new=weighted_other(rng,a['vowel_freq'],old);bm[t]=new;affected=ib[t];rev=lambda:bm.__setitem__(t,old)
            elif kind=='nr':
                t=int(rng.choice(obsn,p=wn));old=int(nm[t]);new=weighted_other(rng,a['run_freq'],old);nm[t]=new;affected=inn[t];rev=lambda:nm.__setitem__(t,old)
            elif kind=='bs':
                ii=rng.choice(len(obsb),size=2,replace=False,p=wb);t,u=obsb[int(ii[0])],obsb[int(ii[1])];x,y=int(bm[t]),int(bm[u]);bm[t],bm[u]=y,x;affected=sorted(set(ib[t])|set(ib[u]));rev=lambda:(bm.__setitem__(t,x),bm.__setitem__(u,y))
            else:
                ii=rng.choice(len(obsn),size=2,replace=False,p=wn);t,u=obsn[int(ii[0])],obsn[int(ii[1])];x,y=int(nm[t]),int(nm[u]);nm[t],nm[u]=y,x;affected=sorted(set(inn[t])|set(inn[u]));rev=lambda:(nm.__setitem__(t,x),nm.__setitem__(u,y))
            dll,dn,pack=eval_affected(affected);new_ratio=(totll+dll)/max(1,totn+dn);delta=new_ratio-old_ratio
            accept=delta>=0 or (delta/T>-745 and rng.random()<math.exp(delta/T))
            if accept:commit(pack,dll,dn)
            else:rev()
    greedy(99)
    return {'chain':chain_id,'fit':totll/max(1,totn),'bmap':bm.tolist(),'nmap':nm.tolist()}

def fit_global(lines,a,tag,fixed_b=None,fixed_n=None,smoke=False):
    global _WORK
    cb,cn,ib,inn=counts_indexes(lines);_WORK={'lines':lines,'asset':a,'fixed_b':fixed_b or {},'fixed_n':fixed_n or {},'tag':tag,'cb':cb,'cn':cn,'ib':ib,'inn':inn,'smoke':smoke}
    nch=1 if smoke else 8
    if nch==1:res=[chain_fit(0)]
    else:
        ctx=mp.get_context('fork')
        with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=ctx) as ex:res=list(ex.map(chain_fit,range(8)))
    best=sorted(res,key=lambda z:(-z['fit'],z['chain']))[0]
    return np.asarray(best['bmap'],dtype=np.int16),np.asarray(best['nmap'],dtype=np.int16),best['fit'],best['chain']

def reveal_maps(fit,key):
    cb,cn,_,_=counts_indexes(fit);ob=[i for i in range(KB) if cb[i]>0];on=[i for i in range(KN) if cn[i]>0]
    ob=sorted(ob,key=lambda i:(-cb[i],i));on=sorted(on,key=lambda i:(-cn[i],i));nb=math.ceil(.25*len(ob));nn=math.ceil(.25*len(on));return {i:int(key['bmap'][i]) for i in ob[:nb]},{i:int(key['nmap'][i]) for i in on[:nn]}

def rand_hold(hold,a,tag):
    z=[]
    for r in range(20):
        bm,nm=init_map(a,f'{tag}:RAND:{r}');z.append(score_lines(hold,a,bm,nm)[0])
    return float(np.median(z))

def recovery(bm,nm,key,fit,hold,a):
    cb,cn,_,_=counts_indexes(fit);tb=tn=gb=gn=0;tb5=tn5=gb5=gn5=0;cc=ct=cc5=ct5=0;seenb=seenn=0;allb=alln=0
    for L in hold:
        for b in L['b']:
            allb+=1;seenb+=int(cb[b]>0);ok=int(bm[b]==key['bmap'][b]);tb+=ok;gb+=1
            cc+=ok;ct+=1
            if cb[b]>=5:tb5+=ok;gb5+=1;cc5+=ok;ct5+=1
        for n in L['n']:
            if n<0:continue
            alln+=1;seenn+=int(cn[n]>0);tr=a['runs'][int(key['nmap'][n])];pr=a['runs'][int(nm[n])];ok=int(pr==tr);tn+=ok;gn+=1
            M=max(len(tr),len(pr));cc+=sum(i<len(tr) and i<len(pr) and tr[i]==pr[i] for i in range(M));ct+=M
            if cn[n]>=5:
                tn5+=ok;gn5+=1;cc5+=sum(i<len(tr) and i<len(pr) and tr[i]==pr[i] for i in range(M));ct5+=M
    return {'REC_B':tb/max(1,gb),'REC_N':tn/max(1,gn),'REC_CHAR':cc/max(1,ct),'REC_B5':tb5/max(1,gb5),'REC_N5':tn5/max(1,gn5),'REC_CHAR5':cc5/max(1,ct5),'COV_B':seenb/max(1,allb),'COV_N':seenn/max(1,alln),'N_B5':gb5,'N_N5':gn5,'N_CHAR5':ct5}

def oracle_block(name,bm,nm,key,fit,hold,a,tag,fit_obj=None,chain=None):
    hlm=score_lines(hold,a,bm,nm)[0];rnd=rand_hold(hold,a,tag);r=recovery(bm,nm,key,fit,hold,a);return {'oracle':name,'FIT_LM':fit_obj,'BEST_CHAIN':chain,'HOLD_LM':hlm,'RAND_HOLD_LM':rnd,'HOLD_ADV':hlm-rnd,**r}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lang',choices=['DE','IT'],required=True);ap.add_argument('--rep',type=int,choices=[0,1,2],required=True);ap.add_argument('--size',type=int,choices=[100,250,500,1000,2000],required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    a=asset(args.lang);pl=plaintext_lines(a,args.lang,args.rep,2000);key=codebook(a,args.lang,args.rep);all_lines=encode(pl,a,key,args.lang,args.rep);lines=all_lines[:args.size];cut=int(.8*args.size);fit=lines[:cut];hold=lines[cut:]
    print('V10START',json.dumps({'lang':args.lang,'rep':args.rep,'size':args.size,'fit':len(fit),'hold':len(hold),'smoke':args.smoke,'runs':a['runs'][:10]}),flush=True)
    o0=oracle_block('O0_TRUE_KEY',key['bmap'],key['nmap'],key,fit,hold,a,f'O0:{args.lang}:{args.rep}:{args.size}',fit_obj=None,chain=None)
    fb,fn=reveal_maps(fit,key);b1,n1,f1,c1=fit_global(fit,a,f'A:{args.lang}:R{args.rep}:S{args.size}:O1',fb,fn,args.smoke);o1=oracle_block('O1_25PCT_REVEALED',b1,n1,key,fit,hold,a,f'O1:{args.lang}:{args.rep}:{args.size}',f1,c1);o1['REVEALED_B']=len(fb);o1['REVEALED_N']=len(fn)
    print('V10O1',json.dumps({k:o1[k] for k in ['REC_B','REC_N','REC_CHAR','REC_B5','REC_N5','REC_CHAR5','HOLD_ADV','REVEALED_B','REVEALED_N']},sort_keys=True),flush=True)
    b2,n2,f2,c2=fit_global(fit,a,f'A:{args.lang}:R{args.rep}:S{args.size}:O2',{}, {},args.smoke);o2=oracle_block('O2_TRUE_LANGUAGE_ZERO_KEY',b2,n2,key,fit,hold,a,f'O2:{args.lang}:{args.rep}:{args.size}',f2,c2)
    o3={'oracle':'O3_TRUE_LANGUAGE_FREQUENT_ONLY','REC_B5':o2['REC_B5'],'REC_N5':o2['REC_N5'],'REC_CHAR5':o2['REC_CHAR5'],'N_B5':o2['N_B5'],'N_N5':o2['N_N5'],'N_CHAR5':o2['N_CHAR5']}
    out={'stage':'A_POSITIVE','lang':args.lang,'rep':args.rep,'size':args.size,'fit_lines':len(fit),'hold_lines':len(hold),'smoke':args.smoke,'O0':o0,'O1':o1,'O2':o2,'O3':o3,'H1_C1_OPENED':False}
    print('VBM_V10_RESULT='+json.dumps(out,sort_keys=True,separators=(',',':')))
if __name__=='__main__':main()

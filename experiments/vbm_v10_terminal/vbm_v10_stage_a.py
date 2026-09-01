# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, collections, hashlib, json, math, time
import numpy as np
from unidecode import unidecode
from wordfreq import top_n_list, zipf_frequency

NS='VBMV10TERMINAL'
VOW='aeiou'; NV=5; KB=30; KR=32; KN=96
SIZES=[100,250,500,1000,2000]
ALPHA=.05; LM_CHARS=700_000; PT_CHARS=900_000
CHAINS=8; COORD_PRE=2; ANNEAL_SWEEPS=12; COORD_POST=1
T0=.35; T1=.01


def seed(*xs):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,xs)).encode()).digest()[:8],'big') & 0x7fffffff

def norm(s): return ''.join(c for c in unidecode(s).lower() if 'a'<=c<='z')

def bank(lang,tag,nchars):
    ws=[]; wt=[]
    for w in top_n_list(lang,30000):
        q=norm(w)
        if not q or len(q)>24: continue
        z=zipf_frequency(w,lang)
        if not np.isfinite(z): continue
        ws.append(q); wt.append(10.0**(.45*z))
    p=np.asarray(wt,float); p/=p.sum(); rng=np.random.default_rng(seed(NS,'BANK',lang,tag)); out=[]; n=0
    while n<nchars:
        ix=rng.choice(len(ws),4096,p=p); s=''.join(ws[int(i)] for i in ix); out.append(s); n+=len(s)
    return ''.join(out)[:nchars]

def decomp(s):
    runs=[]; vs=[]; cur=[]
    for ch in s:
        if ch in VOW: runs.append(''.join(cur)); cur=[]; vs.append(ch)
        else: cur.append(ch)
    runs.append(''.join(cur)); return runs,vs

class LM:
    def __init__(self,s):
        self.ctx=collections.defaultdict(collections.Counter); self.tot=collections.Counter()
        for i in range(4,len(s)):
            c=s[i-4:i]; x=s[i]; self.ctx[c][x]+=1; self.tot[c]+=1
    def score(self,s):
        if len(s)<5: return (-25.0,1)
        ll=0.; n=0
        for i in range(4,len(s)):
            c=s[i-4:i]; x=s[i]; ll += math.log((self.ctx[c][x]+ALPHA)/(self.tot[c]+26*ALPHA)); n+=1
        return ll,n

def assets(LANG):
    la={'DE':'de','IT':'it'}[LANG]
    lmtext=bank(la,'LM',LM_CHARS); pt=bank(la,'PT',PT_CHARS)
    rr,_=decomp(lmtext); cnt=collections.Counter(r for r in rr if r and len(r)<=5); runs=[r for r,_ in cnt.most_common(KR)]
    if len(runs)!=KR: raise RuntimeError('run inventory shortage')
    rf=np.asarray([cnt[r] for r in runs],float); rf/=rf.sum(); _,vv=decomp(lmtext); vc=collections.Counter(vv); vf=np.asarray([vc[v] for v in VOW],float); vf/=vf.sum()
    return {'lm':LM(lmtext),'pt':pt,'runs':runs,'run_freq':rf,'vowel_freq':vf}

def plaintext_lines(A,tag,nlines=2000):
    runs,vs=decomp(A['pt']); allow=set(A['runs']); rng=np.random.default_rng(seed(NS,'PLAIN',tag)); out=[]; tries=0
    while len(out)<nlines and tries<4_000_000:
        tries+=1; B=int(rng.integers(8,15)); st=int(rng.integers(0,len(vs)-B-1)); rs=runs[st:st+B+1]; vv=vs[st:st+B]
        if any(r and (len(r)>5 or r not in allow) for r in rs): continue
        txt=''.join(rs[i]+(vv[i] if i<B else '') for i in range(B+1))
        out.append({'runs':list(rs),'vowels':list(vv),'plain':txt})
    if len(out)<nlines: raise RuntimeError(('plaintext shortage',len(out),tries))
    return out

def codebook(A,tag):
    rng=np.random.default_rng(seed(NS,'KEY',tag)); bm=np.repeat(np.arange(NV,dtype=np.int16),KB//NV); rng.shuffle(bm); nm=np.repeat(np.arange(KR,dtype=np.int16),KN//KR); rng.shuffle(nm)
    bp={v:np.flatnonzero(bm==v) for v in range(NV)}; npool={r:np.flatnonzero(nm==r) for r in range(KR)}
    bw={v:rng.dirichlet(np.full(len(bp[v]),.35)) for v in bp}; nw={r:rng.dirichlet(np.full(len(npool[r]),.35)) for r in npool}
    return {'bmap':bm,'nmap':nm,'bp':bp,'np':npool,'bw':bw,'nw':nw}

def encode(pl,A,key,tag):
    ridx={r:i for i,r in enumerate(A['runs'])}; rng=np.random.default_rng(seed(NS,'EMIT',tag)); ns=[]; bs=[]
    for i,r in enumerate(pl['runs']):
        if r=='': ns.append(-1)
        else:
            z=ridx[r]; pool=key['np'][z]; ns.append(int(rng.choice(pool,p=key['nw'][z])))
        if i<len(pl['vowels']):
            v=VOW.index(pl['vowels'][i]); pool=key['bp'][v]; bs.append(int(rng.choice(pool,p=key['bw'][v])))
    return {'n':ns,'b':bs,'plain':pl['plain']}

def make_positive(LANG,rep,A):
    tag=f'A:{LANG}:R{rep}'; pl=plaintext_lines(A,tag,2000); key=codebook(A,tag); lines=[encode(x,A,key,f'{tag}:L{i}') for i,x in enumerate(pl)]; return lines,key

def decode_line(L,bm,nm,runs):
    out=[]
    for i,n in enumerate(L['n']):
        if n>=0: out.append(runs[int(nm[n])])
        if i<len(L['b']): out.append(VOW[int(bm[L['b'][i]])])
    return ''.join(out)

def indexes(lines):
    bi=[set() for _ in range(KB)]; ni=[set() for _ in range(KN)]
    for j,L in enumerate(lines):
        for x in set(L['b']): bi[x].add(j)
        for x in set(n for n in L['n'] if n>=0): ni[x].add(j)
    return [sorted(x) for x in bi],[sorted(x) for x in ni]

def counts(lines):
    bc=np.zeros(KB,dtype=int); nc=np.zeros(KN,dtype=int)
    for L in lines:
        for x in L['b']: bc[x]+=1
        for x in L['n']:
            if x>=0: nc[x]+=1
    return bc,nc

def init_map(A,tag,key=None,fixed_b=None,fixed_n=None):
    rng=np.random.default_rng(seed(NS,'INIT',tag)); bm=rng.choice(NV,KB,p=A['vowel_freq']).astype(np.int16); nm=rng.choice(KR,KN,p=A['run_freq']).astype(np.int16)
    if key is not None:
        if fixed_b:
            for x in fixed_b: bm[x]=key['bmap'][x]
        if fixed_n:
            for x in fixed_n: nm[x]=key['nmap'][x]
    return bm,nm

def evaluate_cache(lines,A,bm,nm):
    c=[]; ll=0.; nn=0
    for L in lines:
        a,b=A['lm'].score(decode_line(L,bm,nm,A['runs'])); c.append((a,b)); ll+=a; nn+=b
    return c,ll,nn

def affected_score(lines,A,bm,nm,cache,idxs):
    dll=0.; dn=0; pack=[]
    for j in idxs:
        oldll,oldn=cache[j]; ll,n=A['lm'].score(decode_line(lines[j],bm,nm,A['runs'])); dll+=ll-oldll; dn+=n-oldn; pack.append((j,ll,n))
    return dll,dn,pack

def commit_pack(cache,pack):
    for j,ll,n in pack: cache[j]=(ll,n)

def coord_pass(lines,A,bm,nm,cache,ll,nn,bi,ni,tag,pp,fixed_b,fixed_n):
    items=[('b',i) for i,z in enumerate(bi) if z and i not in fixed_b]+[('n',i) for i,z in enumerate(ni) if z and i not in fixed_n]
    rng=np.random.default_rng(seed(NS,'ORDER',tag,pp)); rng.shuffle(items); changed=0
    for typ,t in items:
        idx=bi[t] if typ=='b' else ni[t]; old=int(bm[t] if typ=='b' else nm[t]); cand=range(NV) if typ=='b' else range(KR); bestv=old; bestscore=ll/max(1,nn); bestpack=None; bestdll=bestdn=0.
        for v in cand:
            if v==old: continue
            if typ=='b': bm[t]=v
            else: nm[t]=v
            dll,dn,pack=affected_score(lines,A,bm,nm,cache,idx); sc=(ll+dll)/max(1,nn+dn)
            if sc>bestscore+1e-12: bestscore=sc; bestv=int(v); bestpack=pack; bestdll=dll; bestdn=dn
        if typ=='b': bm[t]=old
        else: nm[t]=old
        if bestv!=old and bestpack is not None:
            if typ=='b': bm[t]=bestv
            else: nm[t]=bestv
            ll+=bestdll; nn+=bestdn; commit_pack(cache,bestpack); changed+=1
    return ll,nn,changed

def anneal(lines,A,bm,nm,cache,ll,nn,bi,ni,tag,fixed_b,fixed_n):
    active_b=[i for i,z in enumerate(bi) if z and i not in fixed_b]; active_n=[i for i,z in enumerate(ni) if z and i not in fixed_n]; total=max(1,len(active_b)+len(active_n)); rng=np.random.default_rng(seed(NS,'ANNEAL',tag))
    temps=np.geomspace(T0,T1,ANNEAL_SWEEPS)
    accepted=0; proposed=0
    for sw,T in enumerate(temps):
        for _ in range(total):
            proposed+=1
            typ='n' if active_n and (not active_b or rng.random()<len(active_n)/(len(active_n)+len(active_b))) else 'b'; arr=active_n if typ=='n' else active_b
            if not arr: continue
            is_swap=(len(arr)>=2 and rng.random()<0.30)
            if is_swap:
                x,y=rng.choice(arr,size=2,replace=False); x=int(x); y=int(y); oldx=int(nm[x] if typ=='n' else bm[x]); oldy=int(nm[y] if typ=='n' else bm[y])
                if oldx==oldy: continue
                if typ=='n': nm[x],nm[y]=oldy,oldx; idx=sorted(set(ni[x])|set(ni[y]))
                else: bm[x],bm[y]=oldy,oldx; idx=sorted(set(bi[x])|set(bi[y]))
                dll,dn,pack=affected_score(lines,A,bm,nm,cache,idx); oldscore=ll/max(1,nn); newscore=(ll+dll)/max(1,nn+dn); delta=newscore-oldscore
                ok=delta>=0 or rng.random()<math.exp(max(-700.,delta/T))
                if ok: ll+=dll; nn+=dn; commit_pack(cache,pack); accepted+=1
                else:
                    if typ=='n': nm[x],nm[y]=oldx,oldy
                    else: bm[x],bm[y]=oldx,oldy
            else:
                x=int(rng.choice(arr)); old=int(nm[x] if typ=='n' else bm[x]); nv=int(rng.integers(KR if typ=='n' else NV))
                if nv==old: continue
                if typ=='n': nm[x]=nv; idx=ni[x]
                else: bm[x]=nv; idx=bi[x]
                dll,dn,pack=affected_score(lines,A,bm,nm,cache,idx); oldscore=ll/max(1,nn); newscore=(ll+dll)/max(1,nn+dn); delta=newscore-oldscore
                ok=delta>=0 or rng.random()<math.exp(max(-700.,delta/T))
                if ok: ll+=dll; nn+=dn; commit_pack(cache,pack); accepted+=1
                else:
                    if typ=='n': nm[x]=old
                    else: bm[x]=old
    return ll,nn,accepted,proposed

def fit_map(lines,A,tag,key=None,fixed_b=None,fixed_n=None):
    fixed_b=set() if fixed_b is None else set(fixed_b); fixed_n=set() if fixed_n is None else set(fixed_n); bi,ni=indexes(lines); best=None
    for ch in range(CHAINS):
        bm,nm=init_map(A,f'{tag}:C{ch}',key,fixed_b,fixed_n); cache,ll,nn=evaluate_cache(lines,A,bm,nm)
        for pp in range(COORD_PRE): ll,nn,_=coord_pass(lines,A,bm,nm,cache,ll,nn,bi,ni,f'{tag}:C{ch}:PRE',pp,fixed_b,fixed_n)
        ll,nn,acc,prop=anneal(lines,A,bm,nm,cache,ll,nn,bi,ni,f'{tag}:C{ch}',fixed_b,fixed_n)
        for pp in range(COORD_POST): ll,nn,_=coord_pass(lines,A,bm,nm,cache,ll,nn,bi,ni,f'{tag}:C{ch}:POST',pp,fixed_b,fixed_n)
        sc=ll/max(1,nn); z={'bmap':bm.copy(),'nmap':nm.copy(),'fit_score':sc,'anneal_accept':acc/max(1,prop)}
        if best is None or sc>best['fit_score']: best=z
    return best

def score_lines(lines,A,m):
    ll=0.; nn=0
    for L in lines:
        a,b=A['lm'].score(decode_line(L,m['bmap'],m['nmap'],A['runs'])); ll+=a; nn+=b
    return ll/max(1,nn)

def random_baseline(hold,A,tag):
    vals=[]
    for r in range(20):
        bm,nm=init_map(A,f'{tag}:RAND:{r}'); vals.append(score_lines(hold,A,{'bmap':bm,'nmap':nm}))
    return float(np.median(vals))

def recovery(m,key,hold,fit_counts,A,minfit=0):
    bc,nc=fit_counts; be=[x for L in hold for x in L['b'] if bc[x]>=minfit]; ne=[x for L in hold for x in L['n'] if x>=0 and nc[x]>=minfit]
    rb=sum(int(m['bmap'][x])==int(key['bmap'][x]) for x in be)/max(1,len(be)); rn=sum(int(m['nmap'][x])==int(key['nmap'][x]) for x in ne)/max(1,len(ne))
    match=tot=0
    for L in hold:
        parts_dec=[]; parts_true=[]
        for i,n in enumerate(L['n']):
            if n>=0 and nc[n]>=minfit:
                parts_dec.append(A['runs'][int(m['nmap'][n])]); parts_true.append(A['runs'][int(key['nmap'][n])])
            if i<len(L['b']) and bc[L['b'][i]]>=minfit:
                b=L['b'][i]; parts_dec.append(VOW[int(m['bmap'][b])]); parts_true.append(VOW[int(key['bmap'][b])])
        d=''.join(parts_dec); t=''.join(parts_true); M=max(len(d),len(t)); tot+=M; match+=sum(i<len(d) and i<len(t) and d[i]==t[i] for i in range(M))
    return rb,rn,match/max(1,tot)

def coverage(hold,bc,nc,minfit=1):
    be=[x for L in hold for x in L['b']]; ne=[x for L in hold for x in L['n'] if x>=0]; return sum(bc[x]>=minfit for x in be)/max(1,len(be)),sum(nc[x]>=minfit for x in ne)/max(1,len(ne))

def stability(fit,hold,A,tag):
    a=fit[::2]; b=fit[1::2]; ma=fit_map(a,A,tag+':ODD'); mb=fit_map(b,A,tag+':EVEN'); be=[x for L in hold for x in L['b']]; ne=[x for L in hold for x in L['n'] if x>=0]
    sb=sum(int(ma['bmap'][x])==int(mb['bmap'][x]) for x in be)/max(1,len(be)); sn=sum(int(ma['nmap'][x])==int(mb['nmap'][x]) for x in ne)/max(1,len(ne)); return .5*(sb+sn),sb,sn

def top_quarter_fixed(fit,key):
    bc,nc=counts(fit); ab=[i for i,x in enumerate(bc) if x>0]; an=[i for i,x in enumerate(nc) if x>0]; ab=sorted(ab,key=lambda i:(-bc[i],i)); an=sorted(an,key=lambda i:(-nc[i],i)); qb=max(1,math.ceil(.25*len(ab))); qn=max(1,math.ceil(.25*len(an))); return set(ab[:qb]),set(an[:qn])

def one_size(lines,key,A,LANG,rep,size):
    z=lines[:size]; cut=int(.8*size); fit=z[:cut]; hold=z[cut:]; fc=counts(fit); tag=f'A:{LANG}:R{rep}:N{size}'; t0=time.time()
    # O2 headline blind key.
    m2=fit_map(fit,A,tag+':O2'); rb,rn,rc=recovery(m2,key,hold,fc,A,0); rb5,rn5,rc5=recovery(m2,key,hold,fc,A,5); cb,cn=coverage(hold,*fc,1); holdlm=score_lines(hold,A,m2); rand=random_baseline(hold,A,tag); adv=holdlm-rand
    # O1 favourable 25%-revealed diagnostic.
    fb,fn=top_quarter_fixed(fit,key); m1=fit_map(fit,A,tag+':O1',key,fb,fn); o1b,o1n,o1c=recovery(m1,key,hold,fc,A,0)
    # O0 truth.
    mt={'bmap':key['bmap'],'nmap':key['nmap']}; truehold=score_lines(hold,A,mt)
    # Split-fit stability.
    stab,sb,sn=stability(fit,hold,A,tag+':STAB')
    out={'language':LANG,'rep':rep,'size':size,'fit_lines':len(fit),'hold_lines':len(hold),'O2':{'REC_B':rb,'REC_N':rn,'REC_CHAR':rc,'REC_B5':rb5,'REC_N5':rn5,'REC_CHAR5':rc5,'COV_B':cb,'COV_N':cn,'HOLD_LM':holdlm,'RAND_HOLD_LM':rand,'HOLD_ADV':adv,'STAB':stab,'STAB_B':sb,'STAB_N':sn,'FIT_SCORE':m2['fit_score'],'ANNEAL_ACCEPT':m2['anneal_accept']},'O1':{'revealed_bridge_types':len(fb),'revealed_nucleus_types':len(fn),'REC_B':o1b,'REC_N':o1n,'REC_CHAR':o1c},'O0':{'HOLD_LM_TRUE_KEY':truehold},'runtime_s':time.time()-t0}
    print('V10A_ROW='+json.dumps(out,sort_keys=True,separators=(',',':')),flush=True); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--lang',choices=['DE','IT'],required=True); ap.add_argument('--rep',type=int,choices=[0,1,2],required=True); a=ap.parse_args(); A=assets(a.lang); print('V10A_ASSET='+json.dumps({'language':a.lang,'runs':A['runs'],'sizes':SIZES},sort_keys=True),flush=True); lines,key=make_positive(a.lang,a.rep,A); rows=[]
    for size in SIZES: rows.append(one_size(lines,key,A,a.lang,a.rep,size))
    print('VBM_V10_STAGE_A_RESULT='+json.dumps({'language':a.lang,'rep':a.rep,'rows':rows},sort_keys=True,separators=(',',':')),flush=True)
if __name__=='__main__': main()

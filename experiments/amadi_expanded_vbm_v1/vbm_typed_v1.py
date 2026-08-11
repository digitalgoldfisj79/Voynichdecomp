# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse,collections,concurrent.futures,dataclasses,hashlib,json,math,statistics,sys
import numpy as np
from numba import njit
sys.path.insert(0,'experiments/amadi_residuals_v1');sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
import amadi_residuals_v1 as ar
import vbm_structure_v1 as s0
ar.HEADERS={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36','Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8','Accept-Language':'en-GB,en;q=0.9','Referer':'https://www.voynich.nu/transcr.html'}
NS='VBMV1TYPED'; PLAIN=s0.PLAIN; P2I={c:i for i,c in enumerate(PLAIN)}; VIDX=np.array([P2I[c] for c in 'aeiou'],np.int16); CIDX=np.array([i for i,c in enumerate(PLAIN) if c not in 'aeiou'],np.int16); BVAL=len(PLAIN)
H1=s0.H1; C1=s0.C1
PROPS=60000;MAX_RESTARTS=16;BATCH=4

def seed(*x):return int.from_bytes(hashlib.sha256('::'.join(map(str,x)).encode()).digest()[:8],'big')&0x7fffffff
@dataclasses.dataclass
class LM:
 name:str;logtri:np.ndarray;freq:np.ndarray;control:list[str];meta:dict

def build_lm(name,train,control):
 A=len(PLAIN);C=np.full((A+1,A+1,A+1),0.25,float);F=np.full(A,0.25,float);n=0
 for raw in train:
  q=s0.norm(raw)
  if not q:continue
  z=[A,A]+[P2I[x] for x in q]+[A,A];n+=len(q)
  for x in q:F[P2I[x]]+=1
  for a,b,c in zip(z,z[1:],z[2:]):C[a,b,c]+=1
 C/=C.sum(axis=2,keepdims=True);F/=F.sum()
 return LM(name,np.log(C),F,[s0.norm(x) for x in control if s0.norm(x)],{'train_chars':n,'control_chars':sum(len(s0.norm(x)) for x in control)})
def load_lms():
 cs=s0.corpora();return {la:build_lm(la,*cs[la]) for la in ['bavarian','german','italian']}

def vr(w):return s0.vr(w)
def core_units(w):return s0.core_units(w)
def bridge_label(a,b):return a[-1]+'.'+vr(b)
def target_geometry():
 pages,_=ar.parse_rf();T,H,prior,H2,C2=ar.target_split(pages);FIT=T+H;lines=s0.raw_lines();bc=collections.Counter();cc=collections.Counter()
 for f in FIT:
  for ws in lines.get(f,[]):
   for i,w in enumerate(ws):
    cc.update(core_units(w))
    if i+1<len(ws):bc[bridge_label(w,ws[i+1])]+=1
 bord=sorted(bc,key=lambda x:(-bc[x],x));tot=sum(bc.values());cum=0;kb=0
 for i,x in enumerate(bord,1):
  cum+=bc[x]
  if cum/max(1,tot)>=.995:kb=i;break
 core=sorted(cc);bridges=bord[:kb];return lines,FIT,core,bridges,{'core_K':len(core),'bridge_K':len(bridges),'bridge_fit_coverage':sum(bc[x] for x in bridges)/tot,'core_counts':dict(cc),'bridge_top20':[(x,bc[x]) for x in bord[:20]]}
def target_sequences(lines,folios,core,bridges):
 ci={x:i for i,x in enumerate(core)};bi={x:len(core)+i for i,x in enumerate(bridges)};seqs=[];raw=kept=0;drop=0
 for f in folios:
  for ws in lines.get(f,[]):
   q=[];ok=True
   for i,w in enumerate(ws):
    for x in core_units(w):
     raw+=1
     if x not in ci:ok=False;break
     q.append(ci[x]);kept+=1
    if not ok:break
    if i+1<len(ws):
     x=bridge_label(w,ws[i+1]);raw+=1
     if x not in bi:ok=False;break
     q.append(bi[x]);kept+=1
   if ok and q:seqs.append(q)
   elif q or not ok:drop+=1
 return seqs,{'segments':len(seqs),'events':sum(map(len,seqs)),'raw_events_seen':raw,'kept_events':kept,'dropped_segments':drop}

@dataclasses.dataclass
class Stats:
 a:np.ndarray;b:np.ndarray;c:np.ndarray;n:np.ndarray;off:np.ndarray;adj:np.ndarray;freq:np.ndarray;denom:int;N:int;boundary:int

def make_stats(seqs,N):
 bd=N;cnt=collections.Counter();freq=np.zeros(N,np.int64);den=0
 for q in seqs:
  z=[bd,bd]+list(q)+[bd,bd];den+=len(q)
  for x in q:freq[x]+=1
  for a,b,c in zip(z,z[1:],z[2:]):cnt[(a,b,c)]+=1
 ks=list(cnt);aa=np.array([x[0] for x in ks],np.int32);bb=np.array([x[1] for x in ks],np.int32);cc=np.array([x[2] for x in ks],np.int32);nn=np.array([cnt[x] for x in ks],np.int64)
 ls=[[] for _ in range(N)]
 for j,(a,b,c) in enumerate(ks):
  for t in set((a,b,c)):
   if t!=bd:ls[t].append(j)
 off=[0];adj=[]
 for z in ls:adj+=z;off.append(len(adj))
 return Stats(aa,bb,cc,nn,np.array(off,np.int32),np.array(adj,np.int32),freq,max(1,den),N,bd)
@njit(nogil=True)
def score_raw(a,b,c,n,dec,logp):
 z=0.
 for i in range(len(n)):z+=n[i]*logp[dec[a[i]],dec[b[i]],dec[c[i]]]
 return z
@njit(nogil=True)
def d_one(a,b,c,n,dec,off,adj,t,newv,logp):
 oldv=dec[t];d=0.
 for jj in range(off[t],off[t+1]):
  i=adj[jj];x=a[i];y=b[i];z=c[i];ox=dec[x];oy=dec[y];oz=dec[z];nx=newv if x==t else ox;ny=newv if y==t else oy;nz=newv if z==t else oz;d+=n[i]*(logp[nx,ny,nz]-logp[ox,oy,oz])
 return d

def allowed(t,Kc):return CIDX if t<Kc else VIDX
def init_dec(N,Kc,rng):
 d=np.full(N+1,BVAL,np.int16)
 for lo,hi,vals in [(0,Kc,CIDX),(Kc,N,VIDX)]:
  m=hi-lo;base=list(map(int,vals));a=base+[int(rng.choice(vals)) for _ in range(max(0,m-len(base)))];a=a[:m];rng.shuffle(a);d[lo:hi]=a
 return d
def counts_type(dec,Kc,N,core=True):
 vals=CIDX if core else VIDX;sl=dec[:Kc] if core else dec[Kc:N];return {int(v):int(np.sum(sl==v)) for v in vals}
def propose(dec,Kc,N,rng):
 for _ in range(30):
  t=int(rng.integers(0,N));vals=allowed(t,Kc);old=int(dec[t]);new=int(rng.choice(vals))
  if new==old:continue
  sl=dec[:Kc] if t<Kc else dec[Kc:N]
  if int(np.sum(sl==old))<=1:continue
  return t,new
 return None
def agreement(freq,a,b):return float(np.sum(freq*(a[:len(freq)]==b[:len(freq)]))/max(1,np.sum(freq)))
def one_restart(S,lm,Kc,tag,ens,rr,props=PROPS):
 rng=np.random.default_rng(seed(NS,tag,ens,rr));d=init_dec(S.N,Kc,rng);raw=score_raw(S.a,S.b,S.c,S.n,d,lm.logtri);ds=[]
 for _ in range(80):
  p=propose(d,Kc,S.N,rng)
  if p:ds.append(abs(d_one(S.a,S.b,S.c,S.n,d,S.off,S.adj,p[0],p[1],lm.logtri)/S.denom))
 t0=max(1e-6,(float(np.median(ds)) if ds else 1e-4)*5);best=(raw,d.copy())
 for k in range(props):
  p=propose(d,Kc,S.N,rng)
  if not p:continue
  t,nv=p;dr=d_one(S.a,S.b,S.c,S.n,d,S.off,S.adj,t,nv,lm.logtri);dn=dr/S.denom;frac=k/max(1,props-1);temp=max(1e-8,t0*(0.002**frac))
  if dr>=0 or rng.random()<math.exp(max(-60,dn/temp)):
   d[t]=nv;raw+=dr
   if raw>best[0]:best=(raw,d.copy())
 d=best[1];raw=best[0]
 # deterministic coordinate polish, frequent symbols first
 for _ in range(5):
  changed=False
  for t in np.argsort(-S.freq):
   t=int(t);old=int(d[t]);bestd=0.;bestv=old;sl=d[:Kc] if t<Kc else d[Kc:S.N]
   for nv in allowed(t,Kc):
    nv=int(nv)
    if nv==old or int(np.sum(sl==old))<=1:continue
    dr=d_one(S.a,S.b,S.c,S.n,d,S.off,S.adj,t,nv,lm.logtri)
    if dr>bestd+1e-10:bestd=dr;bestv=nv
   if bestv!=old:d[t]=bestv;raw+=bestd;changed=True
  if not changed:break
 return raw/S.denom,d
def paired_fit(S,lm,Kc,tag,props=PROPS):
 best={'A':(-1e99,None),'B':(-1e99,None)};last=None
 for batch in range(MAX_RESTARTS//BATCH):
  for ens in ['A','B']:
   for j in range(BATCH):
    rr=batch*BATCH+j;s,d=one_restart(S,lm,Kc,tag,ens,rr,props)
    if s>best[ens][0]:best[ens]=(s,d.copy())
  gap=abs(best['A'][0]-best['B'][0]);agr=agreement(S.freq,best['A'][1],best['B'][1]);last={'restarts_each':(batch+1)*BATCH,'score_gap':gap,'agreement':agr}
  if gap<=1e-7 and agr>=.95:break
 win=best['A'] if best['A'][0]>=best['B'][0] else best['B'];return {'fit_score':win[0],'dec':win[1],'agreement':last['agreement'],'converged':bool(last['score_gap']<=1e-7 and last['agreement']>=.95),'restarts_each':last['restarts_each'],'score_gap':last['score_gap']}
def fixed(S,lm,dec):return score_raw(S.a,S.b,S.c,S.n,dec,lm.logtri)/S.denom

def plain_span(control,tag,fitn,holdn):
 st=seed(NS,'span',tag)%len(control);fit=[];hold=[];nf=nh=0;j=0
 while nh<holdn:
  q=control[(st+j)%len(control)];j+=1
  if not q:continue
  if nf<fitn:fit.append(q);nf+=len(q)
  else:hold.append(q);nh+=len(q)
  if j>len(control)*30:raise RuntimeError(('span exhausted',tag,nf,nh))
 return fit,hold
def hidden_map(N,Kc,lm,tag):
 rng=np.random.default_rng(seed(NS,'hidden',tag));d=init_dec(N,Kc,rng)
 # diversify extra homophones using language frequency
 for t in range(Kc):
  if t>=len(CIDX):d[t]=int(rng.choice(CIDX,p=np.array([lm.freq[x] for x in CIDX])/sum(lm.freq[x] for x in CIDX)))
 for t in range(Kc,N):
  if t-Kc>=len(VIDX):d[t]=int(rng.choice(VIDX,p=np.array([lm.freq[x] for x in VIDX])/sum(lm.freq[x] for x in VIDX)))
 # restore surjectivity if overwritten
 for vals,lo,hi in [(CIDX,0,Kc),(VIDX,Kc,N)]:
  for j,v in enumerate(vals):d[lo+j]=int(v)
 return d
def encrypt_plain(seqs,dec,Kc,tag):
 pools=collections.defaultdict(list)
 for t,v in enumerate(dec[:-1]):pools[int(v)].append(t)
 out=[];truth=[]
 for i,s in enumerate(seqs):
  rng=np.random.default_rng(seed(NS,'emit',tag,i));q=[];z=[]
  for ch in s:
   v=P2I[ch];pool=pools[v]
   if not pool:continue
   q.append(int(pool[int(rng.integers(0,len(pool)))]));z.append(v)
  if q:out.append(q);truth.append(z)
 return out,truth
def recovery(seqs,truth,dec):
 ok=n=0
 for q,z in zip(seqs,truth):
  for x,y in zip(q,z):ok+=int(int(dec[x])==int(y));n+=1
 return ok/max(1,n)

def qualify(lms,Kc,Kb,workers=12,smoke=False):
 N=Kc+Kb;reps=1 if smoke else 4;jobs=[(la,r) for la in lms for r in range(reps)]
 def one(j):
  la,r=j;lm=lms[la];fw,hw=plain_span(lm.control,f'Q:{la}:{r}',5000 if smoke else 40000,3000 if smoke else 15000);truth=hidden_map(N,Kc,lm,f'Q:{la}:{r}');fc,ft=encrypt_plain(fw,truth,Kc,f'QF:{la}:{r}');hc,ht=encrypt_plain(hw,truth,Kc,f'QH:{la}:{r}');fs=make_stats(fc,N);hs=make_stats(hc,N);cand=[]
  for cl,clm in lms.items():
   sol=paired_fit(fs,clm,Kc,f'Q:{la}:{r}:{cl}',18000 if smoke else PROPS);sc=fixed(hs,clm,sol['dec']);cand.append((sc,cl,sol))
  cand.sort(key=lambda x:(-x[0],x[1]));top=cand[0];margin=top[0]-cand[1][0];truefit=[x for x in cand if x[1]==la][0];rec=recovery(hc,ht,truefit[2]['dec']);return {'truth':la,'rep':r,'selected':top[1],'margin':margin,'true_score':truefit[0],'true_recovery':rec,'true_agreement':truefit[2]['agreement'],'true_converged':truefit[2]['converged'],'ranking':[(x[1],x[0]) for x in cand]}
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
  for z in ex.map(one,jobs):rows.append(z);print('Q1',json.dumps(z,sort_keys=True),flush=True)
 floors={la:float(np.quantile([x['true_score'] for x in rows if x['truth']==la],.05,method='linear')) for la in lms};langacc=sum(x['truth']==x['selected'] and x['margin']>=.02 for x in rows)/len(rows);rec=[x['true_recovery'] for x in rows];agre=[x['true_agreement'] for x in rows];basepass=bool(statistics.median(rec)>=.95 and min(rec)>=.85 and min(agre)>=.90 and all(x['true_converged'] for x in rows) and langacc>=.90)
 if smoke:return {'controls':rows,'floors':floors,'base_pass':basepass,'negative':[],'false_positives':0,'pass':basepass}
 # structured typed negatives; fit 20k / hold 7k events to keep Q4 bounded.
 kinds=['iid','markov','motif','copy','slot'];negjobs=[(k,r) for k in kinds for r in range(10)]
 def neg(q):
  kind,r=q;rng=np.random.default_rng(seed(NS,'neg',kind,r));L1=20000;L2=7000
  # target-like C/V cadence: 24% bridge events; surface frequencies Zipf-like.
  def gen(L,phase):
   out=[];cur=[]
   for i in range(L):
    isv=(i%4==3) if kind=='slot' else rng.random()<.24
    if isv:
     x=Kc+int(rng.zipf(1.5)-1)%Kb
    else:x=int(rng.zipf(1.5)-1)%Kc
    if kind=='motif' and i>=12 and rng.random()<.75:x=cur[i%12]
    if kind=='copy' and i>=64 and rng.random()<.7:x=cur[i-64]
    if kind=='markov' and i and rng.random()<.7:x=(cur[-1]+1+(r%7))%(Kc if not isv else Kb)+(0 if not isv else Kc)
    cur.append(int(x))
   # split deterministic pseudo-lines
   return [cur[i:i+80] for i in range(0,len(cur),80)]
  fw=gen(L1,0);hw=gen(L2,1);fs=make_stats(fw,N);hs=make_stats(hw,N);cand=[]
  for la,lm in lms.items():
   sol=paired_fit(fs,lm,Kc,f'N:{kind}:{r}:{la}',PROPS);sc=fixed(hs,lm,sol['dec']);cand.append((sc,la,sol))
  cand.sort(key=lambda x:(-x[0],x[1]));m=cand[0][0]-cand[1][0];x=cand[0];pos=bool(x[2]['converged'] and x[0]>=floors[x[1]] and m>=.02);return {'kind':kind,'rep':r,'top':x[1],'score':x[0],'margin':m,'converged':x[2]['converged'],'positive':pos}
 negs=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
  for z in ex.map(neg,negjobs):negs.append(z);print('NEG',json.dumps(z,sort_keys=True),flush=True)
 fp=sum(x['positive'] for x in negs);return {'controls':rows,'floors':floors,'language_accuracy':langacc,'median_recovery':statistics.median(rec),'min_recovery':min(rec),'min_agreement':min(agre),'all_converged':all(x['true_converged'] for x in rows),'base_pass':basepass,'negative':negs,'false_positives':fp,'pass':bool(basepass and fp<=1)}

def target(lms,qual,workers=3):
 lines,FIT,core,bridges,geom=target_geometry();N=len(core)+len(bridges);fw,fmeta=target_sequences(lines,FIT,core,bridges);hw,hmeta=target_sequences(lines,H1,core,bridges);fs=make_stats(fw,N);hs=make_stats(hw,N);rows=[]
 def one(la):
  sol=paired_fit(fs,lms[la],len(core),f'TARGET:{la}',PROPS);sc=fixed(hs,lms[la],sol['dec']);return {'language':la,'fit_score':sol['fit_score'],'H1_score':sc,'floor':qual['floors'][la],'abs_pass':bool(sc>=qual['floors'][la]),'agreement':sol['agreement'],'converged':sol['converged'],'restarts_each':sol['restarts_each'],'score_gap':sol['score_gap']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
  for z in ex.map(one,lms):rows.append(z);print('TARGET',json.dumps(z,sort_keys=True),flush=True)
 rows.sort(key=lambda x:(-x['H1_score'],x['language']));margin=rows[0]['H1_score']-rows[1]['H1_score'];top=rows[0];cand=bool(top['abs_pass'] and top['converged'] and margin>=.02 and qual['pass']);return {'geometry':geom,'FIT':fmeta,'H1':hmeta,'VBM_H1':H1,'VBM_C1':C1,'C1_opened':False,'ranking':rows,'top_margin':margin,'candidate':cand,'bavarian_candidate':bool(cand and top['language']=='bavarian'),'verdict':'VBM_H1_CANDIDATE' if cand else ('UNRESOLVED_SEARCH' if not top['converged'] else 'VBM_TYPED_H1_NEGATIVE')}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','qualify','target'],required=True);ap.add_argument('--qual-url');ap.add_argument('--workers',type=int,default=12);a=ap.parse_args();lms=load_lms();lines,FIT,core,bridges,geom=target_geometry();print('GEOMETRY',json.dumps(geom,sort_keys=True),flush=True)
 if len(core)<len(CIDX) or len(bridges)<len(VIDX):raise RuntimeError(('geometry too small',geom))
 if a.mode in ['smoke','qualify']:
  z=qualify(lms,len(core),len(bridges),a.workers,a.mode=='smoke');z['geometry']=geom;z['namespace']=NS;print('RESULT_JSON',json.dumps(z,sort_keys=True));return
 if not a.qual_url:raise RuntimeError('--qual-url required')
 q=json.loads(ar.getb(a.qual_url).decode());assert q['pass'];z=target(lms,q,min(3,a.workers));print('RESULT_JSON',json.dumps(z,sort_keys=True))
if __name__=='__main__':main()

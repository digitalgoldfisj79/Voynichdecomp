"""Frozen reconstruction of raw-EVA joint generator. No PGCS/root decomposition.
Every fitted stage is atomic and serializable. Native code calculates exact ED-shell masses.
"""
from __future__ import annotations
import collections as C, ctypes as ct, hashlib, json, math, os, pickle, re, sys, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'joint_run'; OUT.mkdir(exist_ok=True)
ATOMS=tuple(list('abcdefghijklmnopqrstuvwxyz')+['I','ch','sh','cth','ckh','cph','cfh'])
A=len(ATOMS); B=A; CAP=32
COMPOUNDS=('cfh','cph','ckh','cth','ch','sh')
HARD=('ci','dh','dn','kk','km','kn','kp','lh','ln','pl','pn','pp','pt','tm','tn','tp','tr','tt','yn')
AI={a:i for i,a in enumerate(ATOMS)}
LIB=ct.CDLL(str(ROOT/'joint_native_v2.so'))
LIB.distance_raw.argtypes=[ct.c_char_p,ct.c_char_p]; LIB.distance_raw.restype=ct.c_int
LIB.dist_many.argtypes=[ct.c_char_p,ct.POINTER(ct.c_char_p),ct.c_int,ct.c_void_p]
LIB.close_masses.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_char_p),ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_longlong]
LIB.close_masses.restype=ct.c_longlong
LIB.set_vocab.argtypes=[ct.POINTER(ct.c_char_p),ct.c_int]
LIB.shell_sample.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_char_p),ct.c_void_p,ct.c_void_p,ct.c_ulonglong,ct.c_int,ct.c_void_p]
LIB.draw_conditioned.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_char_p),ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_ulonglong,ct.c_int,ct.c_char_p]
LIB.draw_conditioned.restype=ct.c_int
LIB.set_atom_trie.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int]
LIB.draw_close_exact.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_char_p),ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_ulonglong,ct.c_char_p,ct.c_void_p]
LIB.draw_close_exact.restype=ct.c_longlong
astr=(ct.c_char_p*A)(*[s.encode() for s in ATOMS])

def atomic(obj,path):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with open(tmp,'wb') as f: pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL); f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atoms(t):
 out=[];i=0
 while i<len(t):
  x=next((s for s in COMPOUNDS if t.startswith(s,i)),t[i]);out.append(x);i+=len(x)
 return out
def fnum(f):return int(re.match(r'f(\d+)',f).group(1))
def section(f):
 n=fnum(f)
 return 'HERBAL' if n<=66 else 'ASTRO' if n<=73 else 'BIO' if n<=84 else 'PHARMA' if n<=102 else 'RECIPES'
def ed(a,b):return LIB.distance_raw(a.encode(),b.encode())
def distances(a,vs):
 bs=(ct.c_char_p*len(vs))(*[v.encode() for v in vs]);out=np.empty(len(vs),np.uint8)
 LIB.dist_many(a.encode(),bs,len(vs),out.ctypes.data);return out
def norm(v):
 z=np.sum(v)
 if z<=0:raise ValueError('zero distribution')
 return v/z
def load_lines(layer='ZLZI'):
 slim=json.load(open(ROOT/'voynich_repo/voynich_transcriptions_slim.json'))
 # Independent primary dump is used when present; other layers use the same dump route.
 p=ROOT/('dump'+layer+'.json')
 if p.exists():
  x=json.load(open(p));d=x.get('structuredContent') or json.loads(x['content'][0]['text']);rs=d['data']
 else:
  rs=[dict(folio=f,line=int(ln),tokens=r.get('t',{}).get(layer,'').split()) for f,ls in slim['pages'].items() for ln,r in ls.items() if r.get('t',{}).get(layer,'')]
 raw=(ROOT/'ZL_source.txt').read_text();cur={};para={};ctr=C.Counter()
 for line in raw.splitlines():
  m=re.match(r'<(f\d+[^>.]*)>.*\$L=([AB])',line)
  if m:cur[m[1]]=m[2]
  m=re.match(r'<(f[^.,>]+)\.(\d+),([^>]*)>\s*(.*)',line)
  if m:
   f,ln,u,pay=m.groups()
   if 'P' in u:
    if '<%>' in pay or ctr[f]==0:ctr[f]+=1
    para[f,int(ln)]=(ctr[f],u)
 out=[]
 for r in rs:
  if not r['tokens']:continue
  f=r['folio'];ln=r['line'];pd=para.get((f,ln));rec=slim['pages'].get(f,{}).get(str(ln),{})
  t=r['tokens']; bad=[x for x in t if any(a not in AI for a in atoms(x))]
  if bad:raise ValueError(('unsupported atom',bad))
  out.append(dict(folio=f,line=ln,tokens=t,section=section(f),currier=cur.get(f,'UNK'),
   para=pd[0] if pd else None,paragraph=bool(pd),unit=rec.get('u',''),fold=fnum(f)%5))
 return out
def transition_rows(lines):
 for r in lines:
  ts=r['tokens']; prev=5
  for i in range(1,len(ts)):
   d=min(4,ed(ts[i-1],ts[i]));yield r,i,ts[i-1],ts[i],d,prev;prev=d
def state_id(a,b,q,run):return (((a*(A+1)+b)*2+q)*4+run)
S=(A+1)**2*8;INIT=state_id(B,B,0,0)
def decode(s):
 run=s%4;s//=4;q=s%2;s//=2;b=s%(A+1);a=s//(A+1);return a,b,q,run
def next_st(s,c):
 a,b,q,run=decode(s);return state_id(b,c,int(q or ATOMS[c]=='q'),min(3,run+1) if ATOMS[c]=='i' else 0)
def legal(s,c,hard=True):
 a,b,q,run=decode(s);old=[ATOMS[k] for k in (a,b) if k!=B];new=old+[ATOMS[c]]
 text=''.join(new)
 if atoms(text)!=new:return False
 if hard and any(x in text for x in HARD):return False
 return True

class AtomModel:
 def __init__(self,lines,cap=CAP,special=True,hard=True):
  self.cap=cap;self.special=special;self.hard=hard
  c0=np.full(A+1,.01);c1=C.defaultdict(lambda:np.zeros(A+1));c2=C.defaultdict(lambda:np.zeros(A+1));cx=C.defaultdict(lambda:np.zeros(A+1));cs=C.defaultdict(lambda:np.zeros(A+1))
  for r in lines:
   for t in r['tokens']:
    st=INIT
    for c in [AI[x] for x in atoms(t)]+[A]:
     a,b,q,run=decode(st);c0[c]+=1;c1[b][c]+=1;c2[a,b][c]+=1;cx[st][c]+=1;cs[r['section'],st][c]+=1
     if c!=A:st=next_st(st,c)
  p0=norm(c0);self.kernel={};self.ns=np.zeros((S,A),np.int32)
  for st in range(S):
   for c in range(A):self.ns[st,c]=next_st(st,c)
  sectors=sorted({r['section'] for r in lines})+['UNK']
  for sec in sectors:
   P=np.zeros((S,A+1))
   for st in range(S):
    a,b,q,run=decode(st);p1=norm(c1[b]+4*p0);p2=norm(c2[a,b]+4*p1)
    px=norm(cx[st]+100*p2) if special else p2
    ps=norm(cs[sec,st]+100*px) if sec!='UNK' else px
    for c in range(A):
     if not legal(st,c,hard):ps[c]=0
    if st==INIT:ps[A]=0
    P[st]=norm(ps)
   self.kernel[sec]=P
  self._close={}
 def P(self,sec):return self.kernel.get(sec,self.kernel['UNK'])
 def prob(self,t,sec):
  aa=atoms(t)
  if len(aa)>self.cap:return 0.
  p=1.;st=INIT;P=self.P(sec)
  for c in aa:
   p*=P[st,AI[c]];st=int(self.ns[st,AI[c]])
  return p*(1. if len(aa)==self.cap else P[st,A])
 def draw(self,sec,rng,initial_weights=None):
  st=INIT;out=[];P=self.P(sec)
  for i in range(self.cap):
   p=P[st].copy()
   if i==0 and initial_weights is not None:p[:A]*=initial_weights;p=norm(p)
   c=int(rng.choice(A+1,p=p))
   if c==A:break
   out.append(ATOMS[c]);st=int(self.ns[st,c])
  return ''.join(out)
 def close(self,source,sec):
  key=source,sec
  if key not in self._close:
   P=self.P(sec);out=np.zeros((4,A));t=time.time()
   visits=LIB.close_masses(source.encode(),A,S,INIT,self.cap,astr,P.ctypes.data,self.ns.ctypes.data,out.ctypes.data,5000000)
   if visits<0:raise RuntimeError(('DP state budget',key,visits))
   self._close[key]=out
  return self._close[key]
 def tail_bound(self,sec):
  P=self.P(sec);v=np.zeros(S);v[INIT]=1.
  for i in range(self.cap):
   nxt=np.zeros(S)
   for c in range(A):np.add.at(nxt,self.ns[:,c],v*P[:,c])
   v=nxt
  return v.sum()

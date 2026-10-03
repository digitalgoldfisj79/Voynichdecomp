"""Run frozen joint generator qualification, prediction and simulations."""
import argparse, collections as C, importlib.util, json, math, multiprocessing as mp, pickle, sys, time
import numpy as np
from joint_model import *

def mi(pairs):
 cs=C.Counter(pairs);N=sum(cs.values())
 if not N:return 0.
 ca=C.Counter();cb=C.Counter()
 for (a,b),v in cs.items():ca[a]+=v;cb[b]+=v
 return sum(v/N*math.log2(v*N/(ca[a]*cb[b])) for (a,b),v in cs.items())
def cmi(events):
 by=C.defaultdict(list)
 for k,a,b in events:by[k].append((a,b))
 N=len(events)
 return sum(len(ps)/max(N,1)*mi(ps) for ps in by.values())
def diagnostics(lines):
 ts=[t for r in lines for t in r['tokens']];freq=C.Counter(ts);N=len(ts);out={}
 ds=C.Counter(d for r,i,a,b,d,p in transition_rows(lines))
 for d in range(5):out['ed_share_'+str(d)]=ds[d]/max(sum(ds.values()),1)
 ep=[(p,d) for r,i,a,b,d,p in transition_rows(lines) if p!=5];out['ed_state_MI']=mi(ep)
 for lag in range(1,13):
  hit=n=0
  for r in lines:
   t=r['tokens'];hit+=sum(t[i]==t[i-lag] for i in range(lag,len(t)));n+=max(0,len(t)-lag)
  out['line_repeat_lag'+str(lag)]=hit/max(n,1)
 bypage=C.defaultdict(list)
 for r in lines:bypage[r['folio']]+=r['tokens']
 for lo,hi in [(1,1),(2,5),(6,12),(13,32),(33,64)]:
  hits=n=0
  for t in bypage.values():
   for i in range(lo,len(t)):
    hits+=any(t[i]==t[j] for j in range(max(0,i-hi),i-lo+1));n+=1
  out[f'page_reuse_{lo}_{hi}']=hits/max(n,1)
 out['n_types']=len(freq);out['hapax_type_fraction']=sum(v==1 for v in freq.values())/len(freq)
 out['hapax_token_fraction']=sum(v==1 for v in freq.values())/N
 out['token_length_mean']=float(np.mean([len(t) for t in ts]));out['atom_length_mean']=float(np.mean([len(atoms(t)) for t in ts]))
 out['hard_zero_violation_rate']=sum(any(x in t for x in HARD) for t in ts)/N
 space=[];cross=[];opairs=[];prby={};pos=[];q=[];ir=[]
 for r in lines:
  t=r['tokens'];aa=[atoms(x) for x in t]
  space.extend((a[-1],b[0]) for a,b in zip(aa,aa[1:]))
  for i,a in enumerate(aa):pos.append((min(4,int(5*i/len(aa))),a[0]))
  pr=prby.get(r['folio'])
  if pr and r['line']==pr['line']+1:
   cross.append((atoms(pr['tokens'][-1])[-1],aa[0][0]))
   if r['para'] is not None and r['para']==pr['para']:opairs.append((atoms(pr['tokens'][0])[0],aa[0][0]))
  prby[r['folio']]=r
  for x in t:
   if x.startswith(('qok','qot')):q.append((x[3:],1,x[2]))
   elif x.startswith(('ok','ot')):q.append((x[2:],0,x[1]))
   for m in re.finditer('i+',x):
    a,b=m.span();p=x[max(0,a-2):a];y=x[b:b+1] or '$';ir.append((p,min(3,b-a),y))
 out['space_edge_MI']=mi(space);out['line_break_edge_MI']=mi(cross);out['space_minus_line_edge_MI']=mi(space)-mi(cross)
 out['relative_position_head_MI']=mi(pos);out['q_kt_tail_CMI']=cmi(q);out['q_kt_opportunities_per_token']=len(q)/N
 out['irun_terminator_CMI']=cmi(ir);out['irun_opportunities_per_token']=len(ir)/N
 out['opener_MI']=mi(opairs);out['opener_same_rate']=np.mean([a==b for a,b in opairs]) if opairs else 0.
 F=sum((a,b) in [('d','q'),('q','y'),('y','d')] for a,b in opairs);B0=sum((a,b) in [('q','d'),('y','q'),('d','y')] for a,b in opairs)
 out['opener_direction_per_pair']=(F-B0)/max(len(opairs),1);out['opener_forward_share']=F/max(F+B0,1)
 return {k:float(v) for k,v in out.items()}

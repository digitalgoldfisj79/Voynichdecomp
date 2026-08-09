#!/usr/bin/env python3
import urllib.request,json,hashlib
import numpy as np
PARENT='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/c7c50f74e1c36d1a772af95a40c95b74d7afa58'
# use canonical parent runner instead of blob URL above
RUN='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/c7c50f74e1f1f88004a0f08ea379324a3d42c16d/experiments/bnf_m19_german_confirm_v1_0/run_confirm.py'
src=urllib.request.urlopen(RUN,timeout=90).read().decode();src=src.rsplit("if __name__=='__main__':main()",1)[0]
lib={'__name__':'parent'};exec(compile(src,'run_confirm.py','exec'),lib)
b=lib['b'];inner=lib['inner'];M=lib['M'];SYMS=lib['SYMS']
def seed(*x):return int.from_bytes(hashlib.sha256(('20260809|WHYINC|'+'|'.join(map(str,x))).encode()).digest()[:8],'big')&0xffffffff

def batch_scores(perms,lms,VB,Vst,Ven,Vfreq):
 E0=b['EMIT'];B=len(perms);out=np.empty((B,len(b['LANGS'])),float)
 for li,la in enumerate(b['LANGS']):
  lm=lms[la];E=E0[perms] # B x letters x values
  start=np.einsum('i,biv->bv',lm['st'],E); start=np.maximum(start,1e-15); start/=start.sum(axis=1,keepdims=True)
  den=np.einsum('i,biv->bv',lm['uni'],E);post=(lm['uni'][None,:,None]*E)/np.maximum(den[:,None,:],1e-15)
  nex=np.einsum('ij,bjw->biw',lm['T'],E,optimize=True)
  trans=np.einsum('biv,biw->bvw',post,nex,optimize=True);trans=np.maximum(trans,1e-15);trans/=trans.sum(axis=2,keepdims=True)
  end=np.einsum('biv,i->bv',post,lm['en']);end=np.maximum(end,1e-15)
  # surface-homophone penalty is identical across languages and architectures for frozen M, so omitted for rank/margin.
  out[:,li]=np.einsum('vw,bvw->b',VB,np.log(trans),optimize=True)+np.einsum('v,bv->b',Vst,np.log(start),optimize=True)+np.einsum('v,bv->b',Ven,np.log(end),optimize=True)
 return out

def main():
 lms,_,_=inner['load_fresh']();data=json.loads(b['fetch'](b['SLIM']));sample,hold,pages,_=inner['split_vms'](data);T={f for f,_,_ in sample};H={f for f,_,_ in hold};A={f for f,_,_ in pages};C=sorted(A-T-H);words=lib['words_for'](data,C,'ZLZI');S=b['stats'](words,SYMS)
 # Collapse Voynich cipher counts to numerical-value counts under frozen key.
 V=b['NV'];VB=np.zeros((V,V),np.int64);Vst=np.zeros(V,np.int64);Ven=np.zeros(V,np.int64);Vfreq=np.zeros(V,np.int64)
 for i in range(len(SYMS)):
  vi=int(M[i]);Vst[vi]+=S['st'][i];Ven[vi]+=S['en'][i];Vfreq[vi]+=S['freq'][i]
  for j in range(len(SYMS)):VB[vi,int(M[j])]+=S['B'][i,j]
 real=batch_scores(np.arange(b['N'])[None,:],lms,VB,Vst,Ven,Vfreq)[0];langs=b['LANGS'];g=langs.index('german');bestother=max(real[i] for i in range(len(langs)) if i!=g);realmargin=float(real[g]-bestother);realrank=1+sum(real[i]>real[g] for i in range(len(langs)) if i!=g)
 N=10000;BS=100;rng=np.random.default_rng(seed('profiles'));margins=[];gtop=0;ge=0
 for st in range(0,N,BS):
  m=min(BS,N-st);perms=np.stack([rng.permutation(b['N']) for _ in range(m)]);sc=batch_scores(perms,lms,VB,Vst,Ven,Vfreq)
  go=np.max(np.delete(sc,g,axis=1),axis=1);gm=sc[:,g]-go;margins.extend(map(float,gm));gtop+=int(np.sum(gm>0));ge+=int(np.sum(gm>=realmargin))
  if st%1000==0:print('PROGRESS',st,flush=True)
 q={str(x):float(np.quantile(margins,x)) for x in [0,.01,.05,.25,.5,.75,.95,.99,1]};out={'n':N,'real_rank':int(realrank),'real_margin_raw':realmargin,'real_scores':{langs[i]:float(real[i]) for i in range(len(langs))},'random_german_top_fraction':gtop/N,'random_margin_ge_real_fraction':ge/N,'empirical_p':(ge+1)/(N+1),'margin_quantiles':q};print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
import urllib.request,json,hashlib
import numpy as np
PARENT='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/c7c50f74e1f1f88004a0f08ea379324a3d42c16d/experiments/bnf_m19_german_confirm_v1_0/run_confirm.py'
src=urllib.request.urlopen(PARENT,timeout=90).read().decode();src=src.rsplit("if __name__=='__main__':main()",1)[0]
lib={'__name__':'parent'};exec(compile(src,'run_confirm.py','exec'),lib)
b=lib['b'];inner=lib['inner'];M=lib['M'];SYMS=lib['SYMS']
def seed(*x):return int.from_bytes(hashlib.sha256(('20260809|WHYFAST|'+'|'.join(map(str,x))).encode()).digest()[:8],'big')&0xffffffff

def rank_exact(words,lms,m):
 r=[]
 for la in b['LANGS']:
  sc,n,sk,cov=inner['forward_words'](words,m,SYMS,lms[la]);r.append((la,float(sc)))
 r.sort(key=lambda x:x[1],reverse=True);return r

def gmargin(r):
 g=next(x[1] for x in r if x[0]=='german');bestother=max(x[1] for x in r if x[0]!='german');return g-bestother

def main():
 lms,_,_=inner['load_fresh']();comps={la:b['induced'](lms[la]) for la in b['LANGS']};data=json.loads(b['fetch'](b['SLIM']));sample,hold,pages,_=inner['split_vms'](data);T={f for f,_,_ in sample};H={f for f,_,_ in hold};A={f for f,_,_ in pages};C=sorted(A-T-H);words=lib['words_for'](data,C,'ZLZI');S=b['stats'](words,SYMS)
 def irank(m):
  r=sorted([(la,float(b['score'](S,m,comps[la]))) for la in b['LANGS']],key=lambda x:x[1],reverse=True);return r
 rng=np.random.default_rng(seed('keynull'));N=20000;rec=[]
 for j in range(N):
  m=b['init_map'](rng);r=irank(m);rec.append((gmargin(r),j,m.copy()))
 rec.sort(key=lambda x:x[0]);idx=np.unique(np.linspace(0,N-1,64).round().astype(int));sel=[rec[int(i)] for i in idx]
 base=rank_exact(words,lms,M);bm=gmargin(base);rows=[]
 for k,(im,j,m) in enumerate(sel):
  r=rank_exact(words,lms,m);row={'qindex':int(k),'source_index':int(j),'induced_gmargin':float(im),'exact_rank':1+next(i for i,x in enumerate(r) if x[0]=='german'),'exact_gmargin':gmargin(r),'ranking':r};rows.append(row);print('EXACT_NULL',json.dumps(row,separators=(',',':')),flush=True)
 n=len(rows);top=sum(x['exact_rank']==1 for x in rows);ge=sum(x['exact_gmargin']>=bm for x in rows);out={'n':n,'frozen_exact_ranking':base,'frozen_exact_gmargin':bm,'null_german_top_fraction':top/n,'null_exact_margin_ge_frozen':ge/n,'empirical_p_upper':(ge+1)/(n+1),'exact_margin_quantiles':{str(q):float(np.quantile([x['exact_gmargin'] for x in rows],q)) for q in [0,.05,.25,.5,.75,.95,1]},'rows':rows};print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':main()

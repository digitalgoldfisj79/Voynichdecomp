#!/usr/bin/env python3
import urllib.request,json,csv,io,hashlib,math
from collections import Counter,defaultdict
import numpy as np

PARENT='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/c7c50f74e1f1f88004a0f08ea379324a3d42c16d/experiments/bnf_m19_german_confirm_v1_0/run_confirm.py'
src=urllib.request.urlopen(PARENT,timeout=90).read().decode('utf-8');src=src.rsplit("if __name__=='__main__':main()",1)[0]
lib={'__name__':'parent'};exec(compile(src,'run_confirm.py','exec'),lib)
b=lib['b'];inner=lib['inner'];M=lib['M'];SYMS=lib['SYMS'];RAW=lib['RAW']
SEC_URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_section_map.json'
MAN_URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/daiin_manifest.csv'
EXTRA=['ZLZB','VDRB-1','TTVE','TTIA','TTII','GCGA','GCGI','FFSG','FFSG-1','FFSG-2','RGVN','PCCA','PCCI','JSLI','JGLI']


def seed(*x):return int.from_bytes(hashlib.sha256(('20260809|WHYFAST|'+'|'.join(map(str,x))).encode()).digest()[:8],'big')&0xffffffff

def exact_rank(words,lms,m=M):
 rows=[]
 for la in b['LANGS']:
  sc,n,sk,cov=inner['forward_words'](words,m,SYMS,lms[la]);rows.append((la,float(sc),int(n),int(sk),float(cov)))
 rows.sort(key=lambda x:x[1],reverse=True);return rows

def split_bad(words,bad):
 out=[]
 for w in words:
  cur=[]
  for c in w:
   if c in bad:
    if cur:out.append(''.join(cur));cur=[]
   else:cur.append(c)
  if cur:out.append(''.join(cur))
 return out

def compact_rank(r):return [(x[0],x[1]) for x in r]

def margin_for(r,lang='german'):
 pos=next(i for i,x in enumerate(r) if x[0]==lang)
 if pos==0:return r[0][1]-r[1][1]
 return next(x[1] for x in r if x[0]==lang)-r[0][1]

def main():
 lms,_,_=inner['load_fresh']();comps={la:b['induced'](lms[la]) for la in b['LANGS']}
 data=json.loads(b['fetch'](b['SLIM']));sample,hold,pages,required=inner['split_vms'](data);T={f for f,_,_ in sample};H={f for f,_,_ in hold};A={f for f,_,_ in pages};C=sorted(A-T-H)
 words=lib['words_for'](data,C,'ZLZI');S=b['stats'](words,SYMS);base=exact_rank(words,lms);print('BASE',json.dumps({'letters':sum(map(len,words)),'tokens':len(words),'ranking':compact_rank(base),'margin':margin_for(base)},separators=(',',':')),flush=True)
 out={'base':{'ranking':compact_rank(base),'margin':margin_for(base)}}

 # T1: fast induced legal-key null.
 def induced_rank(m):
  rr=[]
  for la in b['LANGS']:rr.append((la,float(b['score'](S,m,comps[la]))))
  rr.sort(key=lambda x:x[1],reverse=True);return rr
 obs_i=induced_rank(M);obs_m=margin_for(obs_i);rng=np.random.default_rng(seed('keynull'));NNULL=20000;gtop=0;ge=0;marg=[];maps=[]
 for j in range(NNULL):
  m=b['init_map'](rng);r=induced_rank(m);mm=margin_for(r);marg.append(mm);maps.append(m.copy());gtop+=int(r[0][0]=='german');ge+=int(mm>=obs_m)
 order=np.argsort(marg);qs=np.linspace(0,NNULL-1,64).round().astype(int);sel=[maps[int(order[q])].tolist() for q in qs]
 t1={'n':NNULL,'frozen_ranking':obs_i,'frozen_german_margin':obs_m,'random_german_top_fraction':gtop/NNULL,'random_margin_ge_frozen_fraction':ge/NNULL,'margin_quantiles':{str(q):float(np.quantile(marg,q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]},'stratified_exact_null_maps':sel}
 out['T1']=t1;print('T1_KEY_NULL='+json.dumps(t1,separators=(',',':')),flush=True)

 # T2: singleton, all-value and all-symbol ablations.
 vals=sorted(set(RAW.values()));abl=[]
 for val in vals:
  bad={s for s,v in RAW.items() if v==val};w=split_bad(words,bad);r=exact_rank(w,lms);abl.append({'value':val,'symbols':sorted(bad),'letters':sum(map(len,w)),'coverage':sum(map(len,w))/sum(map(len,words)),'ranking':compact_rank(r),'german_rank':1+next(i for i,x in enumerate(r) if x[0]=='german'),'german_margin':margin_for(r)})
 single_symbols={'u','b','g','n','v','x','s'};w=split_bad(words,single_symbols);r=exact_rank(w,lms);joint={'symbols':sorted(single_symbols),'letters':sum(map(len,w)),'coverage':sum(map(len,w))/sum(map(len,words)),'ranking':compact_rank(r),'german_margin':margin_for(r)}
 symabl=[]
 for s in SYMS:
  w=split_bad(words,{s});r=exact_rank(w,lms);symabl.append({'symbol':s,'raw_value':RAW[s],'freq':S['freq'][SYMS.index(s)].item(),'letters':sum(map(len,w)),'german_rank':1+next(i for i,x in enumerate(r) if x[0]=='german'),'german_margin':margin_for(r)})
 out['T2']={'by_value':abl,'all_singletons':joint,'by_symbol':symabl};print('T2_VALUES='+json.dumps(abl,separators=(',',':')),flush=True);print('T2_SYMBOLS='+json.dumps(sorted(symabl,key=lambda z:z['german_margin']),separators=(',',':')),flush=True)

 # T3: score decomposition and boundary tests.
 cnt=np.bincount(M,minlength=b['NV']);hom=-float(np.dot(S['freq'],np.log(cnt[M])))
 parts={}
 for la in b['LANGS']:
  lt,ls,le=comps[la];internal=float(np.sum(S['B']*lt[np.ix_(M,M)]));start=float(np.dot(S['st'],ls[M]));end=float(np.dot(S['en'],le[M]));parts[la]={'internal':internal/S['denom'],'start':start/S['denom'],'end':end/S['denom'],'homophone':hom/S['denom'],'total':(internal+start+end+hom)/S['denom']}
 # component-only ranks (homophone cancels across languages).
 def comp_rank(k):return sorted([(la,parts[la][k]) for la in b['LANGS']],key=lambda x:x[1],reverse=True)
 # unigram numerical channel.
 unir=[]
 for la in b['LANGS']:
  p=np.maximum(lms[la]['uni']@b['EMIT'],1e-15);sc=float(np.dot(S['freq'],np.log(p[M]))+hom)/max(1,S['freq'].sum());unir.append((la,sc))
 unir.sort(key=lambda x:x[1],reverse=True)
 # exact reversed and page-concatenated no-word-boundary conditions.
 rev=[w[::-1] for w in words];rev_rank=exact_rank(rev,lms)
 pagewords=[''.join(lib['words_for'](data,[f],'ZLZI')) for f in C];pagewords=[x for x in pagewords if x];page_rank=exact_rank(pagewords,lms)
 # top internal German-vs-French transition contributions.
 ltg=comps['german'][0];ltf=comps['french'][0];D=S['B']*(ltg[np.ix_(M,M)]-ltf[np.ix_(M,M)])/S['denom'];flat=[]
 for i,a in enumerate(SYMS):
  for j,c in enumerate(SYMS):
   if S['B'][i,j]:flat.append((float(D[i,j]),int(S['B'][i,j]),a,c,RAW[a],RAW[c]))
 flat.sort(reverse=True)
 t3={'parts':parts,'internal_rank':comp_rank('internal'),'start_rank':comp_rank('start'),'end_rank':comp_rank('end'),'unigram_rank':unir,'reversed_rank':compact_rank(rev_rank),'reversed_margin':margin_for(rev_rank),'page_concat_rank':compact_rank(page_rank),'page_concat_margin':margin_for(page_rank),'top_german_vs_french_internal_positive':flat[:30],'top_negative':flat[-30:]}
 out['T3']=t3;print('T3='+json.dumps(t3,separators=(',',':')),flush=True)

 # T5: untouched C10 across additional transliteration surfaces. No refit.
 transfers={}
 for tid in EXTRA:
  ww=lib['words_for'](data,C,tid);tot=sum(map(len,ww));known=sum(sum(c in set(SYMS) for c in w) for w in ww);cov=known/max(1,tot);row={'folios':sum(bool(b['extract_page'](data,f,tid)) for f in C),'letters':tot,'coverage':cov}
  if tot and cov>=.90:
   rr=exact_rank(ww,lms);row.update({'ranking':compact_rank(rr),'german_rank':1+next(i for i,x in enumerate(rr) if x[0]=='german'),'german_margin':margin_for(rr)})
  transfers[tid]=row;print('T5',tid,json.dumps(row,separators=(',',':')),flush=True)
 out['T5']=transfers

 # T7: fixed-key section and Currier diagnostics.
 sec=json.loads(b['fetch'](SEC_URL))['mapping'];bysec=defaultdict(list)
 for f in C:
  if f in sec:bysec[sec[f]].append(f)
 secres={}
 for g,fs in sorted(bysec.items()):
  ww=lib['words_for'](data,fs,'ZLZI');
  if sum(map(len,ww))<500:continue
  rr=exact_rank(ww,lms);secres[g]={'folios':len(fs),'letters':sum(map(len,ww)),'ranking':compact_rank(rr),'german_rank':1+next(i for i,x in enumerate(rr) if x[0]=='german'),'german_margin':margin_for(rr)}
 man=list(csv.DictReader(io.StringIO(b['fetch'](MAN_URL))));cl={r['folio']:r['lang'] for r in man};byc=defaultdict(list)
 for f in C:
  if f in cl:byc[cl[f]].append(f)
 cures={}
 for g,fs in sorted(byc.items()):
  ww=lib['words_for'](data,fs,'ZLZI');rr=exact_rank(ww,lms);cures[g]={'folios':len(fs),'letters':sum(map(len,ww)),'ranking':compact_rank(rr),'german_rank':1+next(i for i,x in enumerate(rr) if x[0]=='german'),'german_margin':margin_for(rr)}
 out['T7']={'sections':secres,'currier':cures};print('T7='+json.dumps(out['T7'],separators=(',',':')),flush=True)
 print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__':main()

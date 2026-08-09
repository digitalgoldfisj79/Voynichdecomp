#!/usr/bin/env python3
import urllib.request,json,hashlib,csv,io,re
import numpy as np
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/185f0c55e910ae075dd945402f590586e1bd02cd/experiments/bnf_m19_hmm_v0_9/run_v09.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8'); marker="ns['main']()"; pos=src.rfind(marker)
if pos<0: raise RuntimeError('v0.9 main marker missing')
lib={'__name__':'v09lib'};exec(compile(src[:pos],'run_v09.py','exec'),lib)
inner=lib['ns'];b=lib['b']
RAW={'a':5,'b':22,'c':6,'d':4,'e':1,'f':16,'g':22,'h':3,'i':10,'j':20,'k':2,'l':12,'m':9,'n':23,'o':1,'p':7,'q':4,'r':24,'s':30,'t':8,'u':0,'v':28,'x':28,'y':5,'z':20}
SYMS=sorted(RAW);M=np.asarray([b['V2I'][RAW[s]] for s in SYMS],dtype=np.int16);assert b['valid_map'](M)
MAN='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/daiin_manifest.csv'

def words_for(data,folios,tid):
 out=[]
 for f in folios:
  if f in data['pages']:out.extend(b['extract_page'](data,f,tid))
 return out

def exact_coverage(words):
 ss=set(SYMS);tot=sum(map(len,words));known=sum(sum(c in ss for c in w) for w in words);return known/max(1,tot),known,tot

def rank_words(words,lms):
 rows=[]
 for la in b['LANGS']:
  sc,n,sk,cov=inner['forward_words'](words,M,SYMS,lms[la]);rows.append((la,sc,n,sk))
 rows.sort(key=lambda x:x[1],reverse=True);return rows

def lexical_z256(words,lm,tag):
 obs,hit,tot=b['lexical'](words,M,SYMS,lm);rng=np.random.default_rng(b['seed']('v10lex',tag));vals=[]
 for _ in range(256):
  x=M.copy();rng.shuffle(x);vals.append(b['lexical'](words,x,SYMS,lm)[0])
 mu=float(np.mean(vals));sd=float(np.std(vals,ddof=1));return {'fraction':obs,'hits':hit,'tokens':tot,'null_mean':mu,'null_sd':sd,'z':(obs-mu)/sd if sd>1e-15 else 0.0}

def bucket_of(f):
 h=hashlib.sha256(('20260809|M19GermanConfirm|bucket|'+f).encode()).digest();return int.from_bytes(h[:8],'big')%4

def canonical_folios(folios):
 try:
  rows=list(csv.DictReader(io.StringIO(b['fetch'](MAN))));page={r['folio']:int(r['page']) for r in rows if r.get('page','').isdigit()}
 except Exception:page={}
 def key(f):
  if f in page:return (0,page[f],f)
  m=re.match(r'f(\d+)([rv])(\d*)',f)
  return (1,int(m.group(1)) if m else 9999,0 if m and m.group(2)=='r' else 1,int(m.group(3) or 0) if m else 0,f)
 return sorted(folios,key=key)

def first_decodes(data,folios,lm,nmax=100):
 out=[];s2i={s:i for i,s in enumerate(SYMS)}
 for f in canonical_folios(folios):
  for lk,line in sorted(data['pages'][f].items(),key=lambda kv:int(kv[0]) if str(kv[0]).isdigit() else 99999):
   for ti,tok in enumerate(line.get('t',{}).get('ZLZI','').split()):
    z=''.join(c.lower() for c in tok if c.isalpha())
    if not z:continue
    dec=b['viterbi'](z,M,SYMS,lm)
    out.append({'folio':f,'line':str(lk),'token_index':ti,'cipher':z,'decode':dec,'dictionary_hit':bool(dec in lm['vocab']) if dec is not None else False})
    if len(out)>=nmax:return out
 return out

def main():
 lms,pools,meta=inner['load_fresh']();data=json.loads(b['fetch'](b['SLIM']));sample,hold,pages,required=inner['split_vms'](data)
 T={f for f,_,_ in sample};H={f for f,_,_ in hold};A={f for f,_,_ in pages};C=sorted(A-T-H)
 if T&H or T&set(C) or H&set(C):raise RuntimeError('split overlap')
 if len(C)!=122:raise RuntimeError(('unexpected C10 size',len(C)))
 print('FRESH_PANEL',json.dumps({'all':len(A),'T09':len(T),'H09':len(H),'C10':len(C),'overlap':0},separators=(',',':')),flush=True)
 out={'protocol':'v1.0','mapping':RAW,'panel':{'T09':sorted(T),'H09':sorted(H),'C10':C},'surfaces':{},'buckets':[]}
 primary_ok=True
 for tid,margin_floor,cov_floor,lex_floor in [('ZLZI',.05,.99,5),('TTLI',.03,.90,3),('VDRB',.03,.90,3)]:
  words=words_for(data,C,tid);cov,known,total=exact_coverage(words);rank=rank_words(words,lms);grank=1+next(i for i,x in enumerate(rank) if x[0]=='german');gscore=next(x[1] for x in rank if x[0]=='german');margin=(rank[0][1]-rank[1][1]) if rank[0][0]=='german' else None;lex=lexical_z256(words,lms['german'],('C10',tid));r={'folios_with_text':sum(bool(b['extract_page'](data,f,tid)) for f in C if f in data['pages']),'coverage':cov,'known_letters':known,'total_letters':total,'ranking':[(x[0],x[1]) for x in rank],'german_rank':grank,'german_score':gscore,'german_margin':margin,'lexical':lex};out['surfaces'][tid]=r;passed=cov>=cov_floor and grank==1 and margin is not None and margin>=margin_floor and lex['z']>=lex_floor;primary_ok=primary_ok and passed;print('SURFACE',tid,'PASS' if passed else 'FAIL',json.dumps(r,separators=(',',':')),flush=True)
 # Binding four-bucket ZLZI stability.
 bucket_ranks=[]
 for k in range(4):
  fs=[f for f in C if bucket_of(f)==k];words=words_for(data,fs,'ZLZI');rank=rank_words(words,lms);grank=1+next(i for i,x in enumerate(rank) if x[0]=='german');margin=(rank[0][1]-rank[1][1]) if rank[0][0]=='german' else -abs(next(x[1] for x in rank if x[0]=='german')-rank[0][1]);r={'bucket':k,'folios':len(fs),'letters':sum(map(len,words)),'ranking':[(x[0],x[1]) for x in rank],'german_rank':grank,'german_margin':margin};bucket_ranks.append(r);print('BUCKET',json.dumps(r,separators=(',',':')),flush=True)
 bucket_ok=sum(r['german_rank']==1 for r in bucket_ranks)>=3 and float(np.median([r['german_margin'] for r in bucket_ranks]))>0;out['buckets']=bucket_ranks;out['bucket_ok']=bucket_ok
 confirmed=primary_ok and bucket_ok;out['verdict']='CONFIRMED FRESH-PANEL GERMAN M19 SIGNAL' if confirmed else 'GERMAN M19 LEAD FAILS FRESH CONFIRMATION'
 if confirmed:
  out['first_100_decodes']=first_decodes(data,C,lms['german'],100)
 print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
import json,csv,io,re,hashlib,urllib.request,urllib.parse
from collections import Counter,defaultdict
import numpy as np
from unidecode import unidecode

SEED0=20260809
ALPH='abcdefghiklmnopqrstuxyz';A2I={c:i for i,c in enumerate(ALPH)}
ORDER=['F','M','G','L','H']
TABLES={
'F':[1,2,3,4,5,6,7,8,9,10,10,2,12,22,4,12,24,6,16,4,20,8,24],
'M':[1,2,3,4,5,28,10,12,1,16,2,12,23,6,2,20,3,30,9,1,20,0,4],
'G':[1,2,6,4,5,8,1,6,7,1,8,8,5,6,5,2,2,1,4,1,1,3,3],
'L':[1,2,6,4,1,8,4,3,10,2,3,8,5,6,8,7,2,6,1,6,5,0,7],
'H':[1,2,6,4,5,6,3,1,3,6,2,4,1,6,7,2,8,6,1,6,1,0,7],
}
VALUE_SETS={t:set(v) for t,v in TABLES.items()}
ALL_VALUES=sorted(set().union(*VALUE_SETS.values()))
assert ALL_VALUES==[0,1,2,3,4,5,6,7,8,9,10,12,16,20,22,23,24,28,30]
SCHEDULES=['CHAR_CONTINUOUS','CHAR_WORD_RESET','WORD_CONTINUOUS','WORD_LINE_RESET','LINE_CONTINUOUS']
LM_URLS={
'latin':'https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-train.conllu',
'italian':'https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu',
'german':'https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu',
'hebrew':'https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/master/he_htb-ud-train.conllu',
}
SEF='https://storage.googleapis.com/sefaria-export/json/Halakhah/Mishneh Torah/Sefer Madda/Mishneh Torah, Torah Study/Hebrew/Torat Emet 363.json'
SLIM='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json'
MAN='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/daiin_manifest.csv'

def seed(*p):
 h=hashlib.sha256(('::'.join(map(str,p))).encode()).digest();return (SEED0+int.from_bytes(h[:8],'big'))&0xffffffff

def fetch(u):
 q=urllib.parse.quote(u,safe=':/?=&%');req=urllib.request.Request(q,headers={'User-Agent':'Mozilla/5.0 BnF-v06'})
 with urllib.request.urlopen(req,timeout=90) as r:return r.read().decode('utf-8','replace')

def conllu_lines(txt):
 out=[];cur=[]
 for ln in txt.splitlines():
  if not ln:
   if cur:out.append(' '.join(cur));cur=[]
   continue
  if ln.startswith('#'):continue
  c=ln.split('\t')
  if len(c)>=2 and c[0].isdigit():cur.append(c[1])
 if cur:out.append(' '.join(cur))
 return out

def norm_line(s):
 s=unidecode(s).lower().replace('j','i').replace('v','u').replace('w','u')
 words=[]
 for w in re.findall(r'[a-z]+',s):
  z=''.join(c for c in w if c in A2I)
  if z:words.append(z)
 return words

def split_hold(lines):return [s for i,s in enumerate(lines) if i%5==0]

def load_control_lines():
 out={}
 for lang,u in LM_URLS.items():
  lines=conllu_lines(fetch(u));hold=split_hold(lines);normed=[norm_line(x) for x in hold];normed=[x for x in normed if x];out[lang]=normed
  print('CONTROL_SOURCE',lang,'lines',len(normed),'letters',sum(len(w) for l in normed for w in l),flush=True)
 # append Sefaria chunks as independent lines for Hebrew, control plaintext only
 obj=json.loads(fetch(SEF));chunks=[]
 def walk(x):
  if isinstance(x,str):chunks.append(x)
  elif isinstance(x,list):
   for y in x:walk(y)
 walk(obj.get('text',[]));extra=[norm_line(x) for x in chunks];extra=[x for x in extra if x];out['hebrew']+=extra
 print('CONTROL_SOURCE_EXT hebrew lines',len(out['hebrew']),'letters',sum(len(w) for l in out['hebrew'] for w in l),flush=True)
 # cap at 150k letters by whole words/lines, preserving line structure; Hebrew uses all if shorter
 capped={}
 for lang,lines in out.items():
  cap=150000;acc=[];n=0
  for line in lines:
   nl=[]
   for w in line:
    if n>=cap:break
    take=w if n+len(w)<=cap else w[:cap-n]
    if take:nl.append(take);n+=len(take)
   if nl:acc.append(nl)
   if n>=cap:break
  capped[lang]=acc
  print('CONTROL_CAPPED',lang,'lines',len(acc),'letters',n,flush=True)
 return capped

def load_vms_lines():
 data=json.loads(fetch(SLIM));rows=list(csv.DictReader(io.StringIO(fetch(MAN))));page={r['folio']:int(r['page']) for r in rows if r.get('page','').isdigit()}
 def fkey(f):
  if f in page:return (0,page[f],f)
  m=re.match(r'f(\d+)([rv])(\d*)',f);return (1,int(m.group(1)) if m else 9999,0 if m and m.group(2)=='r' else 1,int(m.group(3) or 0) if m else 0,f)
 lines=[]
 for f in sorted(data['pages'],key=fkey):
  pg=data['pages'][f]
  for k,line in sorted(pg.items(),key=lambda kv:int(kv[0]) if str(kv[0]).isdigit() else 99999):
   t=line.get('t',{}).get('ZLZI','');words=[]
   for tok in t.split():
    z=''.join(c.lower() for c in tok if c.isalpha())
    if z:words.append(z)
   if words:lines.append((f,str(k),words))
 return lines

def table_for_phase(phase,rot):return ORDER[(phase+rot)%5]

def schedule_occurrences(lines,family,rot,plaintext=False):
 # returns list (symbol-or-letter, table), preserving only characters
 out=[];char_global=0;word_global=0;line_global=0
 for item in lines:
  words=item if plaintext else item[2]
  for wi,w in enumerate(words):
   if family=='WORD_CONTINUOUS':wt=table_for_phase(word_global,rot)
   elif family=='WORD_LINE_RESET':wt=table_for_phase(wi,rot)
   elif family=='LINE_CONTINUOUS':wt=table_for_phase(line_global,rot)
   else:wt=None
   for ci,ch in enumerate(w):
    if family=='CHAR_CONTINUOUS':t=table_for_phase(char_global,rot)
    elif family=='CHAR_WORD_RESET':t=table_for_phase(ci,rot)
    else:t=wt
    out.append((ch,t))
    char_global+=1
   word_global+=1
  line_global+=1
 return out

def control_numeric(lines,family,rot):
 cnt=Counter();total=0
 for ch,t in schedule_occurrences(lines,family,rot,plaintext=True):
  if ch not in A2I:continue
  v=TABLES[t][A2I[ch]];cnt[v]+=1;total+=1
 return cnt,total

def vms_domains(lines,family,rot):
 phases=defaultdict(set);freq=Counter()
 for ch,t in schedule_occurrences(lines,family,rot,plaintext=False):
  phases[ch].add(t);freq[ch]+=1
 domains={g:set.intersection(*(VALUE_SETS[t] for t in ts)) for g,ts in phases.items()}
 return phases,freq,domains

def fill_remaining(glyphs,domains,assign,counts,expected):
 rem=[g for g in glyphs if g not in assign]
 def rec(todo):
  if not todo:return dict(assign)
  # MRV under current counts
  best=None;optsbest=None
  for g in todo:
   opts=[v for v in sorted(domains[g]) if counts[v]<3]
   if not opts:return None
   if optsbest is None or len(opts)<len(optsbest):best,optsbest=g,opts
  rest=[x for x in todo if x!=best]
  for v in optsbest:
   assign[best]=v;counts[v]+=1
   z=rec(rest)
   if z is not None:return z
   counts[v]-=1;del assign[best]
  return None
 return rec(rem)

def exact_assignment(glyphs,domains,expected):
 # each expected value needs its own glyph. Search rarest value first, then fill extras max 3/value.
 cand={v:[g for g in glyphs if v in domains[g]] for v in expected}
 if any(not cand[v] for v in expected):return None,{'missing_values':[v for v in expected if not cand[v]]}
 req=sorted(expected,key=lambda v:(len(cand[v]),v));assign={};counts=Counter()
 nodes=0
 def rec(i,used):
  nonlocal nodes
  nodes+=1
  if i==len(req):return fill_remaining(glyphs,domains,assign,counts,set(expected))
  v=req[i]
  for g in sorted(cand[v],key=lambda x:(len(domains[x]),x)):
   if g in used:continue
   assign[g]=v;counts[v]+=1
   z=rec(i+1,used|{g})
   if z is not None:return z
   counts[v]-=1;del assign[g]
  return None
 z=rec(0,set());return z,{'nodes':nodes,'candidate_counts':{str(v):len(cand[v]) for v in expected}}

def freq_sse(assign,freq,target):
 total=sum(freq.values());agg=Counter()
 for g,v in assign.items():agg[v]+=freq[g]
 return sum(((agg[v]/total)-target.get(v,0.0))**2 for v in ALL_VALUES)

def local_freq_search(first,glyphs,domains,freq,target,expected,tag):
 a=dict(first);cnt=Counter(a.values());best=dict(a);bs=freq_sse(a,freq,target);rng=np.random.default_rng(seed('freq',tag))
 for _ in range(50000):
  if rng.random()<.25:
   g1,g2=rng.choice(glyphs,size=2,replace=False);v1,v2=a[g1],a[g2]
   if v2 in domains[g1] and v1 in domains[g2]:
    old=freq_sse(a,freq,target);a[g1],a[g2]=v2,v1;new=freq_sse(a,freq,target)
    if new<=old or rng.random()<.002:
     if new<bs:bs=new;best=dict(a)
    else:a[g1],a[g2]=v1,v2
  else:
   g=str(rng.choice(glyphs));oldv=a[g];opts=[v for v in domains[g] if v!=oldv and cnt[v]<3 and not (oldv in expected and cnt[oldv]<=1)]
   if not opts:continue
   newv=int(rng.choice(opts));old=freq_sse(a,freq,target);a[g]=newv;cnt[oldv]-=1;cnt[newv]+=1;new=freq_sse(a,freq,target)
   if new<=old or rng.random()<.002:
    if new<bs:bs=new;best=dict(a)
   else:a[g]=oldv;cnt[newv]-=1;cnt[oldv]+=1
 return best,bs

def main():
 controls=load_control_lines();vms=load_vms_lines();alphabet=sorted(set(ch for _,_,ws in vms for w in ws for ch in w));freqall=Counter(ch for _,_,ws in vms for w in ws for ch in w)
 print('VMS', 'lines',len(vms),'letters',sum(freqall.values()),'alphabet',len(alphabet),alphabet,flush=True)
 calib={};results=[]
 for fam in SCHEDULES:
  control_runs=[]
  for lang in ['latin','italian','german','hebrew']:
   for rot in range(5):
    cnt,n=control_numeric(controls[lang],fam,rot);control_runs.append({'lang':lang,'rot':rot,'seen':sorted(cnt),'freq':{str(v):cnt[v]/n for v in ALL_VALUES}})
  expected=sorted(set.intersection(*(set(x['seen']) for x in control_runs)))
  target={v:float(np.median([float(x['freq'][str(v)]) for x in control_runs])) for v in ALL_VALUES}
  calib[fam]={'expected':expected,'n_expected':len(expected),'runs':control_runs,'median_freq':{str(v):target[v] for v in ALL_VALUES}}
  print('CALIB',fam,'expected',expected,flush=True)
  for rot in range(5):
   phases,freq,domains=vms_domains(vms,fam,rot);empty=[g for g in alphabet if not domains[g]]
   first,diag=(None,{'empty_domains':empty}) if empty else exact_assignment(alphabet,domains,expected)
   row={'family':fam,'rotation':rot,'expected':expected,'alphabet':len(alphabet),'empty_domains':empty,'legal_sizes':{g:len(domains[g]) for g in alphabet},'table_support':{g:sorted(phases[g]) for g in alphabet},'status':'REJECT'}
   row['assignment_search']=diag
   if first is not None:
    target2=target;firsts=freq_sse(first,freq,target2);best,bs=local_freq_search(first,alphabet,domains,freq,target2,expected,(fam,rot));row.update({'status':'PASS','assignment':first,'first_frequency_sse':firsts,'approx_frequency_sse':bs,'approx_assignment':best})
   results.append(row);print('SCHEDULE',json.dumps({k:row[k] for k in ['family','rotation','status','empty_domains','assignment_search']},separators=(',',':')),flush=True)
 survivors=[r for r in results if r['status']=='PASS'];verdict='SURVIVORS — REQUIRE v0.7' if survivors else 'ALL FROZEN UNMARKED SCHEDULES STRUCTURALLY REJECTED'
 out={'protocol':'v0.6','all_values':ALL_VALUES,'vms_alphabet':alphabet,'vms_letters':sum(freqall.values()),'calibration':calib,'results':results,'survivors':[(r['family'],r['rotation']) for r in survivors],'verdict':verdict}
 print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__':main()

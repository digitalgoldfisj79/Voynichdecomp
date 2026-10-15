from collections import Counter,defaultdict
from pathlib import Path
import math, random, pickle, json, hashlib, statistics, tempfile, os
P=Path(__file__).resolve().parent
BASE=Path('/mnt/data/vbm_v143_crosscarry')
REF=BASE/'vbm_v142_inband'/'hartlieb.pair.vbm'
ARMS={'baseline':REF,'carry':BASE/'hartlieb.carry.vbm','statefree':BASE/'hartlieb.statefree.vbm'}
CONTRACT=(P/'CONTRACT.md').read_bytes()
def atomic_pickle(name, obj):
 fd,t=tempfile.mkstemp(dir=P,prefix='.tmp')
 with os.fdopen(fd,'wb') as f: pickle.dump(obj,f,protocol=5);f.flush();os.fsync(f.fileno())
 os.replace(t,P/name)

def events(words):
 o=[]
 for i in range(1,len(words)):
  w=words[i];
  if not (5<=len(w)<=20):continue
  j=len(w)//2
  # Include all other characters in shell? NO. Specifically prefix 2 and final 1 with current word length
  shell=(len(w),w[:2],w[-1])
  o.append((i,shell,w[j],words[i-1][-3:]))
 return o

def fit(train):
 gl=Counter(y for _,sh,y,x in train)
 n=sum(gl.values());pr={k:v/n for k,v in gl.items()}
 bysh=defaultdict(Counter);byx=defaultdict(Counter);cnt=Counter()
 for _,sh,y,x in train:
  bysh[sh][y]+=1;byx[(sh,x)][y]+=1;cnt[sh]+=1
 admitted={sh for sh,c in bysh.items() if sum(c.values())>=10 and len(c)>=2}
 return pr,bysh,byx,cnt,admitted

def pred(tup,model):
 i,sh,y,x=tup;pr,bysh,byx,cnt,admitted=model
 a=3;b=15;N=cnt[sh]
 pbase=(bysh[sh][y]+a*pr.get(y,1e-10))/(N+a)
 c=byx[(sh,x)];q=(c[y]+b*pbase)/(sum(c.values())+b)
 return math.log2(q/pbase)

def work(words,seed=7188, reps=100):
 data=events(words); folds=defaultdict(list)
 for e in data:folds[(e[0]//1000)%5].append(e)
 all_holdout=[]
 sizes=[]
 for f in range(5):
  train=[e for k,xs in folds.items() if k!=f for e in xs]
  model=fit(train)
  test=[e for e in folds[f] if e[1] in model[4]]
  sizes.append({'fold':f,'n':len(test),'train':len(train),'eligible_shells':len(model[4])})
  for e in test:all_holdout.append((e,model))
 if not all_holdout:return {'qualified':False,'n':0}
 obs=sum(pred(e,mo) for e,mo in all_holdout)/len(all_holdout)
 # shuffle current test X amongst same shells (across all folds; preserve X marginals and precise shell). Model frozen.
 groups=defaultdict(list)
 for j,(e,mo) in enumerate(all_holdout):groups[(e[1],(e[0]//1000)%5)].append(j)
 allxs=[e[3] for e,mo in all_holdout]
 null=[];rng=random.Random(seed)
 for _ in range(reps):
  xs=allxs[:]
  for g in groups.values():
   z=[xs[j] for j in g];rng.shuffle(z)
   for j,v in zip(g,z):xs[j]=v
  score=0
  for j,(e,mo) in enumerate(all_holdout):
   ee=(e[0],e[1],e[2],xs[j]);score+=pred(ee,mo)
  null.append(score/len(all_holdout))
 sd=statistics.stdev(null) if len(null)>1 else 0
 return {'qualified':True,'n':len(all_holdout),'total_eligible_preholdout':len(data),'folds':sizes,'gain_bits':obs,'null_mean_bits':statistics.mean(null),'null_sd_bits':sd,'effect_bits':obs-statistics.mean(null),'z':(obs-statistics.mean(null))/sd if sd else None,'null_reps':reps,'seed':seed}

result={'contract_sha256':hashlib.sha256(CONTRACT).hexdigest(),'controls':{}}
for arm,path in ARMS.items():
 words=path.read_text().split()
 print('Start',arm,len(words),flush=True)
 res=work(words,reps=40)
 res['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 result['controls'][arm]=res
 atomic_pickle(f'stage_{arm}.pkl',res)
 print(arm,{k:res[k] for k in ['n','effect_bits','null_sd_bits','z']},flush=True)
(P/'RESULTS.json').write_text(json.dumps(result,indent=2))
atomic_pickle('stage_final.pkl',result)

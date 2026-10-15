from pathlib import Path
from collections import Counter,defaultdict
import hashlib,random,statistics,math,pickle,tempfile,os,json
P=Path(__file__).resolve().parent; B=Path('/mnt/data/vbm_v143_crosscarry'); Z=B/'vbm_v142_inband'
paths={'baseline':Z/'hartlieb.pair.vbm','carry':B/'hartlieb.carry.vbm','statefree':B/'hartlieb.statefree.vbm'}
text={k:p.read_text().split() for k,p in paths.items()};N=len(text['baseline'])
assert all(len(x)==N for x in text.values())
assert all(len(text['baseline'][i])==len(text['carry'][i])==len(text['statefree'][i]) and text['baseline'][i][0]==text['carry'][i][0]==text['statefree'][i][0] for i in range(N))
ARMS=list(paths)
# current outcome and earlier tail; shell is invariant across arms
data={};fold={};allpos=[]
for i in range(1,N):
 w=text['baseline'][i]
 if not 5<=len(w)<=20:continue
 sh=(len(w),w[0]);data[i]={arm:(sh,text[arm][i][len(w)//2],text[arm][i-1][-3:]) for arm in ARMS}
 fold[i]=(i//1000)%5;allpos.append(i)

def fit(arm,train):
 gl=Counter(data[i][arm][1] for i in train);nt=sum(gl.values());pri={c:n/nt for c,n in gl.items()}
 bysh=defaultdict(Counter);byx=defaultdict(Counter)
 for i in train:
  sh,y,x=data[i][arm];bysh[sh][y]+=1;byx[sh,x][y]+=1
 admissible={sh for sh,v in bysh.items() if sum(v.values())>=20 and len(v)>=2}
 return pri,bysh,byx,admissible

def score(i,arm,fitmodel,x_override=None):
 pri,bysh,byx,_=fitmodel
 sh,y,x=data[i][arm]
 if x_override is not None:x=x_override
 base=(bysh[sh][y]+3*pri.get(y,1e-12))/(sum(bysh[sh].values())+3)
 ctr=byx[sh,x]
 pred=(ctr[y]+15*base)/(sum(ctr.values())+15)
 return math.log2(pred/base)

holds=[];foldsizes=[]
for f in range(5):
 test=[i for i in allpos if fold[i]==f]
 train=[i for i in allpos if fold[i]!=f]
 models={arm:fit(arm,train) for arm in ARMS}
 elig=[i for i in test if all(data[i][arm][0] in models[arm][3] for arm in ARMS)]
 foldsizes.append({'fold':f,'all':len(test),'admitted':len(elig)})
 holds.extend((i,{k:models[k] for k in ARMS}) for i in elig)
lenh=len(holds);assert lenh>1000
observed={arm:sum(score(i,arm,m[arm]) for i,m in holds)/lenh for arm in ARMS}
# permute index, re-use same reassignment in every arm within invariant blocks
indices=defaultdict(list)
for idx,(i,mod) in enumerate(holds):indices[(data[i]['baseline'][0],i//1000,fold[i])].append(idx)
null={arm:[] for arm in ARMS};rng=random.Random(23091)
for t in range(60):
 assignment=list(range(lenh))
 for idx in indices.values():
  shuffled=idx[:];rng.shuffle(shuffled)
  for origin,source in zip(idx,shuffled):assignment[origin]=source
 vals={k:0.0 for k in ARMS}
 for j,(i,mod) in enumerate(holds):
  src_i=holds[assignment[j]][0]
  for arm in ARMS:vals[arm]+=score(i,arm,mod[arm],data[src_i][arm][2])
 for arm in ARMS:null[arm].append(vals[arm]/lenh)
res={'contract_sha256':hashlib.sha256((P/'CONTRACT_V2.md').read_bytes()).hexdigest(),'n_common':lenh,'n_initial':len(allpos),'folds':foldsizes,'arm_results':{},'contrasts':{}}
for arm in ARMS:
 v=null[arm];mu=statistics.mean(v);sd=statistics.stdev(v)
 res['arm_results'][arm]={'gain_bits':observed[arm],'null_mean_bits':mu,'effect_bits':observed[arm]-mu,'null_sd_bits':sd,'z':(observed[arm]-mu)/sd if sd else None}
for neg in ['baseline','statefree']:
 dd=[null['carry'][j]-null[neg][j] for j in range(60)]
 ob=observed['carry']-observed[neg];avg=statistics.mean(dd);sd=statistics.stdev(dd)
 res['contrasts']['carry_vs_'+neg]={'raw_gain_difference':ob,'null_mean':avg,'effect_bits':ob-avg,'null_sd_bits':sd,'z':(ob-avg)/sd if sd else None}
(P/'RESULTS_V2.json').write_text(json.dumps(res,indent=2)+'\n')
fd,t=tempfile.mkstemp(dir=P,prefix='.atomic-');
with os.fdopen(fd,'wb') as f:pickle.dump(res,f,protocol=5);f.flush();os.fsync(f.fileno())
os.replace(t,P/'stage_V2.pkl')
print(json.dumps({k:res[k] for k in ['n_common','folds','arm_results','contrasts']},indent=2))

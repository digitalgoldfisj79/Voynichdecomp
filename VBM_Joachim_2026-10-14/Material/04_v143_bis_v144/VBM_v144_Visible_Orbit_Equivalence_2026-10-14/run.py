from pathlib import Path
from collections import Counter,defaultdict
import math,statistics,random,json,hashlib,pickle,tempfile,os,time
P=Path(__file__).resolve().parent;B=Path('/mnt/data/vbm_v143_crosscarry');Z=B/'vbm_v142_inband'
files={'base':Z/'hartlieb.pair.vbm','carry':B/'hartlieb.carry.vbm','statefree':B/'hartlieb.statefree.vbm'}
m=json.loads((B/'01_v143_frozen_manifest.json').read_text());vmap={tuple(k.split('|')):v for k,v in m['vowel_by_pair'].items()};cells={v:tuple(z['cells']) for v,z in m['state'].items()}
allwords={k:p.read_text().split() for k,p in files.items()};N=len(allwords['base']);assert all(len(v)==N for v in allwords.values())
def parse(w):
 i=0;v='e';parts=[];firstcell=None;lastv='-';check=[]
 while i<len(w):
  if w[i] not in 'ynr':
   if w.startswith(('ch','sh'),i):br=w[i:i+2];i+=2
   else:br=w[i];i+=1
   if i>=len(w):raise RuntimeError(f'no key {w} {i}')
   pair=(br,w[i]);v=vmap[pair];lastv=v;parts.append(('b',br+w[i]));check.append(pair)
  else:
   possibilities=[z for z in cells[v] if w.startswith(z,i)]
   if len(possibilities)!=1:raise RuntimeError(f'bad parse {w} {i} {v} {len(possibilities)}')
   z=possibilities[0];i+=len(z);parts.append(('c',z[0],len(z)))
   if firstcell is None:firstcell=z
 return tuple(parts),firstcell,lastv,check
arms=list(files);parsed={k:[parse(w) for w in allwords[k]] for k in arms}
assert all(parsed['base'][i][0]==parsed['carry'][i][0]==parsed['statefree'][i][0] for i in range(N))
assert all(parsed['base'][i][2]==parsed['carry'][i][2]==parsed['statefree'][i][2] for i in range(N))
print('parse and invariance complete',N,flush=True)
def checkpoint(name,obj):
 fd,tmp=tempfile.mkstemp(prefix='.stage.',dir=P)
 with os.fdopen(fd,'wb') as f:pickle.dump(obj,f,protocol=5);f.flush();os.fsync(f.fileno())
 os.replace(tmp,P/name)
checkpoint('stage0_parse_invariants.pkl',{'n':N,'manifest_sha256':hashlib.sha256((B/'01_v143_frozen_manifest.json').read_bytes()).hexdigest(),'invariant_pass':True})
obs={i:{k:(parsed[k][i][0],parsed[k][i][1],parsed[k][i-1][2]) for k in arms} for i in range(1,N) if parsed['base'][i][1] is not None};fold={i:(i//1000)%5 for i in obs}
print('positions with cells',len(obs),'unique shells',len(set(x['base'][0] for x in obs.values())),flush=True)
def fit(arm,train):
 gl=Counter(obs[i][arm][1] for i in train); nt=sum(gl.values()); prior={y:c/nt for y,c in gl.items()}
 sh=defaultdict(Counter); sx=defaultdict(Counter)
 for i in train:
  a,b,c=obs[i][arm];sh[a][b]+=1;sx[a,c][b]+=1
 admissible={key for key,c in sh.items() if sum(c.values())>=12 and len(c)>=2}
 return (prior,sh,sx,admissible)
def score(i,arm,model,xx=None):
 pri,sh,sx,_=model;shape,y,x=obs[i][arm];x=x if xx is None else xx
 base=(sh[shape][y]+3*pri.get(y,1e-12))/(sum(sh[shape].values())+3)
 by=sx[shape,x];pred=(by[y]+12*base)/(sum(by.values())+12)
 return math.log2(pred/base)
hold=[];folds=[]
for f in range(5):
 train=[i for i in obs if fold[i]!=f];test=[i for i in obs if fold[i]==f]
 models={arm:fit(arm,train) for arm in arms}
 elig=[i for i in test if all(obs[i][arm][0] in models[arm][3] for arm in arms)]
 hold.extend([(i,models) for i in elig]);folds.append({'fold':f,'n_test':len(test),'n_admitted':len(elig)})
L=len(hold); print('admitted',L,'folds',folds,flush=True)
if L<1000: raise RuntimeError('n insufficient; no test')
G=defaultdict(list)
for j,(i,models) in enumerate(hold):G[(obs[i]['base'][0],i//2000)].append(j)
movegroups=[g for g in G.values() if len(g)>1]
mutable=sum(len(g) for g in movegroups)
assert all(obs[i]['base'][2]==obs[i]['carry'][2]==obs[i]['statefree'][2] for i,_ in hold)
actual={arm:sum(score(i,arm,m[arm]) for i,m in hold)/L for arm in arms}
foldactual={arm:[statistics.mean(score(i,arm,m[arm]) for i,m in hold if fold[i]==f) if folds[f]['n_admitted'] else None for f in range(5)] for arm in arms}
print('observed',actual, 'mutable',mutable,flush=True)
null={arm:[] for arm in arms};rg=random.Random(44331)
for run in range(80):
 assignment=list(range(L))
 for g in movegroups:
  q=g[:];rg.shuffle(q)
  for a,b in zip(g,q):assignment[a]=b
 sums={a:0.0 for a in arms}
 for j,(i,models) in enumerate(hold):
  x=obs[hold[assignment[j]][0]]['base'][2]
  for arm in arms:sums[arm]+=score(i,arm,models[arm],xx=x)
 for arm in arms:null[arm].append(sums[arm]/L)
 if run%20==19: print('replicates',run+1,flush=True)
res={'n':N,'n_eligible':L,'fraction_permutable':mutable/L,'n_shells':len(G),'folds':folds,'observed':actual,'arm':{},'contrast':{},'fold_observed':foldactual,'null_reps':80}
for arm in arms:
 mu=statistics.mean(null[arm]);sd=statistics.stdev(null[arm]);effect=actual[arm]-mu
 res['arm'][arm]={'null_mean':mu,'null_sd':sd,'gain_minus_null':effect,'z':effect/sd if sd else None}
for neg in ['base','statefree']:
 d=[a-b for a,b in zip(null['carry'],null[neg])];dd=actual['carry']-actual[neg];mu=statistics.mean(d);sd=statistics.stdev(d);effect=dd-mu
 res['contrast'][f'carry_minus_{neg}']={'raw':dd,'null_mean':mu,'effect':effect,'null_sd':sd,'z':effect/sd if sd else None}
res['pass']=all(res['contrast'][f'carry_minus_{neg}']['z']>=2 for neg in ['base','statefree']) and sum(foldactual['carry'][j]>max(foldactual['base'][j],foldactual['statefree'][j]) for j in range(5))>=4
(P/'RESULTS.json').write_text(json.dumps(res,indent=2)+'\n');checkpoint('stage1_results.pkl',res)
print(json.dumps(res,indent=2),flush=True)

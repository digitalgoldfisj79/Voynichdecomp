import os,json,math,collections,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/8f73569a1e744eb14f6f19c33d9844bf7ca84311/research/selector_innovation_residual_test_v3_20261004.py"
src=urllib.request.urlopen(URL,timeout=60).read().decode()
ns={"__name__":"innovation_module"}
exec(compile(src,URL,"exec"),ns)

NAMES=ns["NAMES"];TV=ns["TARGET_VEC"];sim_task=ns["sim_task"]
FAMS=("SOURCE","TABLE","SOURCE_TABLE")
# 300 fresh URN for per-metric standardized diagnostics; 120/family at beta .75 and 1.0 for attribution calibration.
jobs=[("URN",202610200000+i,0.0) for i in range(300)]
for beta in (.75,1.0):
  for fi,f in enumerate(FAMS):
    jobs += [(f,202610300000+int(beta*100)*100000+fi*10000+i,beta) for i in range(120)]
with ProcessPoolExecutor(max_workers=min(32,os.cpu_count() or 8)) as ex:
  res=list(ex.map(sim_task,jobs,chunksize=2))
URN=np.array([v for k,b,v in res if k=="URN"],float)
mu=URN.mean(0);sd=URN.std(0,ddof=1);z=(TV-mu)/np.where(sd>1e-12,sd,1)
diag=sorted([{"metric":n,"target":float(t),"null_mean":float(m),"null_sd":float(s),"z":float(zz)}
             for n,t,m,s,zz in zip(NAMES,TV,mu,sd,z)],key=lambda d:-abs(d["z"]))

def calibrate(beta):
  A={f:np.array([v for k,b,v in res if k==f and abs(b-beta)<1e-9],float) for f in FAMS}
  ntr=70;te=range(ntr,120)
  pool=np.vstack([A[f][:ntr] for f in FAMS]);pm=pool.mean(0);ps=pool.std(0,ddof=1);ps=np.where(ps>1e-9,ps,1)
  cen={f:((A[f][:ntr]-pm)/ps).mean(0) for f in FAMS}
  def cl(x,allow=FAMS):
    q=(x-pm)/ps;d={f:float(np.mean((q-cen[f])**2)) for f in allow};return min(d,key=d.get),d
  conf={f:collections.Counter() for f in FAMS}
  for f in FAMS:
    for i in te: conf[f][cl(A[f][i])[0]]+=1
  rec={f:conf[f][f]/50 for f in FAMS}
  ov=sum(conf[f][f] for f in FAMS)/150
  def pair(a,b):
    g=0;n=0
    for f in (a,b):
      for i in te:g+=cl(A[f][i],(a,b))[0]==f;n+=1
    return g/n
  pw={"SOURCE_TABLE":pair("SOURCE","TABLE"),"SOURCE_SOURCE_TABLE":pair("SOURCE","SOURCE_TABLE"),"TABLE_SOURCE_TABLE":pair("TABLE","SOURCE_TABLE")}
  pred,dist=cl(TV)
  return {"overall":ov,"recall":rec,"pairwise":pw,"target_nearest":pred,"target_distance":dist,
          "confusion":{f:dict(conf[f]) for f in FAMS}}
out={"per_metric":diag,"attribution":{"0.75":calibrate(.75),"1.0":calibrate(1.0)}}
print("INNOVATION_DIAG_JSON="+json.dumps(out,separators=(",",":")),flush=True)

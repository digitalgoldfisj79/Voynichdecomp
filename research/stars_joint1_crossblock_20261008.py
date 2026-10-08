#!/usr/bin/env python3
import json,os,urllib.request,numpy as np
from concurrent.futures import ProcessPoolExecutor
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/0ac6de27a4d2dc128561f017b9ace7b63cd90309/research/stars_close1_full_innovation_20261008.py"
s=urllib.request.urlopen(URL,timeout=120).read().decode()
m={"__name__":"close1_module"};exec(compile(s,URL,"exec"),m)
REAL=m["REAL"];analyze=m["analyze"];generate_temp=m["generate_temp"]
SEEDS=list(range(202610090200,202610090650));RIDGE=1.0
N=list(REAL["full_names"])
AN=["repeat_resid_1","repeat_resid_2","repeat_resid_3","repeat_resid_5","repeat_resid_10","transition_resid_norm","sv1_energy","sv12_energy"]
BN=["surprise_ac_1","surprise_ac_2","surprise_ac_3","surprise_ac_5","surprise_ac_10","mean_excess_surprise","pearson_energy"]
AI=[N.index(x) for x in AN];BI=[N.index(x) for x in BN]
def task(seed): return seed,analyze(generate_temp(seed))["full"].tolist()
def rfit(X,Y): return np.linalg.solve(X.T@X+RIDGE*np.eye(X.shape[1]),X.T@Y)
def prec(E):
 S=np.cov(E,rowvar=False);C=.75*S+.25*np.diag(np.diag(S))+np.eye(S.shape[0])*1e-9
 return np.linalg.pinv(C)
if __name__=="__main__":
 with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex: rr=list(ex.map(task,SEEDS,chunksize=2))
 mp={s:np.asarray(v,float) for s,v in rr}
 F=np.vstack([mp[s] for s in SEEDS[:250]]);C=np.vstack([mp[s] for s in SEEDS[250:350]]);B=np.vstack([mp[s] for s in SEEDS[350:]])
 R=np.asarray(REAL["full"],float)
 ma=F[:,AI].mean(0);sa=np.maximum(F[:,AI].std(0,ddof=1),1e-12);mb=F[:,BI].mean(0);sb=np.maximum(F[:,BI].std(0,ddof=1),1e-12)
 def std(V): return (V[:,AI]-ma)/sa,(V[:,BI]-mb)/sb
 AF,BF=std(F);AC,BC=std(C);AB,BB=std(B);Ar=(R[AI]-ma)/sa;Br=(R[BI]-mb)/sb
 Wba=rfit(AF,BF);Wab=rfit(BF,AF);Pba=prec(BF-AF@Wba);Pab=prec(AF-BF@Wab)
 def dist(A,B):
  rb=B-A@Wba;ra=A-B@Wab
  return np.einsum("ni,ij,nj->n",rb,Pba,rb),np.einsum("ni,ij,nj->n",ra,Pab,ra)
 dBc,dAc=dist(AC,BC);dBb,dAb=dist(AB,BB);rb=Br-Ar@Wba;ra=Ar-Br@Wab
 dBr=float(rb@Pba@rb);dAr=float(ra@Pab@ra)
 mB=float(dBc.mean());sB=max(float(dBc.std(ddof=1)),1e-12);mA=float(dAc.mean());sA=max(float(dAc.std(ddof=1)),1e-12)
 zBc=(dBc-mB)/sB;zAc=(dAc-mA)/sA;zBb=(dBb-mB)/sB;zAb=(dAb-mA)/sA;zBr=(dBr-mB)/sB;zAr=(dAr-mA)/sA
 Jc=np.maximum(zBc,zAc);Jb=np.maximum(zBb,zAb);Jr=max(zBr,zAr);q=float(np.quantile(Jc,.99));blind=float(np.mean(Jb<=q))
 p=float((1+np.sum(np.r_[Jc,Jb]>=Jr))/(len(Jc)+len(Jb)+1));passed=blind>=.90 and Jr>q and p<=.01
 qB=float(np.quantile(zBc,.99));qA=float(np.quantile(zAc,.99));rB=bool(passed and zBr>qB);rA=bool(passed and zAr>qA)
 label=("BIDIRECTIONAL_CONDITIONAL_MISMATCH" if rB and rA else "SURPRISE_GIVEN_TOPOLOGY_MISMATCH" if rB else "TOPOLOGY_GIVEN_SURPRISE_MISMATCH" if rA else "GLOBAL_ONLY_NO_DIRECTION_RESOLVED" if passed else None)
 out={"programme":"STARS-JOINT1","status":"complete","real":{"D_B_given_A":dBr,"D_A_given_B":dAr,"Z_B_given_A":zBr,"Z_A_given_B":zAr,"J":Jr},"calibration":{"J_q99":q,"blind_acceptance":blind,"add_one_p":p},"decision":"CROSS_BLOCK_CONDITIONAL_COUPLING_MISMATCH_PRESENT" if passed else "CROSS_BLOCK_COUPLING_NOT_RESOLVED","localization":{"q99_B_given_A":qB,"q99_A_given_B":qA,"surprise_given_topology_resolved":rB,"topology_given_surprise_resolved":rA,"label":label}}
 print("STARS_JOINT1_JSON="+json.dumps(out,separators=(",",":")),flush=True)

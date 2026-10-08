#!/usr/bin/env python3
import collections,json,math,re,urllib.request,numpy as np
U="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/cfebbd657e31cc8a49e1513230e7e2c34514fbcc/research/stars_innov2_position_20261008.py"
m={"__name__":"innov2"};exec(compile(urllib.request.urlopen(U,timeout=120).read().decode(),U,"exec"),m)
rows=m["rows"];para_id=m["para_id"];base_prob=m["base_prob"];fit_pos=m["fit_pos"];tilt_p=m["tilt_p"];K=m["K"];fnum=m["fnum"]
MAIN=set(f"f{n}{s}" for n in list(range(103,109))+list(range(111,115)) for s in ("r","v"))
ALL=MAIN|{"f115r","f115v","f116r","f116v"}
page=collections.defaultdict(lambda:np.zeros(K));para=collections.defaultdict(lambda:np.zeros(K));hist=collections.defaultdict(list);by=collections.OrderedDict()
for r in rows:
 fol=r["folio"];ln=int(r["line"]);lk=(fol,ln);pk=fol;pid=para_id(pk,ln);pq=(pk,pid);y=int(r["start"]);pos=int(r["pos"])
 if pos==0:
  if fol in ALL:by.setdefault(lk,{"folio":fol,"line":ln,"events":[]})
 else:
  prev=hist[lk][-1][1];rc=np.zeros(K)
  for yy,pp in hist[lk][-6:]:rc[int(yy)]+=1
  p=base_prob(r["section"],prev,page[pk],para[pq],rc)
  if fol in ALL:by.setdefault(lk,{"folio":fol,"line":ln,"events":[]})["events"].append({"p":p,"y":y,"folio":fol,"line":lk})
 page[pk][y]+=1;para[pq][y]+=1;hist[lk].append((y,r["final_piece"]))
raw=[[dict(e) for e in v["events"]] for v in by.values() if len(v["events"])>=3]
mods={0:fit_pos(raw,0),1:fit_pos(raw,1)}
def role(i,n):
 if i==0:return "FIRST"
 if i==1:return "SECOND"
 if i==n-1:return "FINAL"
 if i==n-2:return "PENULT"
 rel=(i-2)/max(1,n-5)
 return "EARLY" if rel<1/3 else ("MIDDLE" if rel<2/3 else "LATE")
def lb(n):return "L7" if n<=7 else ("L10" if n<=10 else ("L14" if n<=14 else "L15P"))
out=[]
for rec in by.values():
 if len(rec["events"])<3:continue
 fol=rec["folio"];tr=1-fnum(fol)%2;gb,rb,cb=mods[tr];n=len(rec["events"]);seq=[]
 for i,e in enumerate(rec["events"]):
  ro=role(i,n);b=cb.get((ro,lb(n)),rb.get(ro,gb));seq.append({"p":tilt_p(e["p"],b),"y":int(e["y"])})
 s=0.;nn=0
 for t in range(2,len(seq)):
  a=seq[t-2]["y"];b=seq[t-1]["y"]
  if a==b:continue
  s+=(1. if seq[t]["y"]==a else 0.)-float(seq[t]["p"][a]);nn+=1
 if nn:out.append({"folio":fol,"line":rec["line"],"sum":s,"n":nn,"mean":s/nn})
def agg(xs):
 return {"score":sum(x["sum"] for x in xs)/sum(x["n"] for x in xs),"n":sum(x["n"] for x in xs),"lines":len(xs)}
f115=[x for x in out if x["folio"]=="f115r"]
res={"f115r_all":agg(f115),
 "first12":agg([x for x in f115 if x["line"]<=12]),"after12":agg([x for x in f115 if x["line"]>12]),
 "first18":agg([x for x in f115 if x["line"]<=18]),"after18":agg([x for x in f115 if x["line"]>18]),
 "surfaces":{f:agg([x for x in out if x["folio"]==f]) for f in ("f115r","f115v","f116r") if any(x["folio"]==f for x in out)},
 "f115r_lines":f115}
print("F115_LOCAL="+json.dumps(res,separators=(",",":")))

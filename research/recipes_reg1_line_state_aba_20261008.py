#!/usr/bin/env python3
# RECIPES-REG1 — preregistered 2026-10-08.
import collections,json,math,os,re,urllib.request
import numpy as np

INNOV_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/cfebbd657e31cc8a49e1513230e7e2c34514fbcc/research/stars_innov2_position_20261008.py"
LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
ii={"__name__":"innov2"}; exec(compile(urllib.request.urlopen(INNOV_URL,timeout=120).read().decode(),INNOV_URL,"exec"),ii)
ll={"__name__":"latent"}; exec(compile(urllib.request.urlopen(LAT_URL,timeout=120).read().decode(),LAT_URL,"exec"),ll)

rows=ii["rows"]; para_id=ii["para_id"]; base_prob=ii["base_prob"]; fit_pos=ii["fit_pos"]; tilt_p=ii["tilt_p"]
K=ii["K"]; fnum=ii["fnum"]
MAIN=set(f"f{n}{s}" for n in list(range(103,109))+list(range(111,115)) for s in ("r","v"))
LATE={"f115r","f115v","f116r","f116v"}
ALL=MAIN|LATE
SEED=202610081250
NREP=100000

# Frozen latent line-state model, exact original train folds and hyperparameters.
TR=[x for x in ll["LINES"] if x["fold"] in (2,3,4)]
baseF=ll["fit_struct"](TR)
TRp=ll["attach"](TR,baseF)
MD=ll["fit_mix"](TRp,4,3.0)
B=np.asarray(MD["B"],float); PI=np.asarray(MD["pi"],float); HP={k:np.asarray(v,float) for k,v in MD["hp"].items()}
house=ll["house"]

# Build observed full Recipes target under same compact SELECT baseline.
page=collections.defaultdict(lambda:np.zeros(K,float))
para=collections.defaultdict(lambda:np.zeros(K,float))
hist=collections.defaultdict(list)
by=collections.OrderedDict()
for r in rows:
    fol=r["folio"];ln=int(r["line"]);lk=(fol,ln);pk=fol;pid=para_id(pk,ln);pq=(pk,pid)
    y=int(r["start"]);pos=int(r["pos"])
    if pos==0:
        if fol in ALL:
            by.setdefault(lk,{"folio":fol,"line":ln,"opener":r["token"],"events":[]})
    else:
        prev_piece=hist[lk][-1][1]
        rc=np.zeros(K,float)
        for yy,pp in hist[lk][-6:]: rc[int(yy)]+=1
        p=base_prob(r["section"],prev_piece,page[pk],para[pq],rc)
        if fol in ALL:
            rec=by.setdefault(lk,{"folio":fol,"line":ln,"opener":None,"events":[]})
            rec["events"].append({"p":p,"y":y})
    page[pk][y]+=1;para[pq][y]+=1;hist[lk].append((y,r["final_piece"]))

raw=[]
for rec in by.values():
    if len(rec["events"])>=3:
        raw.append([dict(e,folio=rec["folio"],line=(rec["folio"],rec["line"])) for e in rec["events"]])

# Cross-fitted position/length correction on full Recipes target.
mods={0:fit_pos(raw,0),1:fit_pos(raw,1)}
def role(i,n):
    if i==0:return "FIRST"
    if i==1:return "SECOND"
    if i==n-1:return "FINAL"
    if i==n-2:return "PENULT"
    rel=(i-2)/max(1,n-5)
    return "EARLY" if rel<1/3 else ("MIDDLE" if rel<2/3 else "LATE")
def lbin_pos(n): return "L7" if n<=7 else ("L10" if n<=10 else ("L14" if n<=14 else "L15P"))

lines=[]
for rec0 in by.values():
    if len(rec0["events"])<3: continue
    fol=rec0["folio"];tr=1-fnum(fol)%2;gb,rb,cb=mods[tr];n=len(rec0["events"])
    seq=[]
    for i,e in enumerate(rec0["events"]):
        ro=role(i,n);bb=cb.get((ro,lbin_pos(n)),rb.get(ro,gb))
        q=tilt_p(e["p"],bb)
        seq.append({"p":q,"y":int(e["y"])})
    lines.append({"folio":fol,"line":int(rec0["line"]),"opener":rec0["opener"],"house":house(rec0["opener"]),"seq":seq})

def state_post_and_probs(line):
    seq=line["seq"];prior=HP.get(line["house"],PI)
    A=np.log(np.maximum(prior,1e-15))
    q=2
    for z in range(4):
        for t in range(q):
            p=tilt_p(seq[t]["p"],B[z])
            A[z]+=math.log(max(float(p[seq[t]["y"]]),1e-300))
    mx=A.max();W=np.exp(A-mx);W/=W.sum()
    pls=[]
    for t,e in enumerate(seq):
        if t<2: pls.append(e["p"]); continue
        p=np.zeros(K,float)
        for z in range(4): p+=W[z]*tilt_p(e["p"],B[z])
        p/=p.sum();pls.append(p)
    return W,np.asarray(pls)

def line_metrics(line):
    W,PLS=state_post_and_probs(line)
    seq=line["seq"];s0=sls=0.;n=0
    for t in range(2,len(seq)):
        a=int(seq[t-2]["y"]);b=int(seq[t-1]["y"])
        if a==b:continue
        y=int(seq[t]["y"]);obs=1.0 if y==a else 0.
        s0 += obs-float(seq[t]["p"][a])
        sls += obs-float(PLS[t][a])
        n+=1
    r=W-HP.get(line["house"],PI)
    return s0,sls,n,r

data=[]
for L in lines:
    s0,sls,n,r=line_metrics(L)
    if n<=0:continue
    fol=L["folio"];grp=1 if fol in LATE else 0
    ln=len(L["seq"])
    lb="3-5" if ln<=5 else ("6-9" if ln<=9 else ("10-14" if ln<=14 else "15+"))
    data.append({"folio":fol,"line":L["line"],"group":grp,"house":L["house"],"lb":lb,
                 "s0":s0,"sls":sls,"n":n,"r":r})

def observed_stats(D):
    late=[x for x in D if x["group"]==1];main=[x for x in D if x["group"]==0]
    def emean(xs,key): return sum(x[key] for x in xs)/sum(x["n"] for x in xs)
    d0=emean(main,"s0")-emean(late,"s0")
    dls=emean(main,"sls")-emean(late,"sls")
    rm=np.mean(np.vstack([x["r"] for x in main]),0);rl=np.mean(np.vstack([x["r"] for x in late]),0)
    occ=float(np.sum((rm-rl)**2))
    return d0,dls,occ,{"main_n":sum(x["n"] for x in main),"late_n":sum(x["n"] for x in late),
                        "main_lines":len(main),"late_lines":len(late),
                        "main_aba_q0":emean(main,"s0"),"late_aba_q0":emean(late,"s0"),
                        "main_aba_qLS":emean(main,"sls"),"late_aba_qLS":emean(late,"sls"),
                        "main_R":rm.tolist(),"late_R":rl.tolist()}

def perm_test(D,seed):
    d0,dls,occ,audit=observed_stats(D)
    strata=collections.defaultdict(list)
    for i,x in enumerate(D):strata[(x["house"],x["lb"])].append(i)
    mobility={k:sum(D[i]["group"] for i in ix) for k,ix in strata.items()}
    rng=np.random.default_rng(seed)
    null0=np.empty(NREP);nullls=np.empty(NREP);nullo=np.empty(NREP)
    # Arrays for fast aggregate from sampled late set.
    s0=np.array([x["s0"] for x in D]);sls=np.array([x["sls"] for x in D]);nn=np.array([x["n"] for x in D],float)
    rr=np.vstack([x["r"] for x in D])
    tot0=s0.sum();totls=sls.sum();totn=nn.sum();totr=rr.sum(0);N=len(D)
    for b in range(NREP):
        lix=[]
        for k,ix in strata.items():
            m=mobility[k]
            if m<=0: continue
            if m>=len(ix): lix.extend(ix)
            else: lix.extend(rng.choice(ix,size=m,replace=False).tolist())
        li=np.asarray(lix,int)
        ls0=s0[li].sum();lls=sls[li].sum();ln=nn[li].sum()
        ms0=tot0-ls0;mls=totls-lls;mn=totn-ln
        null0[b]=ms0/mn-ls0/ln
        nullls[b]=mls/mn-lls/ln
        lr=rr[li].sum(0);mr=totr-lr
        dr=mr/(N-len(li))-lr/len(li)
        nullo[b]=float(dr@dr)
    def two(obs,z):
        return float((1+np.sum(np.abs(z)>=abs(obs)))/(len(z)+1))
    po=float((1+np.sum(nullo>=occ))/(len(nullo)+1))
    return {"D0":d0,"DLS":dls,"occ":occ,"p_D0":two(d0,null0),"p_DLS":two(dls,nullls),"p_occ":po,
            "q99_abs_D0":float(np.quantile(np.abs(null0),.99)),
            "q99_abs_DLS":float(np.quantile(np.abs(nullls),.99)),
            "q99_occ":float(np.quantile(nullo,.99)),"audit":audit}

PRIMARY=perm_test(data,SEED)
SENS=perm_test([x for x in data if x["folio"]!="f115r"],SEED+1)

def agg_fol(prefix):
    xs=[x for x in data if re.match(r"f"+str(prefix)+r"[rv]",x["folio"])]
    if not xs:return None
    return {"q0":sum(x["s0"] for x in xs)/sum(x["n"] for x in xs),
            "qLS":sum(x["sls"] for x in xs)/sum(x["n"] for x in xs),
            "n":sum(x["n"] for x in xs),"lines":len(xs)}
CONJOINT={"f103":agg_fol(103),"f116":agg_fol(116),"f104":agg_fol(104),"f115":agg_fol(115)}

# Frozen adjudication.
if PRIMARY["p_D0"]<=.01:
    if abs(PRIMARY["DLS"])<=.5*abs(PRIMARY["D0"]) and PRIMARY["p_DLS"]>.05:
        absorb="FROZEN_LINE_STATE_ABSORBS_ABA_REGIME_DIFFERENCE"
    elif PRIMARY["p_DLS"]<=.01:
        absorb="REGIME_DIFFERENCE_SURVIVES_FROZEN_LINE_STATE"
    else:
        absorb="PARTIAL_OR_UNRESOLVED_ABSORPTION"
else:
    absorb="DIRECT_ABA_REGIME_DIFFERENCE_NOT_RESOLVED"
occdec="LINE_STATE_OCCUPANCY_SHIFT_PRESENT" if PRIMARY["p_occ"]<=.01 else "LINE_STATE_OCCUPANCY_SHIFT_NOT_RESOLVED"

OUT={"programme":"RECIPES-REG1","status":"complete","target":{"main":sorted(MAIN),"late":sorted(LATE)},
     "frozen_line_state":{"K":4,"L2":3.0,"pi":PI.tolist(),"house_priors":{k:v.tolist() for k,v in HP.items()}},
     "primary":PRIMARY,"absorption_decision":absorb,"occupancy_decision":occdec,
     "sensitivity_exclude_f115r":SENS,"conjoint_descriptive":CONJOINT}
print("RECIPES_REG1_JSON="+json.dumps(OUT,separators=(",",":")),flush=True)

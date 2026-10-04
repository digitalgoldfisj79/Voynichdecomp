#!/usr/bin/env python3
import collections,json,math,os,urllib.request
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from concurrent.futures import ProcessPoolExecutor

# Frozen downstream line-state implementation.
LINE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
lns={"__name__":"line_module"}
exec(compile(urllib.request.urlopen(LINE_URL,timeout=60).read().decode(),LINE_URL,"exec"),lns)
LINES=lns["LINES"]; fit_struct=lns["fit_struct"]; attach=lns["attach"]; fit_mix=lns["fit_mix"]; K=12

# Frozen innovation metric implementation.
INNO_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/69ba5470055a94f726fe797ca63f1cc27f410cc4/research/selector_innovation_residual_noed_refit_20261004.py"
ivs={"__name__":"innovation_module"}
exec(compile(urllib.request.urlopen(INNO_URL,timeout=60).read().decode(),INNO_URL,"exec"),ivs)
metric_vector=ivs["metric_vector"]; mahal_setup=ivs["mahal_setup"]; distance=ivs["distance"]

# Canonical physical bifolia + Davis-hand helper.
OCC_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/research/hf_emergent_occupancy_fold.py"
occ={"__name__":"occupancy_module"}
exec(compile(urllib.request.urlopen(OCC_URL,timeout=60).read().decode(),OCC_URL,"exec"),occ)
PAIRS=occ["PAIRS"]; parse_num=occ["parse_num"]; davis_hand=occ["davis_hand"]
MATE={a:b for a,b in PAIRS}; MATE.update({b:a for a,b in PAIRS})

L2_GRID=(1.0,3.0,10.0,30.0)
LAM_GRID=(0.0,.25,.5,.75,1.0,1.5)
B_SIGN=200000
B_PACKET=2000
B_AUDIT=500

def state_q(p,b):
    sc=np.log(np.maximum(p,1e-15))+b
    sc-=sc.max();q=np.exp(sc);return q/q.sum()

def apply_line_model(lines,base,md):
    ap=attach(lines,base);B=md["B"];HP=md["hp"];out=[]
    for l in ap:
        P,Y=l["P"],l["Y"]
        if len(Y)<3: continue
        prior=HP[l["house"]]
        A=np.log(prior+1e-15)
        for z in range(4):
            for i in (0,1):
                q=state_q(P[i],B[z]);A[z]+=math.log(max(q[Y[i]],1e-300))
        A-=A.max();post=np.exp(A);post/=post.sum()
        Q=[]
        for i in range(2,len(Y)):
            q=np.zeros(K,float)
            for z in range(4):q+=post[z]*state_q(P[i],B[z])
            Q.append(q)
        if not Q: continue
        out.append({
            "P":np.stack(Q),"Y":Y[2:].copy(),"previd":int(Y[1]),
            "section":l["events"][0]["section"],"bif":l["bif"],"fold":int(l["fold"]),
            "folio":l["folio"],"line":int(l["line"]),"num":int(parse_num(l["folio"]))
        })
    return out

def build_split(trainfolds,targetfolds):
    tr=[l for l in LINES if l["fold"] in trainfolds]
    tg=[l for l in LINES if l["fold"] in targetfolds]
    base=fit_struct(tr);trp=attach(tr,base);md=fit_mix(trp,4,3.0)
    return apply_line_model(tr,base,md),apply_line_model(tg,base,md)

def fit_tilt(P,Y,l2):
    P=np.asarray(P,float);Y=np.asarray(Y,int)
    def fg(b):
        bb=b-b.mean()
        sc=np.log(np.maximum(P,1e-15))+bb[None,:]
        z=logsumexp(sc,axis=1);q=np.exp(sc-z[:,None])
        loss=-float(np.sum(sc[np.arange(len(Y)),Y]-z))+.5*l2*float(np.dot(bb,bb))
        D=q;D[np.arange(len(Y)),Y]-=1
        g=D.sum(0)+l2*bb;g-=g.mean()
        return loss,g
    r=minimize(lambda b:fg(b),np.zeros(K),jac=True,method="L-BFGS-B",
               options={"maxiter":80,"ftol":1e-10})
    b=r.x-r.x.mean()
    return b

def tilt_prob(p,b,lam):
    sc=np.log(np.maximum(p,1e-15))+lam*b
    sc-=sc.max();q=np.exp(sc);return q/q.sum()

def seq_index(seqs):
    bypage=collections.defaultdict(list)
    for s in seqs:bypage[s["folio"]].append(s)
    for f in bypage:bypage[f].sort(key=lambda x:x["line"])
    return bypage

def line_donors(seqs,l2):
    bypage=seq_index(seqs);rows=[]
    for fol,ls in bypage.items():
        for j,s in enumerate(ls):
            if len(s["Y"])<6 or j==0:continue
            prev=ls[j-1]
            if len(prev["Y"])<3:continue
            same_b=fit_tilt(s["P"][:3],s["Y"][:3],l2)
            prev_b=fit_tilt(prev["P"][-3:],prev["Y"][-3:],l2)
            rows.append((s,same_b,prev_b))
    return rows

def line_score(rows,lam):
    base=same=prev=0.;n=0;blk=collections.defaultdict(lambda:[0.,0.,0.,0])
    for s,bs,bp in rows:
        for i in range(3,len(s["Y"])):
            y=int(s["Y"][i]);p=s["P"][i]
            lb=-math.log2(max(p[y],1e-300))
            ps=tilt_prob(p,bs,lam);pp=tilt_prob(p,bp,lam)
            ls=-math.log2(max(ps[y],1e-300));lp=-math.log2(max(pp[y],1e-300))
            base+=lb;same+=ls;prev+=lp;n+=1
            z=blk[s["bif"]];z[0]+=lb;z[1]+=ls;z[2]+=lp;z[3]+=1
    diffs=[(z[2]-z[1])/z[3] for z in blk.values() if z[3]]
    return {
        "n":n,"base_bits":base/n,"same_bits":same/n,"prev_bits":prev/n,
        "same_gain":(base-same)/n,"prev_gain":(base-prev)/n,
        "same_advantage_over_prev":(prev-same)/n,
        "blocks_n":len(diffs),"blocks_positive":int(sum(x>0 for x in diffs)),
        "block_diffs":diffs
    }

def signflip_p(vals,seed=20261004,B=B_SIGN):
    v=np.asarray(vals,float);obs=float(v.mean());rng=np.random.default_rng(seed)
    hit=0;done=0;batch=10000
    while done<B:
        m=min(batch,B-done);sg=rng.choice((-1.,1.),size=(m,len(v)))
        null=(sg*v).mean(1);hit+=int(np.sum(null>=obs));done+=m
    return {"obs_block_mean":obs,"p_add":float((1+hit)/(B+1)),"B":B,
            "block_sd":float(v.std(ddof=1)) if len(v)>1 else 0.0,
            "mean_over_sd":float(obs/v.std(ddof=1)) if len(v)>1 and v.std(ddof=1)>0 else None}

def folio_groups(seqs):
    g=collections.defaultdict(list)
    for s in seqs:g[s["num"]].append(s)
    out={}
    for n,ls in g.items():
        P=np.vstack([x["P"] for x in ls]);Y=np.concatenate([x["Y"] for x in ls])
        sec=collections.Counter(x["section"] for x in ls).most_common(1)[0][0]
        hc=collections.Counter()
        for x in ls:hc[davis_hand(x["folio"],x["line"])]+=len(x["Y"])
        hand=hc.most_common(1)[0][0]
        out[n]={"num":n,"seqs":ls,"P":P,"Y":Y,"section":sec,"hand":hand,
                "bif":ls[0]["bif"],"fold":ls[0]["fold"]}
    return out

def packet_tilts(seqs,l2):
    f=folio_groups(seqs)
    for n in f:f[n]["tilt"]=fit_tilt(f[n]["P"],f[n]["Y"],l2)
    return f

def donor_bits(target,bd,lam):
    loss=0.;n=0
    for s in target["seqs"]:
        for p,y in zip(s["P"],s["Y"]):
            q=tilt_prob(p,bd,lam);loss+=-math.log2(max(q[int(y)],1e-300));n+=1
    return loss,n

def base_folio_bits(target):
    loss=0.;n=0
    for s in target["seqs"]:
        for p,y in zip(s["P"],s["Y"]):
            loss+=-math.log2(max(p[int(y)],1e-300));n+=1
    return loss,n

def candidate_pool(tnum,folios):
    m=MATE.get(tnum)
    if m not in folios:return []
    target=folios[tnum];mh=folios[m]["hand"]
    c=[n for n,x in folios.items() if n!=tnum and x["section"]==target["section"] and x["hand"]==mh]
    if len(c)<3:c=[n for n,x in folios.items() if n!=tnum and x["section"]==target["section"]]
    if m not in c:c.append(m)
    return sorted(set(c))

def packet_score(folios,lam,do_null=False,seed=20261004):
    rec=[];base=mate=0.;nall=0;blk=collections.defaultdict(lambda:[0.,0.,0])
    for tnum,t in sorted(folios.items()):
        m=MATE.get(tnum)
        if m not in folios:continue
        pool=candidate_pool(tnum,folios)
        if len(pool)<2:continue
        lb,nn=base_folio_bits(t);lm,_=donor_bits(t,folios[m]["tilt"],lam)
        cand=[]
        for d in pool:
            ld,_=donor_bits(t,folios[d]["tilt"],lam);cand.append((d,ld))
        rank=1+sum(ld<lm-1e-12 for d,ld in cand)
        rec.append({"target":tnum,"mate":m,"n":nn,"base_loss":lb,"mate_loss":lm,
                    "pool":[d for d,_ in cand],"losses":[float(ld) for _,ld in cand],"rank":rank})
        base+=lb;mate+=lm;nall+=nn
        z=blk[t["bif"]];z[0]+=lb;z[1]+=lm;z[2]+=nn
    out={"n":nall,"base_bits":base/nall,"mate_bits":mate/nall,"mate_gain":(base-mate)/nall,
         "targets":len(rec),"top1":sum(r["rank"]==1 for r in rec),
         "median_rank":float(np.median([r["rank"] for r in rec])) if rec else None,
         "mean_rank":float(np.mean([r["rank"] for r in rec])) if rec else None,
         "block_gains":[(a-b)/n for a,b,n in blk.values() if n]}
    if do_null and rec:
        rng=np.random.default_rng(seed);null=[]
        for _ in range(B_PACKET):
            loss=0.;den=0
            for r in rec:
                nm=[(d,l) for d,l in zip(r["pool"],r["losses"]) if d!=r["mate"]]
                if not nm:continue
                d,l=nm[int(rng.integers(len(nm)))];loss+=l;den+=r["n"]
            null.append((loss-mate)/den if den else 0.)
        null=np.asarray(null);obs=float(np.mean(null)+0) # only for metadata
        # true-mate advantage = random-nonmate loss minus true-mate loss
        adv=float(np.mean([(np.mean([l for d,l in zip(r["pool"],r["losses"]) if d!=r["mate"]])-r["mate_loss"])/r["n"]
                           for r in rec if any(d!=r["mate"] for d in r["pool"])]))
        # Monte Carlo p: how often random assignment is as good as or better than true mates.
        rand_losses=[]
        rng=np.random.default_rng(seed+1)
        for _ in range(B_PACKET):
            loss=0.;den=0
            for r in rec:
                nm=[l for d,l in zip(r["pool"],r["losses"]) if d!=r["mate"]]
                if not nm:continue
                loss+=nm[int(rng.integers(len(nm)))];den+=r["n"]
            rand_losses.append(loss/den if den else np.inf)
        rand_losses=np.asarray(rand_losses);true_bits=mate/nall
        out["matched_nonmate"]={"mean_bits":float(rand_losses.mean()),"sd_bits":float(rand_losses.std(ddof=1)),
                                "true_mate_advantage_bits":float(rand_losses.mean()-true_bits),
                                "p_add":float((1+np.sum(rand_losses<=true_bits))/(B_PACKET+1)),"B":B_PACKET}
    return out,rec

def validation_select():
    D,V=build_split((2,3),(4,))
    table=[]
    for l2 in L2_GRID:
        lr=line_donors(V,l2)
        lbest=max((line_score(lr,lam) | {"lambda":lam} for lam in LAM_GRID),
                  key=lambda x:(x["same_gain"],-x["lambda"]))
        pf=packet_tilts(V,l2)
        pbest=max((packet_score(pf,lam,False)[0] | {"lambda":lam} for lam in LAM_GRID),
                  key=lambda x:(x["mate_gain"],-x["lambda"]))
        table.append({"l2":l2,"line":lbest,"packet":pbest,"sum_gain":lbest["same_gain"]+pbest["mate_gain"]})
    best=max(table,key=lambda x:(x["sum_gain"],-x["l2"]))
    return table,{"l2":best["l2"],"lambda_line":best["line"]["lambda"],"lambda_packet":best["packet"]["lambda"]}

def combined_sequences(seqs,l2,lam_line,lam_packet):
    fol=packet_tilts(seqs,l2);bypage=seq_index(seqs);out=[]
    for f,ls in bypage.items():
        for s in ls:
            if len(s["Y"])<6:continue
            m=MATE.get(s["num"])
            if m not in fol:continue
            bl=fit_tilt(s["P"][:3],s["Y"][:3],l2);bp=fol[m]["tilt"]
            Pnew=[]
            for i in range(3,len(s["Y"])):
                p=s["P"][i]
                sc=np.log(np.maximum(p,1e-15))+lam_line*bl+lam_packet*bp
                sc-=sc.max();q=np.exp(sc);q/=q.sum();Pnew.append(q)
            if Pnew:
                out.append({"P0":s["P"][3:].copy(),"PH":np.stack(Pnew),"Y":s["Y"][3:].copy(),
                            "previd":int(s["Y"][2]),"fold":s["fold"],"bif":s["bif"],
                            "folio":s["folio"],"line":s["line"]})
    return out

def codelength_comb(rows):
    b=h=0.;n=0;blk=collections.defaultdict(lambda:[0.,0.,0])
    for s in rows:
        for p0,ph,y in zip(s["P0"],s["PH"],s["Y"]):
            y=int(y);lb=-math.log2(max(p0[y],1e-300));lh=-math.log2(max(ph[y],1e-300))
            b+=lb;h+=lh;n+=1;z=blk[s["bif"]];z[0]+=lb;z[1]+=lh;z[2]+=1
    gs=[(a-bb)/nn for a,bb,nn in blk.values() if nn]
    return {"n":n,"base_bits":b/n,"hier_bits":h/n,"gain_bits":(b-h)/n,
            "blocks_n":len(gs),"blocks_positive":int(sum(x>0 for x in gs)),
            "block_mean":float(np.mean(gs)),"block_sd":float(np.std(gs,ddof=1)) if len(gs)>1 else 0.}

AUDIT_ROWS=None;AUDIT_KEY=None
def sim_metric(seed):
    rng=np.random.default_rng(seed);ev=[];ls=[]
    for s in AUDIT_ROWS:
        z=[]
        P=s[AUDIT_KEY];Y=[]
        for p in P:Y.append(int(rng.choice(K,p=p)))
        for i,(p,y) in enumerate(zip(P,Y)):
            prev=s["previd"] if i==0 else Y[i-1]
            e={"p":p,"y":y,"prev":int(prev),"half":int(s["fold"]),"line":(s["folio"],s["line"])}
            ev.append(e);z.append(e)
        if z:ls.append(z)
    return metric_vector(ev,ls,True)[0].tolist()

def actual_metric(rows,key):
    ev=[];ls=[]
    for s in rows:
        z=[]
        for i,(p,y) in enumerate(zip(s[key],s["Y"])):
            prev=s["previd"] if i==0 else int(s["Y"][i-1])
            e={"p":p,"y":int(y),"prev":prev,"half":int(s["fold"]),"line":(s["folio"],s["line"])}
            ev.append(e);z.append(e)
        if z:ls.append(z)
    return metric_vector(ev,ls,True)

def audit(rows,key,seedbase):
    global AUDIT_ROWS,AUDIT_KEY
    AUDIT_ROWS=rows;AUDIT_KEY=key
    tv,names=actual_metric(rows,key)
    with ProcessPoolExecutor(max_workers=min(32,os.cpu_count() or 8)) as ex:
        A=np.array(list(ex.map(sim_metric,[seedbase+i for i in range(B_AUDIT)],chunksize=2)),float)
    mu,inv=mahal_setup(A);ds=np.array([distance(x,mu,inv) for x in A]);td=distance(tv,mu,inv)
    groups={
      "repeat":[i for i,n in enumerate(names) if n.startswith("repeat_resid_")],
      "transition":[i for i,n in enumerate(names) if n in ("transition_resid_norm","half_transition_concordance","sv1_energy","sv12_energy")],
      "surprise":[i for i,n in enumerate(names) if n.startswith("surprise_ac_") or n in ("mean_excess_surprise","pearson_energy")]
    };gd={}
    for g,ix in groups.items():
        M=A[:,ix];m,iv=mahal_setup(M);dd=np.array([distance(x,m,iv) for x in M]);tt=distance(tv[ix],m,iv)
        gd[g]={"distance":float(tt),"null_q99":float(np.quantile(dd,.99)),
               "p_add":float((1+np.sum(dd>=tt))/(B_AUDIT+1))}
    return {"distance":float(td),"null_median":float(np.median(ds)),"null_q99":float(np.quantile(ds,.99)),
            "p_add":float((1+np.sum(ds>=td))/(B_AUDIT+1)),"groups":gd,
            "target_metrics":{n:float(v) for n,v in zip(names,tv)}}

if __name__=="__main__":
    val,sel=validation_select()
    TR,TE=build_split((2,3,4),(0,1))
    l2=sel["l2"];ll=sel["lambda_line"];lp=sel["lambda_packet"]

    # A: line reset.
    lrows=line_donors(TE,l2);lres=line_score(lrows,ll);lres["signflip"]=signflip_p(lres["block_diffs"])
    lres.pop("block_diffs",None)

    # B: packet/bifolium transfer.
    fol=packet_tilts(TE,l2);pres,precords=packet_score(fol,lp,True)
    pres["block_signflip"]=signflip_p(pres["block_gains"],seed=20261005)
    pres.pop("block_gains",None)

    # C: combined hierarchy.
    comb=combined_sequences(TE,l2,ll,lp);cres=codelength_comb(comb)
    abase=audit(comb,"P0",202610800000);ahier=audit(comb,"PH",202610900000)
    red=1-ahier["distance"]/abase["distance"]
    improved={g:ahier["groups"][g]["distance"]<abase["groups"][g]["distance"] for g in ("repeat","transition","surprise")}
    gates={
      "line_pass":bool(lres["same_advantage_over_prev"]>0 and lres["signflip"]["p_add"]<.01),
      "packet_pass":bool(pres.get("matched_nonmate",{}).get("p_add",1)<.01),
      "compression_50":bool(red>=.50),
      "transition_improved":bool(improved["transition"]),
      "surprise_improved":bool(improved["surprise"]),
      "majority_blocks":bool(cres["blocks_positive"]>cres["blocks_n"]/2)
    }
    gates["full_pass"]=bool(all([gates["line_pass"],gates["packet_pass"],gates["compression_50"],
                                 gates["transition_improved"],gates["surprise_improved"],gates["majority_blocks"]]))
    out={
      "frozen_form_anchor":{
        "ZLZI_L3_z":-5.32,"ZLZI_L4_z":-7.75,
        "interpretation":"order2 FORM accounts for period1/2; longer literal copying suppressed"
      },
      "validation":val,"selected":sel,
      "test_A_line_reset":lres,
      "test_B_packet_transfer":pres,
      "test_C_hierarchy":{"codelength":cres,"baseline_innovation":abase,"hierarchy_innovation":ahier,
                          "distance_reduction_fraction":float(red),"groups_improved":improved},
      "gates":gates,
      "counts":{"test_lines":len(TE),"line_test_units":len(lrows),"packet_folios":len(fol),"combined_lines":len(comb)}
    }
    print("BOUNDARY_RESET_HIERARCHY_JSON="+json.dumps(out,separators=(",",":")),flush=True)

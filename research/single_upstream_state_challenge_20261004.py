#!/usr/bin/env python3
import collections,json,math,os,urllib.request
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

# ---------- Frozen downstream line-state model ----------
LINE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
lns={"__name__":"line_module"}
exec(compile(urllib.request.urlopen(LINE_URL,timeout=60).read().decode(),LINE_URL,"exec"),lns)
LINES=lns["LINES"]; fit_struct=lns["fit_struct"]; attach=lns["attach"]; fit_mix=lns["fit_mix"]; K=12

# Innovation machinery
INNO_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/69ba5470055a94f726fe797ca63f1cc27f410cc4/research/selector_innovation_residual_noed_refit_20261004.py"
iv={"__name__":"innovation_module"}
exec(compile(urllib.request.urlopen(INNO_URL,timeout=60).read().decode(),INNO_URL,"exec"),iv)
metric_vector=iv["metric_vector"]; mahal_setup=iv["mahal_setup"]; distance=iv["distance"]

SOURCE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/9ea05f51aef9b2ab00e443bdd0a0e0a295a42ba8/research/data/selector_source_controls_20261004.json"
SOURCE_ROWS=json.loads(urllib.request.urlopen(SOURCE_URL,timeout=60).read().decode())

L2=10.0; SMOOTH=2.0
STATE_GRID=(2,4,6,8,12)
FAMILIES=("AUTO","SECTION","SOURCE")
GROUPS={
 "repeat":None,
 "transition":None,
 "surprise":None
}

def state_q(p,b):
    sc=np.log(np.maximum(p,1e-15))+b
    sc-=sc.max();q=np.exp(sc);return q/q.sum()

def line_posterior_model(train_lines):
    base=fit_struct(train_lines)
    ap=attach(train_lines,base)
    md=fit_mix(ap,4,3.0)
    return base,md

def apply_line_model(lines,base,md):
    ap=attach(lines,base);B=md["B"];HP=md["hp"]
    out=[]
    for l in ap:
        P,Y=l["P"],l["Y"];prior=HP[l["house"]]
        A=np.log(prior+1e-15)
        for z in range(4):
            for i in (0,1):
                q=state_q(P[i],B[z]);A[z]+=math.log(max(q[Y[i]],1e-300))
        A-=A.max();post=np.exp(A);post/=post.sum()
        Q=[]
        for i in range(2,len(Y)):
            q=np.zeros(K)
            for z in range(4):q+=post[z]*state_q(P[i],B[z])
            Q.append(q)
        if Q:
            out.append({"P":np.stack(Q),"Y":Y[2:].copy(),"section":l["events"][0]["section"],"bif":l["bif"],"fold":l["fold"],
                        "folio":l["folio"],"line":l["line"],"prev0":int(Y[1])})
    return out

def build_split(trainfolds,targetfolds):
    tr=[l for l in LINES if l["fold"] in trainfolds]
    te=[l for l in LINES if l["fold"] in targetfolds]
    base,md=line_posterior_model(tr)
    return apply_line_model(tr,base,md),apply_line_model(te,base,md),base,md

# ---------- generic offset-HMM ----------
def emissions(seq,B):
    P,Y=seq["P"],seq["Y"];S=len(B);E=np.zeros((len(Y),S),float)
    for z in range(S):
        sc=np.log(np.maximum(P,1e-15))+B[z]
        mx=sc.max(1,keepdims=True);Q=np.exp(sc-mx);Q/=Q.sum(1,keepdims=True)
        E[:,z]=np.maximum(Q[np.arange(len(Y)),Y],1e-300)
    return E

def fb(seq,B,pi,T):
    E=emissions(seq,B);n,S=E.shape
    a=np.zeros((n,S));scale=np.zeros(n)
    a[0]=pi*E[0];scale[0]=a[0].sum();a[0]/=max(scale[0],1e-300)
    for t in range(1,n):
        a[t]=(a[t-1]@T)*E[t];scale[t]=a[t].sum();a[t]/=max(scale[t],1e-300)
    b=np.ones((n,S))
    for t in range(n-2,-1,-1):
        b[t]=T@(E[t+1]*b[t+1]);b[t]/=max(scale[t+1],1e-300)
    g=a*b;g/=np.maximum(g.sum(1,keepdims=True),1e-300)
    xis=np.zeros((max(n-1,0),S,S))
    for t in range(n-1):
        x=a[t][:,None]*T*(E[t+1]*b[t+1])[None,:];x/=max(x.sum(),1e-300);xis[t]=x
    return g,xis,float(np.log(np.maximum(scale,1e-300)).sum())

def optimize_B(seqs,Gs,B0):
    S=B0.shape[0];B=B0.copy()
    # concatenate once
    Ps=np.vstack([s["P"] for s in seqs]);Ys=np.concatenate([s["Y"] for s in seqs]);G=np.vstack(Gs)
    for z in range(S):
        w=G[:,z]
        def fg(b):
            bb=b-b.mean();sc=np.log(np.maximum(Ps,1e-15))+bb;mx=sc.max(1,keepdims=True);Q=np.exp(sc-mx);Q/=Q.sum(1,keepdims=True)
            loss=-float(np.sum(w*np.log(np.maximum(Q[np.arange(len(Ys)),Ys],1e-300))))+.5*L2*np.dot(bb,bb)
            D=Q.copy();D[np.arange(len(Ys)),Ys]-=1;grad=(D*w[:,None]).sum(0)+L2*bb;grad-=grad.mean()
            return loss,grad
        r=minimize(lambda b:fg(b),B[z],jac=True,method="L-BFGS-B",options={"maxiter":120,"ftol":1e-9})
        B[z]=r.x-r.x.mean()
    return B

def random_model(S,seed):
    rng=np.random.default_rng(seed);B=rng.normal(0,.03,(S,K));B-=B.mean(1,keepdims=True)
    T=rng.dirichlet(np.ones(S)*2,size=S);pi=np.ones(S)/S
    return B,pi,T

def fit_auto(seqs,S,seed=1,fixed=None,maxem=45):
    B,pi,T=random_model(S,seed)
    if fixed is not None:
        pi=np.array(fixed[0],float);T=np.array(fixed[1],float)
    last=-1e100
    for it in range(maxem):
        G=[];X=np.zeros((S,S));pc=np.zeros(S);ll=0
        for s in seqs:
            g,x,l=fb(s,B,pi,T);G.append(g);X+=x.sum(0);pc+=g[0];ll+=l
        B=optimize_B(seqs,G,B)
        if fixed is None:
            T=(X+SMOOTH);T/=T.sum(1,keepdims=True);pi=(pc+SMOOTH);pi/=pi.sum()
        if abs(ll-last)<1e-4:break
        last=ll
    return {"B":B,"pi":pi,"T":T,"ll":ll,"iter":it+1}

def fit_section(seqs,S,seed=1,maxem=45):
    B,pi0,T0=random_model(S,seed);secs=sorted({s["section"] for s in seqs})
    Pis={c:pi0.copy() for c in secs};Ts={c:T0.copy() for c in secs};last=-1e100
    for it in range(maxem):
        G=[];Xs={c:np.zeros((S,S)) for c in secs};pcs={c:np.zeros(S) for c in secs};ll=0
        for s in seqs:
            c=s["section"];g,x,l=fb(s,B,Pis[c],Ts[c]);G.append(g);Xs[c]+=x.sum(0);pcs[c]+=g[0];ll+=l
        B=optimize_B(seqs,G,B)
        for c in secs:
            T=Xs[c]+SMOOTH;T/=T.sum(1,keepdims=True);Ts[c]=T
            p=pcs[c]+SMOOTH;p/=p.sum();Pis[c]=p
        if abs(ll-last)<1e-4:break
        last=ll
    return {"B":B,"Pis":Pis,"Ts":Ts,"ll":ll,"iter":it+1}

def predictive_score(seqs,model,family):
    total_base=total_mod=0.;n=0;blk=collections.defaultdict(lambda:[0.,0.,0])
    for s in seqs:
        P,Y=s["P"],s["Y"];S=len(model["B"])
        if family=="SECTION":
            pi=model["Pis"].get(s["section"],np.ones(S)/S);T=model["Ts"].get(s["section"],np.ones((S,S))/S)
        else:pi=model["pi"];T=model["T"]
        prior=pi.copy()
        for t in range(len(Y)):
            ez=np.array([state_q(P[t],model["B"][z])[Y[t]] for z in range(S)])
            pm=float(np.dot(prior,ez));pb=float(P[t,Y[t]])
            b=-math.log2(max(pb,1e-300));m=-math.log2(max(pm,1e-300));total_base+=b;total_mod+=m;n+=1
            v=blk[s["bif"]];v[0]+=b;v[1]+=m;v[2]+=1
            post=prior*ez;post/=max(post.sum(),1e-300);prior=post@T
    gs=[(a-b)/c for a,b,c in blk.values() if c];sd=float(np.std(gs,ddof=1)) if len(gs)>1 else 0.
    return {"n":n,"base_bits":total_base/n,"model_bits":total_mod/n,"gain_bits":(total_base-total_mod)/n,
            "blocks_n":len(gs),"blocks_positive":int(sum(x>0 for x in gs)),"block_mean":float(np.mean(gs)),
            "block_sd":sd,"block_mean_over_sd":float(np.mean(gs)/sd) if sd>0 else None}

# ---------- source dynamics ----------
def source_sequences(target_marginal,seed=20261004):
    bywit={}
    for row in SOURCE_ROWS:
        rec=collections.defaultdict(list)
        for x,rid in zip(row["ids"],row["recipes"]):rec[int(rid)].append(int(x))
        bywit[row["witness_id"]]=list(rec.values())
    out=[]
    for wi,w in enumerate(sorted(bywit)):
        recipes=bywit[w];ids=[x for q in recipes for x in q];cnt=collections.Counter(ids)
        rng=np.random.default_rng(seed+97*(wi+1));types=list(cnt.items());rng.shuffle(types);types.sort(key=lambda z:-z[1])
        target=target_marginal*len(ids);assigned=np.zeros(K);mp={}
        for typ,n in types:
            cost=((assigned+n-target)/np.maximum(target,1))**2-((assigned-target)/np.maximum(target,1))**2
            a=int(np.argmin(cost));mp[typ]=a;assigned[a]+=n
        out += [[mp[x] for x in q] for q in recipes if len(q)>=2]
    return out

def source_hmm(obsseqs,S,seed=7,maxem=50):
    # ordinary categorical HMM used only to freeze source state dynamics
    rng=np.random.default_rng(seed+S);pi=np.ones(S)/S;T=rng.dirichlet(np.ones(S)*2,size=S)
    E=rng.dirichlet(np.ones(K)*2,size=S)
    last=-1e100
    for it in range(maxem):
        X=np.zeros((S,S));pc=np.zeros(S);EC=np.zeros((S,K));ll=0.
        for y in obsseqs:
            y=np.asarray(y,int);n=len(y);em=E[:,y].T
            a=np.zeros((n,S));sc=np.zeros(n);a[0]=pi*em[0];sc[0]=a[0].sum();a[0]/=max(sc[0],1e-300)
            for t in range(1,n):a[t]=(a[t-1]@T)*em[t];sc[t]=a[t].sum();a[t]/=max(sc[t],1e-300)
            b=np.ones((n,S))
            for t in range(n-2,-1,-1):b[t]=T@(em[t+1]*b[t+1]);b[t]/=max(sc[t+1],1e-300)
            g=a*b;g/=np.maximum(g.sum(1,keepdims=True),1e-300);pc+=g[0];ll+=float(np.log(np.maximum(sc,1e-300)).sum())
            for k in range(K):EC[:,k]+=g[y==k].sum(0)
            for t in range(n-1):
                x=a[t][:,None]*T*(em[t+1]*b[t+1])[None,:];x/=max(x.sum(),1e-300);X+=x
        pi=pc+SMOOTH;pi/=pi.sum();T=X+SMOOTH;T/=T.sum(1,keepdims=True);E=EC+1.;E/=E.sum(1,keepdims=True)
        if abs(ll-last)<1e-4:break
        last=ll
    return pi,T

# ---------- validation fitting ----------
D=[l for l in LINES if l["fold"] in (2,3)];V=[l for l in LINES if l["fold"]==4]
Dseq,Vseq,_,_=build_split((2,3),(4,))
marg=np.bincount(np.concatenate([s["Y"] for s in Dseq]),minlength=K).astype(float);marg/=marg.sum()
SRC_DYN={S:source_hmm(source_sequences(marg),S) for S in STATE_GRID}

def fit_best(seqs,S,family,seedbase):
    best=None
    for j in range(3):
        seed=seedbase+101*j
        if family=="AUTO":m=fit_auto(seqs,S,seed)
        elif family=="SECTION":m=fit_section(seqs,S,seed)
        else:m=fit_auto(seqs,S,seed,fixed=SRC_DYN[S])
        if best is None or m["ll"]>best["ll"]:best=m
    return best

def val_task(arg):
    family,S=arg;m=fit_best(Dseq,S,family,20261004+S*1000+FAMILIES.index(family)*10000);sc=predictive_score(Vseq,m,family)
    return family,S,sc

# ---------- innovation audit ----------
def model_probs_target(seqs,model,family):
    ev=[];ls=[]
    for s in seqs:
        P,Y=s["P"],s["Y"];S=len(model["B"])
        if family=="SECTION":pi=model["Pis"].get(s["section"],np.ones(S)/S);T=model["Ts"].get(s["section"],np.ones((S,S))/S)
        else:pi=model["pi"];T=model["T"]
        prior=pi.copy();zline=[]
        for t in range(len(Y)):
            qz=np.stack([state_q(P[t],model["B"][z]) for z in range(S)])
            pp=prior@qz
            e={"p":pp,"y":int(Y[t]),"prev":int(s["prev0"] if t==0 else Y[t-1]),"half":int(s["fold"]),"line":(s["folio"],s["line"])}
            ev.append(e);zline.append(e)
            ez=qz[:,Y[t]];post=prior*ez;post/=max(post.sum(),1e-300);prior=post@T
        if zline:ls.append(zline)
    return ev,ls

def baseline_target(seqs):
    ev=[];ls=[]
    for s in seqs:
        z=[]
        for t,y in enumerate(s["Y"]):
            e={"p":s["P"][t],"y":int(y),"prev":int(s["prev0"] if t==0 else s["Y"][t-1]),"half":int(s["fold"]),"line":(s["folio"],s["line"])}
            ev.append(e);z.append(e)
        if z:ls.append(z)
    return ev,ls

def simulate_metrics(arg):
    seed,kind,family,model,seqs=arg;rng=np.random.default_rng(seed);ev=[];ls=[]
    for s in seqs:
        P=s["P"];n=len(P);Y=[];zline=[]
        if kind=="baseline":
            for t in range(n):Y.append(int(rng.choice(K,p=P[t])))
            for t,y in enumerate(Y):
                e={"p":P[t],"y":y,"prev":int(s["prev0"] if t==0 else Y[t-1]),"half":int(s["fold"]),"line":(s["folio"],s["line"])};ev.append(e);zline.append(e)
        else:
            S=len(model["B"])
            if family=="SECTION":prior=model["Pis"].get(s["section"],np.ones(S)/S).copy();T=model["Ts"].get(s["section"],np.ones((S,S))/S)
            else:prior=model["pi"].copy();T=model["T"]
            for t in range(n):
                qz=np.stack([state_q(P[t],model["B"][z]) for z in range(S)]);pp=prior@qz;y=int(rng.choice(K,p=pp));Y.append(y)
                e={"p":pp,"y":y,"prev":int(s["prev0"] if t==0 else Y[t-1]),"half":int(s["fold"]),"line":(s["folio"],s["line"])};ev.append(e);zline.append(e)
                ez=qz[:,y];post=prior*ez;post/=max(post.sum(),1e-300);prior=post@T
        if zline:ls.append(zline)
    return metric_vector(ev,ls,True)[0].tolist()

def group_indices(names):
    return {
      "repeat":[i for i,n in enumerate(names) if n.startswith("repeat_resid_")],
      "transition":[i for i,n in enumerate(names) if n in ("transition_resid_norm","half_transition_concordance","sv1_energy","sv12_energy")],
      "surprise":[i for i,n in enumerate(names) if n.startswith("surprise_ac_") or n in ("mean_excess_surprise","pearson_energy")]
    }

def audit_target(seqs,model=None,family=None,baseline=False,B=500):
    if baseline:ev,ls=baseline_target(seqs)
    else:ev,ls=model_probs_target(seqs,model,family)
    tv,names=metric_vector(ev,ls,True);kind="baseline" if baseline else "model"
    jobs=[(202610600000+i,kind,family,model,seqs) for i in range(B)]
    with ProcessPoolExecutor(max_workers=min(32,os.cpu_count() or 8)) as ex:A=np.array(list(ex.map(simulate_metrics,jobs,chunksize=2)),float)
    mu,inv=mahal_setup(A);ds=np.array([distance(x,mu,inv) for x in A]);td=distance(tv,mu,inv)
    gi=group_indices(names);gd={}
    for g,ix in gi.items():
        M=A[:,ix];m,ivv=mahal_setup(M);dd=np.array([distance(x,m,ivv) for x in M]);tt=distance(tv[ix],m,ivv)
        gd[g]={"distance":float(tt),"null_q99":float(np.quantile(dd,.99)),"p_add":float((1+np.sum(dd>=tt))/(B+1))}
    return {"distance":float(td),"null_median":float(np.median(ds)),"null_q99":float(np.quantile(ds,.99)),"p_add":float((1+np.sum(ds>=td))/(B+1)),
            "groups":gd,"target_metrics":{n:float(v) for n,v in zip(names,tv)}}

def param_count(family,S,nsec):
    em=S*(K-1)
    if family=="AUTO":return em+S*(S-1)+(S-1)
    if family=="SECTION":return em+nsec*(S*(S-1)+(S-1))
    return em

if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=min(15,os.cpu_count() or 8)) as ex:
        vals=list(ex.map(val_task,[(f,S) for f in FAMILIES for S in STATE_GRID],chunksize=1))
    val={f:{} for f in FAMILIES}
    for f,S,sc in vals:val[f][str(S)]=sc
    selected={f:max(STATE_GRID,key=lambda S:val[f][str(S)]["gain_bits"]) for f in FAMILIES}

    # Final downstream fit on folds 2/3/4 and frozen source dynamics refit using only those training marginals.
    TRseq,TEseq,_,_=build_split((2,3,4),(0,1))
    marg=np.bincount(np.concatenate([s["Y"] for s in TRseq]),minlength=K).astype(float);marg/=marg.sum()
    SRC_FINAL={S:source_hmm(source_sequences(marg),S) for S in STATE_GRID}
    final_models={};final_scores={};ladders={f:{} for f in FAMILIES}
    for f in FAMILIES:
        for S in STATE_GRID:
            best=None
            for j in range(3):
                seed=20261100+S*1000+FAMILIES.index(f)*10000+101*j
                if f=="AUTO":m=fit_auto(TRseq,S,seed)
                elif f=="SECTION":m=fit_section(TRseq,S,seed)
                else:m=fit_auto(TRseq,S,seed,fixed=SRC_FINAL[S])
                if best is None or m["ll"]>best["ll"]:best=m
            sc=predictive_score(TEseq,best,f);d=param_count(f,S,len(set(x["section"] for x in TRseq)))
            pen=.5*d*math.log2(max(sum(len(x["Y"]) for x in TRseq),2))/max(sc["n"],1)
            sc["target_fit_params"]=d;sc["mdl_param_penalty_bits_per_test_event"]=pen;sc["net_gain_after_mdl_diagnostic"]=sc["gain_bits"]-pen
            ladders[f][str(S)]=sc
            if S==selected[f]:final_models[f]=best;final_scores[f]=sc

    baseline_audit=audit_target(TEseq,baseline=True,B=500)
    audits={}
    for f in FAMILIES:audits[f]=audit_target(TEseq,final_models[f],f,False,B=500)

    gate={}
    for f in FAMILIES:
        S=selected[f];a=audits[f];sc=final_scores[f]
        reduction=1-a["distance"]/baseline_audit["distance"]
        improved=[g for g in ("repeat","transition","surprise") if a["groups"][g]["distance"]<baseline_audit["groups"][g]["distance"]]
        gate[f]={"S":S,"positive_gain":sc["gain_bits"]>0,"majority_blocks":sc["blocks_positive"]>sc["blocks_n"]/2,
                 "distance_reduction_fraction":float(reduction),"improved_groups":improved,
                 "passes":bool(S<=8 and sc["gain_bits"]>0 and sc["blocks_positive"]>sc["blocks_n"]/2 and reduction>=.50 and len(improved)>=3)}

    out={"validation":val,"selected":selected,"final_scores":final_scores,"test_ladders_descriptive":ladders,
         "innovation":{"baseline":baseline_audit,"families":audits},"promotion_gate":gate,
         "n_train_events":sum(len(x["Y"]) for x in TRseq),"n_test_events":sum(len(x["Y"]) for x in TEseq),
         "n_train_lines":len(TRseq),"n_test_lines":len(TEseq)}
    print("UPSTREAM_STATE_RESULT_JSON="+json.dumps(out,separators=(",",":")),flush=True)

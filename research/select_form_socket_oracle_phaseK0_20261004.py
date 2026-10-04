#!/usr/bin/env python3
# Phase K0: reconstruct frozen state-separated FORM socket and measure F0/F1 oracle identifiability.
# Synthetic-only. NO P70. No real Voynich inversion.
import collections,json,math,urllib.request
import numpy as np

LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
m={"__name__":"latent"}
exec(compile(urllib.request.urlopen(LAT_URL,timeout=60).read().decode(),LAT_URL,"exec"),m)
rows=m["rows"]; segment=m["segment"]; ST=m["ST"]; CLASSES=m["CLASSES"]
K12=12; START_IN=12
# Frozen strict K8 continuation response grouping from state-separated FORM audit.
K8_GROUPS=[(0,),(1,),(2,10),(3,7,8),(4,11),(5,),(6,),(9,)]
G_OF={c:g for g,cs in enumerate(K8_GROUPS) for c in cs}
assert len(G_OF)==12

# Convert corpus to piece routes.
routes=[]
for r in rows:
    z=segment(r["token"])
    routes.append(z)
PIECES=sorted({p for z in routes for p in z})
PID={p:i for i,p in enumerate(PIECES)}
PCLASS=np.array([ST[p] for p in PIECES],dtype=int)
NP=len(PIECES)

# ---------- Build frozen FORM socket from the real corpus ----------
start_class=np.full(K12,.5,float)
start_piece=np.full((K12,NP),0.,float)
# continuation class counts by K8 source-response group
cont=np.zeros((8,K12),float)
# exact next-piece realization conditional on source group + destination class
next_piece=np.zeros((8,K12,NP),float)
# STOP counts exact piece x depthbucket x incomingclass/sentinel
stop_n=np.zeros((NP,4,K12+1),float);stop_y=np.zeros_like(stop_n)
# lower-order stop priors
pd_n=np.zeros((NP,4),float);pd_y=np.zeros_like(pd_n)
p_n=np.zeros(NP,float);p_y=np.zeros(NP,float)

for z in routes:
    ids=[PID[p] for p in z]; cls=[ST[p] for p in z]
    start_class[cls[0]]+=1
    start_piece[cls[0],ids[0]]+=1
    for i,(pp,cc) in enumerate(zip(ids,cls)):
        dep=min(i,3);inc=START_IN if i==0 else cls[i-1];y=1.0 if i==len(ids)-1 else 0.0
        stop_n[pp,dep,inc]+=1;stop_y[pp,dep,inc]+=y
        pd_n[pp,dep]+=1;pd_y[pp,dep]+=y;p_n[pp]+=1;p_y[pp]+=y
        if i<len(ids)-1:
            dc=cls[i+1];g=G_OF[cc]
            cont[g,dc]+=1;next_piece[g,dc,ids[i+1]]+=1

P_START_CLASS=start_class/start_class.sum()
# exact piece within start class, Jeffreys on pieces in class that actually occur at START
P_START_PIECE=np.zeros_like(start_piece)
for c in range(K12):
    ok=(PCLASS==c)&(start_piece[c]>0)
    if ok.any():
        v=start_piece[c,ok]+.5;P_START_PIECE[c,ok]=v/v.sum()

# Continuation edges: rowwise retain smallest destination set covering 99% empirical mass.
LEGAL=np.zeros((8,K12),bool)
P_CONT=np.zeros((8,K12),float)
for g in range(8):
    order=np.argsort(cont[g])[::-1];tot=cont[g].sum();cum=0.
    for c in order:
        if cont[g,c]<=0: continue
        LEGAL[g,c]=True;cum+=cont[g,c]
        if cum>=.99*tot:break
    v=cont[g,LEGAL[g]]+.5
    P_CONT[g,LEGAL[g]]=v/v.sum()

P_NEXT=np.zeros_like(next_piece)
for g in range(8):
    for c in range(K12):
        if not LEGAL[g,c]:continue
        ok=(PCLASS==c)&(next_piece[g,c]>0)
        if not ok.any():
            ok=(PCLASS==c)
            v=np.ones(ok.sum(),float)
        else:v=next_piece[g,c,ok]+.5
        P_NEXT[g,c,ok]=v/v.sum()

# Hierarchical fixed stop hazard:
# exact piece+depth+incoming shrunk to piece+depth, itself to piece.
global_stop=(p_y.sum()+1)/(p_n.sum()+2)
P_STOP=np.zeros_like(stop_n)
for pp in range(NP):
    p_piece=(p_y[pp]+4*global_stop)/(p_n[pp]+4)
    for dep in range(4):
        p_pd=(pd_y[pp,dep]+6*p_piece)/(pd_n[pp,dep]+6)
        for inc in range(K12+1):
            P_STOP[pp,dep,inc]=(stop_y[pp,dep,inc]+5*p_pd)/(stop_n[pp,dep,inc]+5)
P_STOP=np.clip(P_STOP,.005,.995)

def row_soft(base,bias,mask):
    q=np.zeros_like(base,float);ok=mask & (base>0)
    x=np.log(np.maximum(base[ok],1e-30))+bias[ok];mx=x.max();v=np.exp(x-mx);v/=v.sum();q[ok]=v
    return q

# ---------- Hidden upstream source ----------
def source_graph(K,d,rng):
    A=np.zeros((K,K),float)
    for i in range(K):
        keep={i}
        while len(keep)<d:keep.add(int(rng.integers(K)))
        js=sorted(keep);w=rng.gamma(1.5,1.,len(js));w[js.index(i)]+=2.;w/=w.sum();A[i,js]=w
    return A,np.ones(K)/K

def hidden_seq(A,pi,N,rng):
    K=len(pi);z=np.empty(N,int);z[0]=rng.choice(K,p=pi)
    for t in range(1,N):z[t]=rng.choice(K,p=A[z[t-1]])
    return z

def controls(K,rank,strength,rng,family):
    U=rng.normal(size=(K,rank));U-=U.mean(0);U/=np.maximum(U.std(0),1e-9)
    Ve=rng.normal(size=(rank,K12));Ve-=Ve.mean(1,keepdims=True);Ve/=np.maximum(Ve.std(),1e-9)
    Vr=rng.normal(size=(rank,K12));Vr-=Vr.mean(1,keepdims=True);Vr/=np.maximum(Vr.std(),1e-9)
    return U,Ve*strength,(Vr*strength if family=="F1" else np.zeros_like(Vr))

def gen_token(x,U,Ve,Vr,rng,maxlen=30):
    be=U[x]@Ve;br=U[x]@Vr
    q=row_soft(P_START_CLASS,be,np.ones(K12,bool));c=int(rng.choice(K12,p=q))
    qp=P_START_PIECE[c];pp=int(rng.choice(NP,p=qp))
    out=[pp];inc=START_IN
    for depth in range(maxlen):
        dep=min(depth,3)
        if rng.random()<P_STOP[pp,dep,inc]:return out
        g=G_OF[PCLASS[pp]]
        qd=row_soft(P_CONT[g],br,LEGAL[g]);dc=int(rng.choice(K12,p=qd))
        qn=P_NEXT[g,dc];npp=int(rng.choice(NP,p=qn))
        inc=PCLASS[pp];pp=npp;out.append(pp)
    # Pathological long synthetic trajectory: reject rather than forge STOP.
    return None

def generate(K,d,rank,strength,N,seed,family):
    rng=np.random.default_rng(seed);A,pi=source_graph(K,d,rng);U,Ve,Vr=controls(K,rank,strength,rng,family)
    z=hidden_seq(A,pi,N,rng);obs=[]
    for x in z:
        for _ in range(100):
            t=gen_token(int(x),U,Ve,Vr,rng)
            if t is not None:obs.append(t);break
        else:raise RuntimeError("synthetic token failed to terminate")
    return A,pi,U,Ve,Vr,z,obs

def token_loglik(tok,x,U,Ve,Vr):
    be=U[x]@Ve;br=U[x]@Vr
    c0=PCLASS[tok[0]]
    q=row_soft(P_START_CLASS,be,np.ones(K12,bool))
    lp=math.log(max(q[c0],1e-300))+math.log(max(P_START_PIECE[c0,tok[0]],1e-300))
    inc=START_IN
    for i,pp in enumerate(tok):
        dep=min(i,3);ps=P_STOP[pp,dep,inc]
        if i==len(tok)-1:
            lp+=math.log(max(ps,1e-300));break
        lp+=math.log(max(1-ps,1e-300))
        g=G_OF[PCLASS[pp]];dc=PCLASS[tok[i+1]]
        qd=row_soft(P_CONT[g],br,LEGAL[g])
        lp+=math.log(max(qd[dc],1e-300))+math.log(max(P_NEXT[g,dc,tok[i+1]],1e-300))
        inc=PCLASS[pp]
    return lp

def emission(obs,U,Ve,Vr):
    E=np.empty((len(obs),len(U)),float)
    for t,tok in enumerate(obs):
        for x in range(len(U)):E[t,x]=token_loglik(tok,x,U,Ve,Vr)
    return E

def fb(E,A,pi):
    N,K=E.shape;la=np.log(np.maximum(A,1e-300));lp=np.log(np.maximum(pi,1e-300))
    al=np.empty((N,K));sc=np.empty(N)
    a=lp+E[0];mx=a.max();sc[0]=mx+math.log(np.exp(a-mx).sum());al[0]=a-sc[0]
    for t in range(1,N):
        M=al[t-1][:,None]+la;mm=M.max(0);pr=mm+np.log(np.exp(M-mm).sum(0))
        a=pr+E[t];mx=a.max();sc[t]=mx+math.log(np.exp(a-mx).sum());al[t]=a-sc[t]
    be=np.zeros((N,K))
    for t in range(N-2,-1,-1):
        M=la+E[t+1][None,:]+be[t+1][None,:];mm=M.max(1)
        be[t]=mm+np.log(np.exp(M-mm[:,None]).sum(1))-sc[t+1]
    lg=al+be;mm=lg.max(1,keepdims=True);g=np.exp(lg-mm);g/=g.sum(1,keepdims=True)
    return float(sc.sum()),g

def nmi(a,b):
    a=np.asarray(a);b=np.asarray(b);n=len(a);ua,ia=np.unique(a,return_inverse=True);ub,ib=np.unique(b,return_inverse=True)
    C=np.zeros((len(ua),len(ub)),float);np.add.at(C,(ia,ib),1);P=C/n;pa=P.sum(1);pb=P.sum(0);mi=0.
    for i in range(len(pa)):
        for j in range(len(pb)):
            if P[i,j]>0:mi+=P[i,j]*math.log(P[i,j]/(pa[i]*pb[j]))
    ha=-sum(x*math.log(x) for x in pa if x>0);hb=-sum(x*math.log(x) for x in pb if x>0)
    return float(mi/max((ha+hb)/2,1e-15))

def ari(a,b):
    a=np.asarray(a);b=np.asarray(b);n=len(a);ua,ia=np.unique(a,return_inverse=True);ub,ib=np.unique(b,return_inverse=True)
    C=np.zeros((len(ua),len(ub)),dtype=np.int64);np.add.at(C,(ia,ib),1)
    comb=lambda x:x*(x-1)/2
    sij=comb(C).sum();sa=comb(C.sum(1)).sum();sb=comb(C.sum(0)).sum();tot=comb(n);ex=sa*sb/max(tot,1);mx=.5*(sa+sb)
    return float((sij-ex)/max(mx-ex,1e-15))

if __name__=="__main__":
    qa={
      "real_tokens":len(routes),"piece_count":NP,"k12":K12,"k8_groups":[list(x) for x in K8_GROUPS],
      "legal_cont_edges":int(LEGAL.sum()),
      "mean_real_piece_len":float(np.mean([len(x) for x in routes])),
      "start_classes_used":int((P_START_CLASS>0).sum()),
      "stop_min":float(P_STOP.min()),"stop_max":float(P_STOP.max())
    }
    print("SELECT_FORM_SOCKET_QA_JSON="+json.dumps(qa,separators=(",",":")),flush=True)
    panel=[]
    for family in ("F0","F1"):
      for strength in (.5,1.,1.5,2.,3.,4.,5.):
        reps=[]
        for rep,seed in enumerate((20261004,20261005,20261006)):
          A,pi,U,Ve,Vr,z,obs=generate(16,4,2,strength,4000,seed,family)
          E=emission(obs,U,Ve,Vr);ll,g=fb(E,A,pi);pred=g.argmax(1)
          rr={"seed":seed,"nmi":nmi(z,pred),"ari":ari(z,pred),"ll":ll,
              "mean_len":float(np.mean([len(x) for x in obs]))}
          reps.append(rr)
        q={"family":family,"strength":strength,
           "median_nmi":float(np.median([x["nmi"] for x in reps])),
           "min_nmi":float(min(x["nmi"] for x in reps)),"reps":reps}
        print("SELECT_FORM_ORACLE_JSON="+json.dumps(q,separators=(",",":")),flush=True);panel.append(q)
    print("SELECT_FORM_PHASEK0_JSON="+json.dumps({"qa":qa,"panel":panel},separators=(",",":")),flush=True)

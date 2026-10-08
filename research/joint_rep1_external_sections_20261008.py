#!/usr/bin/env python3
# JOINT-REP1  preregistered 2026-10-08.
import collections,json,math,os,urllib.request
import numpy as np
from scipy.optimize import minimize_scalar
from concurrent.futures import ProcessPoolExecutor

ABAURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/dd58810db1b46d8f2d891228195cf6de7cb776ec/research/aba_rep1_external_sections_20261008.py"
src=urllib.request.urlopen(ABAURL,timeout=120).read().decode()
a={"__name__":"aba_module"};exec(compile(src,ABAURL,"exec"),a)

TARGETS=a["TARGETS"];HA=a["HA"];Q13=a["Q13"];rows=a["rows"];K=a["K"]
para_id=a["para_id"];base_prob=a["base_prob"];fnum=a["fnum"];tilt_p=a["tilt_p"]
position_adjust_target=a["position_adjust_target"];fit_pos_target=a["fit_pos_target"];pos_bias=a["pos_bias"]
SC_LEN=a["SC_LEN"];metric_vector=a["ns"]["metric_vector"];L2=10.0
REAL_BASE=a["REAL_BASE"]
SECTIONS=("Herbal-A","Q13")
SEEDS=list(range(202610090650,202610091100))
TEMP_L2_HALF=5.0
RIDGE=1.0

def relation_F(x,y):
    F=np.zeros((K,3),float)
    if x==y:F[y,0]=1.
    else:F[x,1]=1.;F[y,2]=1.
    return F

def arrays_E(lines,parity):
    P=[];Y=[];FF=[]
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        for t in range(2,len(seq)):
            x=int(seq[t-2]["y"]);y=int(seq[t-1]["y"])
            P.append(seq[t]["p"]);Y.append(int(seq[t]["y"]));FF.append(relation_F(x,y))
    return np.asarray(P,float),np.asarray(Y,int),np.asarray(FF,float)

def fit_E(lines,parity):
    P,Y,F=arrays_E(lines,parity);b=np.zeros(3,float)
    if len(Y)==0:return b
    def ngh(beta):
        sc=np.log(np.maximum(P,1e-15))+np.tensordot(F,beta,axes=([2],[0]))
        mx=sc.max(1,keepdims=True);Q=np.exp(sc-mx);Q/=Q.sum(1,keepdims=True)
        loss=-float(np.sum(np.log(np.maximum(Q[np.arange(len(Y)),Y],1e-300))))+.5*L2*float(beta@beta)
        Ef=np.einsum("nk,nkj->nj",Q,F);Fy=F[np.arange(len(Y)),Y,:]
        g=(Ef-Fy).sum(0)+L2*beta
        E2=np.einsum("nk,nkj,nkl->jl",Q,F,F)
        H=np.eye(3)*L2+E2-Ef.T@Ef
        return loss,g,H
    for _ in range(25):
        loss,g,H=ngh(b)
        if np.max(np.abs(g))<1e-9:break
        step=np.linalg.solve(H+np.eye(3)*1e-10,g);tt=1.
        while tt>1e-6:
            c=b-tt*step;nl,_,_=ngh(c)
            if nl<=loss+1e-12:b=c;break
            tt*=.5
        if tt<=1e-6:break
    return b

def apply_E(p,beta,x,y):
    F=relation_F(int(x),int(y))
    sc=np.log(np.maximum(np.asarray(p,float),1e-15))+F@np.asarray(beta,float)
    sc-=sc.max();q=np.exp(sc);return q/q.sum()

def augment_E(lines):
    E={0:fit_E(lines,0),1:fit_E(lines,1)};out=[]
    for seq in lines:
        if not seq:continue
        te=fnum(seq[0]["folio"])%2;tr=1-te;zz=[]
        for t,e in enumerate(seq):
            p=np.asarray(e["p"],float)
            if t>=2:p=apply_E(p,E[tr],int(seq[t-2]["y"]),int(seq[t-1]["y"]))
            x=dict(e);x["p"]=p;zz.append(x)
        out.append(zz)
    return out,E

def temp_p(p,theta):
    lam=math.exp(float(theta));sc=lam*np.log(np.maximum(np.asarray(p,float),1e-15))
    sc-=sc.max();q=np.exp(sc);return q/q.sum()

def fit_theta(lines,parity):
    P=[];Y=[]
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        for e in seq:P.append(e["p"]);Y.append(int(e["y"]))
    P=np.asarray(P,float);Y=np.asarray(Y,int);LP=np.log(np.maximum(P,1e-15))
    if len(Y)==0:return 0.
    def obj(th):
        lam=math.exp(float(th));sc=lam*LP;mx=sc.max(1,keepdims=True)
        z=mx[:,0]+np.log(np.exp(sc-mx).sum(1))
        return -float(np.sum(sc[np.arange(len(Y)),Y]-z))+TEMP_L2_HALF*float(th*th)
    return float(minimize_scalar(obj,bounds=(-3.,3.),method="bounded",options={"xatol":1e-10,"maxiter":200}).x)

def augment_temp(lines):
    th={0:fit_theta(lines,0),1:fit_theta(lines,1)};out=[]
    for seq in lines:
        if not seq:continue
        te=fnum(seq[0]["folio"])%2;tr=1-te;zz=[]
        for e in seq:
            x=dict(e);x["p"]=temp_p(e["p"],th[tr]);zz.append(x)
        out.append(zz)
    return out,th

def analyze_section(base):
    q0,posmods=position_adjust_target(base)
    e3,E=augment_E(q0)
    qt,th=augment_temp(e3)
    v,names,n=metric_vector(qt)
    return {"vector":v,"names":names,"n_events":n,"n_lines":len(qt),
            "posmods":posmods,"E":E,"theta":th}

REAL={s:analyze_section(REAL_BASE[s]) for s in SECTIONS}
NAMES=REAL["Herbal-A"]["names"]
AN=["repeat_resid_1","repeat_resid_2","repeat_resid_3","repeat_resid_5","repeat_resid_10","transition_resid_norm","sv1_energy","sv12_energy"]
BN=["surprise_ac_1","surprise_ac_2","surprise_ac_3","surprise_ac_5","surprise_ac_10","mean_excess_surprise","pearson_energy"]
AI=[NAMES.index(x) for x in AN];BI=[NAMES.index(x) for x in BN]

def target_of(fol):
    if fol in HA:return "Herbal-A"
    if fol in Q13:return "Q13"
    return None

def generate_joint(seed):
    rng=np.random.default_rng(seed)
    page=collections.defaultdict(lambda:np.zeros(K,float))
    para=collections.defaultdict(lambda:np.zeros(K,float))
    hist=collections.defaultdict(list)
    by={s:collections.OrderedDict() for s in SECTIONS}
    for r in rows:
        fol=r["folio"];ln=int(r["line"]);lk=(fol,ln);pk=fol;pid=para_id(pk,ln);pq=(pk,pid)
        yobs=int(r["start"]);pos=int(r["pos"])
        if pos==0:y=yobs
        else:
            prev_piece=hist[lk][-1][1];rc=np.zeros(K,float)
            for yy,pp in hist[lk][-6:]:rc[int(yy)]+=1
            pbase=base_prob(r["section"],prev_piece,page[pk],para[pq],rc);pgen=pbase
            sec=target_of(fol)
            if sec is not None:
                seq=by[sec].setdefault(lk,[]);i=len(seq);n=SC_LEN[lk];te=fnum(fol)%2;tr=1-te
                rr=REAL[sec]
                pgen=tilt_p(pgen,pos_bias(rr["posmods"][tr],i,n))
                if i>=2:pgen=apply_E(pgen,rr["E"][tr],int(seq[i-2]["y"]),int(seq[i-1]["y"]))
                pgen=temp_p(pgen,rr["theta"][tr])
                y=int(rng.choice(K,p=pgen))
                seq.append({"p":pbase,"y":y,"prev":int(hist[lk][-1][0]),"folio":fol,"line":lk,"bif":r["bifolium"]})
            else:y=int(rng.choice(K,p=pbase))
        page[pk][y]+=1;para[pq][y]+=1;hist[lk].append((y,r["final_piece"]))
    return {s:list(by[s].values()) for s in SECTIONS}

def task(seed):
    b=generate_joint(seed)
    return seed,{s:analyze_section(b[s])["vector"].tolist() for s in SECTIONS}

def ridge_fit(X,Y):
    return np.linalg.solve(X.T@X+RIDGE*np.eye(X.shape[1]),X.T@Y)
def precision(E):
    S=np.cov(E,rowvar=False);C=.75*S+.25*np.diag(np.diag(S))+np.eye(S.shape[0])*1e-9
    return np.linalg.pinv(C)

def section_joint(F,C,B,R):
    ma=F[:,AI].mean(0);sa=np.maximum(F[:,AI].std(0,ddof=1),1e-12)
    mb=F[:,BI].mean(0);sb=np.maximum(F[:,BI].std(0,ddof=1),1e-12)
    def st(V):return (V[:,AI]-ma)/sa,(V[:,BI]-mb)/sb
    AF,BF=st(F);AC,BC=st(C);AB,BB=st(B)
    Ar=(R[AI]-ma)/sa;Br=(R[BI]-mb)/sb
    Wba=ridge_fit(AF,BF);Wab=ridge_fit(BF,AF)
    Pba=precision(BF-AF@Wba);Pab=precision(AF-BF@Wab)
    def ds(A,B):
        rb=B-A@Wba;ra=A-B@Wab
        return np.einsum("ni,ij,nj->n",rb,Pba,rb),np.einsum("ni,ij,nj->n",ra,Pab,ra)
    dBc,dAc=ds(AC,BC);dBb,dAb=ds(AB,BB)
    rb=Br-Ar@Wba;ra=Ar-Br@Wab
    dBr=float(rb@Pba@rb);dAr=float(ra@Pab@ra)
    mB=float(dBc.mean());sB=max(float(dBc.std(ddof=1)),1e-12)
    mA=float(dAc.mean());sA=max(float(dAc.std(ddof=1)),1e-12)
    zBc=(dBc-mB)/sB;zAc=(dAc-mA)/sA;zBb=(dBb-mB)/sB;zAb=(dAb-mA)/sA
    zBr=float((dBr-mB)/sB);zAr=float((dAr-mA)/sA)
    Jc=np.maximum(zBc,zAc);Jb=np.maximum(zBb,zAb);Jr=max(zBr,zAr)
    return {"Jc":Jc,"Jb":Jb,"Jr":Jr,"zB_c":zBc,"zA_c":zAc,"zB_r":zBr,"zA_r":zAr,
            "D_B_r":dBr,"D_A_r":dAr}

if __name__=="__main__":
    print("REAL",json.dumps({s:{"n_events":REAL[s]["n_events"],"n_lines":REAL[s]["n_lines"],
        "vector":REAL[s]["vector"].tolist()} for s in SECTIONS},separators=(",",":")),flush=True)
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(task,SEEDS,chunksize=2))
    mp={seed:v for seed,v in rr}
    SRES={}
    for s in SECTIONS:
        V=np.vstack([np.asarray(mp[x][s],float) for x in SEEDS])
        F=V[:250];C=V[250:350];B=V[350:]
        SRES[s]=section_joint(F,C,B,REAL[s]["vector"])
    Jc=np.maximum(SRES["Herbal-A"]["Jc"],SRES["Q13"]["Jc"])
    Jb=np.maximum(SRES["Herbal-A"]["Jb"],SRES["Q13"]["Jb"])
    realJ={s:float(SRES[s]["Jr"]) for s in SECTIONS};rmax=max(realJ.values())
    fq=float(np.quantile(Jc,.99));blind=float(np.mean(Jb<=fq))
    p=float((1+np.sum(np.r_[Jc,Jb]>=rmax))/(len(Jc)+len(Jb)+1))
    globalpass=bool(blind>=.90 and rmax>fq and p<=.01)
    ownq={s:float(np.quantile(SRES[s]["Jc"],.99)) for s in SECTIONS}
    resolved={s:bool(globalpass and realJ[s]>ownq[s]) for s in SECTIONS}
    nr=sum(resolved.values())
    if not globalpass:decision="NO_EXTERNAL_COUPLING_REPLICATION"
    elif nr==2:decision="TWO_SECTION_COUPLING_REPLICATION"
    elif nr==1:decision="ONE_SECTION_COUPLING_REPLICATION"
    else:decision="GLOBAL_MAX_PASS_NO_SECTION_OWN_Q99"
    detail={}
    for s in SECTIONS:
        rr=SRES[s];qB=float(np.quantile(rr["zB_c"],.99));qA=float(np.quantile(rr["zA_c"],.99))
        rB=bool(resolved[s] and rr["zB_r"]>qB);rA=bool(resolved[s] and rr["zA_r"]>qA)
        if resolved[s]:
            label="BIDIRECTIONAL_CONDITIONAL_MISMATCH" if rB and rA else "SURPRISE_GIVEN_TOPOLOGY_MISMATCH" if rB else "TOPOLOGY_GIVEN_SURPRISE_MISMATCH" if rA else "GLOBAL_ONLY_NO_DIRECTION_RESOLVED"
        else:label=None
        detail[s]={"J":realJ[s],"own_q99_J":ownq[s],"resolved":resolved[s],
                   "D_B_given_A":float(rr["D_B_r"]),"D_A_given_B":float(rr["D_A_r"]),
                   "Z_B_given_A":float(rr["zB_r"]),"q99_Z_B_given_A":qB,
                   "Z_A_given_B":float(rr["zA_r"]),"q99_Z_A_given_B":qA,
                   "label":label}
    out={"programme":"JOINT-REP1","status":"complete",
         "familywise":{"real_max_J":rmax,"q99_max_J":fq,"blind_acceptance":blind,"add_one_p":p,
                       "global_pass":globalpass,"decision":decision},
         "sections":detail}
    print("JOINT_REP1_JSON="+json.dumps(out,separators=(",",":")),flush=True)

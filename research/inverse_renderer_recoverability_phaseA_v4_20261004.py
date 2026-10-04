#!/usr/bin/env python3
# Inverse-renderer Phase A v4: vectorised variational state discovery -> exact HMM refinement.
# NO P70.
import argparse,json,time,urllib.request
import numpy as np, torch

V3_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/a149e5372d892423e1354eb2ab6062bd4a933a20/research/inverse_renderer_recoverability_phaseA_v3_20261004.py"
m={"__name__":"v3base"}
exec(compile(urllib.request.urlopen(V3_URL,timeout=60).read().decode(),V3_URL,"exec"),m)

generate=m["generate"];flatten_decisions=m["flatten_decisions"];emission_ll=m["emission_ll"]
oracle=m["oracle"];refine=m["refine"];nmi=m["nmi"];ari=m["ari"]
HAZARD_MIX=m["HAZARD_MIX"];BIAS_CAP=m["BIAS_CAP"]

def variational_init(X,K,rank,seed,device,steps=350,lr=.045):
    rng=np.random.default_rng(seed);N=len(X)
    # Free mean-field posterior logits plus exact model parameters.
    Z=torch.nn.Parameter(torch.tensor(rng.normal(0,.12,(N,K)),dtype=torch.float32,device=device))
    U=torch.nn.Parameter(torch.tensor(rng.normal(0,.12,(K,rank)),dtype=torch.float32,device=device))
    V=torch.nn.Parameter(torch.tensor(rng.normal(0,.12,(rank,m["NCTX"],m["NOPT"])),dtype=torch.float32,device=device))
    ALog=torch.nn.Parameter(torch.tensor(rng.normal(0,.10,(K,K))+1.0*np.eye(K),dtype=torch.float32,device=device))
    piLog=torch.nn.Parameter(torch.zeros(K,dtype=torch.float32,device=device))
    opt=torch.optim.Adam([Z,U,V,ALog,piLog],lr=lr)
    best=None;t0=time.time()
    for step in range(steps):
        frac=step/max(steps-1,1)
        ent_w=1.5-.5*frac  # deterministic annealing toward true mean-field ELBO
        opt.zero_grad(set_to_none=True)
        q=torch.softmax(Z,dim=1)
        E=emission_ll(X,U,V,device)
        logA=torch.log_softmax(ALog,dim=1);logpi=torch.log_softmax(piLog,dim=0)
        emit=(q*E).sum()
        trans=torch.einsum("ti,ij,tj->",q[:-1],logA,q[1:])
        init=(q[0]*logpi).sum()
        ent=-(q*torch.log(torch.clamp(q,min=1e-9))).sum()
        occ=q.mean(0);occ_kl=(occ*torch.log(torch.clamp(occ*K,min=1e-9))).sum()
        # late row-entropy penalty gently prepares sparse transition projection.
        A=torch.softmax(ALog,dim=1);aent=-(A*torch.log(torch.clamp(A,min=1e-9))).sum()
        sparse_w=.0 if frac<.55 else .003*(frac-.55)/.45
        elbo=emit+trans+init+ent_w*ent
        loss=-elbo/N + .04*occ_kl + sparse_w*aent + 5e-4*(U.square().mean()+V.square().mean())
        loss.backward();torch.nn.utils.clip_grad_norm_([Z,U,V,ALog,piLog],8.0);opt.step()
        with torch.no_grad():V[:,:,m["END"]]=0.
        if step%50==0 or step==steps-1:
            with torch.no_grad():
                qq=torch.softmax(Z,dim=1);score=float(elbo.detach().cpu())
                if best is None or score>best[0]:
                    best=(score,qq.cpu().numpy().copy(),torch.softmax(ALog,1).cpu().numpy().copy(),
                          torch.softmax(piLog,0).cpu().numpy().copy(),U.cpu().numpy().copy(),V.cpu().numpy().copy(),step+1)
    score,q,A,pi,U0,V0,st=best
    return {"score":score,"q":q,"A":A,"pi":pi,"U":U0,"V":V0,"steps":st,"seconds":time.time()-t0}

def run_one(strength,args,device):
    dat=generate(args.K,args.d,args.rank,strength,args.N,args.seed);X=flatten_decisions(dat["routes"]);orc=oracle(dat,X,device);runs=[]
    for r in range(args.restarts):
        vi=variational_init(X,args.K,args.rank,args.seed+1000*r,device,args.var_steps,args.var_lr)
        vp=vi["q"].argmax(1);vn=nmi(dat["z"],vp)
        # Exact HMM refinement, starting from variationally learned model; dense first then sparse d.
        fit=refine(X,vi["A"],vi["pi"],vi["U"],vi["V"],args.d,device,args.refine_epochs,args.msteps,args.refine_lr,args.dense_epochs)
        ll,A,pi,U,V,g,ep,sec=fit;pred=g.argmax(1)
        z={"restart":r,"variational_nmi":vn,"variational_ari":ari(dat["z"],vp),"var_seconds":vi["seconds"],
           "final_ll":ll,"final_nmi":nmi(dat["z"],pred),"final_ari":ari(dat["z"],pred),
           "refine_epochs":ep,"refine_seconds":sec}
        runs.append(z);print("V4_RESTART_JSON="+json.dumps({"strength":strength,**z},separators=(",",":")),flush=True)
    return {"strength":strength,"oracle":orc,"runs":runs,
            "median_var_nmi":float(np.median([x["variational_nmi"] for x in runs])),
            "median_final_nmi":float(np.median([x["final_nmi"] for x in runs])),
            "best_final_nmi":max(x["final_nmi"] for x in runs),
            "stable_fraction_nmi70":float(np.mean([x["final_nmi"]>=.70 for x in runs]))}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--N",type=int,default=4000);ap.add_argument("--K",type=int,default=16)
    ap.add_argument("--d",type=int,default=4);ap.add_argument("--rank",type=int,default=2);ap.add_argument("--seed",type=int,default=20261004)
    ap.add_argument("--strengths",default="3.0");ap.add_argument("--restarts",type=int,default=3)
    ap.add_argument("--var-steps",type=int,default=350);ap.add_argument("--var-lr",type=float,default=.045)
    ap.add_argument("--refine-epochs",type=int,default=50);ap.add_argument("--dense-epochs",type=int,default=15)
    ap.add_argument("--msteps",type=int,default=15);ap.add_argument("--refine-lr",type=float,default=.035)
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=ap.parse_args();device=torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    out=[run_one(float(s),args,device) for s in args.strengths.split(",")]
    final={"phase":"A_v4","device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
           "hazard_mix":HAZARD_MIX,"bias_cap":BIAS_CAP,"N":args.N,"K":args.K,"d":args.d,"rank":args.rank,
           "results":out,"gate_pass":bool(all(x["oracle"]["nmi"]>=.70 and x["median_final_nmi"]>=.70 and x["stable_fraction_nmi70"]>=.67 for x in out))}
    print("INVERSE_PHASEA_V4_JSON="+json.dumps(final,separators=(",",":")),flush=True)

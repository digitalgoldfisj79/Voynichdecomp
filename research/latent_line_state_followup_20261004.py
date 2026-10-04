#!/usr/bin/env python3
import collections,json,math,os,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

LINE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
ns={"__name__":"line_module"}
exec(compile(urllib.request.urlopen(LINE_URL,timeout=60).read().decode(),LINE_URL,"exec"),ns)
LINES=ns["LINES"];fit_struct=ns["fit_struct"];attach=ns["attach"];fit_mix=ns["fit_mix"];score=ns["score"];K=12
TR=[l for l in LINES if l['fold'] in (2,3,4)];TE=[l for l in LINES if l['fold'] in (0,1)]
base=fit_struct(TR);TRp=attach(TR,base);TEp=attach(TE,base)
MD=fit_mix(TRp,4,3.0);B=MD['B'];PI=MD['pi'];HP=MD['hp']

# Generic innovation metrics imported from the prior preregistered residual test.
INNO_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/69ba5470055a94f726fe797ca63f1cc27f410cc4/research/selector_innovation_residual_noed_refit_20261004.py"
iv={"__name__":"innovation_module"}
exec(compile(urllib.request.urlopen(INNO_URL,timeout=60).read().decode(),INNO_URL,"exec"),iv)
metric_vector=iv["metric_vector"];mahal_setup=iv["mahal_setup"];distance=iv["distance"]

def state_q(p,z):
    sc=np.log(np.maximum(p,1e-15))+B[z];sc-=sc.max();q=np.exp(sc);return q/q.sum()
def infer_post(l,prior,Y=None):
    P=l['P'];Y=l['Y'] if Y is None else np.asarray(Y,int);A=np.log(prior+1e-15)
    for z in range(4):
        q0=state_q(P[0],z);q1=state_q(P[1],z)
        A[z]+=math.log(max(q0[Y[0]],1e-300))+math.log(max(q1[Y[1]],1e-300))
    A-=A.max();w=np.exp(A);return w/w.sum()
def mixp(p,post):
    q=np.zeros(K)
    for z in range(4):q+=post[z]*state_q(p,z)
    return q

# Exact heldout line loss under each of the five possible opener priors, so permutation is cheap.
HN=['H1','H2','H3','H4','O'];hi={h:i for i,h in enumerate(HN)}
loss=np.zeros((len(TEp),len(HN)));base_loss=np.zeros(len(TEp));weights=np.zeros(len(TEp),int)
for j,l in enumerate(TEp):
    Y,P=l['Y'],l['P'];weights[j]=len(Y)-2
    base_loss[j]=sum(-math.log2(max(P[i,Y[i]],1e-300)) for i in range(2,len(Y)))
    for hidx,h in enumerate(HN):
        post=infer_post(l,HP[h])
        loss[j,hidx]=sum(-math.log2(max(mixp(P[i],post)[Y[i]],1e-300)) for i in range(2,len(Y)))
obs_h=np.array([hi[l['house']] for l in TEp],int)
obs_gain=(base_loss.sum()-loss[np.arange(len(TEp)),obs_h].sum())/weights.sum()
global_loss=0.
for j,l in enumerate(TEp):
    post=infer_post(l,PI);Y,P=l['Y'],l['P'];global_loss+=sum(-math.log2(max(mixp(P[i],post)[Y[i]],1e-300)) for i in range(2,len(Y)))
global_gain=(base_loss.sum()-global_loss)/weights.sum();obs_increment=(global_loss-loss[np.arange(len(TEp)),obs_h].sum())/weights.sum()

# Matched permutation of opener houses within section x scored-line-length bin.
def lbin(n):return '3-5' if n<=5 else ('6-9' if n<=9 else ('10-14' if n<=14 else '15+'))
groups=collections.defaultdict(list)
for i,l in enumerate(TEp):groups[(l['events'][0]['section'],lbin(len(l['Y'])))].append(i)
rng=np.random.default_rng(20261004);vals=[]
for b in range(2000):
    ph=obs_h.copy()
    for ix in groups.values():
        if len(ix)>1:
            a=ph[ix].copy();rng.shuffle(a);ph[ix]=a
    pm=loss[np.arange(len(TEp)),ph].sum()
    vals.append((global_loss-pm)/weights.sum())
vals=np.array(vals);perm={'obs_increment_bits':obs_increment,'null_mean':float(vals.mean()),'null_sd':float(vals.std(ddof=1)),'z':float((obs_increment-vals.mean())/vals.std(ddof=1)),'p_add':float((1+np.sum(vals>=obs_increment))/(len(vals)+1)),'B':len(vals)}

# Build actual innovation event lines after the first two ordinary tokens.
def actual_events(use_state=True):
    ev=[];ls=[]
    for l in TEp:
        Y,P=l['Y'],l['P'];prior=HP[l['house']]
        post=infer_post(l,prior)
        z=[]
        for i in range(2,len(Y)):
            pp=mixp(P[i],post) if use_state else P[i]
            e={'p':pp,'y':int(Y[i]),'prev':int(Y[i-1]),'half':int(l['fold']),'line':(l['folio'],l['line'])}
            ev.append(e);z.append(e)
        if z:ls.append(z)
    return ev,ls
ACT_LINE,ACT_LINESEQ=actual_events(True);ACT_BASE,ACT_BASESEQ=actual_events(False)
TV_LINE,NAMES=metric_vector(ACT_LINE,ACT_LINESEQ,True);TV_BASE,_=metric_vector(ACT_BASE,ACT_BASESEQ,True)

def sim_one(arg):
    seed,use_state=arg;rng=np.random.default_rng(seed);ev=[];ls=[]
    for l in TEp:
        P=l['P'];prior=HP[l['house']]
        if use_state:
            ztrue=int(rng.choice(4,p=prior));Y=[]
            for i in range(len(P)):Y.append(int(rng.choice(K,p=state_q(P[i],ztrue))))
            post=infer_post(l,prior,Y)
        else:
            Y=[int(rng.choice(K,p=P[i])) for i in range(len(P))];post=None
        qline=[]
        for i in range(2,len(P)):
            pp=mixp(P[i],post) if use_state else P[i]
            e={'p':pp,'y':Y[i],'prev':Y[i-1],'half':int(l['fold']),'line':(l['folio'],l['line'])};ev.append(e);qline.append(e)
        if qline:ls.append(qline)
    return metric_vector(ev,ls,True)[0].tolist()

def audit(use_state,target):
    with ProcessPoolExecutor(max_workers=min(32,os.cpu_count() or 8)) as ex:
        A=np.array(list(ex.map(sim_one,[(202610500000+i,use_state) for i in range(500)],chunksize=2)),float)
    mu,inv=mahal_setup(A);ds=np.array([distance(x,mu,inv) for x in A]);td=distance(target,mu,inv)
    return {'distance':float(td),'null_median':float(np.median(ds)),'null_q99':float(np.quantile(ds,.99)),'p_add':float((1+np.sum(ds>=td))/(len(ds)+1)),'target_metrics':{n:float(v) for n,v in zip(NAMES,target)}}

if __name__=="__main__":
    aud_base=audit(False,TV_BASE);aud_line=audit(True,TV_LINE)
    out={'predictive':{'base_bits':float(base_loss.sum()/weights.sum()),'global_gain_bits':float(global_gain),'house_gain_bits':float(obs_gain),'house_increment_over_global':float(obs_increment),'n_scored':int(weights.sum()),'lines':len(TEp)},'opener_permutation':perm,'innovation_same_subset':{'baseline':aud_base,'line_state':aud_line,'distance_reduction_fraction':float(1-aud_line['distance']/aud_base['distance'])},'model':{'K':4,'l2':3.0,'B':B.tolist(),'pi':PI.tolist(),'house_priors':{k:v.tolist() for k,v in HP.items()}}}
    print("LATENT_LINE_FOLLOWUP_JSON="+json.dumps(out,separators=(",",":")),flush=True)

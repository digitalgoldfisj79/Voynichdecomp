# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, collections, hashlib, json, math, statistics, sys
import numpy as np
sys.path.insert(0,'experiments/amadi_residuals_v1')
sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
sys.path.insert(0,'experiments/vbm_hmm_v2')
sys.path.insert(0,'experiments/vbm_amadi_homophone_v3')
sys.path.insert(0,'experiments/vbm_key_transfer_v6')
sys.path.insert(0,'experiments/vbm_crite_v7')
import vbm_key_transfer_v6 as v6
import vbm_crite_v7 as v7

NS='VBMEDGECOMP71'
v6.NS=NS;v6.b.NS=NS;v6.q3.NS=NS;v6.q3.b.NS=NS
v7.NS=NS;v7.v6.NS=NS;v7.v6.b.NS=NS
A=v6.A;K=v6.K
POS=['BAV_GLOBAL','GER_GLOBAL','BAV_GLOBAL_SWAP']
NEG=['BAV_FRESH','GER_FRESH','MARKOV1','MARKOV2','MARKOV3','SLOT5']
FAMS=POS+NEG


def seed(*xs):return int.from_bytes(hashlib.sha256('::'.join(map(str,xs)).encode()).digest()[:8],'big')&0x7fffffff

def flat(folios):return v6.flatten_folios(folios)

def counts(seqs):
    C=np.zeros((A,A),float);n=0
    for q in seqs:
        z=np.asarray(q,int)
        for x,y in zip(z,z[1:]):C[int(x),int(y)]+=1;n+=1
    return C,n

def make_mask(C,tag,frac=.20,min_count=3):
    mask=np.zeros((A,A),bool); blocks=[(0,K,0,K),(0,K,K,A),(K,A,0,K),(K,A,K,A)]
    block_sizes=[]
    for bi,(r0,r1,c0,c1) in enumerate(blocks):
        cells=[]
        for i in range(r0,r1):
            for j in range(c0,c1):
                if C[i,j]>=min_count:
                    h=hashlib.sha256(f'{NS}::{tag}::MASK::{bi}::{i}::{j}'.encode()).hexdigest();cells.append((h,i,j))
        cells.sort();m=max(1,int(math.floor(frac*len(cells)))) if cells else 0
        for _,i,j in cells[:m]:mask[i,j]=True
        block_sizes.append({'block':bi,'eligible':len(cells),'masked':m})
    if int(mask.sum())<10:raise RuntimeError(('mask too small',int(mask.sum()),block_sizes))
    return mask,block_sizes

def masked_moment_fit(C,mask,lm,tag,start,steps=700,lr=.05):
    total=max(1.,float(C.sum()));P=C/total;W=(~mask).astype(float)
    rng=np.random.default_rng(seed(NS,'MMF',tag,start));Z=np.zeros((v6.b.A,v6.b.NOBS),float)
    for s in v6.b.CIDX:Z[int(s),:K]=rng.normal(0,.35,K)
    for s in v6.b.VIDX:Z[int(s),K:]=rng.normal(0,.35,A-K)
    m=np.zeros_like(Z);vv=np.zeros_like(Z);b1=.9;b2=.999;eps=1e-8;J=(lm.pi[:,None]*lm.T);J/=J.sum();best=(1e99,None,None)
    for t in range(1,steps+1):
        E=v6.m.softmax_rows(Z);M=E.T@J@E;R=W*(M-P);loss=float(np.sum(R*R))
        if loss<best[0]:best=(loss,E.copy(),M.copy())
        G=J@E@R.T+J.T@E@R;GZ=np.zeros_like(Z)
        for s in v6.b.CIDX:
            ss=int(s);e=E[ss,:K];g=G[ss,:K];GZ[ss,:K]=e*(g-float(np.dot(g,e)))
        for s in v6.b.VIDX:
            ss=int(s);e=E[ss,K:];g=G[ss,K:];GZ[ss,K:]=e*(g-float(np.dot(g,e)))
        m=b1*m+(1-b1)*GZ;vv=b2*vv+(1-b2)*(GZ*GZ);mh=m/(1-b1**t);vh=vv/(1-b2**t);Z-=lr*mh/(np.sqrt(vh)+eps)
        if t%100==0 and t>=400:Z*=.999
    M=np.maximum(best[2],1e-15);M/=M.sum();return {'loss':float(best[0]),'E':best[1],'M':M}

def fit_latent(C,mask,lms,tag,steps):
    rows=[]
    for la in ['bavarian','german']:
        zs=[masked_moment_fit(C,mask,lms[la],f'{tag}:{la}',st,steps) for st in range(2)];z=min(zs,key=lambda q:q['loss']);rows.append((z['loss'],la,z))
    rows.sort(key=lambda x:(x[0],x[1]));loss,la,z=rows[0];return {'language':la,'loss':float(loss),'M':z['M'],'candidate_losses':{x[1]:float(x[0]) for x in rows}}

def independence(C,mask):
    X=C.copy();X[mask]=0.;r=X.sum(1)+.5;c=X.sum(0)+.5;M=np.outer(r,c);M=np.maximum(M,1e-15);M/=M.sum();return M

def als_fit(C,mask,rank,tag,iters=40,ridge=1e-7):
    P=C/max(1.,C.sum());W=(~mask)
    rng=np.random.default_rng(seed(NS,'ALS',tag,rank));scale=math.sqrt(max(float(P.mean()),1e-12)/max(1,rank));U=np.maximum(rng.normal(scale,scale*.2,(A,rank)),1e-8);V=np.maximum(rng.normal(scale,scale*.2,(A,rank)),1e-8);I=np.eye(rank)
    for _ in range(iters):
        for i in range(A):
            idx=np.flatnonzero(W[i]);X=V[idx];y=P[i,idx]
            if len(idx):
                try:u=np.linalg.solve(X.T@X+ridge*I,X.T@y)
                except np.linalg.LinAlgError:u=np.linalg.lstsq(X.T@X+ridge*I,X.T@y,rcond=None)[0]
                U[i]=np.maximum(u,0.)
        for j in range(A):
            idx=np.flatnonzero(W[:,j]);X=U[idx];y=P[idx,j]
            if len(idx):
                try:v=np.linalg.solve(X.T@X+ridge*I,X.T@y)
                except np.linalg.LinAlgError:v=np.linalg.lstsq(X.T@X+ridge*I,X.T@y,rcond=None)[0]
                V[j]=np.maximum(v,0.)
    M=np.maximum(U@V.T,1e-15);M/=M.sum();return M

def masked_score(M,H,mask):
    h=H[mask];n=float(h.sum())
    if n<=0:return float('-inf'),0
    p=np.maximum(M[mask],1e-15);return float(np.dot(h,np.log(p))/n),int(n)

def one_split(folios,fold,lms,tag,nfold=4,steps=700,als_iters=40):
    tr=[f for i,f in enumerate(folios) if i%nfold!=fold];ho=[f for i,f in enumerate(folios) if i%nfold==fold];C,nc=counts(flat(tr));H,nh=counts(flat(ho));mask,blocks=make_mask(C,f'{tag}:F{fold}')
    lat=fit_latent(C,mask,lms,f'{tag}:F{fold}:LAT',steps);ls,hev=masked_score(lat['M'],H,mask)
    bas={'INDEP':independence(C,mask)}
    for r in [8,19,32]:bas[f'ALS_R{r}']=als_fit(C,mask,r,f'{tag}:F{fold}',als_iters)
    scores={};masses={}
    for k,M in bas.items():scores[k]=masked_score(M,H,mask)[0];masses[k]=float(M[mask].sum())
    best=max(scores,key=scores.get);adv=float(ls-scores[best])
    return {'fold':fold,'selected_language':lat['language'],'candidate_losses':lat['candidate_losses'],'masked_train_loss':lat['loss'],'mask_cells':int(mask.sum()),'mask_blocks':blocks,'held_mask_events':hev,
            'latent_score':ls,'baseline_scores':scores,'best_baseline':best,'best_baseline_score':float(scores[best]),'EDGE_ADV':adv,
            'latent_mask_mass':float(lat['M'][mask].sum()),'baseline_mask_masses':masses,'train_pairs':int(nc),'hold_pairs':int(nh)}

def family_dataset(family,phase,rep,lms):return v7.family_dataset(family,phase,rep,lms)

def replicate(family,phase,rep,lms,smoke=False):
    folios=family_dataset(family,phase,rep,lms);nfold=2 if smoke else 4;steps=300 if smoke else 700;iters=20 if smoke else 40
    rows=[one_split(folios,k,lms,f'{phase}:{family}:R{rep}',nfold,steps,iters) for k in range(nfold)]
    return {'phase':phase,'family':family,'rep':rep,'median_EDGE_ADV':float(statistics.median(x['EDGE_ADV'] for x in rows)),'positive_folds':sum(x['EDGE_ADV']>0 for x in rows),
            'min_held_mask_events':min(x['held_mask_events'] for x in rows),'selected_languages':[x['selected_language'] for x in rows],'best_baselines':[x['best_baseline'] for x in rows],'folds':rows}

def brief(z):return {k:z[k] for k in ['phase','family','rep','median_EDGE_ADV','positive_folds','min_held_mask_events','selected_languages','best_baselines']}

def smoke(lms):
    rows=[]
    for fam in FAMS:
        z=replicate(fam,'SMOKE',0,lms,True);rows.append(z);print('V71CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
    return {'namespace':NS,'stage':'SMOKE','rows':rows}

def calibrate(lms):
    rows=[]
    for fam in FAMS:
        for r in range(3):
            z=replicate(fam,'CAL',r,lms,False);rows.append(z);print('V71CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
    p=[x for x in rows if x['family'] in POS];n=[x for x in rows if x['family'] in NEG];minp=min(x['median_EDGE_ADV'] for x in p);maxn=max(x['median_EDGE_ADV'] for x in n)
    if not minp>maxn:return {'namespace':NS,'stage':'CAL','pass':False,'reason':'nonseparable','min_positive_EDGE_ADV':minp,'max_negative_EDGE_ADV':maxn,'CAL':rows}
    tau=float((minp+maxn)/2);pp=sum(x['median_EDGE_ADV']>0 for x in p);passed=pp>=8
    return {'namespace':NS,'stage':'CAL','pass':passed,'reason':None if passed else 'positive_sign_gate','TAU_EDGE':tau,'min_positive_EDGE_ADV':minp,'max_negative_EDGE_ADV':maxn,'positive_above_zero':pp,'CAL':rows}

def validate(lms,tau):
    rows=[];pc=collections.Counter();seen=collections.Counter()
    for fam in POS:
        for r in range(3):
            z=replicate(fam,'VAL',r,lms,False);rows.append(z);print('V71CTRL',json.dumps(brief(z),sort_keys=True),flush=True);seen[fam]+=1
            if z['median_EDGE_ADV']>=tau:pc[fam]+=1
            if seen[fam]-pc[fam]>1:return {'namespace':NS,'stage':'VAL','pass':False,'reason':'positive_family_irrecoverable','TAU_EDGE':tau,'VAL':rows}
    if sum(pc.values())<8:return {'namespace':NS,'stage':'VAL','pass':False,'reason':'positive_total','TAU_EDGE':tau,'positive_pass':dict(pc),'VAL':rows}
    for fam in NEG:
        for r in range(3):
            z=replicate(fam,'VAL',r,lms,False);rows.append(z);print('V71CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
            if z['median_EDGE_ADV']>=tau:return {'namespace':NS,'stage':'VAL','pass':False,'reason':'negative_false_positive','TAU_EDGE':tau,'false_positive_family':fam,'VAL':rows}
    return {'namespace':NS,'stage':'VAL','pass':True,'TAU_EDGE':tau,'positive_pass':dict(pc),'VAL':rows}

def q0(lms):
    cal=calibrate(lms);print('CAL_RESULT',json.dumps({k:v for k,v in cal.items() if k!='CAL'},sort_keys=True),flush=True)
    if not cal['pass']:return cal
    val=validate(lms,cal['TAU_EDGE']);val['CAL']=cal['CAL'];return val

def fit_target(lms,tau):
    folios,labs,meta=v6.target_folios();folios,labs=v6.balanced_hash_order(folios,labs,6);rows=[]
    for k in range(6):
        z=one_split(folios,k,lms,'VOYNICH_FIT_V71',6,900,40);z['hold_folios']=[labs[i] for i in range(len(labs)) if i%6==k];rows.append(z);print('V71FIT',json.dumps(z,sort_keys=True),flush=True)
    ma=float(statistics.median(x['EDGE_ADV'] for x in rows));langs=collections.Counter(x['selected_language'] for x in rows);passed=bool(ma>=tau and sum(x['EDGE_ADV']>0 for x in rows)>=5 and min(x['held_mask_events'] for x in rows)>=100)
    return {'namespace':NS,'stage':'FIT','TAU_EDGE':tau,'median_EDGE_ADV':ma,'positive_folds':sum(x['EDGE_ADV']>0 for x in rows),'min_held_mask_events':min(x['held_mask_events'] for x in rows),'language_counts':dict(langs),'pass':passed,'confirmatory':False,'meta':meta,'folds':rows}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','q0','fit'],required=True);ap.add_argument('--tau-edge',type=float);a=ap.parse_args();lms=v6.b.load_lms()
    if a.mode=='smoke':out=smoke(lms)
    elif a.mode=='q0':out=q0(lms)
    else:
        if a.tau_edge is None:raise SystemExit('fit requires frozen --tau-edge')
        out=fit_target(lms,a.tau_edge)
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()

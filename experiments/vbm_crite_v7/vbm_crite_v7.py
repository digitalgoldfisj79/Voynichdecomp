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
sys.path.insert(0,'experiments/vbm_key_transfer_v61')
sys.path.insert(0,'experiments/vbm_discriminative_v4')
import vbm_key_transfer_v6 as v6
import vbm_discriminative_v4 as v4

NS='VBMCRITEV7'
v6.NS=NS; v6.b.NS=NS; v6.q3.NS=NS; v6.q3.b.NS=NS
v4.NS=NS; v4.b.NS=NS
A=v6.A; K=v6.K
POS=['BAV_GLOBAL','GER_GLOBAL','BAV_GLOBAL_SWAP']
NEG=['BAV_FRESH','GER_FRESH','MARKOV1','MARKOV2','MARKOV3','SLOT5']
FAMS=POS+NEG


def seed(*xs):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,xs)).encode()).digest()[:8],'big') & 0x7fffffff


def flat(folios): return v6.flatten_folios(folios)


def fit_fixed_latent(train,lm,tag,steps=500,em_iters=8):
    seqs=flat(train); cands=[]
    for st in range(2):
        z=v6.fit_moment(seqs,lm,f'{tag}:M',1,steps) if st==0 else v6.fit_moment(seqs,lm,f'{tag}:M2',1,steps)
        cands.append(z)
    z=min(cands,key=lambda q:q['loss']); E=np.asarray(z['E'],float).copy()
    obs,offs=v6.b.flatten(seqs); score=-1e99
    for _ in range(em_iters):
        ll,counts,_,n=v6.b._fb_counts(obs,offs,lm.T,lm.pi,E,v6.b.CIDX,v6.b.VIDX,False)
        E=v6.b._norm_emission(counts,0.05,v6.b.CIDX,v6.b.VIDX); score=float(ll/max(1,n))
    # score after final update for comparable train selection
    ll,_,_,n=v6.b._fb_counts(obs,offs,lm.T,lm.pi,E,v6.b.CIDX,v6.b.VIDX,False)
    score=float(ll/max(1,n))
    return {'E':E,'moment_loss':float(z['loss']),'train_score':score}


def latent_score(seqs,lm,E):
    return float(v6.b.eval_E(seqs,lm,E,None)['score'])


def fit_periodic(seqs,p):
    C=np.full((p,A),0.5,float)
    for q in seqs:
        for i,y in enumerate(q): C[i%p,int(y)]+=1
    C/=C.sum(1,keepdims=True); return C


def score_periodic(seqs,P):
    ll=0.;n=0;p=len(P)
    for q in seqs:
        for i,y in enumerate(q): ll+=math.log(max(float(P[i%p,int(y)]),1e-300));n+=1
    return ll/max(1,n)


def choose_surface(train,tag):
    a=train[::2];b=train[1::2]
    if not a or not b: a=train[:-1];b=train[-1:]
    sa=flat(a);sb=flat(b); cand=[]
    h=v4._fit_hier(sa,5,typed=False); ht=v4._fit_hier(sa,5,typed=True)
    for o in range(1,6):
        cand.append((v4._score_hier(h,sb,o,False),('hier',o,False)))
        cand.append((v4._score_hier(ht,sb,o,True),('hier',o,True)))
    for p in range(2,9):
        P=fit_periodic(sa,p);cand.append((score_periodic(sb,P),('slot',p,False)))
    cand.sort(key=lambda x:(-x[0],str(x[1]))); desc=cand[0][1]; allseq=flat(train)
    if desc[0]=='hier':
        typ=desc[2]; model=v4._fit_hier(allseq,desc[1],typed=typ)
    else:model=fit_periodic(allseq,desc[1])
    return {'desc':desc,'inner_score':float(cand[0][0]),'model':model}


def surface_score(seqs,s):
    d=s['desc']
    if d[0]=='hier':return float(v4._score_hier(s['model'],seqs,d[1],d[2]))
    return float(score_periodic(seqs,s['model']))


def euler_surrogate_seq(q,tag,r,fi,si):
    z=list(map(int,q));n=len(z)
    if n<3:return np.asarray(z,np.int32)
    outs=[[] for _ in range(A)]
    for x,y in zip(z,z[1:]):outs[x].append(y)
    rng=np.random.default_rng(seed(NS,tag,'EULER',r,fi,si))
    for xs in outs:
        if len(xs)>1:rng.shuffle(xs)
    idx=[0]*A; stack=[z[0]]; path=[]
    while stack:
        v=stack[-1]
        if idx[v]<len(outs[v]):
            y=outs[v][idx[v]];idx[v]+=1;stack.append(y)
        else:path.append(stack.pop())
    path=path[::-1]
    if len(path)!=n or path[0]!=z[0] or path[-1]!=z[-1]:
        raise RuntimeError(('Euler surrogate invalid',tag,r,fi,si,n,len(path),z[0],z[-1],path[:1],path[-1:]))
    return np.asarray(path,np.int32)


def euler_surrogate(folios,tag,r):
    return [[euler_surrogate_seq(q,tag,r,fi,si) for si,q in enumerate(f)] for fi,f in enumerate(folios)]


def fit_ngram_generator(source,order):
    seqs=flat(source); levels=[]
    base=np.full(A,0.25,float)
    for q in seqs:base+=np.bincount(np.asarray(q,int),minlength=A)
    base/=base.sum();levels.append(None)
    for o in range(1,order+1):
        D={}
        for q in seqs:
            z=list(map(int,q))
            for i in range(o,len(z)):
                ctx=tuple(z[i-o:i]);a=D.get(ctx)
                if a is None:a=np.full(A,0.05,float);D[ctx]=a
                a[z[i]]+=1
        for a in D.values():a/=a.sum()
        levels.append(D)
    return base,levels


def generate_ngram(source,template,order,tag):
    base,levels=fit_ngram_generator(source,order);rng=np.random.default_rng(seed(NS,tag,'GEN'));out=[]
    for fi,f in enumerate(template):
        ff=[]
        for si,q0 in enumerate(f):
            n=len(q0)
            if n<=0:continue
            z=[]
            for i in range(n):
                p=base
                for o in range(min(order,i),0,-1):
                    x=levels[o].get(tuple(z[i-o:i]))
                    if x is not None:p=x;break
                z.append(int(rng.choice(A,p=p)))
            ff.append(np.asarray(z,np.int32))
        out.append(ff)
    return out


def generate_slot(source,template,p,tag):
    P=fit_periodic(flat(source),p);rng=np.random.default_rng(seed(NS,tag,'SLOTGEN'));out=[]
    for fi,f in enumerate(template):
        ff=[]
        for si,q0 in enumerate(f):
            z=np.asarray([int(rng.choice(A,p=P[i%p])) for i in range(len(q0))],np.int32);ff.append(z)
        out.append(ff)
    return out


def family_dataset(family,phase,rep,lms):
    if family in {'BAV_GLOBAL','GER_GLOBAL','BAV_GLOBAL_SWAP','BAV_FRESH','GER_FRESH'}:
        return v6.dataset_family(family,phase,rep,lms)
    tag=f'{phase}:{family}:R{rep}'
    src=v6.dataset_family('BAV_GLOBAL',phase+':ADV',rep,lms)
    tmpl=v6.dataset_family('GER_FRESH',phase+':TMPL',rep,lms)
    if family=='MARKOV1':return generate_ngram(src,tmpl,1,tag)
    if family=='MARKOV2':return generate_ngram(src,tmpl,2,tag)
    if family=='MARKOV3':return generate_ngram(src,tmpl,3,tag)
    if family=='SLOT5':return generate_slot(src,tmpl,5,tag)
    raise ValueError(family)


def one_split(folios,fold,lms,tag,nfold=4,steps=500,perms=24):
    tr=[f for i,f in enumerate(folios) if i%nfold!=fold];ho=[f for i,f in enumerate(folios) if i%nfold==fold]
    trseq=flat(tr);hoseq=flat(ho);lat=[]
    for la in ['bavarian','german']:
        z=fit_fixed_latent(tr,lms[la],f'{tag}:F{fold}:{la}',steps,8);lat.append((z['train_score'],la,z))
    lat.sort(key=lambda x:(-x[0],x[1]));_,la,lz=lat[0];surf=choose_surface(tr,f'{tag}:F{fold}:SURF')
    lobs=latent_score(hoseq,lms[la],lz['E']);sobs=surface_score(hoseq,surf);gobs=lobs-sobs
    gs=[];ls=[];ss=[]
    for r in range(perms):
        hp=euler_surrogate(ho,f'{tag}:F{fold}',r);hs=flat(hp)
        lv=latent_score(hs,lms[la],lz['E']);sv=surface_score(hs,surf);ls.append(lv);ss.append(sv);gs.append(lv-sv)
    crite=float(gobs-statistics.median(gs));lex=float(lobs-statistics.median(ls));sex=float(sobs-statistics.median(ss))
    return {'fold':fold,'selected_language':la,'surface_model':list(surf['desc']),'surface_inner_score':surf['inner_score'],
            'latent_train_score':float(lz['train_score']),'latent_moment_loss':float(lz['moment_loss']),
            'latent_obs':float(lobs),'surface_obs':float(sobs),'PRED_ADV':float(gobs),'CRITE':crite,
            'latent_surrogate_excess':lex,'surface_surrogate_excess':sex,
            'surrogate_gap_median':float(statistics.median(gs)),'surrogate_gap_max':float(max(gs)),
            'hold_events':int(sum(len(q) for q in hoseq))}


def replicate(family,phase,rep,lms,smoke=False):
    folios=family_dataset(family,phase,rep,lms);nfold=2 if smoke else 4;steps=220 if smoke else 500;perms=6 if smoke else 24
    rows=[one_split(folios,k,lms,f'{phase}:{family}:R{rep}',nfold,steps,perms) for k in range(nfold)]
    return {'phase':phase,'family':family,'rep':rep,
            'median_CRITE':float(statistics.median(x['CRITE'] for x in rows)),
            'median_PRED_ADV':float(statistics.median(x['PRED_ADV'] for x in rows)),
            'positive_CRITE_folds':sum(x['CRITE']>0 for x in rows),
            'positive_PRED_ADV_folds':sum(x['PRED_ADV']>0 for x in rows),
            'selected_languages':[x['selected_language'] for x in rows],
            'surface_models':[x['surface_model'] for x in rows],'folds':rows}


def brief(z):return {k:z[k] for k in ['phase','family','rep','median_CRITE','median_PRED_ADV','positive_CRITE_folds','positive_PRED_ADV_folds','selected_languages','surface_models']}


def smoke(lms):
    rows=[]
    for fam in FAMS:
        z=replicate(fam,'SMOKE',0,lms,True);rows.append(z);print('V7CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
    return {'namespace':NS,'stage':'SMOKE','rows':rows}


def calibrate(lms):
    rows=[]
    for fam in FAMS:
        for r in range(3):
            z=replicate(fam,'CAL',r,lms,False);rows.append(z);print('V7CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
    p=[x for x in rows if x['family'] in POS];n=[x for x in rows if x['family'] in NEG]
    minp=min(x['median_CRITE'] for x in p);maxn=max(x['median_CRITE'] for x in n);sep=minp>maxn
    if not sep:return {'namespace':NS,'stage':'CAL','pass':False,'reason':'CRITE_nonseparable','min_positive_CRITE':minp,'max_negative_CRITE':maxn,'CAL':rows}
    tau=float((minp+maxn)/2);pp=sum(x['median_PRED_ADV']>0 for x in p);nj=sum(x['median_PRED_ADV']>0 and x['median_CRITE']>=tau for x in n)
    passed=bool(pp>=8 and nj<=1)
    return {'namespace':NS,'stage':'CAL','pass':passed,'reason':None if passed else 'predictive_gate',
            'TAU_CRITE':tau,'min_positive_CRITE':minp,'max_negative_CRITE':maxn,
            'positive_pred_adv':pp,'negative_joint_pred_crite':nj,'CAL':rows}


def validate(lms,tau):
    rows=[];pospass=collections.Counter();seen=collections.Counter();ppred=0
    for fam in POS:
        for r in range(3):
            z=replicate(fam,'VAL',r,lms,False);rows.append(z);print('V7CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
            seen[fam]+=1
            if z['median_CRITE']>=tau:pospass[fam]+=1
            if z['median_PRED_ADV']>0:ppred+=1
            # irrecoverable family >=2/3 or total >=8/9 checks
            if seen[fam]-pospass[fam]>1:
                return {'namespace':NS,'stage':'VAL','pass':False,'reason':'positive_family_irrecoverable','TAU_CRITE':tau,'VAL':rows}
    if sum(pospass.values())<8 or ppred<8:
        return {'namespace':NS,'stage':'VAL','pass':False,'reason':'positive_total_or_pred','TAU_CRITE':tau,'positive_pass':dict(pospass),'positive_pred_adv':ppred,'VAL':rows}
    for fam in NEG:
        for r in range(3):
            z=replicate(fam,'VAL',r,lms,False);rows.append(z);print('V7CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
            if z['median_CRITE']>=tau:
                return {'namespace':NS,'stage':'VAL','pass':False,'reason':'negative_false_positive','TAU_CRITE':tau,'false_positive_family':fam,'VAL':rows}
    return {'namespace':NS,'stage':'VAL','pass':True,'TAU_CRITE':tau,'positive_pass':dict(pospass),'positive_pred_adv':ppred,'VAL':rows}


def q0(lms):
    cal=calibrate(lms);print('CAL_RESULT',json.dumps({k:v for k,v in cal.items() if k!='CAL'},sort_keys=True),flush=True)
    if not cal['pass']:return cal
    val=validate(lms,cal['TAU_CRITE']);val['CAL']=cal['CAL'];return val


def fit_target(lms,tau):
    folios,labs,meta=v6.target_folios();folios,labs=v6.balanced_hash_order(folios,labs,6);rows=[]
    for k in range(6):
        z=one_split(folios,k,lms,'VOYNICH_FIT_V7',6,700,24);z['hold_folios']=[labs[i] for i in range(len(labs)) if i%6==k]
        rows.append(z);print('V7FIT',json.dumps(z,sort_keys=True),flush=True)
    mc=float(statistics.median(x['CRITE'] for x in rows));mp=float(statistics.median(x['PRED_ADV'] for x in rows));langs=collections.Counter(x['selected_language'] for x in rows)
    passed=bool(mc>=tau and sum(x['CRITE']>0 for x in rows)>=5 and mp>0 and sum(x['PRED_ADV']>0 for x in rows)>=5)
    return {'namespace':NS,'stage':'FIT','TAU_CRITE':tau,'median_CRITE':mc,'median_PRED_ADV':mp,
            'positive_CRITE_folds':sum(x['CRITE']>0 for x in rows),'positive_PRED_ADV_folds':sum(x['PRED_ADV']>0 for x in rows),
            'language_counts':dict(langs),'pass':passed,'confirmatory':False,'meta':meta,'folds':rows}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','q0','fit'],required=True);ap.add_argument('--tau-crite',type=float);a=ap.parse_args();lms=v6.b.load_lms()
    if a.mode=='smoke':out=smoke(lms)
    elif a.mode=='q0':out=q0(lms)
    else:
        if a.tau_crite is None:raise SystemExit('fit requires frozen --tau-crite')
        out=fit_target(lms,a.tau_crite)
    print('RESULT_JSON',json.dumps(out,sort_keys=True))

if __name__=='__main__':main()

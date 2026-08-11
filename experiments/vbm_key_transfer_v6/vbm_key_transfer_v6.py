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
import vbm_typed_v1 as tv
import vbm_hmm_moment_v2 as m
import vbm_amadi_q0_v3 as q3
b=m.b
NS='VBMKEYTRANSFERV6'; b.NS=NS; q3.NS=NS; q3.b.NS=NS
K=b.KCORE; A=b.NOBS
H1={'f28v','f31v','f88r','f5r','f34r','f81v'}
C1=['f85r1','f53v','f33r','f10r','f23r','f111r']
POS={'BAV_GLOBAL','GER_GLOBAL','BAV_GLOBAL_SWAP'}
NEG={'BAV_FRESH','GER_FRESH','STABLE_MARKOV'}

def seed(*xs):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,xs)).encode()).digest()[:8],'big') & 0x7fffffff

def flatten_folios(folios):
    return [q for f in folios for q in f]

def plain_folios(lm,tag,nfolio=12,chars=1800,seglen=120):
    need=nfolio*chars+seglen
    fit,_=b.plain_span(lm.control,f'V6:{tag}',need,seglen)
    s=''.join(fit)
    if len(s)<nfolio*chars: raise RuntimeError(('plain shortage',tag,len(s)))
    out=[];p=0
    for _ in range(nfolio):
        block=s[p:p+chars];p+=chars
        out.append([block[j:j+seglen] for j in range(0,len(block),seglen) if len(block[j:j+seglen])>=20])
    return out

def enc_one_folio(words,lm,keytag,emit_tag):
    p,u,ph,_=q3.make_key(lm,'FREQ_PROP','CYCLE',keytag)
    z,_=q3.encrypt_v3(words,p,u,ph,'CYCLE',emit_tag)
    return [np.asarray(x,np.int32) for x in z]

def encrypted_dataset(lm,tag,fresh=False,swap=False):
    plain=plain_folios(lm,tag)
    out=[]
    if not fresh:
        p,u,ph,_=q3.make_key(lm,'FREQ_PROP','CYCLE',f'{tag}:GLOBALKEY')
    for fi,words in enumerate(plain):
        if fresh:
            p,u,ph,_=q3.make_key(lm,'FREQ_PROP','CYCLE',f'{tag}:KEY:{fi}')
        z,_=q3.encrypt_v3(words,p,u,ph,'CYCLE',f'{tag}:EMIT:{fi}')
        ff=[np.asarray(x,np.int32).copy() for x in z]
        if swap:
            rng=np.random.default_rng(seed(NS,tag,'swap',fi))
            for q in ff:
                i=0
                while i+1<len(q):
                    if ((q[i]<K)==(q[i+1]<K)) and rng.random()<0.05:
                        q[i],q[i+1]=q[i+1],q[i];i+=2
                    else:i+=1
        out.append(ff)
    return out

def fit_surface_markov(folios):
    c=np.full(A,0.25,float);T=np.full((A,A),0.25,float)
    for q in flatten_folios(folios):
        z=np.asarray(q,int)
        if len(z): c+=np.bincount(z,minlength=A)
        for x,y in zip(z,z[1:]):T[int(x),int(y)]+=1
    p=c/c.sum();T/=T.sum(1,keepdims=True)
    return p,T

def generate_markov_like(source,tag,template):
    p,T=fit_surface_markov(source);rng=np.random.default_rng(seed(NS,tag,'markov'));out=[]
    for fi,f in enumerate(template):
        ff=[]
        for si,q0 in enumerate(f):
            n=len(q0)
            if n<=0:continue
            q=np.empty(n,np.int32);q[0]=int(rng.choice(A,p=p))
            for j in range(1,n):q[j]=int(rng.choice(A,p=T[q[j-1]]))
            ff.append(q)
        out.append(ff)
    return out

def dataset_family(family,phase,rep,lms):
    tag=f'{phase}:{family}:R{rep}'
    if family=='BAV_GLOBAL':return encrypted_dataset(lms['bavarian'],tag,False,False)
    if family=='GER_GLOBAL':return encrypted_dataset(lms['german'],tag,False,False)
    if family=='BAV_GLOBAL_SWAP':return encrypted_dataset(lms['bavarian'],tag,False,True)
    if family=='BAV_FRESH':return encrypted_dataset(lms['bavarian'],tag,True,False)
    if family=='GER_FRESH':return encrypted_dataset(lms['german'],tag,True,False)
    if family=='STABLE_MARKOV':
        src=encrypted_dataset(lms['bavarian'],tag+':SRC',False,False)
        tmpl=encrypted_dataset(lms['german'],tag+':TMPL',True,False)
        return generate_markov_like(src,tag,tmpl)
    raise ValueError(family)

def surface_bigram_matrix(seqs):
    C=np.zeros((A,A),float);n=0
    for q in seqs:
        z=np.asarray(q,int)
        for x,y in zip(z,z[1:]):C[int(x),int(y)]+=1;n+=1
    if n:C/=n
    return C,n

def fit_moment(seqs,lm,tag,starts=2,steps=500):
    P,_=surface_bigram_matrix(seqs);best=(1e99,None)
    for st in range(starts):
        E,loss=m.moment_fit(P,lm,tag,st,steps=steps,lr=0.05)
        if loss<best[0]:best=(float(loss),E)
    return {'loss':best[0],'E':best[1]}

def predicted_M(lm,E):
    J=(lm.pi[:,None]*lm.T);J/=J.sum();M=E.T@J@E;M=np.maximum(M,1e-12);M/=M.sum();return M

def score_M(seqs,M):
    ll=0.;n=0
    for q in seqs:
        z=np.asarray(q,int)
        for x,y in zip(z,z[1:]):ll+=math.log(float(M[int(x),int(y)]));n+=1
    return ll/max(1,n)

def permuted(folios,tag,r):
    out=[]
    for fi,f in enumerate(folios):
        rng=np.random.default_rng(seed(NS,tag,'perm',r,fi));pc=np.arange(K);pv=np.arange(K,A);rng.shuffle(pc);rng.shuffle(pv)
        mp=np.empty(A,np.int32);mp[:K]=pc;mp[K:]=pv
        out.append([mp[np.asarray(q,int)] for q in f])
    return out

def map_labels(lm,E):
    post=lm.pi[:,None]*E
    return np.argmax(post,axis=0).astype(np.int32)

def surf_freq(seqs):
    c=np.zeros(A,float)
    for q in seqs:c+=np.bincount(np.asarray(q,int),minlength=A)
    return c

def stability(train_folios,lm,tag,steps=350):
    a=train_folios[::2];bb=train_folios[1::2]
    if not a or not bb:return 0.0
    ea=fit_moment(flatten_folios(a),lm,tag+':SA',1,steps)['E'];eb=fit_moment(flatten_folios(bb),lm,tag+':SB',1,steps)['E']
    la=map_labels(lm,ea);lb=map_labels(lm,eb);w=surf_freq(flatten_folios(train_folios));return float(np.sum(w*(la==lb))/max(1.,w.sum()))

def one_split(folios,fold,lms,tag,nfold=4,steps=500,perms=24):
    tr=[f for i,f in enumerate(folios) if i%nfold!=fold];ho=[f for i,f in enumerate(folios) if i%nfold==fold]
    trseq=flatten_folios(tr);hoseq=flatten_folios(ho);cand=[]
    for la in ['bavarian','german']:
        z=fit_moment(trseq,lms[la],f'{tag}:F{fold}:{la}',2,steps);cand.append((z['loss'],la,z['E']))
    cand.sort(key=lambda x:(x[0],x[1]));loss,la,E=cand[0];M=predicted_M(lms[la],E);obs=score_M(hoseq,M);ps=[]
    for r in range(perms):ps.append(score_M(flatten_folios(permuted(ho,f'{tag}:F{fold}',r)),M))
    ite=float(obs-statistics.median(ps));stab=stability(tr,lms[la],f'{tag}:F{fold}:{la}',max(250,steps//2))
    return {'fold':fold,'selected_language':la,'train_moment_loss':float(loss),'hold_score':float(obs),'perm_median':float(statistics.median(ps)),'perm_max':float(max(ps)),'ITE':ite,'STAB':stab,'hold_events':int(sum(len(q) for q in hoseq))}

def replicate(family,phase,rep,lms,smoke=False):
    folios=dataset_family(family,phase,rep,lms);nfold=2 if smoke else 4;steps=220 if smoke else 500;perms=6 if smoke else 24
    rows=[one_split(folios,k,lms,f'{phase}:{family}:R{rep}',nfold,steps,perms) for k in range(nfold)]
    return {'phase':phase,'family':family,'rep':rep,'median_ITE':float(statistics.median(x['ITE'] for x in rows)),'median_STAB':float(statistics.median(x['STAB'] for x in rows)),'selected_languages':[x['selected_language'] for x in rows],'folds':rows}

def q0(lms,smoke=False):
    fams=['BAV_GLOBAL','GER_GLOBAL','BAV_GLOBAL_SWAP','BAV_FRESH','GER_FRESH','STABLE_MARKOV'];reps=1 if smoke else 3
    phases=['SMOKE'] if smoke else ['CAL','VAL'];allout={}
    for ph in phases:
        rows=[]
        for fam in fams:
            for r in range(reps):
                z=replicate(fam,ph,r,lms,smoke);rows.append(z);print('V6CTRL',json.dumps({k:z[k] for k in ['phase','family','rep','median_ITE','median_STAB','selected_languages']},sort_keys=True),flush=True)
        allout[ph]=rows
        if smoke:continue
        if ph=='CAL':
            pos=[x for x in rows if x['family'] in POS];neg=[x for x in rows if x['family'] in NEG];fresh=[x for x in rows if x['family'] in {'BAV_FRESH','GER_FRESH'}]
            sep_ite=min(x['median_ITE'] for x in pos)>max(x['median_ITE'] for x in neg);sep_stab=min(x['median_STAB'] for x in pos)>max(x['median_STAB'] for x in fresh)
            if not (sep_ite and sep_stab):
                return {'namespace':NS,'pass':False,'stage':'CAL','reason':'nonseparable','CAL':rows,'sep_ite':sep_ite,'sep_stab':sep_stab}
            tau_ite=(min(x['median_ITE'] for x in pos)+max(x['median_ITE'] for x in neg))/2;tau_stab=(min(x['median_STAB'] for x in pos)+max(x['median_STAB'] for x in fresh))/2
            allout['TAU_ITE']=float(tau_ite);allout['TAU_STAB']=float(tau_stab)
    if smoke:return {'namespace':NS,'smoke':True,'rows':allout['SMOKE']}
    tau_i=allout['TAU_ITE'];tau_s=allout['TAU_STAB'];val=allout['VAL']
    def passes(x):return x['median_ITE']>=tau_i and x['median_STAB']>=tau_s
    p=[x for x in val if x['family'] in POS];n=[x for x in val if x['family'] in NEG];by={f:sum(passes(x) for x in val if x['family']==f) for f in fams}
    passed=bool(sum(passes(x) for x in p)>=8 and sum(passes(x) for x in n)==0 and by['BAV_GLOBAL']>=2 and by['GER_GLOBAL']>=2 and by['BAV_GLOBAL_SWAP']>=2 and by['STABLE_MARKOV']==0)
    return {'namespace':NS,'pass':passed,'stage':'VAL','TAU_ITE':tau_i,'TAU_STAB':tau_s,'family_pass_counts':by,'CAL':allout['CAL'],'VAL':val}

def target_folios():
    lines,FIT,core,bridges,meta=tv.target_geometry()
    if len(core)!=21 or len(bridges)!=123:raise RuntimeError(('geometry',len(core),len(bridges)))
    if H1.intersection(FIT) or set(C1).intersection(FIT):raise RuntimeError(('sealed overlap',H1.intersection(FIT),set(C1).intersection(FIT)))
    out=[];labs=[]
    for f in FIT:
        z,_=tv.target_sequences(lines,[f],core,bridges)
        if z:out.append([np.asarray(q,np.int32) for q in z]);labs.append(f)
    return out,labs,{'FIT_total':len(FIT),'FIT_retained':len(out),'geometry':meta}

def balanced_hash_order(folios,labs,nfold=6):
    pairs=sorted(zip(labs,folios),key=lambda x:hashlib.sha256(f'V6FOLD::{x[0]}'.encode()).hexdigest())
    buckets=[[] for _ in range(nfold)];bl=[[] for _ in range(nfold)]
    for i,(lab,f) in enumerate(pairs):buckets[i%nfold].append(f);bl[i%nfold].append(lab)
    # interleave buckets so generic one_split modulo assignment reproduces these exact folds
    flat=[];flabs=[]
    mx=max(map(len,buckets))
    for j in range(mx):
        for k in range(nfold):
            if j<len(buckets[k]):flat.append(buckets[k][j]);flabs.append(bl[k][j])
    return flat,flabs

def fit_target(lms,tau_ite,tau_stab):
    folios,labs,meta=target_folios();folios,labs=balanced_hash_order(folios,labs,6);rows=[]
    for k in range(6):
        z=one_split(folios,k,lms,'VOYNICH_FIT',6,700,24);z['hold_folios']=[labs[i] for i in range(len(labs)) if i%6==k];rows.append(z);print('V6FIT',json.dumps(z,sort_keys=True),flush=True)
    med_i=float(statistics.median(x['ITE'] for x in rows));med_s=float(statistics.median(x['STAB'] for x in rows));langs=collections.Counter(x['selected_language'] for x in rows);plural,pc=langs.most_common(1)[0]
    passed=bool(med_i>=tau_ite and sum(x['ITE']>0 for x in rows)>=5 and med_s>=tau_stab and sum(x['STAB']>=tau_stab for x in rows)>=5 and pc>=4)
    return {'namespace':NS,'stage':'FIT','TAU_ITE':tau_ite,'TAU_STAB':tau_stab,'median_ITE':med_i,'median_STAB':med_s,'positive_ITE_folds':sum(x['ITE']>0 for x in rows),'stability_pass_folds':sum(x['STAB']>=tau_stab for x in rows),'language_counts':dict(langs),'plurality_language':plural,'plurality_count':pc,'pass':passed,'C1_opened':False,'meta':meta,'folds':rows}

def c1_score(lms,tau_ite,plurality):
    lines,FIT,core,bridges,meta=tv.target_geometry();train,_=tv.target_sequences(lines,FIT,core,bridges);hold,_=tv.target_sequences(lines,C1,core,bridges);cands=[]
    for la in ['bavarian','german']:
        z=fit_moment(train,lms[la],f'C1FINAL:{la}',3,850);cands.append((z['loss'],la,z['E']))
    cands.sort(key=lambda x:(x[0],x[1]));loss,la,E=cands[0];M=predicted_M(lms[la],E);obs=score_M(hold,M)
    # group C1 folios for topology-preserving independent per-folio permutations
    hf=[]
    for f in C1:
        z,_=tv.target_sequences(lines,[f],core,bridges);hf.append([np.asarray(q,np.int32) for q in z])
    ps=[score_M(flatten_folios(permuted(hf,'C1FINAL',r)),M) for r in range(64)];ite=float(obs-statistics.median(ps));passed=bool(ite>=tau_ite and obs>max(ps) and la==plurality)
    return {'namespace':NS,'stage':'C1','TAU_ITE':tau_ite,'selected_language':la,'FIT_plurality_language':plurality,'train_moment_loss':float(loss),'C1_score':float(obs),'perm_median':float(statistics.median(ps)),'perm_max':float(max(ps)),'ITE':ite,'rank_vs_permutations':1+sum(x>=obs for x in ps),'pass':passed,'C1_opened':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','q0','fit','c1'],required=True);ap.add_argument('--tau-ite',type=float);ap.add_argument('--tau-stab',type=float);ap.add_argument('--plurality',choices=['bavarian','german']);a=ap.parse_args();lms=b.load_lms()
    if a.mode=='smoke':out=q0(lms,True)
    elif a.mode=='q0':out=q0(lms,False)
    elif a.mode=='fit':
        if a.tau_ite is None or a.tau_stab is None:raise SystemExit('fit requires frozen thresholds')
        out=fit_target(lms,a.tau_ite,a.tau_stab)
    else:
        if a.tau_ite is None or a.plurality is None:raise SystemExit('c1 requires frozen tau-ite and FIT plurality')
        out=c1_score(lms,a.tau_ite,a.plurality)
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()

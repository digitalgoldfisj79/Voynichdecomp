# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, collections, json, math, statistics, sys
import numpy as np
sys.path.insert(0,'experiments/amadi_residuals_v1')
sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
sys.path.insert(0,'experiments/vbm_hmm_v2')
sys.path.insert(0,'experiments/vbm_amadi_homophone_v3')
sys.path.insert(0,'experiments/vbm_key_transfer_v6')
import vbm_key_transfer_v6 as v6

NS='VBMKEYTRANSFERV61'
v6.NS=NS
v6.b.NS=NS
v6.q3.NS=NS
v6.q3.b.NS=NS
POS=v6.POS
NEG=v6.NEG
FAMS=['BAV_GLOBAL','GER_GLOBAL','BAV_GLOBAL_SWAP','BAV_FRESH','GER_FRESH','STABLE_MARKOV']


def fro_cos(A,B):
    a=np.asarray(A,float).ravel();b=np.asarray(B,float).ravel()
    da=float(np.linalg.norm(a));db=float(np.linalg.norm(b))
    if da<=0 or db<=0:return 0.0
    return float(np.dot(a,b)/(da*db))


def emission_kernel(lm,E):
    K=E.T@(lm.pi[:,None]*E)
    K=np.maximum(K,0.0)
    n=float(np.linalg.norm(K))
    return K/max(n,1e-300)


def geometry_stability(train_folios,lm,tag,steps=350):
    a=train_folios[::2];c=train_folios[1::2]
    if not a or not c:return {'EKS':0.0,'PMS':0.0}
    za=v6.fit_moment(v6.flatten_folios(a),lm,tag+':GA',1,steps)
    zb=v6.fit_moment(v6.flatten_folios(c),lm,tag+':GB',1,steps)
    Ka=emission_kernel(lm,za['E']);Kb=emission_kernel(lm,zb['E'])
    Ma=v6.predicted_M(lm,za['E']);Mb=v6.predicted_M(lm,zb['E'])
    return {'EKS':fro_cos(Ka,Kb),'PMS':fro_cos(Ma,Mb),'half_loss_A':float(za['loss']),'half_loss_B':float(zb['loss'])}


def one_split(folios,fold,lms,tag,nfold=4,steps=500,perms=24):
    tr=[f for i,f in enumerate(folios) if i%nfold!=fold]
    ho=[f for i,f in enumerate(folios) if i%nfold==fold]
    trseq=v6.flatten_folios(tr);hoseq=v6.flatten_folios(ho);cand=[]
    for la in ['bavarian','german']:
        z=v6.fit_moment(trseq,lms[la],f'{tag}:F{fold}:{la}',2,steps)
        cand.append((z['loss'],la,z['E']))
    cand.sort(key=lambda x:(x[0],x[1]));loss,la,E=cand[0]
    M=v6.predicted_M(lms[la],E);obs=v6.score_M(hoseq,M)
    ps=[]
    for r in range(perms):
        hp=v6.permuted(ho,f'{tag}:F{fold}',r)
        ps.append(v6.score_M(v6.flatten_folios(hp),M))
    ite=float(obs-statistics.median(ps))
    gs=geometry_stability(tr,lms[la],f'{tag}:F{fold}:{la}',max(250,steps//2))
    return {'fold':fold,'selected_language':la,'train_moment_loss':float(loss),'hold_score':float(obs),
            'perm_median':float(statistics.median(ps)),'perm_max':float(max(ps)),'ITE':ite,
            'EKS':float(gs['EKS']),'PMS_diag':float(gs['PMS']),
            'half_loss_A':gs.get('half_loss_A'),'half_loss_B':gs.get('half_loss_B'),
            'hold_events':int(sum(len(q) for q in hoseq))}


def replicate(family,phase,rep,lms,smoke=False):
    folios=v6.dataset_family(family,phase,rep,lms)
    nfold=2 if smoke else 4;steps=220 if smoke else 500;perms=6 if smoke else 24
    rows=[one_split(folios,k,lms,f'{phase}:{family}:R{rep}',nfold,steps,perms) for k in range(nfold)]
    return {'phase':phase,'family':family,'rep':rep,
            'median_ITE':float(statistics.median(x['ITE'] for x in rows)),
            'median_EKS':float(statistics.median(x['EKS'] for x in rows)),
            'median_PMS_diag':float(statistics.median(x['PMS_diag'] for x in rows)),
            'selected_languages':[x['selected_language'] for x in rows],'folds':rows}


def brief(z):
    return {k:z[k] for k in ['phase','family','rep','median_ITE','median_EKS','median_PMS_diag','selected_languages']}


def smoke(lms):
    rows=[]
    for fam in FAMS:
        z=replicate(fam,'SMOKE',0,lms,True);rows.append(z);print('V61CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
    return {'namespace':NS,'stage':'SMOKE','rows':rows}


def calibrate(lms):
    cal=[]
    for fam in FAMS:
        for r in range(4):
            z=replicate(fam,'CAL',r,lms,False);cal.append(z);print('V61CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
    pos=[x for x in cal if x['family'] in POS];neg=[x for x in cal if x['family'] in NEG]
    fresh=[x for x in cal if x['family'] in {'BAV_FRESH','GER_FRESH'}]
    minpi=min(x['median_ITE'] for x in pos);maxni=max(x['median_ITE'] for x in neg)
    minpe=min(x['median_EKS'] for x in pos);maxfe=max(x['median_EKS'] for x in fresh)
    sep_i=minpi>maxni;sep_e=minpe>maxfe
    if not (sep_i and sep_e):
        return {'namespace':NS,'pass':False,'stage':'CAL','reason':'nonseparable','sep_ite':sep_i,'sep_eks':sep_e,
                'min_positive_ITE':minpi,'max_negative_ITE':maxni,'min_positive_EKS':minpe,'max_fresh_EKS':maxfe,'CAL':cal}
    ti=float((minpi+maxni)/2);te=float((minpe+maxfe)/2)
    return {'namespace':NS,'pass':True,'stage':'CAL','TAU_ITE':ti,'TAU_EKS':te,
            'min_positive_ITE':minpi,'max_negative_ITE':maxni,'min_positive_EKS':minpe,'max_fresh_EKS':maxfe,'CAL':cal}


def validation(lms,tau_i,tau_e):
    rows=[];pos_total=pos_pass=0;fam_pos_seen=collections.Counter();fam_pos_pass=collections.Counter()
    # positives first to enable irrecoverable early stop
    for fam in ['BAV_GLOBAL','GER_GLOBAL','BAV_GLOBAL_SWAP']:
        for r in range(4):
            z=replicate(fam,'VAL',r,lms,False);rows.append(z);print('V61CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
            ok=z['median_ITE']>=tau_i and z['median_EKS']>=tau_e
            pos_total+=1;fam_pos_seen[fam]+=1
            if ok:pos_pass+=1;fam_pos_pass[fam]+=1
            fails=pos_total-pos_pass
            if fails>1 or (fam_pos_seen[fam]-fam_pos_pass[fam])>1:
                return {'namespace':NS,'pass':False,'stage':'VAL','reason':'positive_gate_irrecoverable','TAU_ITE':tau_i,'TAU_EKS':tau_e,'VAL':rows,
                        'positive_seen':pos_total,'positive_pass':pos_pass,'family_positive_pass':dict(fam_pos_pass)}
    # negatives: a single pass is binding failure
    neg_pass=collections.Counter()
    for fam in ['BAV_FRESH','GER_FRESH','STABLE_MARKOV']:
        for r in range(4):
            z=replicate(fam,'VAL',r,lms,False);rows.append(z);print('V61CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
            ok=z['median_ITE']>=tau_i and z['median_EKS']>=tau_e
            if ok:
                neg_pass[fam]+=1
                return {'namespace':NS,'pass':False,'stage':'VAL','reason':'negative_false_positive','TAU_ITE':tau_i,'TAU_EKS':tau_e,'VAL':rows,
                        'positive_pass':pos_pass,'negative_pass':dict(neg_pass)}
    return {'namespace':NS,'pass':True,'stage':'VAL','TAU_ITE':tau_i,'TAU_EKS':tau_e,'VAL':rows,
            'positive_pass':pos_pass,'family_positive_pass':dict(fam_pos_pass),'negative_pass':dict(neg_pass)}


def q0(lms):
    cal=calibrate(lms)
    print('CAL_RESULT',json.dumps({k:v for k,v in cal.items() if k!='CAL'},sort_keys=True),flush=True)
    if not cal['pass']:return cal
    val=validation(lms,cal['TAU_ITE'],cal['TAU_EKS'])
    val['CAL']=cal['CAL'];return val


def fit_target(lms,tau_i,tau_e):
    folios,labs,meta=v6.target_folios();folios,labs=v6.balanced_hash_order(folios,labs,6);rows=[]
    for k in range(6):
        z=one_split(folios,k,lms,'VOYNICH_FIT_V61',6,700,24)
        z['hold_folios']=[labs[i] for i in range(len(labs)) if i%6==k]
        rows.append(z);print('V61FIT',json.dumps(z,sort_keys=True),flush=True)
    mi=float(statistics.median(x['ITE'] for x in rows));me=float(statistics.median(x['EKS'] for x in rows))
    langs=collections.Counter(x['selected_language'] for x in rows);plural,pc=langs.most_common(1)[0]
    passed=bool(mi>=tau_i and sum(x['ITE']>0 for x in rows)>=5 and me>=tau_e and sum(x['EKS']>=tau_e for x in rows)>=5 and pc>=4)
    return {'namespace':NS,'stage':'FIT','TAU_ITE':tau_i,'TAU_EKS':tau_e,'median_ITE':mi,'median_EKS':me,
            'positive_ITE_folds':sum(x['ITE']>0 for x in rows),'eks_pass_folds':sum(x['EKS']>=tau_e for x in rows),
            'language_counts':dict(langs),'plurality_language':plural,'plurality_count':pc,'pass':passed,'C1_opened':False,'meta':meta,'folds':rows}


def c1_score(lms,tau_i,plurality):
    # inherited v6 final C1 test; only ITE is a held-out C1 endpoint
    return v6.c1_score(lms,tau_i,plurality)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','q0','fit','c1'],required=True)
    ap.add_argument('--tau-ite',type=float);ap.add_argument('--tau-eks',type=float);ap.add_argument('--plurality',choices=['bavarian','german'])
    a=ap.parse_args();lms=v6.b.load_lms()
    if a.mode=='smoke':out=smoke(lms)
    elif a.mode=='q0':out=q0(lms)
    elif a.mode=='fit':
        if a.tau_ite is None or a.tau_eks is None:raise SystemExit('fit requires frozen thresholds')
        out=fit_target(lms,a.tau_ite,a.tau_eks)
    else:
        if a.tau_ite is None or a.plurality is None:raise SystemExit('c1 requires tau-ite and FIT plurality')
        out=c1_score(lms,a.tau_ite,a.plurality)
    print('RESULT_JSON',json.dumps(out,sort_keys=True))

if __name__=='__main__':main()

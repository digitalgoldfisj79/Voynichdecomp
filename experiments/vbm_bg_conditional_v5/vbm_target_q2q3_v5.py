# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, json, sys
import numpy as np
sys.path.insert(0,'experiments/amadi_residuals_v1')
sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
sys.path.insert(0,'experiments/vbm_hmm_v2')
sys.path.insert(0,'experiments/vbm_discriminative_v4')
sys.path.insert(0,'experiments/vbm_bg_conditional_v5')
import amadi_residuals_v1 as ar
import vbm_structure_v1 as s0
import vbm_typed_v1 as tv
import vbm_hmm_moment_v2 as m
import vbm_discriminative_v4 as v4
import vbm_bg_topology_q0_v5 as q0

TAU_LANG=0.21806702545882306
TAU_BG=1.6272712366587183
H1=['f28v','f31v','f88r','f5r','f34r','f81v']
C1=['f85r1','f53v','f33r','f10r','f23r','f111r']
Q0_W=np.array([0.5220402096578485,0.6444040005641153,0.6809244639394224,0.6973198040052558,0.6909626888732108,0.6849660157263239,0.6784615241587469,0.6850418160535259,0.6120726206433902],float)
Q0_MU=np.array([0.0021992225481279213,0.0026442549261169227,0.0029570652954769507,0.0036001649550905456,0.003546809989170084,0.0036791754363502157,0.002649801454890856,0.011197155744838547],float)
Q0_SD=np.array([0.013012950008421538,0.020426136185307024,0.021242313315687354,0.022974652549234677,0.023728639439757587,0.023891903593312894,0.025919652755627083,0.04885536340269976],float)

def frozen_topology_model():
    corp=q0.load_discovery();train={la:corp[la][0] for la in corp};models=q0.fit_feature_models(train)
    # Rebuild classifier only to verify deterministic Q0 freeze; use frozen numbers for target scoring.
    cal={}
    for la in ['bavarian','german']:
        a,_=q0.split_controls(corp[la][1],la);cal[la]=q0.windows(a,1800,'cal:'+la,48)
    nfit=min(len(cal['bavarian']),len(cal['german']),48);X=[];y=[]
    for s in cal['bavarian'][:nfit]:X.append(q0.feat(s,models));y.append(1)
    for s in cal['german'][:nfit]:X.append(q0.feat(s,models));y.append(0)
    chk=q0.ridge_logistic(X,y,1.0)
    err=max(float(np.max(np.abs(chk['w']-Q0_W))),float(np.max(np.abs(chk['mu']-Q0_MU))),float(np.max(np.abs(chk['sd']-Q0_SD))))
    if err>1e-8:raise RuntimeError(('Q0 freeze reproduction failed',err))
    return models,{'mu':Q0_MU,'sd':Q0_SD,'w':Q0_W},err

def topo_logit(cvseq,models,clf):
    x=q0.feat(cvseq,models);z=(x-clf['mu'])/clf['sd'];return float(clf['w'][0]+np.dot(z,clf['w'][1:]))

def geometry_and_surface(folios):
    lines,FIT,core,bridges,meta=tv.target_geometry()
    if len(core)!=21 or len(bridges)!=123:raise RuntimeError(('frozen surface geometry mismatch',len(core),len(bridges)))
    fit,fitmeta=tv.target_sequences(lines,FIT,core,bridges)
    hold,holdmeta=tv.target_sequences(lines,folios,core,bridges)
    return lines,FIT,core,bridges,fit,hold,{'geometry':meta,'fit':fitmeta,'hold':holdmeta}

def hmm_score(fit,hold):
    lms=m.b.load_lms();null=v4.best_null(fit,hold);cand=[]
    # Same fit tags in H1 and C1 modes => identical FIT-only initialisation/training path.
    for la in ['bavarian','german']:
        r=m.paired_fit_moment(fit,hold,lms[la],f'VBMBGCONDV5:TARGETFIT:{la}',None,40)
        cand.append({'language':la,'score':float(r['score']),'score_A':float(r['A_eval']['score']),'score_B':float(r['B_eval']['score']),'score_gap':float(r['score_gap']),'decode_agreement':float(r['decode_agreement']),'fit_converged':bool(r['converged'])})
    win=max(cand,key=lambda x:x['score']);delta=float(win['score']-null['score'])
    return {'DELTA_LANG':delta,'TAU_LANG':TAU_LANG,'language_gate':bool(delta>=TAU_LANG),'winner':win['language'],'winner_score':win['score'],'null_score':float(null['score']),'null_model':null['model'],'candidates':cand}

def topology_score(lines,folios,models,clf):
    seqs,meta=s0.vbm_types(lines,folios);agg=''.join(seqs);agglog=topo_logit(agg,models,clf)
    per={}
    for f in folios:
        fs,fm=s0.vbm_types(lines,[f]);q=''.join(fs);per[f]={'logit':topo_logit(q,models,clf) if q else None,'events':fm['events']}
    vals=[x['logit'] for x in per.values() if x['logit'] is not None];pos=sum(x>0 for x in vals);med=float(np.median(vals)) if vals else float('-inf')
    return {'aggregate_logit':agglog,'TAU_BG':TAU_BG,'aggregate_gate':bool(agglog>=TAU_BG),'positive_folios':pos,'folio_total':len(vals),'median_folio_logit':med,'folio_gate':bool(pos>=5 and med>0),'per_folio':per,'geometry':meta}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--set',choices=['h1','c1'],required=True);a=ap.parse_args();folios=H1 if a.set=='h1' else C1
    lines,FIT,core,bridges,fit,hold,smeta=geometry_and_surface(folios)
    models,clf,freeze_err=frozen_topology_model();top=topology_score(lines,folios,models,clf);hmm=hmm_score(fit,hold)
    passed=bool(hmm['language_gate'] and top['aggregate_gate'] and top['folio_gate'])
    out={'namespace':'VBMBGCONDV5','set':a.set.upper(),'folios':folios,'FIT_folios':len(FIT),'surface':smeta,'topology':top,'hmm':hmm,'Q0_freeze_max_abs_error':freeze_err,'pass':passed,'C1_opened':bool(a.set=='c1')}
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()

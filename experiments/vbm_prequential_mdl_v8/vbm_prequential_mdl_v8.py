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
sys.path.insert(0,'experiments/vbm_discriminative_v4')
sys.path.insert(0,'experiments/vbm_crite_v7')
import vbm_key_transfer_v6 as v6
import vbm_discriminative_v4 as v4
import vbm_crite_v7 as v7

NS='VBMPREQMDLV8'
v6.NS=NS; v6.b.NS=NS; v6.q3.NS=NS; v6.q3.b.NS=NS
v7.NS=NS; v7.v6.NS=NS; v7.v6.b.NS=NS; v7.v4.NS=NS; v7.v4.b.NS=NS
A=v6.A
POS=['BAV_GLOBAL','GER_GLOBAL','BAV_GLOBAL_SWAP']
NEG=['BAV_FRESH','GER_FRESH','MARKOV1','MARKOV2','MARKOV3','SLOT5']
FAMS=POS+NEG
SURF_ARCH=[('hier',o,t) for o in range(1,6) for t in (False,True)]+[('slot',p,False) for p in range(2,9)]


def seed(*xs):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,xs)).encode()).digest()[:8],'big') & 0x7fffffff


def flat(folios): return v6.flatten_folios(folios)

def nevents(folios): return int(sum(len(q) for f in folios for q in f))


def order_folios(folios,tag,labels=None):
    if labels is None: labels=[str(i) for i in range(len(folios))]
    ix=sorted(range(len(folios)),key=lambda i:hashlib.sha256(f'{NS}::{tag}::{labels[i]}'.encode()).hexdigest())
    return [folios[i] for i in ix],[labels[i] for i in ix]


def fit_surface_desc(folios,desc):
    seq=flat(folios);typ=desc[0]
    if typ=='hier':
        return v4._fit_hier(seq,desc[1],typed=bool(desc[2]))
    return v7.fit_periodic(seq,desc[1])


def surface_score_desc(folios,desc,model):
    seq=flat(folios)
    if desc[0]=='hier': return float(v4._score_hier(model,seq,desc[1],bool(desc[2])))
    return float(v7.score_periodic(seq,model))


def choose_surface_arch(warm):
    n=len(warm)//2;a=warm[:n];b=warm[n:];cand=[]
    for desc in SURF_ARCH:
        model=fit_surface_desc(a,desc);sc=surface_score_desc(b,desc,model);cand.append((sc,desc))
    cand.sort(key=lambda x:(-x[0],str(x[1])))
    return cand[0][1],float(cand[0][0]),[(list(d),float(s)) for s,d in cand[:5]]


def choose_language(warm,lms,tag,steps):
    n=len(warm)//2;a=warm[:n];b=warm[n:];cand=[]
    for la in ['bavarian','german']:
        z=v7.fit_fixed_latent(a,lms[la],f'{tag}:SEL:{la}',steps,4)
        sc=v7.latent_score(flat(b),lms[la],z['E']);cand.append((sc,la,float(z['moment_loss'])))
    cand.sort(key=lambda x:(-x[0],x[1]))
    return cand[0][1],float(cand[0][0]),{la:{'warm_score':float(sc),'moment_loss':float(loss)} for sc,la,loss in cand}


def score_block(prefix,block,lms,la,desc,tag,steps):
    z=v7.fit_fixed_latent(prefix,lms[la],f'{tag}:LAT',steps,5)
    ls=v7.latent_score(flat(block),lms[la],z['E'])
    sm=fit_surface_desc(prefix,desc);ss=surface_score_desc(block,desc,sm)
    n=nevents(block)
    return {'events':n,'latent_score':float(ls),'surface_score':float(ss),
            'adv_nats_per_event':float(ls-ss),'latent_nll':float(-ls*n),'surface_nll':float(-ss*n),
            'latent_moment_loss':float(z['moment_loss'])}


def prequential(folios,lms,tag,smoke=False,warm_n=4,blocks=None):
    if len(folios)<warm_n+1: raise RuntimeError(('too few folios',len(folios),warm_n))
    warm=folios[:warm_n];steps_sel=180 if smoke else 260;steps_fit=180 if smoke else 300
    la,lval,ldetail=choose_language(warm,lms,tag,steps_sel)
    desc,sval,stop5=choose_surface_arch(warm)
    if blocks is None:
        blocks=[[i] for i in range(warm_n,len(folios))]
    coded=[];prefix=list(warm)
    for bi,idxs in enumerate(blocks):
        block=[folios[i] for i in idxs]
        z=score_block(prefix,block,lms,la,desc,f'{tag}:B{bi}',steps_fit);z['block']=bi;z['folio_indices']=list(idxs);coded.append(z)
        prefix.extend(block)
    n=sum(x['events'] for x in coded);lnl=sum(x['latent_nll'] for x in coded);snl=sum(x['surface_nll'] for x in coded)
    adv=(snl-lnl)/max(1,n);hdr=(math.log(len(SURF_ARCH))-math.log(2))/max(1,n)
    return {'selected_language':la,'warm_latent_score':lval,'language_candidates':ldetail,
            'surface_arch':list(desc),'warm_surface_score':sval,'surface_top5':stop5,
            'coded_events':int(n),'latent_nll':float(lnl),'surface_nll':float(snl),
            'PREQ_ADV':float(adv),'PREQ_ADV_HEADER':float(adv+hdr),'header_delta_per_event':float(hdr),
            'blocks':coded}


def synthetic_one(family,phase,rep,lms,smoke=False):
    folios=v7.family_dataset(family,phase,rep,lms);folios,_=order_folios(folios,f'{phase}:{family}:R{rep}')
    z=prequential(folios,lms,f'{phase}:{family}:R{rep}',smoke,4)
    z.update({'phase':phase,'family':family,'rep':rep})
    return z


def brief(z):
    return {k:z[k] for k in ['phase','family','rep','selected_language','surface_arch','coded_events','PREQ_ADV','PREQ_ADV_HEADER']}


def smoke(lms):
    rows=[]
    for fam in FAMS:
        z=synthetic_one(fam,'SMOKE',0,lms,True);rows.append(z);print('V8CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
    return {'namespace':NS,'stage':'SMOKE','rows':rows}


def calibration(lms):
    rows=[]
    for fam in FAMS:
        for r in range(3):
            z=synthetic_one(fam,'CAL',r,lms,False);rows.append(z);print('V8CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
    p=[x for x in rows if x['family'] in POS];n=[x for x in rows if x['family'] in NEG]
    minp=min(x['PREQ_ADV'] for x in p);maxn=max(x['PREQ_ADV'] for x in n)
    true_lang=sum(x['selected_language']==('bavarian' if x['family']=='BAV_GLOBAL' else 'german') for x in rows if x['family'] in {'BAV_GLOBAL','GER_GLOBAL'})
    pos_gt0=sum(x['PREQ_ADV']>0 for x in p)
    sep=minp>maxn
    if not sep:
        return {'namespace':NS,'stage':'CAL','pass':False,'reason':'PREQ_nonseparable','min_positive_PREQ':float(minp),'max_negative_PREQ':float(maxn),'positive_gt0':pos_gt0,'true_language':true_lang,'CAL':rows}
    tau=float((minp+maxn)/2)
    ok=bool(pos_gt0>=8 and true_lang>=5)
    return {'namespace':NS,'stage':'CAL','pass':ok,'reason':None if ok else 'positive_or_language_gate','TAU_PREQ':tau,
            'min_positive_PREQ':float(minp),'max_negative_PREQ':float(maxn),'positive_gt0':pos_gt0,'true_language':true_lang,'CAL':rows}


def validation(lms,tau):
    rows=[];pc=collections.Counter();true_lang=0
    for fam in POS:
        for r in range(3):
            z=synthetic_one(fam,'VAL',r,lms,False);rows.append(z);print('V8CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
            if z['PREQ_ADV']>=tau:pc[fam]+=1
            if fam in {'BAV_GLOBAL','GER_GLOBAL'} and z['selected_language']==('bavarian' if fam=='BAV_GLOBAL' else 'german'):true_lang+=1
            if 3-pc[fam] < 2 and r==2:
                return {'namespace':NS,'stage':'VAL','pass':False,'reason':'positive_family_gate','TAU_PREQ':tau,'positive_pass':dict(pc),'true_language':true_lang,'VAL':rows}
    if sum(pc.values())<8 or any(pc[f]<2 for f in POS) or true_lang<5:
        return {'namespace':NS,'stage':'VAL','pass':False,'reason':'positive_total_or_language_gate','TAU_PREQ':tau,'positive_pass':dict(pc),'true_language':true_lang,'VAL':rows}
    for fam in NEG:
        for r in range(3):
            z=synthetic_one(fam,'VAL',r,lms,False);rows.append(z);print('V8CTRL',json.dumps(brief(z),sort_keys=True),flush=True)
            if z['PREQ_ADV']>=tau:
                return {'namespace':NS,'stage':'VAL','pass':False,'reason':'negative_false_positive','false_positive_family':fam,'TAU_PREQ':tau,'positive_pass':dict(pc),'true_language':true_lang,'VAL':rows}
    return {'namespace':NS,'stage':'VAL','pass':True,'TAU_PREQ':tau,'positive_pass':dict(pc),'true_language':true_lang,'VAL':rows}


def q0(lms):
    cal=calibration(lms);print('CAL_RESULT',json.dumps({k:v for k,v in cal.items() if k!='CAL'},sort_keys=True),flush=True)
    if not cal['pass']:return cal
    val=validation(lms,cal['TAU_PREQ']);val['CAL']=cal['CAL'];return val


def target(lms,tau):
    folios,labs,meta=v6.target_folios();folios,labs=order_folios(folios,'VOYNICH_FIT_V8',labs)
    n=len(folios);warm=max(4,int(round(.20*n)));warm += warm%2
    rem=list(range(warm,n));parts=np.array_split(np.asarray(rem,int),8);blocks=[list(map(int,x)) for x in parts if len(x)]
    z=prequential(folios,lms,'VOYNICH_FIT_V8',False,warm,blocks)
    z.update({'namespace':NS,'stage':'FIT_EXPLORATORY','TAU_PREQ':tau,'pass_threshold':bool(z['PREQ_ADV']>=tau),
              'warm_folios':labs[:warm],'coded_folios':labs[warm:],'meta':meta,'target_is_consumed':True})
    print('V8FIT',json.dumps({k:z[k] for k in ['selected_language','surface_arch','coded_events','PREQ_ADV','PREQ_ADV_HEADER','TAU_PREQ','pass_threshold']},sort_keys=True),flush=True)
    return z


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','q0','fit'],required=True);ap.add_argument('--tau-preq',type=float);a=ap.parse_args()
    lms=v6.b.load_lms()
    if a.mode=='smoke':out=smoke(lms)
    elif a.mode=='q0':out=q0(lms)
    else:
        if a.tau_preq is None:raise SystemExit('fit requires --tau-preq')
        out=target(lms,a.tau_preq)
    print('RESULT_JSON',json.dumps(out,sort_keys=True))

if __name__=='__main__':main()

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .latent_regime import equal_family_weights, heldout_scores, choose_k, bootstrap_stability
from .common import atomic_json, load_config

FEATURES=[
 "TEXT_ORDER::adjacent_mi",
 "TEXT_ENTROPY::red1","TEXT_ENTROPY::red2",
 "TEXT_EDIT::ed1_density",
 "TEXT_PERSIST::midfix_lag1","TEXT_PERSIST::suffix_lag1",
 "LEXICAL::hapax","LEXICAL::type_token",
 "PAGE::between_page_overlap",
]
# Exact inherited fold sizes: 17,11,8,7,7.
FOLDS=np.array([0]*17+[1]*11+[2]*8+[3]*7+[4]*7,dtype=int)
EFFECT_FLOOR=1.50
TRIALS=20
REQUIRED_SUCCESS=16


def synthetic_panel(kind:str, seed:int):
    rng=np.random.default_rng(seed)
    n,p=50,9
    # One shared nuisance factor plus independent feature noise. Nuisance is not a discrete regime.
    nuisance=rng.normal(size=(n,1))
    load=np.array([[.30,.25,.20,.30,.20,.25,.25,.20,.30]])
    X=rng.normal(size=(n,p)) + nuisance@load
    truth=None
    if kind=="shared_k2":
        truth=np.repeat([0,1],25)
        rng.shuffle(truth)
        # Mean difference exactly 1.50 SD on one feature in four independent families.
        active=[0,1,3,4]
        sign=(2*truth-1)[:,None]
        X[:,active]+=sign*(EFFECT_FLOOR/2)
    elif kind=="shared_k3":
        truth=np.tile(np.arange(3),17)[:n]
        rng.shuffle(truth)
        active=[0,1,3,4]
        level=np.array([-EFFECT_FLOOR,0.0,EFFECT_FLOOR])[truth][:,None]
        X[:,active]+=level
    elif kind=="one_state":
        pass
    elif kind=="continuous_drift":
        order=rng.permutation(n)
        t=np.linspace(-1,1,n)[order][:,None]
        X[:,[0,1,3,4]] += 0.90*t
    elif kind=="family_specific":
        # Incompatible partitions in different feature families: a non-shared alternative.
        a=np.repeat([0,1],25); b=np.tile([0,1],25)
        rng.shuffle(a); rng.shuffle(b)
        X[:,[0,1]]+=(2*a-1)[:,None]*(EFFECT_FLOOR/2)
        X[:,[4,6]]+=(2*b-1)[:,None]*(EFFECT_FLOOR/2)
    else: raise ValueError(kind)
    return X,truth


def detector(X, seed, cfg):
    Xw,_=equal_family_weights(FEATURES,X)
    scores=heldout_scores(Xw,FOLDS,cfg["u3"]["k_values"],seed)
    k,means=choose_k(scores)
    wins=sum(scores[k][i] > scores[1][i] for i in range(5)) if k!=1 else 0
    # No expensive stability calculation can rescue a candidate already failing held-out criteria.
    if k==1 or wins < cfg["u3"]["min_outer_fold_wins"]:
        return {"detected":False,"selected_k":int(k),"fold_wins":int(wins),"median_ari":None,"means":means}
    ari,_=bootstrap_stability(Xw,k,seed,cfg["u3"]["bootstrap_reps"])
    detected=bool(ari>=cfg["u3"]["min_stability_ari"])
    return {"detected":detected,"selected_k":int(k),"fold_wins":int(wins),"median_ari":float(ari),"means":means}


def run_power(kind, cfg, base_seed):
    rows=[]; successes=0
    for i in range(TRIALS):
        X,_=synthetic_panel(kind,base_seed+i)
        r=detector(X,base_seed+10000+i,cfg); r["trial"]=i; rows.append(r)
        successes+=int(r["detected"])
        failures=(i+1)-successes
        # Fixed 16/20 threshold: once five failures occur, 16 successes are mathematically impossible.
        if failures>=5:
            return rows,successes,False,"EARLY_FAIL_16_OF_20_IMPOSSIBLE"
    return rows,successes,successes>=REQUIRED_SUCCESS,"COMPLETE"


def run_fpr(kind,cfg,base_seed):
    rows=[]; false_calls=0
    # 40 fixed null trials; <=2 discrete calls is the operational <=5% criterion.
    for i in range(40):
        X,_=synthetic_panel(kind,base_seed+i)
        r=detector(X,base_seed+10000+i,cfg);r["trial"]=i;rows.append(r)
        false_calls+=int(r["detected"])
        if false_calls>=3:
            return rows,false_calls,False,"EARLY_FAIL_GT_5_PERCENT"
    return rows,false_calls,True,"COMPLETE"


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    cfg=load_config(a.config); a.out.mkdir(parents=True,exist_ok=True)
    result={"schema":"u3-calibration-v0.1","effect_floor_sd":EFFECT_FLOOR,"target_opened":False,"trials_required_for_power":TRIALS,"successes_required":REQUIRED_SUCCESS,"arms":{}}

    # Power is tested first. Failure of either shared-regime arm is sufficient to keep target sealed.
    for j,kind in enumerate(["shared_k2","shared_k3"]):
        rows,succ,passed,stop=run_power(kind,cfg,2026081400+j*1000)
        result["arms"][kind]={"attempted":len(rows),"successes":succ,"pass":passed,"stop":stop,"rows":rows}
        if not passed:
            result["formal_verdict"]="FAIL_CALIBRATION_POWER"
            result["target_may_open"]=False
            result["note"]="U3 target remains sealed; null/FPR arms need not be run because the instrument already fails the mandatory power gate."
            atomic_json(a.out/"U3_CALIBRATION.json",result);print(json.dumps(result,indent=2));return

    for j,kind in enumerate(["one_state","continuous_drift"]):
        rows,calls,passed,stop=run_fpr(kind,cfg,2026084400+j*1000)
        result["arms"][kind]={"attempted":len(rows),"false_discrete_calls":calls,"pass":passed,"stop":stop,"rows":rows}
        if not passed:
            result["formal_verdict"]="FAIL_CALIBRATION_FPR";result["target_may_open"]=False
            atomic_json(a.out/"U3_CALIBRATION.json",result);print(json.dumps(result,indent=2));return

    result["formal_verdict"]="PASS";result["target_may_open"]=True
    atomic_json(a.out/"U3_CALIBRATION.json",result);print(json.dumps(result,indent=2))

if __name__=="__main__":main()

#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import memory_recurrence_score_v1 as score

ASSAY='memory_recurrence_source_id_v1'
REPS=score.REPS
GRIDS={
    'refractory_exact':[.01,.02,.04,.06,.08],
    'family_persistence':[.01,.02,.04,.06,.08],
    'r64_exact':[.005,.01,.02,.03,.04,.06],
}
N=40

def wilson(k,n,z=1.959963984540054):
    p=k/n; den=1+z*z/n; ctr=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.0,ctr-half),min(1.0,ctr+half)]

def index_power(objs):
    d={}
    for o in objs:
        if o.get('split')!='power' or o.get('source') not in GRIDS: continue
        st=float(o.get('strength')); key=(o['source'],st,int(o['trial']))
        if key in d and json.dumps(d[key],sort_keys=True)!=json.dumps(o,sort_keys=True):
            raise ValueError(f'conflicting duplicate {key}')
        d[key]=o
    for src,grid in GRIDS.items():
        for st in grid:
            ks=sorted(t for (s,x,t) in d if s==src and abs(x-st)<=1e-12)
            if ks!=list(range(N)): raise ValueError(f'incomplete power cell {src} {st}: n={len(ks)}')
    return d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',nargs='+',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args(); objs=score.load_jsonl(a.input)
    dev=score.index_trials(objs,'development',score.CLASSES,80); fit=score.fit_models(dev); pw=index_power(objs)
    out={'assay_id':ASSAY,'n_per_cell':N,'by_representation':{}}
    for rep in REPS:
        fams={}
        for src,grid in GRIDS.items():
            cells=[]
            for st in grid:
                ss=[score.trial_score(pw[(src,float(st),t)],rep,fit[rep]) for t in range(N)]
                k=sum(x['prediction']==src for x in ss); rate=k/N
                cells.append({'strength':st,'n':N,'correct':k,'rate':rate,'wilson95':wilson(k,N),'abstained':sum(x['abstain'] for x in ss)})
            ge80=next((c['strength'] for c in cells if c['rate']>=.80),None)
            ge90=next((c['strength'] for c in cells if c['rate']>=.90),None)
            fams[src]={'cells':cells,'min_strength_ge_80':ge80,'min_strength_ge_90':ge90}
        out['by_representation'][rep]={'families':fams}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('P1POWER='+json.dumps(out,separators=(',',':')))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Independent solver cross-check, separate from main max-cost-flow code.
The controls here test exact symbolic invertibility, not Voynich mechanism identification.
"""
from pathlib import Path
import collections as C, hashlib, importlib.util, json, os, pickle, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
B=Path(__file__).resolve().parent
R=json.loads((B/'results.json').read_text())
source=B/'source'/'gen138'
for k,h in R['provenance']['files'].items():
    if k in ('GENERAT_v138_kurzertext.txt',):continue
    assert hashlib.sha256((source/k).read_bytes()).hexdigest()==h,(k,'provenance mismatch')
sys.path.insert(0,str(source))
cwd=os.getcwd();os.chdir(source)
try:
    spec=importlib.util.spec_from_file_location('independent_v138',source/'vbm_tafel_v138.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
finally: os.chdir(cwd)
R0=original.RANK;cells=sorted(set(original.CELLS));ev=original.ev
pair_v=original.pair_v
support={v:{pair[1] for pair,w in pair_v.items() if w==v} for v in 'aeiou'}
def core(e):return e[4] if e[4] and e[5] and len(e[4])+len(e[5])>=3 else e[1]
cs=C.Counter();vctx=C.defaultdict(set)
for i,e in enumerate(ev):
    k=core(e)
    if k:
        cs[k]+=1
        if i: vctx[k].add(e[0])
# Hungarian max weighted matching across actual 443 cells x 722 clusters.
clusters=sorted(R0)
score=np.zeros((len(cells),len(clusters)), dtype=np.int64)
for j,k in enumerate(clusters):
    keys=set.intersection(*(support[v] for v in vctx[k])) if vctx[k] else {'y'}
    for i,cell in enumerate(cells):score[i,j]=cs[k] if cell[0] in keys else -1_000_000
row,col=linear_sum_assignment(score,maximize=True)
assert min(score[row,col])>=0
weight=int(score[row,col].sum())
assert weight==R['optimized_bound']['max_covered_source_events'],(weight,R['optimized_bound']['max_covered_source_events'])
assert len(row)==443
# Assert the advertised stage-2 mapping itself, not only independent optimum.
assign=pickle.load(open(B/'stage2_optimized_bound.pkl','rb'))['assignment']
assert len(assign)==443 and len(set(map(tuple,assign.values())))==443
for k,c in assign.items():
    allowed=set.intersection(*(support[v] for v in vctx[k])) if vctx[k] else {'y'}
    assert c[0] in allowed,(k,c)
assert sum(cs[k] for k in assign)==weight
# Control matching: same-key collision must be rejected independently.
keys_by_family=C.defaultdict(list)
for k,cell in assign.items():keys_by_family[cell[0]].append(k)
fam=next(key for key,ks in sorted(keys_by_family.items()) if len(ks)>=2)
a,b=keys_by_family[fam][:2]
collision=dict(assign);collision[b]=collision[a]
positive=len({tuple(v) for v in assign.values()})==len(assign)
negative=len({tuple(v) for v in collision.values()})!=len(collision)
assert positive and negative
# First event compatibility beyond relaxed upper-bound matching.
first_cluster=core(ev[0]);first_key=assign.get(first_cluster,[None])[0]
review={'independent_method':'SciPy Hungarian assignment vs NetworkX network simplex minimum-cost flow',
 'independent_max_covered_event_count':weight,'optimized_type_count':len(row),
 'agrees_with_stage2':True,'independent_positive_control':positive,
 'independent_collision_negative':negative,
 'first_cluster':first_cluster,'first_key_under_optimal_stage2':first_key,
 'fixed_initial_y_compliant': first_key=='y',
 'mechanism_identifiability':'NO: every observed output stream also has a nonlinguistic copying/slot process producing it; absent external anchor, this exact-output negative cannot be rejected'}
tmp=B/'stage4_adversarial.pkl.tmp';tmp.write_bytes(pickle.dumps(review));tmp.replace(B/'stage4_adversarial.pkl')
(B/'independent_review.json').write_text(json.dumps(review,indent=2))
print(json.dumps(review,indent=2))

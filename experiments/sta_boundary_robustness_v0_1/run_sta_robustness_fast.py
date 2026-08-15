#!/usr/bin/env python3
"""Performance-only wrapper for the frozen STA robustness runner.

Replaces two expensive implementations with mathematically identical cached forms:
1. exact null probabilities are computed once per permutation-group pair per line;
2. held-out training tables are built once per model/fold and reused across replicates.
No statistic, threshold, corpus rule, seed family, fold, model, or null changes.
"""
import importlib.util, os, random, sys
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE/'run_sta_robustness.py'
spec=importlib.util.spec_from_file_location('frozen_sta',BASE)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def pair_vector(tokens, ia, ib):
    a,b=tokens[ia],tokens[ib]
    v=[0.0]*10
    if a==b:v[0]=1.0
    if m.is_ed1(a,b):
        v[1]=1.0
        lc=m.lenclass(a,b); v[2+('short','mid','long').index(lc)]=1.0
        dk=m.ed_kind(a,b); v[5+('acc','red','sub_first','sub_second').index(dk)]=1.0
    return v


def group_pair_probs(tokens, groups):
    pos_to_g={int(p):gi for gi,g in enumerate(groups) for p in g}
    cache={}
    for ga,g1 in enumerate(groups):
        for gb,g2 in enumerate(groups):
            key=(ga,gb); sums=[0.0]*10; den=0
            if ga==gb:
                for ia in g1:
                    for ib in g1:
                        if int(ia)==int(ib):continue
                        den+=1; v=pair_vector(tokens,int(ia),int(ib))
                        for k,x in enumerate(v):sums[k]+=x
            else:
                for ia in g1:
                    for ib in g2:
                        den+=1; v=pair_vector(tokens,int(ia),int(ib))
                        for k,x in enumerate(v):sums[k]+=x
            cache[key]=[x/den if den else 0.0 for x in sums]
    return pos_to_g,cache


def fast_exact_null_means(lines,null):
    out=Counter()
    names=['EQ','ED1','LEN_short','LEN_mid','LEN_long','DIR_acc','DIR_red','DIR_sub_first','DIR_sub_second']
    for x in lines:
        t=x['tokens']; n=len(t); groups=m.groups_for(n,null); pg,cache=group_pair_probs(t,groups)
        def pv(i,j):return cache[(pg[i],pg[j])]
        for i in range(n-1):
            v=pv(i,i+1); out['ED1_whole']+=v[1]
            for k,c in enumerate(('short','mid','long')):out['LEN_'+c]+=v[2+k]
            for k,d in enumerate(('acc','red','sub_first','sub_second')):out['DIR_'+d]+=v[5+k]
            if n>=6 and 2<=i<=n-4:out['ED1_N3']+=v[1]
        for i in range(n-2):
            v=pv(i,i+2); out['E2_whole']+=v[0]
            if n>=7 and 2<=i<=n-5:out['E2_N3']+=v[0]
    return out

m.exact_null_means=fast_exact_null_means

# Persistent fold/table cache across all 30 replicates.
_MODEL_CACHE={}
def fast_generate_oof(lines,model,rep):
    key=(id(lines),model)
    if key not in _MODEL_CACHE:
        byfold={f:[] for f in range(m.NFOLD)}
        for x in lines:byfold[m.fold_of(x['folio'])].append(x)
        fold_tables={}
        for f in range(m.NFOLD):
            train=[x for g in range(m.NFOLD) if g!=f for x in byfold[g]]
            fold_tables[f]=m.build_tables(train,model)
        _MODEL_CACHE[key]=(byfold,fold_tables)
    byfold,fold_tables=_MODEL_CACHE[key]
    out=[]
    for f in range(m.NFOLD):
        test=byfold[f]; tables=fold_tables[f]
        rng=random.Random(m.SEED+100000*rep+1000*f+{'S0':0,'S1':100,'S2':200}[model])
        for x in test:
            n=len(x['tokens']); sec=x['section']; lb=m.lbucket(n); g=[]
            for i in range(n):
                p=None
                if model=='S1':p=m.coarse_pos(i,n)
                elif model=='S2':p=m.edge_pos(i,n)
                c=m.choose_counter(tables,sec,lb,p,model); g.append(m.sample_counter(c,rng))
            out.append({'folio':x['folio'],'section':sec,'tokens':tuple(g)})
    return out
m.generate_oof=fast_generate_oof

if __name__=='__main__':
    m.main()

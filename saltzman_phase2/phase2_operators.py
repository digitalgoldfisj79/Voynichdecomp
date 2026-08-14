#!/usr/bin/env python3
import numpy as np
from permutation_operators import transform_lines as base_transform
NEW_OPS=('RANDOM_REVERSAL','RANDOM_ROTATION','BLOCK2_SHUFFLE','BLOCK3_SHUFFLE','BLOCK4_SHUFFLE')

def transform_lines(lines,op,seed):
    if op in ('BLOCK2_SHUFFLE','BLOCK3_SHUFFLE','BLOCK4_SHUFFLE'):
        return base_transform(lines,op,seed)
    rng=np.random.default_rng(seed);out=[]
    for line in lines:
        q=[]
        for w in line:
            n=len(w)
            if n<2: q.append(w); continue
            if op=='RANDOM_REVERSAL':
                q.append(w[::-1] if int(rng.integers(0,2)) else w)
            elif op=='RANDOM_ROTATION':
                k=int(rng.integers(0,n));q.append(w[k:]+w[:k])
            else: raise ValueError(op)
        out.append(q)
    return out

def self_tests():
    for op in NEW_OPS:
        x=transform_lines([['abcdef','abc']],op,123)[0]
        assert sorted(x[0])==sorted('abcdef') and sorted(x[1])==sorted('abc')
    return True

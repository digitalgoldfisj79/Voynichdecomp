#!/usr/bin/env python3
import numpy as np

OPERATORS=(
    'IDENTITY',
    'REVERSE',
    'ROTATE_LEFT_1',
    'ADJACENT_PAIR_SWAP',
    'ODD_EVEN',
    'OUTSIDE_IN',
    'HALF_SWAP',
    'BLOCK2_SHUFFLE',
    'BLOCK3_SHUFFLE',
    'BLOCK4_SHUFFLE',
    'FULL_SHUFFLE',
)

COMPLEXITY={
    'IDENTITY':0,
    'REVERSE':1,
    'ROTATE_LEFT_1':1,
    'ADJACENT_PAIR_SWAP':1,
    'ODD_EVEN':2,
    'OUTSIDE_IN':2,
    'HALF_SWAP':2,
    'BLOCK2_SHUFFLE':3,
    'BLOCK3_SHUFFLE':4,
    'BLOCK4_SHUFFLE':5,
    'FULL_SHUFFLE':99,
}

STOCHASTIC={'BLOCK2_SHUFFLE','BLOCK3_SHUFFLE','BLOCK4_SHUFFLE','FULL_SHUFFLE'}


def _outside_in_idx(n):
    out=[]; i=0; j=n-1
    while i<=j:
        out.append(i)
        if j!=i: out.append(j)
        i+=1; j-=1
    return out


def transform_word(w, op, rng=None):
    n=len(w)
    if n<2 or op=='IDENTITY': return w
    if op=='REVERSE': return w[::-1]
    if op=='ROTATE_LEFT_1': return w[1:]+w[:1]
    if op=='ADJACENT_PAIR_SWAP':
        a=list(w)
        for i in range(0,n-1,2): a[i],a[i+1]=a[i+1],a[i]
        return ''.join(a)
    if op=='ODD_EVEN': return w[0::2]+w[1::2]
    if op=='OUTSIDE_IN': return ''.join(w[i] for i in _outside_in_idx(n))
    if op=='HALF_SWAP':
        k=(n+1)//2
        return w[k:]+w[:k]
    if op.startswith('BLOCK') and op.endswith('_SHUFFLE'):
        if rng is None: raise ValueError('rng required')
        b=int(op.removeprefix('BLOCK').removesuffix('_SHUFFLE'))
        out=[]
        for s in range(0,n,b):
            chunk=list(w[s:s+b])
            if len(chunk)>1:
                idx=rng.permutation(len(chunk)); chunk=[chunk[i] for i in idx]
            out.extend(chunk)
        return ''.join(out)
    if op=='FULL_SHUFFLE':
        if rng is None: raise ValueError('rng required')
        idx=rng.permutation(n); return ''.join(w[i] for i in idx)
    raise ValueError(op)


def transform_lines(lines, op, seed):
    rng=np.random.default_rng(seed) if op in STOCHASTIC else None
    return [[transform_word(w,op,rng) for w in line] for line in lines]


def moved_fraction(n,op):
    if n<2 or op=='IDENTITY': return 0.0
    base=''.join(chr(0x1000+i) for i in range(n))
    # Deterministic operators only. Stochastic values are theoretical labels elsewhere.
    if op in STOCHASTIC: return None
    out=transform_word(base,op,None)
    return sum(a!=b for a,b in zip(base,out))/n


def self_tests():
    assert transform_word('abcd','REVERSE')=='dcba'
    assert transform_word('abcd','ROTATE_LEFT_1')=='bcda'
    assert transform_word('abcde','ADJACENT_PAIR_SWAP')=='badce'
    assert transform_word('abcdef','ODD_EVEN')=='acebdf'
    assert transform_word('abcde','OUTSIDE_IN')=='aebdc'
    assert transform_word('abcdef','HALF_SWAP')=='defabc'
    for op in OPERATORS:
        for w in ('a','ab','abc','abcdefg'):
            out=transform_word(w,op,np.random.default_rng(123) if op in STOCHASTIC else None)
            assert sorted(out)==sorted(w), (op,w,out)
            assert len(out)==len(w)
    return True

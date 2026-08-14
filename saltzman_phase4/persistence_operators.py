#!/usr/bin/env python3
import hashlib
from fractions import Fraction

def _u01(*parts):
    s="|".join(map(str,parts)).encode("utf-8")
    x=int(hashlib.sha256(s).hexdigest()[:14],16)
    return x/float(16**14)

def _bit(*parts):
    return 1 if _u01(*parts) >= 0.5 else 0

def fixed_run_states(n_tokens, run_length, seed):
    L=int(run_length)
    if L < 1:
        raise ValueError("run_length must be >=1")
    return [_bit("P4","FIXED",seed,L,i//L) for i in range(n_tokens)]

def occurrence_states(n_tokens, seed):
    return fixed_run_states(n_tokens,1,seed)

def markov_states(n_tokens, p_num, p_den, seed):
    p=Fraction(int(p_num),int(p_den))
    if not (Fraction(0,1) <= p < Fraction(1,1)):
        raise ValueError("p_stay must be in [0,1)")
    if n_tokens <= 0:
        return []
    out=[_bit("P4","MARKOV",seed,p.numerator,p.denominator,"INIT")]
    threshold=float(p)
    for i in range(1,n_tokens):
        stay=_u01("P4","MARKOV",seed,p.numerator,p.denominator,"STEP",i) < threshold
        out.append(out[-1] if stay else 1-out[-1])
    return out

def _rotate(w,k):
    if len(w)<2 or k%len(w)==0:
        return w
    k%=len(w)
    return w[k:]+w[:k]

def apply_k2_states(lines, states):
    flat_n=sum(len(line) for line in lines)
    if flat_n != len(states):
        raise ValueError((flat_n,len(states)))
    out=[]; j=0
    for line in lines:
        q=[]
        for w in line:
            state=states[j]; j+=1
            off=0 if (state==0 or len(w)<2) else min(len(w)-1, len(w)//2)
            q.append(_rotate(w,off))
        out.append(q)
    return out

def occurrence_all(lines, seed):
    out=[]; j=0
    for line in lines:
        q=[]
        for w in line:
            if len(w)<2:
                q.append(w); j+=1; continue
            u=_u01("P4","OCCURRENCE_ALL",seed,j)
            off=min(len(w)-1,int(u*len(w)))
            q.append(_rotate(w,off)); j+=1
        out.append(q)
    return out

def states_for_arm(arm,n_tokens,seed):
    if arm in ("OCCURRENCE_K2","FIXED_RUN1_K2"):
        return occurrence_states(n_tokens,seed)
    if arm.startswith("FIXED_RUN") and arm.endswith("_K2"):
        L=int(arm[len("FIXED_RUN"):-len("_K2")])
        return fixed_run_states(n_tokens,L,seed)
    if arm.startswith("MARKOV_M") and arm.endswith("_K2"):
        label=arm[len("MARKOV_"):-len("_K2")]
        lookup={"M2":(1,2),"M3":(2,3),"M4":(3,4),"M5":(4,5),"M8":(7,8),"M12":(11,12)}
        if label not in lookup:
            raise ValueError(arm)
        return markov_states(n_tokens,*lookup[label],seed)
    return None

def self_tests():
    n=64; seed=12345
    assert occurrence_states(n,seed)==fixed_run_states(n,1,seed)
    for L in (1,2,3,4,6,8,12):
        s=fixed_run_states(n,L,seed)
        assert len(s)==n and set(s)<=set((0,1))
        for i in range(0,n,L):
            assert len(set(s[i:min(i+L,n)]))==1
    for num,den in ((1,2),(2,3),(3,4),(4,5),(7,8),(11,12)):
        s=markov_states(n,num,den,seed)
        assert len(s)==n and set(s)<=set((0,1))
    lines=[["abcdef","abcde","x"],["wxyz","mnopqr"]]
    s=[1,0,1,1,0]
    z=apply_k2_states(lines,s)
    assert z[0][0]=="defabc"
    assert z[0][1]=="abcde"
    assert z[0][2]=="x"
    assert z[1][0]=="yzwx"
    assert sorted("".join(sum(z,[])))==sorted("".join(sum(lines,[])))
    return True

if __name__=="__main__":
    assert self_tests()
    print("persistence_operators self-tests: OK")

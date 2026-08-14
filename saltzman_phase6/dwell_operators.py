#!/usr/bin/env python3
import hashlib, math
from fractions import Fraction
try:
    from persistence_operators import occurrence_states, fixed_run_states, markov_states
except ImportError:
    from saltzman_phase4.persistence_operators import occurrence_states, fixed_run_states, markov_states

def _u01(*parts):
    s="|".join(map(str,parts)).encode("utf-8")
    x=int(hashlib.sha256(s).hexdigest()[:14],16)
    return x/float(16**14)

def _bit(*parts):
    return 1 if _u01(*parts) >= 0.5 else 0

def semi_refresh_states(n_tokens, tau, seed):
    tau=int(tau)
    if tau==3:
        support=(2,4); threshold=Fraction(2,3)
    elif tau==4:
        support=(3,5); threshold=Fraction(5,8)
    else:
        raise ValueError("tau must be 3 or 4")
    out=[]; block=0
    while len(out)<n_tokens:
        u=_u01("P6-SEMI","DURATION",seed,tau,block)
        d=support[0] if u < float(threshold) else support[1]
        state=_bit("P6-SEMI","STATE",seed,tau,block)
        out.extend([state]*min(d,n_tokens-len(out)))
        block+=1
    return out

def states_for_phase6_arm(arm,n_tokens,seed):
    if arm=="OCCURRENCE_K2":
        return occurrence_states(n_tokens,seed)
    if arm=="TAU3_FIXED_K2":
        return fixed_run_states(n_tokens,3,seed)
    if arm=="TAU3_SEMI_K2":
        return semi_refresh_states(n_tokens,3,seed)
    if arm=="TAU3_GEOM_K2":
        return markov_states(n_tokens,3,4,seed)
    if arm=="TAU4_FIXED_K2":
        return fixed_run_states(n_tokens,4,seed)
    if arm=="TAU4_SEMI_K2":
        return semi_refresh_states(n_tokens,4,seed)
    if arm=="TAU4_GEOM_K2":
        return markov_states(n_tokens,4,5,seed)
    if arm=="IDENTITY":
        return None
    raise ValueError(arm)

def support_moments(tau):
    tau=int(tau)
    if tau==3:
        vals=(2,4); probs=(Fraction(2,3),Fraction(1,3))
    elif tau==4:
        vals=(3,5); probs=(Fraction(5,8),Fraction(3,8))
    else:
        raise ValueError(tau)
    ed=sum(Fraction(v)*p for v,p in zip(vals,probs))
    ed2=sum(Fraction(v*v)*p for v,p in zip(vals,probs))
    var=ed2-ed*ed
    return {"E_D":ed,"E_D2":ed2,"Var_D":var,"tau_int":ed2/ed}

def empirical_rho(states,maxlag=12):
    x=[1 if s else -1 for s in states]
    n=len(x); mu=sum(x)/n
    den=sum((v-mu)**2 for v in x)
    out=[]
    for h in range(maxlag+1):
        if h==0:
            out.append(1.0); continue
        num=sum((x[i]-mu)*(x[i+h]-mu) for i in range(n-h))
        out.append(num/den)
    return out

def self_tests():
    m3=support_moments(3); m4=support_moments(4)
    assert m3["E_D"]==Fraction(8,3)
    assert m3["E_D2"]==Fraction(8,1)
    assert m3["tau_int"]==Fraction(3,1)
    assert m4["E_D"]==Fraction(15,4)
    assert m4["E_D2"]==Fraction(15,1)
    assert m4["tau_int"]==Fraction(4,1)
    n=10000; seed=123456
    for tau in (3,4):
        s=semi_refresh_states(n,tau,seed)
        assert len(s)==n and set(s)<=set((0,1))
    assert states_for_phase6_arm("TAU3_FIXED_K2",100,seed)==fixed_run_states(100,3,seed)
    assert states_for_phase6_arm("TAU3_GEOM_K2",100,seed)==markov_states(100,3,4,seed)
    assert states_for_phase6_arm("TAU4_FIXED_K2",100,seed)==fixed_run_states(100,4,seed)
    assert states_for_phase6_arm("TAU4_GEOM_K2",100,seed)==markov_states(100,4,5,seed)
    for arm in ("TAU3_FIXED_K2","TAU3_SEMI_K2","TAU3_GEOM_K2",
                "TAU4_FIXED_K2","TAU4_SEMI_K2","TAU4_GEOM_K2"):
        r=empirical_rho(states_for_phase6_arm(arm,100000,seed),12)
        assert abs(r[1]) < 0.8
        assert abs(r[-1]) < 0.1
    return True

if __name__=="__main__":
    assert self_tests()
    for tau in (3,4):
        m=support_moments(tau)
        print("TAU",tau,{k:str(v) for k,v in m.items()})
    print("dwell_operators self-tests: OK")

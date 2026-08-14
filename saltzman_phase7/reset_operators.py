#!/usr/bin/env python3
import hashlib

try:
    from persistence_operators import occurrence_states, fixed_run_states, markov_states
except ImportError:
    from saltzman_phase4.persistence_operators import occurrence_states, fixed_run_states, markov_states

def _seed(*parts):
    s="|".join(map(str,parts)).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16],16)

def line_reset_fixed_states(n_tokens, line_width, run_length, seed):
    if n_tokens % line_width != 0:
        raise ValueError("Phase 7 requires exact line partition")
    out=[]
    for j in range(n_tokens//line_width):
        ss=_seed("P7-LINE-RESET","FIXED",seed,j)
        out.extend(fixed_run_states(line_width,run_length,ss))
    return out

def line_reset_markov_states(n_tokens, line_width, p_num, p_den, seed):
    if n_tokens % line_width != 0:
        raise ValueError("Phase 7 requires exact line partition")
    out=[]
    for j in range(n_tokens//line_width):
        ss=_seed("P7-LINE-RESET","MARKOV",seed,j)
        out.extend(markov_states(line_width,p_num,p_den,ss))
    return out

def states_for_phase7_arm(arm,n_tokens,line_width,seed):
    if arm=="OCCURRENCE_K2":
        return occurrence_states(n_tokens,seed)
    if arm=="TAU3_FIXED_CONTINUOUS_K2":
        return fixed_run_states(n_tokens,3,seed)
    if arm=="TAU3_FIXED_LINE_RESET_K2":
        return line_reset_fixed_states(n_tokens,line_width,3,seed)
    if arm=="TAU3_GEOM_CONTINUOUS_K2":
        return markov_states(n_tokens,3,4,seed)
    if arm=="TAU3_GEOM_LINE_RESET_K2":
        return line_reset_markov_states(n_tokens,line_width,3,4,seed)
    if arm=="IDENTITY":
        return None
    raise ValueError(arm)

def self_tests():
    n=200; w=10; seed=12345
    f=states_for_phase7_arm("TAU3_FIXED_CONTINUOUS_K2",n,w,seed)
    assert f==fixed_run_states(n,3,seed)
    g=states_for_phase7_arm("TAU3_GEOM_CONTINUOUS_K2",n,w,seed)
    assert g==markov_states(n,3,4,seed)
    fr=states_for_phase7_arm("TAU3_FIXED_LINE_RESET_K2",n,w,seed)
    gr=states_for_phase7_arm("TAU3_GEOM_LINE_RESET_K2",n,w,seed)
    assert len(fr)==len(gr)==n and set(fr)<=set((0,1)) and set(gr)<=set((0,1))
    assert fr==line_reset_fixed_states(n,w,3,seed)
    assert gr==line_reset_markov_states(n,w,3,4,seed)
    for j in range(n//w):
        ss=_seed("P7-LINE-RESET","FIXED",seed,j)
        assert fr[j*w:(j+1)*w]==fixed_run_states(w,3,ss)
    return True

if __name__=="__main__":
    assert self_tests()
    print("Phase 7 reset-operator self-tests: OK")

#!/usr/bin/env python3
"""SVT v0.1 core: order-free stateful heads + hidden 1--3 glyph units."""
from __future__ import annotations
import dataclasses, hashlib, json, math, random, statistics, sys
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from numba import njit

HERE=Path(__file__).resolve().parent; TERMINAL=HERE.parent/"terminal_cipher_v0_6"; V05=HERE.parent/"recoverability_frontier_v0_5"
for p in (TERMINAL,V05):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import recoverability_v050 as core
import mono_solver_v051 as mono
import v060_family_p_stage_a as pbase

MODES=("periodic","line_reset"); DEV_PERIODS=(2,3,4,6,8); TEST_PERIODS=(5,7,9,10,12); CANDIDATE_PERIODS=tuple(range(2,13))
CODE_LENGTHS=(1,2,3); LENGTH_PRIOR=(0.30,0.45,0.25); SEGMENTATION_BEAM=8; BOUNDARY_WEIGHT=0.35; CONTINUATION_NOISE=0.08; COARSE_CLASSES=4; STATE_SWAP_FRACTION=0.12

@dataclasses.dataclass
class StatefulHeadTrial:
    iso:str; split:str; length:int; mode:str; replicate:int; seed:int; plain:list[int]; cipher:list[int]; period:int; line_starts:list[int]; forward_maps:list[list[int]]
@dataclasses.dataclass
class SegmentationPath:
    starts:list[int]; head_line_starts:list[int]; score:float
@dataclasses.dataclass
class SVTTrial:
    head:StatefulHeadTrial; surface:list[int]; head_positions:list[int]; code_lengths:list[int]; surface_line_starts:list[int]; continuation_offsets_1:list[list[int]]; continuation_offsets_2:list[list[int]]; family:str="FSVT"
@dataclasses.dataclass
class HeadSolution:
    mode:str; period:int; score:float; raw_score:float; prediction:list[int]; inverses:list[list[int]]
@dataclasses.dataclass
class SurfaceSolution:
    path:SegmentationPath; head_solution:HeadSolution; joint_score:float; head_score_per_symbol:float; boundary_score_per_surface:float

def stable_hash_json(obj:Any)->str:
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _phase(length:int,period:int,mode:str,line_starts:list[int])->np.ndarray:
    return pbase.phase_array(length,period,mode,line_starts)

def _fresh_maps(rng:random.Random,a:int,period:int)->list[list[int]]:
    base=list(range(a)); rng.shuffle(base); out=[]; swaps=max(2,round(STATE_SWAP_FRACTION*a))
    for state in range(period):
        m=base.copy()
        for _ in range(swaps+(state%2)):
            i,j=rng.sample(range(a),2); m[i],m[j]=m[j],m[i]
        out.append(m)
    return out

def make_stateful_head_trial(language:core.LanguageData,split:str,length:int,mode:str,replicate:int)->StatefulHeadTrial:
    chunks=core.source_chunks(language,split,length)
    if not chunks: raise RuntimeError(f"no chunks for {language.iso}/{split}/{length}")
    plain=list(chunks[replicate%len(chunks)]); seed=core.stable_seed("svt-v01-head",language.iso,split,length,mode,replicate); rng=random.Random(seed)
    period=rng.choice(TEST_PERIODS if split=="test" else DEV_PERIODS); line_rng=random.Random(core.stable_seed("svt-v01-lines",seed)); line_starts=pbase.make_line_starts(line_rng,length)
    ph=_phase(length,period,mode,line_starts); maps=_fresh_maps(rng,len(language.alphabet),period); cipher=[maps[int(ph[i])][int(x)] for i,x in enumerate(plain)]
    return StatefulHeadTrial(language.iso,split,length,mode,replicate,seed,plain,cipher,period,line_starts,maps)

def choose_length(rng:random.Random,state:int,cls:int)->int:
    p1,p2,p3=LENGTH_PRIOR; k=(state+2*cls)%3
    if k==0: p1+=.06; p2-=.03; p3-=.03
    elif k==1: p2+=.06; p1-=.03; p3-=.03
    else: p3+=.06; p1-=.03; p2-=.03
    u=rng.random(); return 1 if u<p1 else (2 if u<p1+p2 else 3)

def make_svt_trial(language:core.LanguageData,split:str,length:int,mode:str,replicate:int)->SVTTrial:
    head=make_stateful_head_trial(language,split,length,mode,replicate); a=len(language.alphabet); ph=_phase(length,head.period,head.mode,head.line_starts); rng=random.Random(core.stable_seed("svt-v01-surface",head.seed))
    o1=[]; o2=[]
    for _ in range(head.period):
        o1.append([1+rng.randrange(max(1,a-1)) for _ in range(COARSE_CLASSES)]); o2.append([1+rng.randrange(max(1,a-1)) for _ in range(COARSE_CLASSES)])
    surface=[]; starts=[]; lens=[]; p2s=[]
    for i,h in enumerate(head.cipher):
        p2s.append(len(surface)); starts.append(len(surface)); state=int(ph[i]); cls=int(head.plain[i])%COARSE_CLASSES; L=choose_length(rng,state,cls); lens.append(L); surface.append(int(h)); prev=int(h)
        if L>=2:
            t=rng.randrange(a) if rng.random()<CONTINUATION_NOISE else (prev+o1[state][cls])%a; surface.append(t); prev=t
        if L>=3:
            t=rng.randrange(a) if rng.random()<CONTINUATION_NOISE else (prev+o2[state][cls])%a; surface.append(t)
    return SVTTrial(head,surface,starts,lens,[p2s[i] for i in head.line_starts],o1,o2)

def make_nonfactorable_control(trial:SVTTrial,language:core.LanguageData)->SVTTrial:
    rng=random.Random(core.stable_seed("svt-v01-nonfact",trial.head.seed)); a=len(language.alphabet); ph=_phase(trial.head.length,trial.head.period,trial.head.mode,trial.head.line_starts); book={}
    for s in range(trial.head.period):
        for x in range(a): book[(s,x)]=tuple(rng.randrange(a) for _ in range(rng.choice(CODE_LENGTHS)))
    surface=[]; starts=[]; lens=[]; p2s=[]
    for i,x in enumerate(trial.head.plain):
        p2s.append(len(surface)); starts.append(len(surface)); unit=book[(int(ph[i]),int(x))]; surface.extend(unit); lens.append(len(unit))
    return SVTTrial(trial.head,surface,starts,lens,[p2s[i] for i in trial.head.line_starts],[],[],"NONFACT")
def make_shuffled_control(trial:SVTTrial)->SVTTrial:
    rng=random.Random(core.stable_seed("svt-v01-shuffle",trial.head.seed)); starts=trial.surface_line_starts; ends=starts[1:]+[len(trial.surface)]; surface=[]
    for l,r in zip(starts,ends):
        line=trial.surface[l:r].copy(); rng.shuffle(line); surface.extend(line)
    return dataclasses.replace(trial,surface=surface,family="SHUFFLED")

def _transition_logp(values:list[int],a:int,alpha:float=.5)->np.ndarray:
    c=np.full((a,a),alpha,dtype=np.float64)
    for x,y in zip(values,values[1:]): c[int(x),int(y)]+=1
    c/=c.sum(axis=1,keepdims=True); return np.log(c)
def _unit_score(line:list[int],start:int,L:int,T:np.ndarray)->float:
    end=start+L
    if end>len(line): return -1e300
    s=math.log(LENGTH_PRIOR[L-1])
    for i in range(start+1,end): s+=float(T[line[i-1],line[i]])
    if start>0: s+=-float(T[line[start-1],line[start]])
    return s
def _top_one_line(line:list[int],a:int,beam:int)->list[tuple[float,list[int]]]:
    if not line: return [(0.0,[])]
    T=_transition_logp(line,a); dp={0:[(0.0,[])]}
    for pos in range(len(line)):
        for score,starts in dp.get(pos,[]):
            for L in CODE_LENGTHS:
                end=pos+L
                if end>len(line): continue
                b=dp.setdefault(end,[]); b.append((score+_unit_score(line,pos,L,T),starts+[pos])); b.sort(key=lambda z:z[0],reverse=True); del b[beam:]
    return dp.get(len(line),[])[:beam]
def top_segmentations(surface:list[int],surface_line_starts:list[int],a:int,beam:int=SEGMENTATION_BEAM)->list[SegmentationPath]:
    starts=list(surface_line_starts)
    if not starts or starts[0]!=0: starts=[0]+starts
    ends=starts[1:]+[len(surface)]; combined=[(0.0,[],[])]
    for left,right in zip(starts,ends):
        local=_top_one_line(surface[left:right],a,beam); new=[]
        for total,gstarts,hlines in combined:
            for ls,lstarts in local: new.append((total+ls,gstarts+[left+x for x in lstarts],hlines+[len(gstarts)]))
        new.sort(key=lambda z:z[0],reverse=True); combined=new[:beam]
    return [SegmentationPath(s,l,float(sc)) for sc,s,l in combined]
def boundary_f1(pred:Iterable[int],truth:Iterable[int])->float:
    p={int(x) for x in pred if int(x)!=0}; t={int(x) for x in truth if int(x)!=0}
    if not p and not t: return 1.0
    if not p or not t: return 0.0
    tp=len(p&t); pr=tp/len(p); rc=tp/len(t); return 0.0 if pr+rc==0 else 2*pr*rc/(pr+rc)

@njit(cache=True,nogil=True)
def _rng_step(s:np.uint64)->np.uint64:
    s^=s>>np.uint64(12); s^=s<<np.uint64(25); s^=s>>np.uint64(27); return s*np.uint64(2685821657736338717)
@njit(cache=True,nogil=True)
def _rng_int(s:np.uint64,upper:int)->tuple[np.uint64,int]:
    s=_rng_step(s); return s,int(s%np.uint64(upper))
@njit(cache=True,nogil=True)
def _rng_float(s:np.uint64)->tuple[np.uint64,float]:
    s=_rng_step(s); return s,float(s>>np.uint64(11))*(1.0/9007199254740992.0)
@njit(cache=True,nogil=True)
def score_stateful(cipher:np.ndarray,phase:np.ndarray,inverses:np.ndarray,trigram:np.ndarray,unigram:np.ndarray)->float:
    n=cipher.shape[0]
    if n==0: return -1e300
    x=inverses[phase[0],cipher[0]]; score=.15*unigram[x]
    if n==1: return score
    y=inverses[phase[1],cipher[1]]; score+=.15*unigram[y]
    for i in range(2,n):
        z=inverses[phase[i],cipher[i]]; score+=trigram[x,y,z]+.15*unigram[z]; x=y; y=z
    return score
@njit(cache=True,nogil=True)
def anneal_stateful(cipher:np.ndarray,phase:np.ndarray,initial:np.ndarray,trigram:np.ndarray,unigram:np.ndarray,iterations:int,restarts:int,seed:int)->tuple[np.ndarray,float]:
    states,a=initial.shape; rs=np.uint64(seed if seed>0 else 1); best=initial.copy(); best_score=score_stateful(cipher,phase,best,trigram,unigram)
    for restart in range(restarts):
        inv=initial.copy()
        for _ in range(2+restart):
            rs,s=_rng_int(rs,states); rs,i=_rng_int(rs,a); rs,j=_rng_int(rs,a)
            if i!=j: tmp=inv[s,i]; inv[s,i]=inv[s,j]; inv[s,j]=tmp
        current=score_stateful(cipher,phase,inv,trigram,unigram)
        if current>best_score: best_score=current; best=inv.copy()
        temp=10.0; cooling=math.exp(math.log(.08/10.0)/max(1,iterations))
        for _ in range(iterations):
            rs,s=_rng_int(rs,states); rs,i=_rng_int(rs,a); rs,j=_rng_int(rs,a)
            if i==j: continue
            tmp=inv[s,i]; inv[s,i]=inv[s,j]; inv[s,j]=tmp; cand=score_stateful(cipher,phase,inv,trigram,unigram); delta=cand-current; accept=delta>=0
            if not accept: rs,u=_rng_float(rs); accept=u<math.exp(delta/max(temp,1e-9))
            if accept:
                current=cand
                if cand>best_score: best_score=cand; best=inv.copy()
            else: tmp=inv[s,i]; inv[s,i]=inv[s,j]; inv[s,j]=tmp
            temp*=cooling
    return best,best_score

def _initial_inverses(heads:list[int],phase:np.ndarray,period:int,language:core.LanguageData)->np.ndarray:
    a=len(language.alphabet); out=np.empty((period,a),dtype=np.int32)
    for s in range(period):
        subset=[int(heads[i]) for i in range(len(heads)) if int(phase[i])==s]; out[s]=np.asarray(mono.frequency_key(subset if subset else heads,language),dtype=np.int32)
    return out
def decode_stateful(heads:list[int],phase:np.ndarray,inverses:np.ndarray)->list[int]:
    return [int(inverses[int(phase[i]),int(x)]) for i,x in enumerate(heads)]
def stateful_mdl_score(raw:float,period:int,n:int,a:int)->float:
    return raw-.5*max(0,period-1)*max(1,a-1)*math.log(max(2,n))
def solve_head_stream(heads:list[int],head_line_starts:list[int],language:core.LanguageData,model:tuple[np.ndarray,np.ndarray],iterations:int,restarts:int,seed:int)->HeadSolution:
    if heads and max(heads)>=len(language.alphabet): raise ValueError("cipher head alphabet exceeds candidate plaintext alphabet")
    cipher=np.asarray(heads,dtype=np.int32); trigram,unigram=model; candidates=[]
    for mode in MODES:
        for period in CANDIDATE_PERIODS:
            ph=_phase(len(heads),period,mode,head_line_starts or [0]); initial=_initial_inverses(heads,ph,period,language); inv,raw=anneal_stateful(cipher,ph,initial,trigram,unigram,iterations,restarts,int(core.stable_seed("svt-v01-orderfree",seed,mode,period)&0x7fffffffffffffff)); pred=decode_stateful(heads,ph,inv)
            candidates.append(HeadSolution(mode,period,float(stateful_mdl_score(float(raw),period,len(heads),len(language.alphabet))),float(raw),pred,[[int(x) for x in row] for row in inv]))
    return max(candidates,key=lambda x:x.score)
def solve_surface(surface:list[int],line_starts:list[int],language:core.LanguageData,model:tuple[np.ndarray,np.ndarray],iterations:int=80000,restarts:int=8,beam:int=SEGMENTATION_BEAM,boundary_weight:float=BOUNDARY_WEIGHT,seed:int=1)->SurfaceSolution:
    paths=top_segmentations(surface,line_starts,len(language.alphabet),beam)
    if not paths: raise RuntimeError("boundary lattice produced no complete segmentation")
    cand=[]
    for rank,path in enumerate(paths):
        heads=[surface[i] for i in path.starts]; sol=solve_head_stream(heads,path.head_line_starts,language,model,iterations,restarts,int(core.stable_seed("svt-surface",seed,rank))); hn=sol.score/max(1,len(heads)); bn=path.score/max(1,len(surface)); cand.append(SurfaceSolution(path,sol,float(hn+boundary_weight*bn),float(hn),float(bn)))
    return max(cand,key=lambda x:x.joint_score)
def levenshtein_distance(a:list[int],b:list[int])->int:
    if len(a)<len(b): a,b=b,a
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]
def sequence_recovery(truth:list[int],prediction:list[int])->float:
    return 1.0-levenshtein_distance(truth,prediction)/max(1,len(truth),len(prediction))
def solve_svt_trial(trial:SVTTrial,language:core.LanguageData,model:tuple[np.ndarray,np.ndarray],iterations:int=80000,restarts:int=8,beam:int=SEGMENTATION_BEAM,boundary_weight:float=BOUNDARY_WEIGHT)->dict[str,Any]:
    sel=solve_surface(trial.surface,trial.surface_line_starts,language,model,iterations,restarts,beam,boundary_weight,trial.head.seed); paths=top_segmentations(trial.surface,trial.surface_line_starts,len(language.alphabet),beam)
    return {"family":trial.family,"iso":trial.head.iso,"split":trial.head.split,"length":trial.head.length,"replicate":trial.head.replicate,"surface_length":len(trial.surface),"true_mode":trial.head.mode,"true_period":trial.head.period,"selected_mode":sel.head_solution.mode,"selected_period":sel.head_solution.period,"mode_correct":sel.head_solution.mode==trial.head.mode,"period_correct":sel.head_solution.period==trial.head.period,"structure_correct":sel.head_solution.mode==trial.head.mode and sel.head_solution.period==trial.head.period,"boundary_f1":boundary_f1(sel.path.starts,trial.head_positions),"best_lattice_boundary_f1":max(boundary_f1(p.starts,trial.head_positions) for p in paths),"recovery":sequence_recovery(trial.head.plain,sel.head_solution.prediction),"predicted_plain_length":len(sel.head_solution.prediction),"true_plain_length":len(trial.head.plain),"joint_score":sel.joint_score,"head_score_per_symbol":sel.head_score_per_symbol,"boundary_score_per_surface":sel.boundary_score_per_surface,"predicted_starts":sel.path.starts,"head_line_starts":sel.path.head_line_starts}
def solve_true_heads(trial:SVTTrial,language:core.LanguageData,model:tuple[np.ndarray,np.ndarray],iterations:int,restarts:int)->dict[str,Any]:
    heads=[trial.surface[i] for i in trial.head_positions]; sol=solve_head_stream(heads,trial.head.line_starts,language,model,iterations,restarts,trial.head.seed)
    return {"family":trial.family,"iso":trial.head.iso,"split":trial.head.split,"length":trial.head.length,"replicate":trial.head.replicate,"recovery":sequence_recovery(trial.head.plain,sol.prediction),"selected_mode":sol.mode,"selected_period":sol.period,"mode_correct":sol.mode==trial.head.mode,"period_correct":sol.period==trial.head.period,"structure_correct":sol.mode==trial.head.mode and sol.period==trial.head.period}
def summarize_joint(rows:list[dict[str,Any]])->dict[str,Any]:
    rec=[float(r["recovery"]) for r in rows]; bf=[float(r["boundary_f1"]) for r in rows]
    return {"trials":len(rows),"recovery":{"mean":statistics.fmean(rec),"median":statistics.median(rec),"minimum":min(rec),"at_least_70":sum(v>=.70 for v in rec),"at_least_80":sum(v>=.80 for v in rec)},"boundary":{"mean_f1":statistics.fmean(bf),"median_f1":statistics.median(bf)},"structure_accuracy":statistics.fmean(bool(r["structure_correct"]) for r in rows),"mode_accuracy":statistics.fmean(bool(r["mode_correct"]) for r in rows),"period_accuracy":statistics.fmean(bool(r["period_correct"]) for r in rows)}

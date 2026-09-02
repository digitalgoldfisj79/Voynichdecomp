# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2", "triton>=3.0"]
# ///
"""Execution-only patch 2 for frozen V10 evolutionary runner.

The first binding production-path run exposed only an efficiency defect: the
same exact 4,096-key elite set was being fully copied to CPU and lexicographically
sorted after every 32,768-candidate batch and merge. This patch preserves the
frozen exact score cutoff and lexicographic tie rule, but applies lexicographic
work only when a cutoff tie must be resolved and once to order the final elite
set at the end of each generation. Candidate generation, scores, seeds,
population size, mutation schedule, fresh fraction, chains, data and gates are
unchanged. It also includes patch1's Triton constexpr literal fix.
"""
import importlib.util, sys, urllib.request
from pathlib import Path
import numpy as np, torch
URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v10-terminal-identifiability-20260901/experiments/vbm_v10_terminal/vbm_v10_stage_a_gpu_evolution.py'
with urllib.request.urlopen(URL,timeout=120) as r: src=r.read().decode('utf-8')
old='maps+offs*STRIDE+KB+surf'; new='maps+offs*STRIDE+30+surf'
if old not in src: raise RuntimeError('expected frozen source fragment missing')
src=src.replace(old,new,1)
p=Path('/tmp/vbm_v10_stage_a_gpu_evolution_fixed2.py'); p.write_text(src,encoding='utf-8')
spec=importlib.util.spec_from_file_location('v10ev_fixed2',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def lex_np(keys, scores=None):
    cols=tuple(keys[:,j] for j in range(keys.shape[1]-1,-1,-1))
    if scores is None: return np.lexsort(cols)
    return np.lexsort(cols+(-scores,))


def choose_top_exact(scores,maps,k):
    n=int(scores.numel())
    if n<=k: return scores,maps
    vals,idx=torch.topk(scores,k,largest=True,sorted=True)
    cut=vals[-1]
    nge=int((scores>=cut).sum().item())
    if nge==k: return scores[idx],maps[idx]
    gt=torch.nonzero(scores>cut,as_tuple=False).flatten(); need=k-int(gt.numel())
    eq=torch.nonzero(scores==cut,as_tuple=False).flatten()
    eqm=maps[eq].detach().cpu().numpy(); od=lex_np(eqm)
    take=eq[torch.tensor(od[:need],device=eq.device,dtype=torch.long)]
    sel=torch.cat([gt,take])
    return scores[sel],maps[sel]


def merge_top_exact(cur_scores,cur_maps,new_scores,new_maps,k=None):
    if k is None:k=m.ELITE
    if cur_scores is None:return choose_top_exact(new_scores,new_maps,k)
    return choose_top_exact(torch.cat([cur_scores,new_scores]),torch.cat([cur_maps,new_maps]),k)


def score_population_exact(engine,make_batch,previous_scores=None,previous_maps=None):
    gs=previous_scores; gm=previous_maps; base=0
    while base<m.POP:
        q=min(m.BATCH,m.POP-base); maps=make_batch(base,q); sc=engine.score(maps)
        ls,lm=choose_top_exact(sc,maps,min(m.ELITE,q)); gs,gm=merge_top_exact(gs,gm,ls,lm,m.ELITE); base+=q
    # One exact ranking operation per generation: score descending, then 126-int lex ascending.
    sc=gs.detach().cpu().numpy(); km=gm.detach().cpu().numpy(); od=lex_np(km,sc)
    ix=torch.tensor(od,device=gm.device,dtype=torch.long)
    return gs[ix],gm[ix]

m.choose_top=choose_top_exact
m.merge_top=merge_top_exact
m.score_population=score_population_exact
m.main()

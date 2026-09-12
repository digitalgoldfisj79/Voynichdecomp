#!/usr/bin/env python3
"""Blind solver worker for the full Yiddish M0 control.

Inputs contain ciphertext and the registered Yiddish BUILD language model corpus only.
This worker never receives plaintext, keys, root seeds, or the private truth bundle.
It is shardable for parallel execution.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
V03=REPO/'experiments'/'yiddish_qualification_v03'
V1B=REPO/'experiments'/'yiddish_qualification_v1b'
sys.path.insert(0,str(V03)); sys.path.insert(0,str(V1B))
import run_v03_search_repair as v03  # noqa: E402
import finite_panel_r_v02 as base  # noqa: E402

A=base.A; ALPHABET=base.ALPHABET
CFG={'id':'S1E_FULL_FROZEN','restarts':8,'steps':5000,'greedy_passes':60}


def h64(s): return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'big')
def append(row,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f: f.write(json.dumps(row,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())

def relabel_words(words,perm): return [''.join(ALPHABET[perm[ord(c)-97]] for c in w) for w in words]

def canonicalise_fit(words):
 seen=[]; S=set()
 for w in words:
  for c in w:
   if c=='~': continue
   oi=ord(c)-97
   if oi not in S: S.add(oi); seen.append(oi)
 remaining=[i for i in range(A) if i not in S]; order=seen+remaining
 o2c={o:c for c,o in enumerate(order)}
 out=[]
 for w in words:
  out.append(''.join('~' if x=='~' else ALPHABET[o2c[ord(x)-97]] for x in w))
 return out,o2c,len(seen)

def solve_equivariant(fit,lp,uni,seed):
 canon,o2c,nseen=canonicalise_fit(fit)
 cmap,obj=v03.blind_solve_budget(canon,lp,uni,seed,CFG)
 out=[0]*A
 for orig,canon_id in o2c.items(): out[orig]=cmap[canon_id]
 return out,obj,nseen

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--public-dir',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--shard-index',type=int,default=0); ap.add_argument('--shard-count',type=int,default=1)
 a=ap.parse_args();
 if not (0<=a.shard_index<a.shard_count): raise SystemExit('invalid shard')
 manifest=json.loads((a.public_dir/'pre_manifest.json').read_text())
 if manifest.get('voynich_loaded') or manifest.get('target_access_allowed'): raise RuntimeError('target boundary violated')
 training=json.loads((a.public_dir/'training_words.json').read_text())
 normal_lp,normal_uni=base.lm_from_words(training); lm_cache={None:(normal_lp,normal_uni)}
 out=a.out
 if out.exists(): out.unlink()
 n=0
 for line in (a.public_dir/'cases.jsonl').open(encoding='utf-8'):
  case=json.loads(line); cid=case['case_id']
  if h64(cid)%a.shard_count != a.shard_index: continue
  rel=case.get('lm_relabel'); key=None if rel is None else tuple(rel)
  if key not in lm_cache:
   tw=relabel_words(training,list(key)); lm_cache[key]=base.lm_from_words(tw)
  lp,uni=lm_cache[key]
  search_seed=h64('yiddish-full-control|'+cid+'|S1E')
  t=time.time(); mapping,obj,nseen=solve_equivariant(case['fit_cipher'],lp,uni,search_seed); dt=time.time()-t
  append({'case_id':cid,'certificate':case['certificate'],'family':case['family'],'mapping':mapping,'objective':obj,'runtime_s':dt,'fit_symbols_seen':nseen,'search_seed':search_seed,'solver_config':CFG},out); n+=1
 print(json.dumps({'status':'SOLVER_SHARD_COMPLETE','shard_index':a.shard_index,'shard_count':a.shard_count,'cases_solved':n,'output':str(out)},indent=2))

if __name__=='__main__': main()

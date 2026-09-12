#!/usr/bin/env python3
"""Yiddish qualification v1b — repaired fresh-source finite-panel R benchmark.

Scope is intentionally narrow: blind M0 recovery on Penn's lossy Romanized
historical-Yiddish representation. It is NOT the primary diplomatic representation,
NOT language discrimination (L), NOT target transfer (T), and never loads Voynich.

Repairs relative to diagnostic v01:
- fresh confirmation sources;
- separately hashed encoder and independently written decoder;
- round-trip assertions before blind solver invocation;
- prospective exact/near passage-overlap audit with conservative source exclusion;
- hidden planting root with pre-outcome SHA-256 commitment and post-run reveal;
- atomic pickle checkpoints and complete trial rows.
"""
from __future__ import annotations
import hashlib, hmac, json, math, os, pickle, random, re, secrets, subprocess, tempfile, time
from pathlib import Path
from collections import defaultdict
from encoder_v02 import encode_words
from independent_decoder_v02 import invert_permutation, decode_words, assert_roundtrip_non_erased

PROTOCOL_VERSION="yiddish_qualification_v1_20260912"
RUN_VERSION="finite_panel_r_normalized_v02"
PPCHY_URL="https://github.com/beatrice57/penn-parsed-corpus-of-historical-yiddish.git"
ALPHABET=tuple("abcdefghijklmnopqrstuvwxyz"); A=len(ALPHABET); BOUNDARY=A
BUFFER_WORDS=32; LENGTHS=(512,2048); ERASURE_RATES=(0.0,0.01); N_KEYS=32
ATOM_PASS=.90; WORD_PASS=.80; WORK_TRIAL_PASS=29
SEARCH_RESTARTS=3; SEARCH_STEPS=2500; GREEDY_PASSES=40; ALPHA=.25
TRAIN_MIN_YEAR=1600; TRAIN_MAX_YEAR=1750
DEVELOPMENT_WORKS=("1507w-bovo.psd",)
CONFIRMATION_FULL=("1697e-purim.psd",)
CONFIRMATION_SHORT=("1620e-lev-tov-1.psd",)
HELDOUT=set(DEVELOPMENT_WORKS+CONFIRMATION_FULL+CONFIRMATION_SHORT)

HERE=Path(__file__).resolve().parent; OUT=HERE/"run_output_v02"; CHECKPOINT=OUT/"checkpoint.pkl"; ROWS=OUT/"trial_rows.jsonl"; MANIFEST=OUT/"manifest.json"; SUMMARY=OUT/"summary.json"
LEAF_RE=re.compile(r"\(([A-Z][A-Z0-9$=*-]*)\s+([^()\s]+)\)"); YEAR_RE=re.compile(r"^(\d{4})")

def sha(b): return hashlib.sha256(b).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()
def atomic_pickle(obj,path):
 path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=str(path.parent))
 try:
  with os.fdopen(fd,'wb') as f: pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL); f.flush(); os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)
def atomic_json(obj,path):
 path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=str(path.parent),text=True)
 try:
  with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)
def public_seed(*parts): return int.from_bytes(hashlib.sha256('|'.join([PROTOCOL_VERSION,RUN_VERSION,*map(str,parts)]).encode()).digest()[:8],'big')
def secret_seed(root,*parts): return int.from_bytes(hmac.new(root,'|'.join(map(str,parts)).encode(),hashlib.sha256).digest()[:8],'big')

def normalize_leaf(raw):
 if raw.startswith('*') or raw in {'0','-NONE-'}: return []
 raw=raw.replace('@','').split('^',1)[0]; out=[]
 for part in raw.split('_'):
  w=''.join(c for c in part.lower() if 'a'<=c<='z')
  if w: out.append(w)
 return out
def extract_words(path):
 out=[]
 for tag,raw in LEAF_RE.findall(Path(path).read_text(encoding='utf-8',errors='replace')):
  if tag.startswith(('ID','CODE','PUNC')): continue
  out.extend(normalize_leaf(raw))
 return out
def year_from_name(n):
 m=YEAR_RE.match(n); return int(m.group(1)) if m else None

def ensure_corpus(root):
 repo=root/'ppchy'
 if not repo.exists(): subprocess.run(['git','clone','--depth','1',PPCHY_URL,str(repo)],check=True)
 commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 return repo,commit

def ngrams(words,n): return {tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1))}
def leakage_audit(data,candidates,heldout_names):
 held={n:extract_words(data/n) for n in heldout_names}
 held8={n:ngrams(w,8) for n,w in held.items()}; held5={n:ngrams(w,5) for n,w in held.items()}
 kept=[]; excluded=[]; overlaps5=[]
 for p in candidates:
  w=extract_words(p); g8=ngrams(w,8); g5=ngrams(w,5)
  bad=[]
  for hn in heldout_names:
   k=len(g8 & held8[hn])
   if k: bad.append((hn,k))
   k5=len(g5 & held5[hn])
   if k5: overlaps5.append({'training':p.name,'heldout':hn,'shared_5gram_types':k5})
  if bad: excluded.append({'training':p.name,'shared_8gram':bad})
  else: kept.append(p)
 return kept,excluded,overlaps5

def build_training(data):
 cand=[]
 companion=set(HELDOUT)
 for n in list(HELDOUT): companion.add(n[:-4]+'-preface.psd')
 for p in sorted(data.glob('*.psd')):
  y=year_from_name(p.name)
  if y is not None and TRAIN_MIN_YEAR<=y<=TRAIN_MAX_YEAR and p.name not in companion: cand.append(p)
 kept,excluded,overlaps5=leakage_audit(data,cand,sorted(HELDOUT))
 words=[]; hashes={}
 for p in kept: words.extend(extract_words(p)); hashes[p.name]=file_sha(p)
 if not words: raise RuntimeError('empty training corpus')
 return kept,words,hashes,excluded,overlaps5

def lm_from_words(words):
 counts=[[ALPHA for _ in range(A+1)] for _ in range(A+1)]; uni=[ALPHA]*A
 for w in words:
  ids=[ord(c)-97 for c in w]
  for x in ids: uni[x]+=1
  seq=[BOUNDARY]+ids+[BOUNDARY]
  for x,y in zip(seq,seq[1:]): counts[x][y]+=1
 lp=[]
 for row in counts:
  s=sum(row); lp.append([math.log(v/s) for v in row])
 return lp,uni

def cipher_counts(words):
 C=[[0.0 for _ in range(A+1)] for _ in range(A+1)]; uni=[0]*A; pairs=0
 for w in words:
  ids=[]
  for c in w:
   if c=='~': ids.append(None)
   else: x=ord(c)-97; ids.append(x); uni[x]+=1
  seq=[BOUNDARY]+ids+[BOUNDARY]
  for x,y in zip(seq,seq[1:]):
   if x is None or y is None: continue
   C[x][y]+=1; pairs+=1
 if not pairs: raise RuntimeError('no eligible pairs')
 inv=1/pairs
 for i in range(A+1):
  for j in range(A+1): C[i][j]*=inv
 return C,uni

def mapping_score(C,lp,m):
 s=0.0
 for i in range(A+1):
  mi=BOUNDARY if i==BOUNDARY else m[i]
  for j in range(A+1):
   mj=BOUNDARY if j==BOUNDARY else m[j]; s+=C[i][j]*lp[mi][mj]
 return s

def swap_delta(C,lp,m,a,b):
 ma,mb=m[a],m[b]; d=0.0
 for k in range(A+1):
  if k in (a,b): continue
  mk=BOUNDARY if k==BOUNDARY else m[k]
  d+=C[a][k]*(lp[mb][mk]-lp[ma][mk])+C[b][k]*(lp[ma][mk]-lp[mb][mk])
  d+=C[k][a]*(lp[mk][mb]-lp[mk][ma])+C[k][b]*(lp[mk][ma]-lp[mk][mb])
 d+=C[a][a]*(lp[mb][mb]-lp[ma][ma])+C[b][b]*(lp[ma][ma]-lp[mb][mb])
 d+=C[a][b]*(lp[mb][ma]-lp[ma][mb])+C[b][a]*(lp[ma][mb]-lp[mb][ma])
 return d
def frequency_initial(cuni,tuni):
 cr=sorted(range(A),key=lambda x:(-cuni[x],x)); pr=sorted(range(A),key=lambda x:(-tuni[x],x)); m=[0]*A
 for c,p in zip(cr,pr): m[c]=p
 return m

def blind_solve(fit_cipher,lp,tuni,search_seed):
 C,cuni=cipher_counts(fit_cipher); base=frequency_initial(cuni,tuni); rng=random.Random(search_seed); best=None; bs=-1e100
 for r in range(SEARCH_RESTARTS):
  m=base.copy(); rr=random.Random(public_seed('restart',search_seed,r))
  for _ in range(8*r): a,b=rr.sample(range(A),2); m[a],m[b]=m[b],m[a]
  s=mapping_score(C,lp,m)
  if s>bs: bs=s; best=m.copy()
  for step in range(SEARCH_STEPS):
   a,b=rng.sample(range(A),2); d=swap_delta(C,lp,m,a,b); frac=step/max(1,SEARCH_STEPS-1); temp=.006*(1-frac)+.00003
   if d>=0 or rng.random()<math.exp(max(-50,d/temp)):
    m[a],m[b]=m[b],m[a]; s+=d
    if s>bs: bs=s; best=m.copy()
  m=best.copy(); s=bs
  for _ in range(GREEDY_PASSES):
   bd=0.0; bp=None
   for a in range(A):
    for b in range(a+1,A):
     d=swap_delta(C,lp,m,a,b)
     if d>bd+1e-12: bd=d; bp=(a,b)
   if bp is None: break
   a,b=bp; m[a],m[b]=m[b],m[a]; s+=bd
   if s>bs: bs=s; best=m.copy()
 return best,bs

def make_key(seed):
 r=random.Random(seed); p=list(range(A)); r.shuffle(p); return p

def score_recovery(truth,decoded):
 correct=eligible=total=erased=ew=okword=0
 for t,d in zip(truth,decoded):
  total+=len(t)
  for a,b in zip(t,d):
   if b=='~': erased+=1
   else: eligible+=1; correct+=int(a==b)
  if '~' not in d: ew+=1; okword+=int(t==d)
 return {'atom_recovery':correct/eligible if eligible else None,'word_recovery':okword/ew if ew else None,'atom_coverage':eligible/total if total else None,'eligible_atoms':eligible,'total_atoms':total,'eligible_words':ew,'total_words':len(truth),'erased_atoms':erased}
def append_row(row):
 with ROWS.open('a',encoding='utf-8') as f: f.write(json.dumps(row,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
def trial_id(role,work,n,e,k): return f'{role}|{work}|{n}|{e:.3f}|{k:02d}'

def main():
 OUT.mkdir(parents=True,exist_ok=True); src=OUT/'sources'; src.mkdir(exist_ok=True); repo,commit=ensure_corpus(src); data=repo/'data'
 # Separate source code hashes are frozen before outcomes.
 source_hashes={x:file_sha(HERE/x) for x in ('finite_panel_r_v02.py','encoder_v02.py','independent_decoder_v02.py')}
 train_files,train_words,train_hashes,leak_exclusions,overlaps5=build_training(data); lp,tuni=lm_from_words(train_words)
 held_meta={}
 roles=[('development',DEVELOPMENT_WORKS),('confirmation_full',CONFIRMATION_FULL),('confirmation_short',CONFIRMATION_SHORT)]
 for role,names in roles:
  for n in names:
   ws=extract_words(data/n); held_meta[n]={'role':role,'words_pipeline_units':len(ws),'sha256':file_sha(data/n),'eligible_512':len(ws)>=1056,'eligible_2048':len(ws)>=4128}
 # Secret planting root is generated and committed before any blind output exists.
 cp={}
 if CHECKPOINT.exists():
  with CHECKPOINT.open('rb') as f: cp=pickle.load(f)
 root=cp.get('plant_root') or secrets.token_bytes(32); commitment=sha(root)
 cp.setdefault('plant_root',root); cp.setdefault('done',{}); cp['state']='EXECUTION_FROZEN_R_NORMALIZED'; atomic_pickle(cp,CHECKPOINT)
 manifest={'protocol_version':PROTOCOL_VERSION,'run_version':RUN_VERSION,'scope':'R_ONLY_NORMALIZED_FINITE_PANEL_NO_L_NO_T','target_access_allowed':False,'voynich_loaded':False,
  'representation':'Penn/YIVO-style Romanisation, explicitly lossy, then a-z only; secondary normalized representation',
  'ppchy_commit':commit,'source_code_hashes':source_hashes,'plant_root_commitment_sha256':commitment,'plant_root_revealed_pre_outcome':False,
  'conditions':{'lengths':LENGTHS,'erasure_rates':ERASURE_RATES,'keys_per_cell':N_KEYS,'buffer_words':BUFFER_WORDS},
  'thresholds':{'atom':ATOM_PASS,'word':WORD_PASS,'cell_successes_required':WORK_TRIAL_PASS},'heldout':held_meta,
  'training_files':[p.name for p in train_files],'training_hashes':train_hashes,'training_word_count':len(train_words),
  'leakage_audit':{'exact_8gram_training_sources_excluded':leak_exclusions,'shared_5gram_diagnostics':overlaps5},
  'search':{'restarts':SEARCH_RESTARTS,'steps':SEARCH_STEPS,'greedy_passes':GREEDY_PASSES,'objective':'conditional character-bigram cross-entropy'},
  'fresh_confirmation_note':'v01 confirmation sources 1579e-shir and 1589e-ester are consumed and not reused in v02',
  'limitations':['not diplomatic source atoms','no L German/Hebrew discriminator','finite panel','historical population inference prohibited']}
 atomic_json(manifest,MANIFEST)
 # Fixed encoder/independent-decoder fixture before blind trials.
 fixture=['abc','zebra','mish']; fkey=list(range(A)); fkey=fkey[5:]+fkey[:5]
 fenc,_,_=encode_words(fixture,fkey,0.0,random.Random(1)); fdec=decode_words(fenc,invert_permutation(fkey)); assert_roundtrip_non_erased(fixture,fdec)
 done=cp['done']
 for role,names in roles:
  for name in names:
   truth_all=extract_words(data/name)
   for n in LENGTHS:
    if len(truth_all)<2*n+BUFFER_WORDS: continue
    fit_truth=truth_all[:n]; audit_truth=truth_all[n+BUFFER_WORDS:n+BUFFER_WORDS+n]
    for er in ERASURE_RATES:
     for k in range(N_KEYS):
      tid=trial_id(role,name,n,er,k)
      if tid in done: continue
      key_seed=secret_seed(root,'plant',role,name,n,er,k); key=make_key(key_seed)
      ef=random.Random(secret_seed(root,'erase_fit',role,name,n,er,k)); ea=random.Random(secret_seed(root,'erase_audit',role,name,n,er,k))
      fit_cipher,_,_=encode_words(fit_truth,key,er,ef); audit_cipher,_,_=encode_words(audit_truth,key,er,ea)
      oracle=invert_permutation(key)
      # Round-trip is asserted before solver invocation, but key/oracle never enters solver arguments.
      assert_roundtrip_non_erased(fit_truth,decode_words(fit_cipher,oracle)); assert_roundtrip_non_erased(audit_truth,decode_words(audit_cipher,oracle))
      search_seed=public_seed('search',role,name,n,er,k); t0=time.time(); returned,ret_obj=blind_solve(fit_cipher,lp,tuni,search_seed); runtime=time.time()-t0
      # Only now reveal truth to scoring/oracle diagnostics.
      C,_=cipher_counts(fit_cipher); oracle_obj=mapping_score(C,lp,oracle); decoded=decode_words(audit_cipher,returned); rec=score_recovery(audit_truth,decoded)
      passed=rec['atom_recovery'] is not None and rec['word_recovery'] is not None and rec['atom_recovery']>=ATOM_PASS and rec['word_recovery']>=WORD_PASS
      row={'trial_id':tid,'role':role,'work':name,'fit_words':n,'audit_words':n,'erasure_rate':er,'key_index':k,'pass':passed,'runtime_s':runtime,
       'returned_objective':ret_obj,'oracle_objective':oracle_obj,'oracle_minus_returned':oracle_obj-ret_obj,'search_miss_witness':oracle_obj>ret_obj+1e-10,**rec,'search_seed':search_seed}
      append_row(row); done[tid]=row; cp['state']='RUNNING_R_NORMALIZED_V02'; atomic_pickle(cp,CHECKPOINT)
 cells={}
 for r in done.values():
  key='|'.join(map(str,(r['role'],r['work'],r['fit_words'],r['erasure_rate']))); x=cells.setdefault(key,{'n':0,'pass':0,'search_miss':0,'atom_sum':0.0,'word_sum':0.0}); x['n']+=1; x['pass']+=int(r['pass']); x['search_miss']+=int(r['search_miss_witness']); x['atom_sum']+=r['atom_recovery']; x['word_sum']+=r['word_recovery']
 for x in cells.values(): x['mean_atom_recovery']=x.pop('atom_sum')/x['n']; x['mean_word_recovery']=x.pop('word_sum')/x['n']; x['cell_pass']=x['n']==N_KEYS and x['pass']>=WORK_TRIAL_PASS
 works={}
 for role,names in roles:
  for name in names:
   eligible=[]
   for n in LENGTHS:
    if held_meta[name][f'eligible_{n}']:
     for er in ERASURE_RATES: eligible.append(cells.get('|'.join(map(str,(role,name,n,er)))))
   full_matrix=held_meta[name]['eligible_2048']
   works[name]={'role':role,'eligible_cells':len(eligible),'all_eligible_cells_pass':bool(eligible) and all(x and x['cell_pass'] for x in eligible),'full_matrix_eligible':full_matrix,'full_matrix_R_pass':full_matrix and all(x and x['cell_pass'] for x in eligible)}
 reveal=OUT/'plant_root_reveal.hex'; reveal.write_text(root.hex()+'\n',encoding='ascii')
 summary={'status':'R_NORMALIZED_V02_COMPLETE','scope':'FINITE_PANEL_ONLY_NO_L_NO_T_NO_VOYNICH_INFERENCE','cells':cells,'works':works,
  'headline_bound':'exact 32-key outcomes conditional on named works; no population confidence interval','null_sd':'not applicable: R uses exact recovery thresholds, not an effect/null-SD decision','plant_root_commitment_sha256':commitment,'plant_root_reveal_sha256':sha(reveal.read_bytes()),'manifest_sha256':file_sha(MANIFEST)}
 atomic_json(summary,SUMMARY); cp['state']='R_NORMALIZED_V02_COMPLETE'; atomic_pickle(cp,CHECKPOINT); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()

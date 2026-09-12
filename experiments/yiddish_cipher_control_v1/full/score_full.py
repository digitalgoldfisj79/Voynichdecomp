#!/usr/bin/env python3
"""Adjudicate the blinded full Yiddish M0 control after solver commitments exist."""
from __future__ import annotations

import argparse, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
V1B=REPO/'experiments'/'yiddish_qualification_v1b'
sys.path.insert(0,str(V1B))
import finite_panel_r_v02 as base  # noqa: E402
from independent_decoder_v02 import decode_words  # noqa: E402

A=base.A; ALPHABET=base.ALPHABET
ATOM_PASS=.90; WORD_PASS=.80; CELL_PASS=29; FAMILY_ALPHA=.05/6


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def read_jsonl(p):
 with p.open(encoding='utf-8') as f:
  for line in f:
   if line.strip(): yield json.loads(line)
def write_json(x,p): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def cp_upper(k,n,alpha=FAMILY_ALPHA):
 if n<=0:return None
 if k>=n:return 1.0
 def cdf(p): return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
 lo,hi=0.,1.
 for _ in range(90):
  m=(lo+hi)/2
  if cdf(m)>alpha:lo=m
  else:hi=m
 return (lo+hi)/2
def cp_lower(k,n,alpha=.05):
 if n<=0:return None
 if k<=0:return 0.
 def sf(p): return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k,n+1))
 lo,hi=0.,1.
 for _ in range(90):
  m=(lo+hi)/2
  if sf(m)<alpha:lo=m
  else:hi=m
 return (lo+hi)/2

def relabel_words(words,perm): return [''.join(ALPHABET[perm[ord(c)-97]] for c in w) for w in words]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--public-dir',type=Path,required=True); ap.add_argument('--private-dir',type=Path,required=True); ap.add_argument('--results-dir',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
 a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((a.public_dir/'pre_manifest.json').read_text()); training=json.loads((a.public_dir/'training_words.json').read_text())
 if manifest.get('voynich_loaded') or manifest.get('target_access_allowed'): raise RuntimeError('target boundary violation')
 root=bytes.fromhex((a.private_dir/'plant_root_reveal.hex').read_text().strip())
 root_ok=sha_bytes(root)==manifest['plant_root_commitment_sha256']
 cases={x['case_id']:x for x in read_jsonl(a.public_dir/'cases.jsonl')}; truth={x['case_id']:x for x in read_jsonl(a.private_dir/'truth.jsonl')}
 results={}
 for p in sorted(a.results_dir.glob('*.jsonl')):
  for r in read_jsonl(p):
   if r['case_id'] in results: raise RuntimeError('duplicate solver result '+r['case_id'])
   results[r['case_id']]=r
 complete=(set(cases)==set(truth)==set(results))

 lm_cache={None:base.lm_from_words(training)}
 rows=[]
 for cid,c in cases.items():
  t=truth[cid]; r=results.get(cid)
  if r is None: continue
  rel=c.get('lm_relabel'); lk=None if rel is None else tuple(rel)
  if lk not in lm_cache: lm_cache[lk]=base.lm_from_words(relabel_words(training,list(lk)))
  lp,uni=lm_cache[lk]
  dec=decode_words(c['audit_cipher'],r['mapping']); rec=base.score_recovery(t['audit_truth'],dec)
  positive=(rec['atom_recovery'] is not None and rec['word_recovery'] is not None and rec['atom_recovery']>=ATOM_PASS and rec['word_recovery']>=WORD_PASS)
  oracle=t.get('oracle'); oracle_obj=None; key_acc=None; search_miss=None; obj_misalign=None
  if oracle is not None:
   C,_=base.cipher_counts(c['fit_cipher']); oracle_obj=base.mapping_score(C,lp,oracle)
   key_acc=sum(int(x==y) for x,y in zip(r['mapping'],oracle))/A
   search_miss=oracle_obj>r['objective']+1e-12
   obj_misalign=(not positive and not search_miss)
  row={'case_id':cid,'certificate':c['certificate'],'family':c['family'],'work':c.get('work'),'length':c.get('length'),'erasure':c.get('erasure'),
       'pair_id':c.get('pair_id'),'variant':c.get('variant'),'positive_call':positive,'key_accuracy':key_acc,'returned_objective':r['objective'],'oracle_objective':oracle_obj,
       'oracle_minus_returned':None if oracle_obj is None else oracle_obj-r['objective'],'search_miss_witness':search_miss,'objective_misalignment_witness':obj_misalign,
       'fit_symbols_seen':r.get('fit_symbols_seen'),**rec}
  rows.append(row)
 with (a.out/'scored_rows.jsonl').open('w',encoding='utf-8') as f:
  for x in rows:f.write(json.dumps(x,sort_keys=True)+'\n')

 # C0/C1/C2
 c0='PASS' if root_ok and complete and manifest.get('private_truth_withheld_from_solver') and not manifest.get('voynich_loaded') else 'FAIL'
 c1='PASS' if all(x['shared_8gram_types']==0 for x in manifest['build_development_overlap']) and manifest['build_words']>=10000 else 'FAIL'
 c2='PASS' if manifest['kat_rows'] and all(x['pass'] for x in manifest['kat_rows']) else 'FAIL'

 # C3 power map. Only >=512 words and <=1% erasure define the registered primary qualification envelope; all other cells remain reported.
 c3rows=[x for x in rows if x['certificate']=='C3']; cells={}
 by=defaultdict(list)
 for x in c3rows:by[(x['work'],x['length'],x['erasure'])].append(x)
 for k,v in sorted(by.items()):
  s=sum(x['positive_call'] for x in v); cells['|'.join(map(str,k))]={'n':len(v),'successes':s,'lower95':cp_lower(s,len(v)),'cell_pass':len(v)==32 and s>=CELL_PASS,
    'mean_atom_recovery':sum(x['atom_recovery'] for x in v)/len(v),'mean_word_recovery':sum(x['word_recovery'] for x in v)/len(v),
    'search_miss_failures':sum(bool(x['search_miss_witness']) and not x['positive_call'] for x in v),'objective_misalignment_failures':sum(bool(x['objective_misalignment_witness']) for x in v)}
 primary=[z for key,z in cells.items() if int(key.split('|')[1])>=512 and float(key.split('|')[2])<=.01]
 c3status='PASS' if primary and all(z['cell_pass'] for z in primary) else 'FAIL'
 c3={'status':c3status,'cells':cells,'primary_cells':len(primary)}

 # C4a mechanism negatives; false positive = apparent recovery of original plaintext despite no single global key.
 c4a={}; by=defaultdict(list)
 for x in rows:
  if x['certificate']=='C4a':by[x['family']].append(x)
 for fam,v in sorted(by.items()):
  fp=sum(x['positive_call'] for x in v); ub=cp_upper(fp,len(v)); c4a[fam]={'n':len(v),'false_positive_calls':fp,'bonferroni_upper':ub,'pass':len(v)>=94 and ub<=.05}
 c4astatus='PASS' if c4a and all(z['pass'] for z in c4a.values()) else 'FAIL_OR_UNBOUNDED'

 # C4b language/nuisance diagnostics: a global M0 exists, so recovery is not a mechanism false positive.
 c4b={}; by=defaultdict(list)
 for x in rows:
  if x['certificate']=='C4b':by[(x['family'],x.get('work'))].append(x)
 for k,v in sorted(by.items(),key=lambda z:str(z[0])):
  s=sum(x['positive_call'] for x in v); c4b['|'.join(str(q) for q in k)]={'n':len(v),'recovery_successes':s,'rate':s/len(v),'lower95':cp_lower(s,len(v))}

 # C5 deterministic MRs plus paired stochastic MRs.
 det=manifest['deterministic_metamorphic_rows']; detpass=all(x['pass'] for x in det)
 pairgroups=defaultdict(dict)
 for x in rows:
  if x['certificate']=='C5' and x.get('pair_id'):pairgroups[(x['family'],x['pair_id'])][x['variant']]=x
 pair_results=[]
 for (fam,pid),d in sorted(pairgroups.items()):
  if set(d)!={'base','relabel'}: pair_results.append({'family':fam,'pair_id':pid,'pass':False,'reason':'missing variant'}); continue
  x,y=d['base'],d['relabel']
  # Paired recovery should be invariant to arbitrary label names after corresponding representation transformation.
  ok=(x['positive_call']==y['positive_call'] and abs(x['atom_recovery']-y['atom_recovery'])<=1e-12 and abs(x['word_recovery']-y['word_recovery'])<=1e-12)
  pair_results.append({'family':fam,'pair_id':pid,'pass':ok,'base_atom':x['atom_recovery'],'other_atom':y['atom_recovery'],'base_word':x['word_recovery'],'other_word':y['word_recovery']})
 famcounts=defaultdict(list)
 for x in pair_results:famcounts[x['family']].append(x)
 c5families={fam:{'n':len(v),'passes':sum(x['pass'] for x in v),'pass':len(v)>=32 and all(x['pass'] for x in v)} for fam,v in famcounts.items()}
 c5status='PASS' if detpass and c5families and all(z['pass'] for z in c5families.values()) else 'FAIL'
 c5={'status':c5status,'deterministic':det,'paired_families':c5families,'pair_failures':[x for x in pair_results if not x['pass']]}

 mechanism_req=[c0,c1,c2,c3status,c4astatus,c5status]
 mechanism_pass=all(x=='PASS' for x in mechanism_req)
 overall='C0_C5_M0_MECHANISM_PASS__LANGUAGE_L_REQUIRED__C6_NOT_RUN__C7_SEALED' if mechanism_pass else 'M0_INSTRUMENT_UNQUALIFIED__TARGET_SEALED'
 summary={'overall_status':overall,'certificates':{'C0':c0,'C1':c1,'C2':c2,'C3':c3,'C4a':{'status':c4astatus,'families':c4a},'C4b_language_diagnostic':c4b,'C5':c5,'C6':'NOT_RUN','C7':'SEALED'},
          'root_commitment_verified':root_ok,'solver_results_complete':complete,'n_cases':len(cases),'n_results':len(results),
          'interpretation':{'mechanism':'C0-C5 can qualify only fixed monoalphabetic-substitution recovery in the declared secondary representation and operating envelope.',
                            'language':'C4b is diagnostic. Yiddish identity still requires a separately qualified L certificate; current historical L remains unresolved.',
                            'transfer':'Primary-script historical transfer is C6 and has not been run.','target':'Voynich remains sealed and receives no inference from this control.'}}
 write_json(summary,a.out/'summary.json'); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()

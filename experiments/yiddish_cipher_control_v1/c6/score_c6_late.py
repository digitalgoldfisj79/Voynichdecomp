#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
V1B=REPO/'experiments'/'yiddish_qualification_v1b'
sys.path.insert(0,str(V1B))
import finite_panel_r_v02 as base  # noqa:E402
from independent_decoder_v02 import decode_words  # noqa:E402

ATOM_PASS=.90; WORD_PASS=.80; CELL_PASS=29

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def read_jsonl(p):
 with p.open(encoding='utf-8') as f:
  for line in f:
   if line.strip(): yield json.loads(line)
def write_json(x,p): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def cp_lower(k,n,alpha=.05):
 if n<=0:return None
 if k<=0:return 0.0
 def sf(p):return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k,n+1))
 lo,hi=0.,1.
 for _ in range(90):
  m=(lo+hi)/2
  if sf(m)<alpha:lo=m
  else:hi=m
 return (lo+hi)/2

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--public-dir',type=Path,required=True); ap.add_argument('--private-dir',type=Path,required=True); ap.add_argument('--results-dir',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((a.public_dir/'pre_manifest.json').read_text())
 if manifest['preflight_status']=='CONTAMINATION_FAIL':
  write_json({'overall_status':'C6_LATE_NORMALIZED_CONTAMINATION_FAIL','manifest':manifest},a.out/'summary.json'); return
 if manifest['preflight_status']=='CORPUS_LIMITED':
  write_json({'overall_status':'C6_LATE_NORMALIZED_CORPUS_LIMITED','manifest':manifest},a.out/'summary.json'); return
 root=bytes.fromhex((a.private_dir/'plant_root_reveal.hex').read_text().strip()); root_ok=sha_bytes(root)==manifest['plant_root_commitment_sha256']
 cases={x['case_id']:x for x in read_jsonl(a.public_dir/'cases.jsonl')}; truth={x['case_id']:x for x in read_jsonl(a.private_dir/'truth.jsonl')}
 results={}
 for p in sorted(a.results_dir.glob('*.jsonl')):
  for r in read_jsonl(p):
   if r['case_id'] in results: raise RuntimeError('duplicate '+r['case_id'])
   results[r['case_id']]=r
 complete=set(cases)==set(truth)==set(results)
 training=json.loads((a.public_dir/'training_words.json').read_text()); lp,uni=base.lm_from_words(training)
 rows=[]
 for cid,c in cases.items():
  t=truth[cid]; r=results[cid]; dec=decode_words(c['audit_cipher'],r['mapping']); rec=base.score_recovery(t['audit_truth'],dec)
  pos=rec['atom_recovery']>=ATOM_PASS and rec['word_recovery']>=WORD_PASS
  C,_=base.cipher_counts(c['fit_cipher']); oracle_obj=base.mapping_score(C,lp,t['oracle']); search_miss=oracle_obj>r['objective']+1e-12; obj_mis=(not pos and not search_miss)
  rows.append({'case_id':cid,'erasure':c['erasure'],'positive_call':pos,'returned_objective':r['objective'],'oracle_objective':oracle_obj,
               'oracle_minus_returned':oracle_obj-r['objective'],'search_miss_witness':search_miss,'objective_misalignment_witness':obj_mis,**rec})
 with (a.out/'scored_rows.jsonl').open('w',encoding='utf-8') as f:
  for x in rows:f.write(json.dumps(x,sort_keys=True)+'\n')
 by=defaultdict(list)
 for x in rows:by[x['erasure']].append(x)
 cells={}
 for er,v in sorted(by.items()):
  succ=sum(x['positive_call'] for x in v)
  cells[str(er)]={'n':len(v),'successes':succ,'cell_pass':len(v)==32 and succ>=CELL_PASS,'lower95':cp_lower(succ,len(v)),
                  'mean_atom_recovery':sum(x['atom_recovery'] for x in v)/len(v),'mean_word_recovery':sum(x['word_recovery'] for x in v)/len(v),
                  'search_miss_failures':sum((not x['positive_call']) and x['search_miss_witness'] for x in v),
                  'objective_misalignment_failures':sum(x['objective_misalignment_witness'] for x in v)}
 valid=root_ok and complete and manifest.get('private_truth_withheld_from_solver') and not manifest.get('voynich_loaded') and not manifest.get('target_access_allowed')
 status='C6_LATE_NORMALIZED_PASS' if valid and len(cells)==2 and all(z['cell_pass'] for z in cells.values()) else ('C6_LATE_NORMALIZED_EXECUTION_INVALID' if not valid else 'C6_LATE_NORMALIZED_FAIL')
 summary={'overall_status':status,'root_commitment_verified':root_ok,'solver_results_complete':complete,'source':manifest['source'],'source_words':manifest['source_words'],
          'shared_8gram_types_build_vs_source':manifest['shared_8gram_types_build_vs_source'],'shared_5gram_types_build_vs_source':manifest['shared_5gram_types_build_vs_source'],
          'cells':cells,'interpretation':'Late 1783 source-family transfer in the same secondary normalized Penn representation only. This does not qualify early historical or primary Hebrew-script transfer and does not unseal Voynich.'}
 write_json(summary,a.out/'summary.json'); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()

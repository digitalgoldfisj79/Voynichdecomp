import json,collections,sys,importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
P=Path(__file__).resolve().parent;B=(Path(__file__).resolve().parent/'inputs')
spec=importlib.util.spec_from_file_location('vb',str(P/'build_v145.py'));vb=importlib.util.module_from_spec(spec);spec.loader.exec_module(vb)
counts={v:collections.Counter() for v in 'aeiou'};first={v:collections.Counter() for v in 'aeiou'}
words=(B/'train.reference.txt').read_text().splitlines()
for w in words:
 lead, rr=vb.runs(w);counts['e'].update(vb.pieces(lead))
 for v,run in rr:
  parts=vb.pieces(run) if run else [''];counts[v].update(parts);first[v][parts[0]]+=1
for typ in ['A','B']:
 m=json.loads((P/f'v145_{typ}_manifest.json').read_text());C={};maps={}
 for v in 'aeiou':
  codes=m['state'][v]['unitcode'];b=m['bridge_for_vowel_and_next_key'][v]
  a=np.array([counts[v][u] for u in vb.U],dtype=np.int64)
  f=np.array([first[v][u] for u in vb.U],dtype=np.int64)
  lens=np.array([len(c) for c in codes],dtype=np.int64)
  brlen=np.array([len(b[c[0]]) for c in codes],dtype=np.int64)
  cost=a[:,None]*lens[None,:]+f[:,None]*brlen[None,:]
  i,j=linear_sum_assignment(cost)
  codes1=[codes[z] for z in j]
  maps[v]=codes1
  C[v]={'minimum_chars':int(cost[i,j].sum()),'current_chars':int(sum(cost[z,z] for z in range(222))),'units_occurrences':int(a.sum()),'first_count':int(f.sum()),'max_code_len':max(map(len,codes))}
 (P/f'v145_{typ}_minlength_mapping.json').write_text(json.dumps(maps,indent=2));print(typ,C,'total min',sum(x['minimum_chars'] for x in C.values()),'mean',sum(x['minimum_chars'] for x in C.values())/len(words))
(P/'source_unit_weights.json').write_text(json.dumps({'total':{k:dict(v) for k,v in counts.items()},'first':{k:dict(v) for k,v in first.items()}},indent=2))

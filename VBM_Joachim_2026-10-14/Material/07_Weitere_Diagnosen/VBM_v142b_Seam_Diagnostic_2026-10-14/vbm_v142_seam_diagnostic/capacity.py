import json,collections,hashlib
from pathlib import Path
B=Path('/mnt/data/vbm_v143_crosscarry/vbm_v142_inband')
REF=Path('/mnt/data/vbm_v143_crosscarry/VMS_B_REVA_reference.txt')
m=json.loads((B/'04_pair_bridge_manifest.json').read_text());units=m['unit_list']; words=[x for l in REF.read_text().splitlines() if '\t' in l for x in l.split('\t',1)[1].split()];tri={w[i:i+3] for w in words for i in range(len(w)-2)};bi={w[i:i+2] for w in words for i in range(len(w)-1)}
bridge=m['bridge_for_vowel_and_next_key']
res={}
for v,state in m['state'].items():
  cells=state['cells']
  allN=0;goodN=0;good=[];byKey={}
  for s in cells:
    br=bridge[v].get(s[0]);allN+=1
    if br is None:continue
    ok=all((br+s)[i:i+3] in tri for i in range(len(br+s)-2)) and all((br+s)[i:i+2] in bi for i in range(len(br+s)-1))
    byKey.setdefault(s[0],{'total':0,'good':0});byKey[s[0]]['total']+=1;byKey[s[0]]['good']+=int(ok)
    if ok:goodN+=1;good.append(s)
  # Original prefix-free cells can be pruned without breaking prefix-freeness
  res[v]={'total_v142_prefixfree_cells':len(cells),'join_compatible_prefixfree_cells':len(good),'by_first_key':byKey,'capacity_for_direct_src_units_no_escape':len(good),'source_units_including_empty':len(units),'current_direct':state['direct'],'current_escape':state['escape']}
res['note']='Necessary bridge|first-cell admissibility bound only. It does not address cell-cell seams, escape decoding, exact word distribution, A inventory or historical model.'
Path('/mnt/data/vbm_v142_seam_diagnostic/CAPACITY.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
for v,state in m['state'].items():
  good=[s for s in state['cells'] if (lambda w: all(w[i:i+3] in tri for i in range(len(w)-2)) and all(w[i:i+2] in bi for i in range(len(w)-1)))(bridge[v][s[0]]+s)]
  options=[]
  for esc in good:
    # Prefix-free expansion: each non-escape code cell standalone, escape followed by one cell.
    continuations=sum(all((esc+c)[j:j+3] in tri for j in range(len(esc+c)-2)) and all((esc+c)[j:j+2] in bi for j in range(len(esc+c)-1)) for c in good)
    options.append((len(good)-1+continuations,continuations,esc))
  if options:
    mx=max(options)
    res[v].update({'max_capacity_one_escape_within_existing_cells':mx[0],'max_continuations':mx[1],'best_escape':mx[2], 'would_fit_222':mx[0]>=len(units)})
Path('/mnt/data/vbm_v142_seam_diagnostic/CAPACITY.json').write_text(json.dumps(res,indent=2))
print('ONE-ESCAPE BOUND',[(k,res[k]['max_capacity_one_escape_within_existing_cells'],res[k]['would_fit_222'],res[k]['best_escape']) for k in 'aeiou'])
# Corrected optimistic maximum: the second cell in an escape code need not itself be bridge-compatible.
for v,state in m['state'].items():
  good=[s for s in state['cells'] if (lambda w: all(w[i:i+3] in tri for i in range(len(w)-2)) and all(w[i:i+2] in bi for i in range(len(w)-1)))(bridge[v][s[0]]+s)]
  opts=[]
  for esc in good:
    continued=sum(all((esc+c)[j:j+3] in tri for j in range(len(esc+c)-2)) and all((esc+c)[j:j+2] in bi for j in range(len(esc+c)-1)) for c in state['cells'])
    opts.append((len(good)-1+continued,continued,esc))
  if opts:
    max_cap=max(opts)
    res[v].update({'corrected_best_one_escape_capacity':max_cap[0],'corrected_continuations_any_v142_cell':max_cap[1],'corrected_best_escape':max_cap[2],'corrected_passes_222_capacity':max_cap[0]>=len(units)})
Path('/mnt/data/vbm_v142_seam_diagnostic/CAPACITY.json').write_text(json.dumps(res,indent=2))
print('CORRECTED ONE-ESCAPE UPPER BOUND',[(k,res[k]['corrected_best_one_escape_capacity'],res[k]['corrected_passes_222_capacity']) for k in 'aeiou'])

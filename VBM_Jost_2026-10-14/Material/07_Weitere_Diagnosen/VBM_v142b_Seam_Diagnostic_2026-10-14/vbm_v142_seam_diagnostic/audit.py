#!/usr/bin/env python3
"""Frozen v142b support decomposition, no fitting, source: existing v142 generated ciphertext and B reference."""
import json, collections, hashlib
from pathlib import Path
B=Path('/mnt/data/vbm_v143_crosscarry/vbm_v142_inband')
REF=Path('/mnt/data/vbm_v143_crosscarry/VMS_B_REVA_reference.txt')
m=json.loads((B/'04_pair_bridge_manifest.json').read_text())
ref=[w for line in REF.read_text().splitlines() if '\t' in line for w in line.split('\t',1)[1].split()]
trigrams={w[i:i+3] for w in ref for i in range(len(w)-2)}
state={v:(set(x['cells']),max(map(len,x['cells']))) for v,x in m['state'].items()}
pairs={tuple(k.split('|')):v for k,v in m['vowel_by_pair'].items()}

def parse(w):
    pos=0; st=m['initial_state']; parts=[]
    def readcell(pos,st):
      cs,mx=state[st]; options=[w[pos:pos+j] for j in range(1,min(mx,len(w)-pos)+1) if w[pos:pos+j] in cs]
      if len(options)!=1:raise ValueError(('invalid cell',w,pos,st,options))
      c=options[0];return c,pos+len(c)
    while pos<len(w):
      if w[pos] not in 'ynr':
        br=w[pos:pos+2] if w.startswith(('ch','sh'),pos) else w[pos];
        if br not in ('ch','sh','o','d','a','p','l'):raise ValueError(('invalid bridge',w,pos))
        parts.append(('bridge',br));pos+=len(br)
        if pos>=len(w):raise ValueError(('dangling bridge',w))
        st=pairs.get((br,w[pos]));
        if st is None:raise ValueError(('bridge state',w,pos,br,w[pos]))
      cell,pos=readcell(pos,st)
      parts.append(('cell',cell));
      if m['state'][st]['escape'] is not None and cell==m['state'][st]['escape']:
        x,pos=readcell(pos,st);parts.append(('cell',x))
    if ''.join(t for _,t in parts)!=w:raise ValueError(('not exact',w,parts))
    return parts

def audit(label):
  words=(B/f'{label}.pair.vbm').read_text().split()
  nbad=collections.Counter(); vkind=collections.Counter(); examples={}; lengths=collections.Counter()
  for w in words:
    parts=parse(w)
    pieceid=[]; piecetype=[]
    for j,(kind,text) in enumerate(parts):
      pieceid.extend([j]*len(text));piecetype.extend([kind]*len(text))
    miss_internal=miss_boundary=False
    for i in range(len(w)-2):
      tri=w[i:i+3]
      if tri in trigrams:continue
      if pieceid[i]==pieceid[i+2]:
        kind='internal_'+piecetype[i];miss_internal=True
      else:
        miss_boundary=True
        ids=set(pieceid[i:i+3]); types={piecetype[z] for z in range(i,i+3)}
        kind='boundary_bridge' if 'bridge' in types else 'boundary_cell_cell'
      vkind[kind]+=1
      if kind not in examples:examples[kind]={'word':w,'bad_trigram':tri,'pieces':parts,'index':i}
    lengths[len(parts)]+=1
    nbad['invalid_any']+=int(miss_internal or miss_boundary)
    nbad['invalid_only_internal']+=int(miss_internal and not miss_boundary)
    nbad['invalid_only_boundary']+=int(miss_boundary and not miss_internal)
    nbad['invalid_both']+=int(miss_internal and miss_boundary)
    nbad['invalid_internal']+=int(miss_internal)
    nbad['invalid_boundary']+=int(miss_boundary)
  n=len(words)
  return {'words':n,'word_count_invalid':dict(nbad),'invalid_word_proportions':{k:v/n for k,v in nbad.items()},'bad_trigram_occurrences_by_origin':dict(vkind),'total_bad_trigram_occurrences':sum(vkind.values()),'examples':examples,'sample_mean_components':sum(j*k for j,k in lengths.items())/n}

out={'source_sha256':{'B_reference':hashlib.sha256(REF.read_bytes()).hexdigest(),'manifest':hashlib.sha256((B/'04_pair_bridge_manifest.json').read_bytes()).hexdigest()},'reference_words':len(ref),'reference_trigram_support':len(trigrams),'audits':{}}
for label in ['train','hartlieb']:
  out['audits'][label]=audit(label)
  print(label,out['audits'][label]['words'],out['audits'][label]['invalid_word_proportions'],out['audits'][label]['bad_trigram_occurrences_by_origin'])
assert out['audits']['train']['word_count_invalid']['invalid_any']==60454
assert out['audits']['hartlieb']['word_count_invalid']['invalid_any']==11745
Path('/mnt/data/vbm_v142_seam_diagnostic/RESULTS.json').write_text(json.dumps(out,indent=2))
print('verified invariants, done')

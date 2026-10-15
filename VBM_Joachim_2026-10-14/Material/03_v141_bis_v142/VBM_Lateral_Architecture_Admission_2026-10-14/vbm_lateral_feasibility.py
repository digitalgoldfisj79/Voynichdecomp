#!/usr/bin/env python3
"""Exploratory admission only: 2-character compositional alphabet and bridge-key
capacity. NOT an implementation of a VBM encoder or decoder; no VMS claim.
Never mutates the supplied ZIP; write checkpoints atomically.
"""
import os,sys,importlib.util,json,hashlib,bisect,pickle,tempfile
from pathlib import Path
from collections import defaultdict,Counter
SRC=Path('/mnt/data/vbm_v141_gate/source/gen138')
ZIP=Path('/mnt/data/Generator_v138_Pruefpaket_2026-10-14.zip')
OUT=Path('/mnt/data/vbm_lateral_gate');OUT.mkdir(exist_ok=True)
def save(stage,data):
 f=OUT/f'{stage}.pkl'
 with tempfile.NamedTemporaryFile(dir=OUT,delete=False) as x:
  pickle.dump(data,x,protocol=pickle.HIGHEST_PROTOCOL);x.flush();os.fsync(x.fileno()); tmp=x.name
 os.replace(tmp,f)
 return f.name
provenance={'zip_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),'original_unchanged':True,'admission_scope':'symbolic capacity only, no end-to-end decoding or Voynich fit'}
save('stage0_source',provenance)
os.chdir(SRC);sys.path.insert(0,str(SRC))
spec=importlib.util.spec_from_file_location('admission_author_src',SRC/'vbm_tafel_v138.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
# 1) Independently reconstruct author event extraction on two sources.
def make_events(path):
 words=g.gen.words_of(Path(path).read_text(encoding='utf-8',errors='ignore'))
 stream=''.join(words);ends=[];n=0
 for w in words: n+=len(w);ends.append(n)
 vi=[i for i,c in enumerate(stream) if c in 'aeiou']
 out=[]
 for p,q in zip(vi,vi[1:]):
  K=stream[p+1:q];i=bisect.bisect_right(ends,p)
  if i<len(ends) and ends[i]<q and K.replace('§',''):
   b=ends[i];aus=stream[p+1:b];an=stream[b:q]
  else:aus=an=''
  out.append((stream[p],K.replace('§',''),stream[q],'§' in K,aus.replace('§',''),an.replace('§','')))
 return out, len(words)
train,_=make_events(SRC/'klartext/bairisch_kurzertext_GROSS.txt')
assert train==g.ev, 'Independent event reconstruction disagrees with v138!'
heldout,hn=make_events(SRC/'klartext/hartlieb_klartext.txt')

def event_parts(events):
 rows=[]
 for v,K,v2,fl,aus,an in events:
  x=aus if aus and an and len(aus)+len(an)>=3 else K
  if not x:continue
  chunks=[x[j:j+2] for j in range(0,len(x),2)]
  rows.append({'v':v,'K':x,'chunks':chunks})
 return rows
tr=event_parts(train);ho=event_parts(heldout)
trchunks=Counter(s for e in tr for s in e['chunks']);byv=defaultdict(set)
for e in tr:byv[e['v']].update(e['chunks'])
letters=sorted(set(''.join(e['K'] for e in tr)))
# Make single consonants universally encodable (open-class fallbacks), with all
# observed two-character chunks only when attested in the given vowel state.
state_vocab={v:(byv[v] | set(letters)) for v in 'aeiou'}
key_cnt=Counter(c[0] for c in g.CELLS)
key_support={v:{p[1] for p,w in g.pair_v.items() if w==v} for v in 'aeiou'}
cap={v:sum(key_cnt.get(k,0) for k in key_support[v]) for v in 'aeiou'}
assert all(len(state_vocab[v])<=cap[v] for v in 'aeiou')
# Bound: each state assigns every currently admitted primitive to a distinct cell
# chosen from bridge-compatible cell families. Does not test token seams.
cell_order=sorted(g.CELLS)
state_mapping={}
for v in 'aeiou':
 elig=[c for c in cell_order if c[0] in key_support[v]]
 state_mapping[v]={ch:tuple(elig[i]) for i,ch in enumerate(sorted(state_vocab[v]))}
 assert len(set(state_mapping[v].values()))==len(state_mapping[v])
# Negative: collision in same state must be detected; positive: per-state inverse unique.
neg=state_mapping['i'].copy();a,b=list(neg)[:2];neg[b]=neg[a]
assert len(set(neg.values()))<len(neg)
# For Hartlieb, unknown pairs use two singletons within the same vowel state.
def cover(rows):
 unsupported_pairs=0;fallback_events=0;out_units=0; unseen_letters=set();nparts=0
 for e in rows:
  v=e['v'];usedfb=False
  for c in e['chunks']:
   nparts+=1
   if c in state_mapping[v]:out_units+=1
   elif len(c)==2:
    if c[0] not in state_mapping[v]:unseen_letters.add(c[0])
    if c[1] not in state_mapping[v]:unseen_letters.add(c[1])
    unsupported_pairs+=1;out_units+=2;usedfb=True
   else:
    unseen_letters.add(c)
  fallback_events+=usedfb
 return {'nonempty_cluster_events':len(rows),'chunk_units_before_fallback':nparts,
  'unsupported_pair_occurrences':unsupported_pairs,'events_with_fallback':fallback_events,
  'encoded_units_after_fallback':out_units,'unseen_letters':sorted(unseen_letters)}
trcov=cover(tr);hocov=cover(ho)
# Segmentation-level ambiguity alone is NOT evidence of plaintext ambiguity:
# decoding 'bc' and 'b'+'c' yields the same letters. This test merely
# demonstrates that the chunk segmentation is not uniquely determined by
# source characters. Full grammar parsing is untested.
ambig=next(((v,c,[c[0],c[1]]) for v in 'aeiou' for c in sorted(state_vocab[v])
            if len(c)==2 and c[0] in state_vocab[v] and c[1] in state_vocab[v]),None)
assert ambig
results={'provenance':provenance,'training':{'events':len(train),'nonempty_cluster_events':len(tr),
 'unique_cluster_types':len({e['K'] for e in tr}),'unique_2char_primitives':len(trchunks),
 'primitive_letters':letters,'chunk_units':trcov['chunk_units_before_fallback'],
 'overhead_events':trcov['chunk_units_before_fallback']-len(tr),
 'states':{v:{'needed':len(state_vocab[v]),'eligible_cells':cap[v],
                  'margin':cap[v]-len(state_vocab[v])} for v in 'aeiou'},'coverage':trcov},
 'holdout':{'source':'supplied Hartlieb reading transcription, not independent of author package',
            'source_word_count':hn,'event_count':len(heldout),'coverage':hocov},
 'controls':{'independent_event_reconstruction':True,'exactly_injective_per_vowel':True,
             'same_vowel_cell_collision_detected':True,
             'segmentation_not_unique_but_same_plaintext':{'vowel':ambig[0],'single_chunk':ambig[1],'alternate_split':ambig[2], 'NOT_a_plaintext_collision':True}},
 'disqualifications': ['No actual VMS token emission or atom admissibility test',
                       'No independently specified decoder or final-token seam validation',
                       'No word-boundary encoding and no normalisation recovery',
                       'No Voynich-mechanism identification or section holdout',
                       'Per-vowel distinct assignments depend on knowing that vowel from ciphertext, not proven for complete stream']}
save('stage1_training_capacity',results['training']);save('stage2_holdout_coverage',results['holdout']);save('stage3_controls',results['controls'])
(OUT/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(results,indent=2,ensure_ascii=False))

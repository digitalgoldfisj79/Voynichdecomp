"""Independent diagnostic bounds, no model fitting. Frozen manifest is input."""
from pathlib import Path
import json,collections,os,pickle,tempfile
P=Path(__file__).resolve().parent
m=json.loads((P/'01_frozen_manifest.json').read_text()); V='aeiou'
vocab=set(m['units']); singles={x for x in vocab if len(x)==1}
def runs(w):
 i=0
 while i<len(w) and w[i] not in V:i+=1
 out=[w[:i]]
 while i<len(w):
  j=i+1
  while j<len(w) and w[j] not in V:j+=1
  out.append(w[i+1:j]);i=j
 return out
out={}
for label in ['train','hartlieb']:
 words=(P/(label+'.reference.txt')).read_text().splitlines();
 co=collections.Counter(); extra=0; unseen=set()
 for w in words:
  for r in runs(w):
   for i in range(0,len(r),2):
    u=r[i:i+2]
    if len(u)==2:
     co['pair_total']+=1
     if u not in vocab:co['pair_unknown']+=1;extra+=1
    elif u and u not in singles:unseen.add(u)
 out[label]={'word_count':len(words),'und_macro_count':words.count('und'),
 'two_character_chunks':co['pair_total'],'unrepresented_pairs':co['pair_unknown'],
 'fallback_extra_cell_tokens':extra,'unknown_letters':sorted(unseen)}
# Source-reference exact oracle check on decoded results not used by decoder.
for l in ['train','hartlieb']:
 assert (P/(l+'.decoded.txt')).read_bytes()==(P/(l+'.reference.txt')).read_bytes()
# Five disjoint sequential Hartlieb sections produce exact same recovered words
from independent_decoder import decode
cipher=(P/'hartlieb.vbm').read_text().split(); pieces=[];start=0
for i,t in enumerate(cipher):
 if t=='s':pieces.append(' '.join(cipher[start:i+1]));start=i+1
assert start==len(cipher)
size=(len(pieces)+4)//5; sections=[pieces[i:i+size] for i in range(0,len(pieces),size)]
assert sum((decode(' '.join(section),m) for section in sections),[]) == (P/'hartlieb.reference.txt').read_text().splitlines()
out['holdout_independent_blocks']={'number_of_sections':len(sections),'all_exact':True,'segmentation':'source-word-blocked; not Voynich folio blocks'}
with tempfile.NamedTemporaryFile(delete=False,dir=P) as f:
 pickle.dump(out,f);f.flush();os.fsync(f.fileno());q=f.name
os.replace(q,P/'stage6_diagnostic_bounds.pkl')
(P/'03_diagnostic_bounds.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

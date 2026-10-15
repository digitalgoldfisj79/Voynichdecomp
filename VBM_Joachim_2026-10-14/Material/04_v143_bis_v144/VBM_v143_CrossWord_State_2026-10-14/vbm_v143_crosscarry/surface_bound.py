#!/usr/bin/env python3
"""Necessary diagnostic only. VMS reference is in-sample for cell inventory; no historical model test."""
from pathlib import Path
import json,zipfile,pickle,os,tempfile,collections
P=Path(__file__).resolve().parent;B=P/'vbm_v142_inband';Z=Path('/mnt/data/Generator_v138_Pruefpaket_2026-10-14.zip')
local=P/'VMS_B_REVA_reference.txt'
if local.exists():r=local.read_text()
else:
 with zipfile.ZipFile(Z) as z:r=z.read('gen138/vms_B_REVA_atomstream.txt').decode()
ref=[]
for l in r.splitlines():
 if '\t' in l:ref.extend(l.split('\t',1)[1].split())
bi={w[i:i+2] for w in ref for i in range(len(w)-1)}
tri={w[i:i+3] for w in ref for i in range(len(w)-2)}
res={'reference_words':len(ref),'raw_bigrams':len(bi),'raw_trigrams':len(tri),'reference_self_bad_bigrams':0,'reference_self_bad_trigrams':0,'null_sd_for_reference_token_shuffle':0,
     'note':'In-sample necessary support bound, not a qualified matched historical model test. Reference self-support is degenerate by construction.','results':{}}
for label in ('train','hartlieb'):
 for arm,path in [('baseline',B/f'{label}.pair.vbm'),('carry',P/f'{label}.carry.vbm'),('statefree',P/f'{label}.statefree.vbm')]:
  x=path.read_text().split();bad_b=sum(any(w[i:i+2] not in bi for i in range(len(w)-1)) for w in x)
  bad_t=sum(any(w[i:i+3] not in tri for i in range(len(w)-2)) for w in x)
  res['results'][label+'_'+arm]={'n_words':len(x),'invalid_bigram':bad_b,'invalid_trigram':bad_t,'invalid_bigram_fraction':bad_b/len(x),'invalid_trigram_fraction':bad_t/len(x), 'avg_length':sum(map(len,x))/len(x)}
(P/'03_surface_bound.json').write_text(json.dumps(res,indent=2)+'\n')
fd,temp=tempfile.mkstemp(dir=P,prefix='.stage5-')
with os.fdopen(fd,'wb') as f:pickle.dump(res,f,protocol=5); f.flush();os.fsync(f.fileno())
os.replace(temp,P/'stage5_surface_bound.pkl')
print(json.dumps(res,indent=2))

#!/usr/bin/env python3
"""Bounding test: repeat v142 and v142b support under raw (non-atomised) EVA characters."""
from pathlib import Path
import json,collections,pickle,os,tempfile,statistics
P=Path(__file__).resolve().parent;S=Path(os.environ.get('VBM_SOURCE_DIR','/mnt/data/vbm_work/gen138'))
reference=[]
for line in (S/'vms_B_REVA_atomstream.txt').read_text().splitlines():
 if '\t' in line:reference.extend(line.split('\t',1)[1].split())
sets={n:set(k for w in reference for k in zip(*(w[i:] for i in range(n)))) for n in (2,3)}
res={}
for label,f in [('native_b',None),('v142a_dev','train.vbm'),('v142a_hartlieb','hartlieb.vbm'),('v142b_dev','train.pair.vbm'),('v142b_hartlieb','hartlieb.pair.vbm')]:
 words=reference if f is None else (P/f).read_text().split()
 rec={'words':len(words)}
 for n in (2,3):
  viol=sum(any(ng not in sets[n] for ng in zip(*(w[i:] for i in range(n)))) for w in words)
  rec[f'unattested_char_{n}gram_words']=viol
  rec[f'fraction_char_{n}gram_violation']=viol/len(words)
 res[label]=rec
(P/'07_raw_character_bound.json').write_text(json.dumps(res,indent=2)+'\n')
with tempfile.NamedTemporaryFile('wb',dir=P,delete=False) as f:
 pickle.dump(res,f,protocol=5);f.flush();os.fsync(f.fileno());tmp=f.name
os.replace(tmp,P/'stage9_raw_character_bound.pkl')
print(json.dumps(res,indent=2))

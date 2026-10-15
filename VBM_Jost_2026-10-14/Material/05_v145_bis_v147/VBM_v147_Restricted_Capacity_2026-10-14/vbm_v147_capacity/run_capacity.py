#!/usr/bin/env python3
"""VBM v147: exact necessary-support cardinality for fixed positionwise bijective atom substitutions.

An intentionally restricted theorem; not an inference about other encoders.
Run from anywhere: python run_capacity.py
"""
from pathlib import Path
import json,re,hashlib,collections
ROOT=Path(__file__).resolve().parent
I=ROOT/'in'; atom=re.compile(r'ckh|cth|cph|cfh|ch|sh|.')
FILES=['train.reference.txt','hartlieb.reference.txt','A_REVA_reference.txt','VMS_B_REVA_reference.txt']
def target(name):
    lines=(I/name).read_text().splitlines()
    if name.startswith('A_'):
        return [w for line in lines for w in line.split() if re.fullmatch('[a-z]+',w)]
    return [w for line in lines if '\t' in line for w in line.split('\t',1)[1].split() if re.fullmatch('[a-z]+',w)]
def source(name):
    return (I/name).read_text().split()
def kinds(seq):
    out={k:set() for k in ('all','start_imm','interior_mmm','end_mmf','single_triple_imf')}
    for w in seq:
        n=len(w)
        for i in range(n-2):
            tri=tuple(w[i:i+3]);out['all'].add(tri)
            if i==0 and n>=4:out['start_imm'].add(tri)
            if i>=1 and i+3<n:out['interior_mmm'].add(tri)
            if i>=1 and i+3==n:out['end_mmf'].add(tri)
            if i==0 and n==3:out['single_triple_imf'].add(tri)
    return {k:len(v) for k,v in out.items()}
source_data={k:source(k) for k in FILES[:2]};target_data={k:target(k) for k in FILES[2:]}
result={'date':'2026-10-14','scope':'restricted necessary-condition feasibility: 1 atom per source letter, distinct 23 glyph atoms per positional class (I,M,F); all encoded atom-trigrams must be attested in corresponding A or B original REVA inventory. No claim about finite-state code or unknown actual source.','source_inputs':{k:{'sha256':hashlib.sha256((I/k).read_bytes()).hexdigest(),'tokens':len(v),'types':len(set(v)),'mean_letters':sum(map(len,v))/len(v),'trigrams':kinds(v)} for k,v in source_data.items()},'reference_inputs':{k:{'sha256':hashlib.sha256((I/k).read_bytes()).hexdigest(),'tokens':len(v),'types':len(set(v)),'mean_raw_characters':sum(map(len,v))/len(v),'atom_trigrams':kinds([atom.findall(w) for w in v])}for k,v in target_data.items()},'bounds':{}}
for src,v in result['source_inputs'].items():
 for tg,t in result['reference_inputs'].items():
  a=v['trigrams'];b=t['atom_trigrams']
  # Note each position-aligned bijection maps distinct source triples to distinct target triples.
  result['bounds'][src+' vs '+tg]={'plain_1x1_any_missing_at_least':max(0,a['all']-b['all']), 'position3_missing_at_least':{k:max(0,a[k]-b[k]) for k in ['start_imm','interior_mmm','end_mmf','single_triple_imf']}, 'fixed_three_position_perfect_support_possible':all(a[k]<=b[k] for k in ['start_imm','interior_mmm','end_mmf','single_triple_imf'])}
# An idealised lookup per distinct word has sufficient vocabulary capacity on the diagnostic Hartlieb source,
# but not on the larger development corpus when restricted to exact observed native word types.
result['oracle_vocabulary_capacity']={src+' vs '+tg:{'source_types':len(set(words)),'reference_types':len(set(ref)),'distinct_word_dictionary_possible':len(set(words))<=len(set(ref))}for src,words in source_data.items() for tg,ref in target_data.items()}
(ROOT/'RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'source':{k:v['trigrams'] for k,v in result['source_inputs'].items()},'reference':{k:v['atom_trigrams'] for k,v in result['reference_inputs'].items()},'bounds':result['bounds'],'oracle':result['oracle_vocabulary_capacity']},indent=2))

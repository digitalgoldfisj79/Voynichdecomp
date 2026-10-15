#!/usr/bin/env python3
"""Independent, deterministic architectural lower bound for VBM v142b-v145.
No encoder, optimiser, decoder or Voynich fitting is imported. Only frozen inputs.
The mathematical lower bound is source-specific and representation-specific.
"""
import re,json,hashlib,statistics,collections
from pathlib import Path
P=Path(__file__).resolve().parent; I=P/'inputs'
M=json.loads((I/'04_pair_bridge_manifest.json').read_text()); U=set(M['unit_list'])
assert len(U)==222
assert min(len(c) for v in M['state'].values() for c in v['cells'])==2
assert min(len(x) for v in M['bridge_for_vowel_and_next_key'].values() for x in v.values())==1
ATOMS=re.compile(r'cth|ckh|cph|cfh|ch|sh|.')
def pieces(s):
    a=[]
    for i in range(0,len(s),2):
        p=s[i:i+2]
        a.extend([p] if p in U else list(p))
    if any(v not in U for v in a):raise RuntimeError('Source unit inventory mismatch')
    return a
def burden(w):
    i=0
    while i<len(w) and w[i] not in 'aeiou': i+=1
    cells=len(pieces(w[:i]));bridges=0;use=collections.Counter((('e',x) for x in pieces(w[:i])))
    while i<len(w):
        v=w[i];j=i+1
        while j<len(w) and w[j] not in 'aeiou':j+=1
        pp=pieces(w[i+1:j]) if j>i+1 else [''];
        cells+=len(pp);bridges+=1;use.update((v,x) for x in pp);i=j
    assert cells>=1
    return cells,bridges,use

def stats(words):
    lengths=[len(w) for w in words];atoms=[len(ATOMS.findall(w)) for w in words];n=len(words)
    return {'n':n,'mean_raw_chars':sum(lengths)/n,'mean_EVA_atoms':sum(atoms)/n,
      'pct_raw_at_least_10':100*sum(x>=10 for x in lengths)/n,
      'pct_raw_at_least_15':100*sum(x>=15 for x in lengths)/n,
      'pct_atoms_at_least_10':100*sum(x>=10 for x in atoms)/n,
      'distinct_words':len(set(words))}

out={'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in I.iterdir() if p.is_file()},'assumptions':{
 'source_word_count_equals_output_word_count':True,'segmentation':'two-consonant source units, one explicit bridge and at least one explicit code cell per vowel',
 'cell_min_raw_EVA_chars':2,'bridge_min_raw_EVA_chars':1,
 'bound':'output_length(word) >= 2*source_cell_count(word) + vowel_count(word)'
},'reference':{},'sources':{}}
A=(I/'A_REVA_reference.txt').read_text().splitlines()
B=[w for line in (I/'VMS_B_REVA_reference.txt').read_text().splitlines() if '\t' in line for w in line.split('\t',1)[1].split()]
for name,w in [('Currier_A_REVA',A),('Currier_B_REVA',B)]:out['reference'][name]=stats(w)
for source in ['train','hartlieb']:
    words=(I/(source+'.reference.txt')).read_text().splitlines();n=len(words)
    bur=[burden(w) for w in words];C=sum(x for x,y,_ in bur);V=sum(y for x,y,_ in bur);lower=[2*x+y for x,y,_ in bur]
    union=set();freq=collections.Counter()
    for _,_,cnt in bur: union.update(cnt);freq.update(cnt)
    obj={'source_words':n,'source_mean_letters':sum(map(len,words))/n,
        'mandatory_cells':C,'mandatory_bridges':V,'mean_cells':C/n,'mean_bridges':V/n,
        'hard_raw_mean_lower_bound':sum(lower)/n,
        'hard_min_ge_10_pct':sum(x>=10 for x in lower)*100/n,
        'hard_min_ge_15_pct':sum(x>=15 for x in lower)*100/n,
        'reachable_assignments':len(union),'min_hard_output_chars_total':sum(lower),
        'gap_to_native_B_mean_chars':sum(lower)/n-out['reference']['Currier_B_REVA']['mean_raw_chars'],
        'gap_to_native_A_mean_chars':sum(lower)/n-out['reference']['Currier_A_REVA']['mean_raw_chars']}
    if source=='hartlieb':
        for lang in ['A','B']:
            g=(I/f'v145_{lang}_hartlieb_length_constrained.vbm').read_text().split()
            assert len(g)==n
            obj['v145_'+lang+'_synthetic']=stats(g)
            native=set(A if lang=='A' else B)
            obj['v145_'+lang+'_exact_word_matches']=sum(w in native for w in g)
    out['sources'][source]=obj

# Exact theoretical envelope when no codebook can emit shorter than a two-character cell.
assert out['sources']['hartlieb']['hard_raw_mean_lower_bound'] > out['reference']['Currier_B_REVA']['mean_raw_chars']
assert out['sources']['hartlieb']['hard_raw_mean_lower_bound'] > out['reference']['Currier_A_REVA']['mean_raw_chars']
(P/'RESULTS.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))

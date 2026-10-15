#!/usr/bin/env python3
"""Independent reference decoder. Requires ONLY ciphertext and frozen JSON manifest.
Does not import or read author code, source German, event ledgers or encoder.
Usage: python independent_decoder.py manifest.json input.vbm output.txt
"""
import json,sys
from pathlib import Path

def build(m):
 units=m['units']
 assert len(units)==len(set(units)), 'primitive collision'
 cells=m['cells']
 assert len(cells)==len(set(map(tuple,cells))), 'duplicate source-cell tuple'
 assert all(len(c)==3 and c[0] in 'ynrl' for c in cells)
 out={}
 for v,keys in m['key_eligibility'].items():
  permitted=[c for c in cells if c[0] in keys]
  assert len(permitted)>=len(units), 'under-capacity'
  surfaces=[''.join(c) for c in permitted[:len(units)]]
  assert len(set(surfaces))==len(surfaces),'same-vowel codeword collision'
  out[v]=dict(zip(surfaces,units))
 pair={tuple(x.split('|')):v for x,v in m['pair_to_vowel'].items()}
 assert all(len(p)==2 for p in pair)
 assert all(k in 'ynrl' and v in out for (br,k),v in pair.items())
 assert all(not token.startswith(('m','s')) for d in out.values() for token in d)
 return out,pair

def decode(ciphertext,manifest):
 tables,pairs=build(manifest)
 toks=ciphertext.split()
 if not toks:raise ValueError('empty ciphertext')
 out=[]; buf=''; current=manifest['initial_vowel_state'];pending=None;following_empty=False
 for i,t in enumerate(toks):
  if t=='s':
   if pending is not None:raise ValueError(f'pending bridge at word end {i}')
   if not buf:raise ValueError(f'empty/duplicate word end {i}')
   out.append(buf);buf='';current=manifest['initial_vowel_state'];following_empty=False
  elif t.startswith('m'):
   if pending is not None:raise ValueError(f'consecutive bridge {i}')
   pending=t[1:] or '-';following_empty=False
  elif t[0] in 'ynrl':
   if pending is not None:
    pair=(pending,t[0]); vowel=pairs.get(pair)
    if vowel is None:raise ValueError(f'unsupported bridge pair {pair} at {i}')
    current=vowel;buf+=vowel;pending=None
    if t not in tables[current]:raise ValueError(f'unknown cell {t} in state {current} at {i}')
    frag=tables[current][t]
    following_empty=not frag
    buf+=frag
   else:
    if following_empty:raise ValueError(f'cell after empty cell, no new vowel {i}')
    if t not in tables[current]:raise ValueError(f'unknown continuation {t} in state {current} at {i}')
    frag=tables[current][t]
    if not frag:raise ValueError(f'bare empty cell at {i}')
    buf+=frag
  else:raise ValueError(f'illegal token type {t} at {i}')
 if pending is not None:raise ValueError('bridge without cell at EOF')
 if buf:raise ValueError('missing final word terminator')
 return out

if __name__=='__main__':
 if len(sys.argv)!=4:raise SystemExit('usage: decoder.py MANIFEST INPUT OUTPUT')
 manifest=json.loads(Path(sys.argv[1]).read_text())
 out=decode(Path(sys.argv[2]).read_text(),manifest)
 Path(sys.argv[3]).write_text('\n'.join(out)+'\n',encoding='utf-8')
 print(f'Independent decoding completed: {len(out)} words')

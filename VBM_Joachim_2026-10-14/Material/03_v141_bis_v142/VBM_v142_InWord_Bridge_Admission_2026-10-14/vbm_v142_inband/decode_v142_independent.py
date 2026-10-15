#!/usr/bin/env python3
"""Independent in-band parser; imports no generator, original text, or author package."""
import json,sys
from pathlib import Path
VOW='aeiou'

def grammar(m):
 units=m['unit_list']; rev_mark={g:v for v,g in m['source_vowel_to_glyph'].items()}
 if len(units)!=len(set(units)) or len(rev_mark)!=5:raise ValueError('manifest collision: units/vowels')
 T={}
 for v,surfs in m['cell_tables'].items():
  if len(surfs)!=len(units) or len(set(surfs))!=len(surfs):raise ValueError('manifest cell collision')
  if any(x.startswith(y) for i,x in enumerate(surfs) for j,y in enumerate(surfs) if i!=j):raise ValueError('manifest prefix collision')
  if any(not s or s[0] in rev_mark for s in surfs):raise ValueError('non-separable cell/marker start')
  T[v]={x:u for x,u in zip(surfs,units)}
 if not set(rev_mark).isdisjoint(set(s[0] for d in T.values() for s in d)):
  raise ValueError('marker/codeword overlap')
 return T,rev_mark

def decode(cipher,m):
 T,rev=grammar(m); allwords=[]
 for n,token in enumerate(cipher.split()):
  if not token:raise ValueError('empty word')
  state=m['default_initial_vowel_state'];pos=0;plain=''
  while pos<len(token):
   if token[pos] in rev:
    state=rev[token[pos]];plain+=state;pos+=1;continue
   found=[(surface,unit) for surface,unit in T[state].items() if token.startswith(surface,pos)]
   if len(found)!=1:raise ValueError(f'unknown/ambiguous cell at word {n}, offset {pos}, state {state}, matches {len(found)}')
   s,u=found[0];plain+=u;pos+=len(s)
  if not plain:raise ValueError('empty decoded word')
  allwords.append(plain)
 return allwords
if __name__=='__main__':
 if len(sys.argv)!=4:raise SystemExit('usage: decode_v142_independent.py MANIFEST INPUT OUTPUT')
 m=json.loads(Path(sys.argv[1]).read_text()); res=decode(Path(sys.argv[2]).read_text(),m)
 Path(sys.argv[3]).write_text('\n'.join(res)+'\n')
 print('independent decoded word count',len(res))

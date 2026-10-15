#!/usr/bin/env python3
"""Independent strict decoder and surface validator for frozen v145 artifacts.
This file does NOT import the encoder, optimiser, or implementation decoder.
"""
import json,sys,re,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent

def compile_tree(m):
 trees={}
 for vowel,entry in m['state'].items():
  tree={}
  codes=entry['unitcode']
  if len(codes)!=len(set(codes)) or len(codes)!=len(m['unit_list']):raise AssertionError('nonunique codes')
  for j,c in enumerate(codes):
   node=tree
   if c[0] not in 'ynr':raise AssertionError('bad code initial')
   for k in c:
    if 'ix' in node:raise AssertionError('not prefix-free')
    node=node.setdefault(k,{})
   if node:raise AssertionError('duplicate or prefix collision')
   node['ix']=j
  trees[vowel]=tree
 return trees

def decode(text,m):
 trees=compile_tree(m);pairs={tuple(k.split('|')):v for k,v in m['vowel_by_pair'].items()}
 output=[]
 for word in text.split():
  i=0;v=m['initial_state'];source='';first=False;epsilon=False
  while i<len(word):
   if word[i] not in 'ynr':
    if word.startswith('ch',i) or word.startswith('sh',i):bridge=word[i:i+2];i+=2
    else:bridge=word[i];i+=1
    if i>=len(word):raise ValueError('trailing bridge')
    v=pairs.get((bridge,word[i]))
    if v is None:raise ValueError('invalid vowel bridge')
    source+=v;first=True;epsilon=False
   else:
    if epsilon:raise ValueError('extra content after empty unit')
    node=trees[v]
    while 'ix' not in node:
     if i>=len(word) or word[i] not in node:raise ValueError('bad cell parse')
     node=node[word[i]];i+=1
    unit=m['unit_list'][node['ix']]
    if unit=='':
     if not first:raise ValueError('empty non-first unit')
     epsilon=True
    else:source+=unit
    first=False
  if not source or first:raise ValueError('unfinished word')
  output.append(source)
 return output

def run(directory):
 root=Path(directory)
 results={}
 for letter in 'AB':
  m=json.loads((root/f'v145_{letter}_length_constrained_manifest.json').read_text())
  for src in ['train','hartlieb']:
   source=(root/'inputs'/f'{src}.reference.txt').read_text().splitlines()
   cipher=(root/f'v145_{letter}_{src}_length_constrained.vbm').read_text()
   decoded=decode(cipher,m)
   if source != decoded:raise ValueError((letter,src,'mismatch',next((i for i,(x,y) in enumerate(zip(source,decoded))if x!=y),None)))
   results[letter+'_'+src]={'decoded_exact':len(source),'cipher_sha256':hashlib.sha256(cipher.encode()).hexdigest()}
 return results
if __name__=='__main__':
 print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else D),indent=2))

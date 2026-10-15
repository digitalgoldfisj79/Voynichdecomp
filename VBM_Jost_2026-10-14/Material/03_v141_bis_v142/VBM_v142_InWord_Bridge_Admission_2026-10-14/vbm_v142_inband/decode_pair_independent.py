#!/usr/bin/env python3
"""Pure independently authored parser for original bridge pairs embedded inside words."""
import json,sys
from pathlib import Path

def build(m):
 units=m['unit_list'];pairs={tuple(k.split('|')):v for k,v in m['vowel_by_pair'].items()}
 if len(units)!=len(set(units)) or not units or units[0]!='':raise ValueError('bad units')
 state={}
 for v,x in m['state'].items():
  c=x['cells'];d=x['direct'];esc=x['escape']
  if len(c)!=len(set(c)) or any(s.startswith(t) for i,s in enumerate(c) for j,t in enumerate(c) if i!=j):
   raise ValueError('duplicate/prefix overlap')
  if not 0<d<=len(units):raise ValueError('bad direct bound')
  if esc is not None and (esc!=c[-1] or d!=len(c)-1 or len(units)-d>len(c)):
   raise ValueError('bad escape protocol')
  if esc is None and d!=len(units):raise ValueError('missing escape')
  if any(s[0] not in 'ynr' for s in c):raise ValueError('non-cell initial')
  state[v]={'cells':c,'by_cell':{s:i for i,s in enumerate(c)},'direct':d,'esc':esc}
 return units,state,pairs

def decode(text,m):
 units,state,pairs=build(m); out=[]
 # Registered marker syntax: (ch,sh) or single (o,d,a,p,l). These are disjoint from cell initials ynr.
 def next_cell(w,pos,st):
  candidates=[s for s in state[st]['cells'] if w.startswith(s,pos)]
  if len(candidates)!=1:raise ValueError(f'invalid cell at {pos}, state {st}')
  s=candidates[0];return s,pos+len(s)
 for word in text.split():
  pos=0;st=m['initial_state'];buf='';need_first=False;after_epsilon=False
  while pos<len(word):
   if word[pos] not in 'ynr':
    if word.startswith('ch',pos) or word.startswith('sh',pos):br=word[pos:pos+2];pos+=2
    elif word[pos] in 'odapl':br=word[pos];pos+=1
    else:raise ValueError('invalid bridge prefix')
    if pos==len(word) or word[pos] not in 'ynr':raise ValueError('dangling bridge')
    k=word[pos];nxt=pairs.get((br,k))
    if nxt is None:raise ValueError('invalid (bridge,next-key) pair')
    st=nxt;buf+=st;need_first=True;after_epsilon=False
   else:
    if after_epsilon:raise ValueError('continuation after epsilon')
    cell,pos=next_cell(word,pos,st);d=state[st]
    ix=d['by_cell'][cell]
    if d['esc'] is not None and cell==d['esc']:
     sec,pos=next_cell(word,pos,st);j=d['by_cell'][sec]
     ix=d['direct']+j
    elif ix>=d['direct']:raise ValueError('reserved cell outside escape')
    if ix>=len(units):raise ValueError('escape overflow')
    u=units[ix]
    if u=='':
     if not need_first:raise ValueError('epsilon outside bridge-first')
     after_epsilon=True
    else:buf+=u
    need_first=False
  if need_first:raise ValueError('dangling bridge without cell')
  if not buf:raise ValueError('empty plaintext')
  out.append(buf)
 return out

if __name__=='__main__':
 if len(sys.argv)!=4:raise SystemExit('usage: decode_pair_independent.py MANIFEST CIPHER OUT')
 m=json.loads(Path(sys.argv[1]).read_text());out=decode(Path(sys.argv[2]).read_text(),m)
 Path(sys.argv[3]).write_text('\n'.join(out)+'\n')
 print('decoded words',len(out))

#!/usr/bin/env python3
"""Independently reconstruct compressed publicly specified state tables and decode."""
import sys,json
from pathlib import Path
from decode_pair_independent import decode

def expand(m):
 whole=[''.join(x) for x in m['cell_inventory']]
 if len(whole)!=len(set(whole)):raise ValueError('duplicate underlying cell')
 bridge=m['bridge_for_vowel_and_next_key']; U=len(m['unit_list']);state={}
 for v,keys in bridge.items():
  allowed=[c for c in whole if c[0] in keys]
  leaves=[c for c in allowed if not any(other!=c and other.startswith(c) for other in allowed)]
  rank={s:i for i,s in enumerate(whole)}
  leaves.sort(key=lambda x:(len(x),rank[x]))
  if len(leaves)>=U:state[v]={'cells':leaves[:U],'direct':U,'escape':None}
  else:
   if 2*len(leaves)-1<U:raise ValueError('insufficient 2-cell code capacity')
   state[v]={'cells':leaves,'direct':len(leaves)-1,'escape':leaves[-1]}
 expanded={'unit_list':m['unit_list'],'vowel_by_pair':m['vowel_by_pair'], 'state':state,'initial_state':m['initial_state']}
 return expanded
if __name__=='__main__':
 if len(sys.argv)!=4:raise SystemExit('usage: decode_pair_compact.py MANIFEST INPUT OUTPUT')
 m=json.loads(Path(sys.argv[1]).read_text()); recovered=decode(Path(sys.argv[2]).read_text(),expand(m))
 Path(sys.argv[3]).write_text('\n'.join(recovered)+'\n')
 print('decoded',len(recovered))

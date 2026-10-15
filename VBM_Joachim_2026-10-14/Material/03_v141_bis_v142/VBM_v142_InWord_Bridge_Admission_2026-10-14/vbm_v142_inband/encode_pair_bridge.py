#!/usr/bin/env python3
"""Pair-preserving in-word VBM experiment, disjoint architecture branch v142b."""
import os,sys,json,collections,importlib.util,hashlib,pickle,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent; SRC=Path(os.environ.get('VBM_SOURCE_DIR','/mnt/data/vbm_work/gen138'))
sys.path.insert(0,str(P))
def out(name,s):
 with tempfile.NamedTemporaryFile('w',dir=P,delete=False,encoding='utf-8') as f:
  f.write(s);f.flush();os.fsync(f.fileno());t=f.name
 os.replace(t,P/name)
def pick(name,x):
 with tempfile.NamedTemporaryFile('wb',dir=P,delete=False) as f:
  pickle.dump(x,f,protocol=5);f.flush();os.fsync(f.fileno());t=f.name
 os.replace(t,P/name)
prior=json.loads((P/'01_frozen_manifest.json').read_text()); srcunits=prior['unit_list'];vows='aeiou'
# Origin cells and pair definitions are imported only from sealed v141 manifest, no re-evaluation by author code.
base=json.loads((P/'parent_v141_manifest.json').read_text())
cells=[''.join(x) for x in base['cells']]
bridge={
'a':{'y':'sh','n':'o'},
'e':{'y':'d','n':'ch','r':'a'},
'i':{'y':'o'},
'o':{'y':'p'},
'u':{'y':'l','r':'ch'}}
for v, ks in bridge.items():
 for k, br in ks.items():
  assert base['pair_to_vowel'][br+'|'+k]==v,(v,br,k)
# One globally shared source-unit ordering determined only by development corpus (source unit set already frozen).
train=(P/'train.reference.txt').read_text().splitlines()
cnt=collections.Counter()
def runs(w):
 i=0
 while i<len(w) and w[i] not in vows:i+=1
 lead=w[:i]; out=[]
 while i<len(w):
  v=w[i];j=i+1
  while j<len(w) and w[j] not in vows:j+=1
  out.append((v,w[i+1:j]));i=j
 return lead,out
def pieces(s):
 a=[]
 for i in range(0,len(s),2):
  p=s[i:i+2]
  a.extend([p] if p in srcunits else list(p))
 assert ''.join(a)==s
 return a
for w in train:
 lead,r=runs(w)
 for s in [lead]+[t for v,t in r]:cnt.update(pieces(s))
units=['']+sorted(srcunits,key=lambda u:(-cnt[u],u))
assert len(units)==222 and len(set(units))==len(units)
rank={u:i for i,u in enumerate(units)}
state={};alloc={}
for v in vows:
 a=[s for s in cells if s[0] in bridge[v]]
 leaves=[s for s in a if not any(t!=s and t.startswith(s) for t in a)]
 oi={s:i for i,s in enumerate(cells)}
 leaves.sort(key=lambda s:(len(s),oi[s]))
 if len(leaves)<len(units):
  # one escape reserves last cell; two-cell code for remaining shared-rank units.
  direct=len(leaves)-1
  assert len(units)-direct<len(leaves)
  state[v]={'cells':leaves,'direct':direct,'escape':leaves[-1]}
 else:state[v]={'cells':leaves[:len(units)],'direct':len(units),'escape':None}
 alloc[v]={'prefixfree_cells':len(leaves),'direct_units':state[v]['direct'],'two_cell_units':len(units)-state[v]['direct']}
 assert all(not x.startswith(y) for i,x in enumerate(state[v]['cells']) for j,y in enumerate(state[v]['cells']) if i!=j)
 # Key is one of the supported nonempty-bridge families; all codecells start y/n/r
 assert all(x[0] in bridge[v] for x in state[v]['cells'])
manifest={'format':'vbm142b_pair_inband','vowel_by_pair':{br+'|'+k:v for v,dd in bridge.items() for k,br in dd.items()},
 'bridge_for_vowel_and_next_key':{v:{k:br for k,br in dd.items()} for v,dd in bridge.items()},
 'unit_list':units,'state':state,'initial_state':'e', 'word_boundary':'whitespace',
 'inventory_origin':'v141 original VMS-B cell inventory, historically fitted; no independent VMS test',
 'semantics':'actual original pairs encode vowel, but all bridge+cell groups are in-token rather than across tokens',
 'development_source_hash':hashlib.sha256('\n'.join(train).encode()).hexdigest()}
out('04_pair_bridge_manifest.json',json.dumps(manifest,sort_keys=True,indent=2)+'\n')
pick('stage5_pair_grammar_capacity.pkl',{'source_units':len(units),'alloc':alloc,'manifest_sha256':hashlib.sha256((P/'04_pair_bridge_manifest.json').read_bytes()).hexdigest()})

def unitcode(v,u):
 ix=rank[u];s=state[v];cc=s['cells']
 return cc[ix] if ix<s['direct'] else s['escape']+cc[ix-s['direct']]
def encode(w):
 lead,rr=runs(w);o=''.join(unitcode('e',u) for u in pieces(lead))
 for v,run in rr:
  parts=pieces(run) if run else ['']
  first=unitcode(v,parts[0]);assert first
  br=bridge[v][first[0]];o+=br+first
  for u in parts[1:]:o+=unitcode(v,u)
 assert o
 return o
for label in ('train','hartlieb'):
 words=(P/(label+'.reference.txt')).read_text().splitlines()
 ctext=' '.join(map(encode,words))+'\n'
 out(label+'.pair.vbm',ctext)
 pick('stage6_pair_emit_'+label+'.pkl',{'words':len(words),'chars':sum(map(len,ctext.split())),'hash':hashlib.sha256(ctext.encode()).hexdigest()})
print(json.dumps({'source_units':len(units),'alloc':alloc,'source_counts':[len(train),len((P/'hartlieb.reference.txt').read_text().splitlines())]},indent=2))

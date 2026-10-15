from pathlib import Path
import collections,hashlib,json,pickle,os,tempfile
P=Path(__file__).resolve().parent;B=Path('/mnt/data/vbm_v143_crosscarry');Z=B/'vbm_v142_inband';
m=json.loads((B/'01_v143_frozen_manifest.json').read_text());STATES=m['carry_spec']['carry_states'];b2v={tuple(k.split('|')):v for k,v in m['vowel_by_pair'].items()}
trie={};tr={} 
for v,info in m['state'].items():
 T={};groups=collections.defaultdict(list)
 for c in info['cells']:
  node=T
  for char in c:node=node.setdefault(char,{})
  node['*']=c
  groups[(c[0],len(c))].append(c)
 trie[v]=T;tr[v]={}
 for state in STATES:
  shift=STATES.index(state);tr[v][state]={c:g[(j+shift)%len(g)] for g in groups.values() for j,c in enumerate(g)}

def parse(word):
 i=0;v='e';seq=[];last='-'
 while i<len(word):
  if word[i] not in 'ynr':
   if word[i:i+2] in ('ch','sh'):br=word[i:i+2];i+=2
   else:br=word[i];i+=1
   if i>=len(word):raise ValueError('dangling bridge')
   pair=(br,word[i]);v=b2v[pair];last=v;seq.append(('b',br));
  else:
   node=trie[v];j=i
   while j<len(word) and word[j] in node:
    node=node[word[j]];j+=1
    if '*' in node:break
   if '*' not in node:raise ValueError(f'invalid cell at {i} in {word}')
   c=node['*'];seq.append(('c',v,c));i=j
 return seq,last

results={}
for label in ['train','hartlieb']:
 base=(Z/f'{label}.pair.vbm').read_text().split()
 output=[];previous='-';lastcounts=collections.Counter()
 for w in base:
  seq,last=parse(w)
  new=''.join(x[1] if x[0]=='b' else tr[x[1]][previous][x[2]] for x in seq)
  output.append(new);lastcounts[previous]+=1
  # Stronger bound: update from the *previous emitted ciphertext*, not its baseline input.
  assert parse(new)[1]==last
  previous=parse(new)[1]
 full=' '.join(output)+'\n'
 actual=(B/f'{label}.carry.vbm').read_text()
 equality=full==actual
 (P/f'{label}.surface_autonomous.vbm').write_text(full)
 results[label]={'words':len(base),'identical':equality,'surface_sha256':hashlib.sha256(full.encode()).hexdigest(),'source_carry_sha256':hashlib.sha256(actual.encode()).hexdigest(),'initial_state':STATES[0], 'state_counts':dict(lastcounts)}
 # This is verification only and not read in producing transduced text
 from_file=(Z/f'{label}.reference.txt').read_text().splitlines()
 assert len(from_file)==len(base)
 last=-1;nsame=0
 for plaintext,cipher in zip(from_file,base):
  pv=next((char for char in reversed(plaintext) if char in 'aeiou'),'-')
  bv=parse(cipher)[1]
  nsame+=pv==bv
 results[label]['last_source_vowel_equals_last_visible_bridge']=nsame
 assert equality

(P/'EQUIVALENCE_RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
fd,t=tempfile.mkstemp(dir=P,prefix='.atomic.')
with os.fdopen(fd,'wb') as f:pickle.dump(results,f,protocol=5);f.flush();os.fsync(f.fileno())
os.replace(t,P/'stage2_autonomous_equivalence.pkl')
print(json.dumps(results,indent=2))

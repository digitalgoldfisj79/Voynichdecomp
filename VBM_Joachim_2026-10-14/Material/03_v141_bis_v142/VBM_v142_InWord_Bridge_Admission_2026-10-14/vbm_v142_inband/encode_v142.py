#!/usr/bin/env python3
"""Frozen-format in-word stateful-code engineering experiment. Not historical Voynich claim."""
import os,sys,json,collections,hashlib,importlib.util,pickle,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent
SRC=Path(os.environ.get('VBM_SOURCE_DIR','/mnt/data/vbm_work/gen138'))
Z=Path(os.environ.get('VBM_SOURCE_ZIP','/mnt/data/Generator_v138_Pruefpaket_2026-10-14.zip'))
VOW='aeiou'
# Single-glyph separator values. NONE begin a cell; this intentionally relaxes author's pair-vowel mechanism.
MARK={'a':'a','e':'e','i':'i','o':'o','u':'s'}

def atomic(name,obj):
 with tempfile.NamedTemporaryFile('wb',dir=P,delete=False) as f:
  pickle.dump(obj,f,protocol=5);f.flush();os.fsync(f.fileno());t=f.name
 os.replace(t,P/name)
def atext(name,s):
 with tempfile.NamedTemporaryFile('w',dir=P,encoding='utf-8',delete=False) as f:
  f.write(s);f.flush();os.fsync(f.fileno());t=f.name
 os.replace(t,P/name)
def j(name,o):atext(name,json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+'\n')

def author():
 prev=os.getcwd(); os.chdir(SRC);sys.path.insert(0,str(SRC))
 try:
  spec=importlib.util.spec_from_file_location('vbm142_inventory',SRC/'vbm_tafel_v138.py')
  v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
 finally:os.chdir(prev)
 return v
G=author()
word=lambda f: ['und' if t=='§' else t for t in G.gen.words_of((SRC/'klartext'/f).read_text(encoding='utf8',errors='ignore'))]
train=word('bairisch_kurzertext_GROSS.txt');hold=word('hartlieb_klartext.txt')
contract=json.loads((P/'00_admission_contract.json').read_text())
atomic('stage0_contract_provenance.pkl',{'contract':contract,'original_archive_sha256':hashlib.sha256(Z.read_bytes()).hexdigest(), 'source_files':{f:hashlib.sha256((SRC/'klartext'/f).read_bytes()).hexdigest() for f in ['bairisch_kurzertext_GROSS.txt','hartlieb_klartext.txt']}})

def runs(w):
 i=0
 while i<len(w) and w[i] not in VOW:i+=1
 initial=w[:i]; vr=[]
 while i<len(w):
  v=w[i]; assert v in VOW
  j=i+1
  while j<len(w) and w[j] not in VOW:j+=1
  vr.append((v,w[i+1:j]));i=j
 return initial,vr
c=collections.Counter()
for w in train:
 lead,rs=runs(w)
 for r in [lead]+[x[1] for x in rs]:
  c.update(r[i:i+2] for i in range(0,len(r),2) if len(r[i:i+2])==2)
letters=sorted(set(''.join(train))-set(VOW))
top216=sorted(c,key=lambda u:(-c[u],u))[:216]
# Never use holdout for fitting or choosing source pieces.
units=sorted(letters)+top216
assert len(set(units))==len(units)
assert len(units)<=235
unit_idx={x:i for i,x in enumerate(units)}
# Length-optimal ordering within public prefix-free VMS-B cell inventory under a shared global unit rank.
# No fitted per-state word tables. Inventory itself was previously fitted to the author's whole B sample.
cells=[tuple(x) for x in G.CELLS]
keys={v:{p[1] for p,w in G.pair_v.items() if w==v} for v in VOW}
state_tables={}
capacities={}
for v in VOW:
 allowed=[''.join(x) for x in cells if x[0] in keys[v]]
 leaves=[s for s in allowed if not any(t!=s and t.startswith(s) for t in allowed)]
 capacities[v]=len(leaves)
 # Deterministic length first and original frequency rank second, no distributional optimisation.
 orig={''.join(x):i for i,x in enumerate(cells)}
 leaves.sort(key=lambda s:(len(s),orig[s]))
 assert len(leaves)>=len(units),(v,len(leaves),len(units))
 state_tables[v]=leaves[:len(units)]
 assert len(set(state_tables[v]))==len(state_tables[v])
 assert all(not a.startswith(b) for i,a in enumerate(state_tables[v]) for j,b in enumerate(state_tables[v]) if i!=j)
assert not set(MARK.values())&set('ynrl')
manifest={'format':'vbm142_inband_engineering_v1','unit_list':units,'source_vowel_to_glyph':MARK,
 'default_initial_vowel_state':'e','cell_tables':state_tables,
 'construction':'same unit order for all vowels; state-specific cells are sorted shortest prefix-free leaves from fixed B cell inventory',
 'original_source_archive_sha256':hashlib.sha256(Z.read_bytes()).hexdigest(),
 'historical_claim':'NONE: this version changes the original pair-bridge mechanism and retains in-sample VMS B inventory'}
assert sum([len(units),len(cells),32])<=697
j('01_frozen_manifest.json',manifest)
atomic('stage1_grammar_capacity.pkl',{'units':len(units),'cell_inventory':len(cells),'prefix_free_capacity':capacities,'vowel_markers':MARK,'manifest_hash':hashlib.sha256((P/'01_frozen_manifest.json').read_bytes()).hexdigest()})

def chunk(s):
 out=[]
 for i in range(0,len(s),2):
  u=s[i:i+2]
  if u in unit_idx:out.append(u)
  else:out.extend(u)
 assert ''.join(out)==s
 return out

def code(v,c):return ''.join(state_tables[v][unit_idx[u]] for u in chunk(c))
def enc(w):
 assert w
 lead,vr=runs(w); token=code('e',lead)
 for v,c in vr:token+=MARK[v]+code(v,c)
 assert token
 return token
for label,data in [('train',train),('hartlieb',hold)]:
 ciphertext=' '.join(map(enc,data))+'\n';atext(label+'.vbm',ciphertext)
 atext(label+'.reference.txt','\n'.join(data)+'\n')
 atomic('stage2_emit_'+label+'.pkl',{'words':len(data),'cipher_chars':sum(map(len,ciphertext.split())), 'cipher_tokens':len(ciphertext.split()),'encoded_hash':hashlib.sha256(ciphertext.encode()).hexdigest(),'source_hash':hashlib.sha256('\n'.join(data).encode()).hexdigest()})
print(json.dumps({'units':len(units),'capacities':capacities,'source_word_counts':(len(train),len(hold))},indent=2))

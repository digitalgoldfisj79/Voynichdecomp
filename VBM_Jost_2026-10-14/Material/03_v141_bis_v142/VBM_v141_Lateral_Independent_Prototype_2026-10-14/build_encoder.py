#!/usr/bin/env python3
"""Frozen-composition feasibility experiment; engineering evidence only.
Actual source package remains untouched. Emits self-describing control tokens,
not claimed as Voynich manuscript-like running text.
"""
import hashlib, json, os, pickle, tempfile, importlib.util, sys, collections, re
from pathlib import Path
P=Path(os.environ.get('VBM_OUTPUT_DIR','/mnt/data/vbm_v141_lateral_run')); P.mkdir(parents=True,exist_ok=True)
S=Path(os.environ.get('VBM_SOURCE_DIR','/mnt/data/vbm_work/gen138'))
ZIP=Path(os.environ.get('VBM_SOURCE_ZIP','/mnt/data/Generator_v138_Pruefpaket_2026-10-14.zip'))
VOW='aeiou'

def atomwrite(filename,blob):
 with tempfile.NamedTemporaryFile(dir=P,delete=False) as f:
  f.write(blob); f.flush(); os.fsync(f.fileno()); tmp=f.name
 os.replace(tmp,P/filename)
def jwrite(filename,obj): atomwrite(filename,json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False).encode())
def checkpoint(name,obj): atomwrite(name+'.pkl',pickle.dumps(obj,protocol=5))

def load_author():
 old=os.getcwd(); os.chdir(S);sys.path.insert(0,str(S))
 try:
  spec=importlib.util.spec_from_file_location('vbm138_source_for_inventory',S/'vbm_tafel_v138.py'); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 finally:os.chdir(old)
 return module
G=load_author()
words=lambda p: ['und' if w=='§' else w for w in G.gen.words_of(Path(p).read_text(encoding='utf8',errors='ignore'))]
train=words(S/'klartext/bairisch_kurzertext_GROSS.txt')
hold=words(S/'klartext/hartlieb_klartext.txt')
# Precommit before any ciphertext operation: <=235 shared units, <=443 target cells,
# <=40 bridge codes, <=12 grammar productions; exact recovery 100% or fail.
contract={
 'date':'2026-10-14','name':'compositional-vbm-lateral-compact-v1',
 'hypotheses':['H0 symbolic stateful substitution is syntactically feasible',
   'H1 independently decodable without source or per-message ledgers',
   'H2 VMS-B surface constraint satisfies held-out manuscript token structure (untested, separate gate)',
   'H3 VBM identifies historical Voynich source (unidentifiable from engineering-only test)'],
 'limits':{'shared_unit_vocabulary':235,'vms_cell_inventory':443,'bridge_pairs':40,'explicit_rules':12},
 'rules':['casefold, diacritic removal, letter normalisations exactly as source package',
  'canonicalise § lexical macro to und; original spelling is irrecoverable',
  'split each word into maximal leading-consonant and vowel-plus-consonant runs',
  'split consonant runs into greedy pairs, then unknown pair into two singleton letters',
  'one shared primitive rank dictionary used for all five vowel states',
  'cell index is primitive rank in vowel-eligible cells (not five handwritten dictionaries)',
  'leading consonants use public default e state',
  'one m+bridge control token introduces each vowel; the NEXT cell key disambiguates its vowel',
  'after a vowel an empty consonant run is encoded by its reserved epsilon cell',
  'consecutive cell tokens are continuation within current consonant run',
  's token explicitly terminates every normalised source word',
  'decoder obtains everything from ciphertext and frozen public manifest; reject invalid grammar'],
 'scope':'Engineering-only synthetic test. m and s markers are not a qualified Voynich surface grammar.',
 'success_requires':['100% exact normalised text and boundaries on development and Hartlieb',
   '100% parse and detection of injected invalid control/collision/undelimited EOF',
   'no decoder source-file or hidden event ledger input',
   'no manifest mutation between development and Hartlieb',
   'no inventory/complexity ceiling exceedance'],
 'no_historical_inference_without':['matched section-conditioned held-out tests',
   'positive historical-language anchors and hard negative matched slot generators',
   'observable separating VBM from autonomous non-plaintext mechanism']}
jwrite('00_frozen_contract.json',contract)
checkpoint('stage0_contract_provenance',{'contract':contract,'original_zip_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest()})
# Consistent with source package: extract 443 cells and original bridge-pair assignments.
cells=G.CELLS
assert len(cells)==443 and len(set(map(tuple,cells)))==len(cells)
keys={v:{pair[1] for pair,vv in G.pair_v.items() if vv==v} for v in VOW}
state_cells={v:[list(c) for c in cells if c[0] in keys[v]] for v in VOW}
assert {v:len(x) for v,x in state_cells.items()}=={'a':357,'e':443,'i':236,'o':236,'u':259}
# Use shared vocabulary: 18 training-source singleton consonants, up to 216
# most common ordered digraph chunks, and epsilon; capacity upper bound 235.
letter_set=sorted(set(''.join(train))-{*VOW})
assert len(letter_set)==18, letter_set
c=collections.Counter()
def split_runs(w):
 # returns initial consonant string and list[(vowel, consonant run)]
 i=0
 while i<len(w) and w[i] not in VOW:i+=1
 initial=w[:i]; runs=[]
 while i<len(w):
  assert w[i] in VOW,(w,i)
  v=w[i];j=i+1
  while j<len(w) and w[j] not in VOW:j+=1
  runs.append((v,w[i+1:j]));i=j
 return initial,runs
for w in train:
 lead,runs=split_runs(w)
 for cluster in [lead]+[run for v,run in runs]:
  c.update(cluster[i:i+2] for i in range(0,len(cluster),2) if len(cluster[i:i+2])==2)
top216=sorted(c,key=lambda x:(-c[x],x))[:216]
assert len(top216)>0
units=['']+sorted(letter_set)+top216
assert len(units)<=235 and len(set(units))==len(units)
unit_index={u:i for i,u in enumerate(units)}
# Pair matrix is never fitted to Hartlieb.
pairs={f'{br}|{k}':v for (br,k),v in G.pair_v.items() if k in 'ynrl'}
assert len(pairs)<=40,len(pairs)
# Compact rank representation uses only one primitive table and original cell inventory.
# The cell spellings must be unique *within the corresponding vowel state*.
def surface(cell):return ''.join(cell)
assert all(len(set(map(surface, cs)))==len(cs) for cs in state_cells.values())
assert all(len(cs)>len(units) for cs in state_cells.values())
# Check marker grammar disjointness by initial glyph, none of the native cells starts m or s.
assert all(surface(c)[0] in 'ynrl' for cs in state_cells.values() for c in cs)
# Create small pair pick table: source vowel and key -> least-complex available bridge.
bridges={}
for v in VOW:
 for k in sorted(keys[v]):
  bs=[br for (br,kk),vv in G.pair_v.items() if kk==k and vv==v]
  assert bs
  bridges[f'{v}|{k}']=min(bs,key=lambda b:(len(b.replace('-','')), b!='-',b))
manifest={'format':'vbm-compact-v1','source_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),
 'cells':list(map(list,cells)),'units':units,'pair_to_vowel':pairs,
 'key_eligibility':{v:sorted(keys[v]) for v in VOW},
 'control':{'bridge_prefix':'m','word_end':'s','no_bridge':'-'},
 'initial_vowel_state':'e','codeword_surface':'concat(key,state,head)',
 'notes':['m+bridge is an explicit separate token, not a real VMS bridge suffix',
         'word boundary is dedicated s token, not VMS whitespace',
         'the state-specific mapping is obtained by shared primitive rank indexing into filtered cell list']}
jwrite('01_frozen_manifest.json',manifest)
checkpoint('stage1_tables',{'inventory':len(cells),'primitives':len(units),'pairs':len(pairs),'bridge_choices':bridges,
 'state_capacities':{v:len(cs) for v,cs in state_cells.items()}, 'top216':top216})

def chunk(s):
 result=[]
 for i in range(0,len(s),2):
  d=s[i:i+2]
  result.append(d) if d in unit_index else result.extend(d)
 assert ''.join(result)==s and all(x in unit_index for x in result)
 return result

def code(v,u):return surface(state_cells[v][unit_index[u]])

def enc_word(w):
 lead,runs=split_runs(w); out=[code('e',u) for u in chunk(lead)]
 for vowel,run in runs:
  parts=chunk(run) if run else ['']
  cell0=state_cells[vowel][unit_index[parts[0]]]
  br=bridges[vowel+'|'+cell0[0]]
  out.append('m'+('' if br=='-' else br))
  out.extend(code(vowel,u) for u in parts)
 out.append('s')
 return out

def encode(words):return ' '.join(t for w in words for t in enc_word(w))+'\n'
# No output stream is used to construct the manifest.
for label,words_ in [('train',train),('hartlieb',hold)]:
 data=encode(words_);atomwrite(label+'.vbm',data.encode('ascii'))
 chunk_units=sum(1 for t in data.split() if t[0] in 'ynrl')
 marker_count=sum(1 for t in data.split() if t[0] in 'ms')
 checkpoint('stage2_emit_'+label,{'word_count':len(words_),'normalized_characters':sum(map(len,words_)),
  'ciphertext_tokens':len(data.split()),'cell_tokens':chunk_units,'control_tokens':marker_count,
  'ciphertext_sha256':hashlib.sha256(data.encode('ascii')).hexdigest(),
  'source_word_hash':hashlib.sha256('\n'.join(words_).encode('ascii')).hexdigest()})
# Holdout for verifier: normalized words provided as clear target only to verifier, not decoder.
for label,sourcewords in [('train',train),('hartlieb',hold)]:
 atomwrite(label+'.reference.txt',('\n'.join(sourcewords)+'\n').encode('ascii'))
print(json.dumps({'frozen_unit_count':len(units),'cell_count':len(cells),'pair_count':len(pairs),
 'training_words':len(train),'hartlieb_words':len(hold),
 'state_capacities':{v:len(cs) for v,cs in state_cells.items()}},indent=2))

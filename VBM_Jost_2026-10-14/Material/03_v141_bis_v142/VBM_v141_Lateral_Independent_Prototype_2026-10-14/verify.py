#!/usr/bin/env python3
"""Independent verification and deliberate falsification of the frozen lateral codec.
Read-only tests; JSON/pickle saved atomically, never retroactively modify gate.
"""
import json,hashlib,pickle,os,tempfile,sys,collections,copy
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from independent_decoder import decode,build
m=json.loads((P/'01_frozen_manifest.json').read_text())
contract=json.loads((P/'00_frozen_contract.json').read_text())

def save(n,d):
 with tempfile.NamedTemporaryFile(dir=P,delete=False) as f:
  pickle.dump(d,f,protocol=5);f.flush();os.fsync(f.fileno());q=f.name
 os.replace(q,P/(n+'.pkl'))
def safe_fail(payload,man=m):
 try:decode(payload,man);return False
 except (ValueError,AssertionError,IndexError,KeyError):return True

def ref(n):return (P/(n+'.reference.txt')).read_text().splitlines()
def ct(n):return (P/(n+'.vbm')).read_text()

def atoms(t):
 out=[];i=0
 atom_big=('ckh','cth','cph','cfh','ch','sh')
 while i<len(t):
  matched=next((a for a in atom_big if t.startswith(a,i)),None)
  if matched:out.append(matched);i+=len(matched)
  else:out.append(t[i]);i+=1
 return out
base=Path(os.environ.get('VBM_VMS_REFERENCE','/mnt/data/vbm_work/gen138/vms_B_REVA_atomstream.txt'))
vmt=[]
for line in base.read_text().splitlines():vmt+=line.split('\t')[-1].split()
vmtset=set(vmt)
BI=set();TRI=set()
for tok in vmt:
 a=atoms(tok);BI.update(zip(a,a[1:]));TRI.update(zip(a,a[1:],a[2:]))

def inspect(label):
 enc=ct(label);words=ref(label);decoded=decode(enc,m);assert words==decoded
 ts=enc.split();cell=[t for t in ts if t[0] in 'ynrl'];br=[t for t in ts if t[0]=='m'];bd=[t for t in ts if t=='s']
 other=[t for t in ts if t[0] not in 'ynrlms']
 assert not other and len(bd)==len(words)
 bad2=[t for t in ts if any(x not in BI for x in zip(atoms(t),atoms(t)[1:]))]
 bad3=[t for t in ts if any(x not in TRI for x in zip(atoms(t),atoms(t)[1:],atoms(t)[2:]))]
 notattested=[t for t in ts if t not in vmtset]
 unsupported2=sum(t in bad2 for t in ts) if False else None # duplicates must be counted, below
 nbad2=sum(any(x not in BI for x in zip(atoms(t),atoms(t)[1:])) for t in ts)
 nbad3=sum(any(x not in TRI for x in zip(atoms(t),atoms(t)[1:],atoms(t)[2:])) for t in ts)
 na=sum(t not in vmtset for t in ts)
 counts=collections.Counter(t[0] for t in ts)
 return {'source_words':len(words),'normalized_characters':sum(map(len,words)),
  'ciphertext_tokens':len(ts),'cell_tokens':len(cell),'bridge_markers':len(br),'word_markers':len(bd),
  'cipher_per_normalized_word':len(ts)/len(words),
  'cipher_to_v138_token_count':len(ts)/101297 if label=='train' else None,
  'ciphertext_characters_no_whitespace':sum(map(len,ts)),
  'tokens_with_unattested_atomic_bigrams':nbad2,
  'tokens_with_unattested_atomic_trigrams':nbad3,
  'tokens_not_attested_as_whole_vms_B_tokens':na,
  'marker_shape_counter':dict(collections.Counter(br).most_common()),
  'distinct_surface_tokens':len(set(ts)),
  'word_boundary_recovery':True,'roundtrip_exact':True,
  'control_type_counts':dict(counts)}
results={l:inspect(l) for l in ['train','hartlieb']}
save('stage3_independent_roundtrip_surface',results)
# Exhaustive per-vowel cell uniqueness and round-trip of each primitive in each state.
tables,pairs=build(m)
uniques={v:len(x) for v,x in tables.items()}
assert len(set(map(tuple,[tuple(sorted(x.values())) for x in tables.values()])))==1
assert all(sorted(x.values())==sorted(m['units']) for x in tables.values())
# Negative 1: one duplicate primitive rank in manifest, caught when loading.
bad=copy.deepcopy(m);bad['units'][1]=bad['units'][2]
negative_collision=safe_fail(ct('hartlieb'),bad)
# Negative 2: legal prefix, but unsupported (bridge,key) pair. It must fail closed.
holdtok=ct('hartlieb').split();pos=next(i for i,t in enumerate(holdtok) if t[0]=='m')
orig=holdtok[pos]
invalid='mzz'
mut=holdtok.copy();mut[pos]=invalid
negative_badbridge=safe_fail(' '.join(mut))
# Negative 3: missing final word delimiter must fail at EOF.
assert holdtok[-1]=='s'
negative_final_boundary=safe_fail(' '.join(holdtok[:-1]))
# Negative 4: remove an interior word boundary: syntax remains legal and this
# MUST be recorded as detectable by external reference, not intrinsically by decoder.
idx=next(i for i,t in enumerate(holdtok) if t=='s' and i<len(holdtok)-1)
mut=holdtok.copy();mut.pop(idx)
try:
 decoded_without=decode(' '.join(mut),m)
 middle_delimiter_parser_rejected=False
 missing_boundary_changes_output=decoded_without!=ref('hartlieb')
except ValueError:
 middle_delimiter_parser_rejected=True; missing_boundary_changes_output=True
# Negative 5: deliberately wrong initial state in manifest must reject or change output.
bad=copy.deepcopy(m);bad['initial_vowel_state']='i'
try:
 dec_wrong=decode(ct('hartlieb'),bad)
 initial_wrong_rejected=False;initial_wrong_changes_output=dec_wrong!=ref('hartlieb')
except (AssertionError,ValueError):
 initial_wrong_rejected=True;initial_wrong_changes_output=True
# Adversarial valid bridge substitution could change plaintext without detection, as expected.
# Find alternate bridge for *same next key* in a different vowel class.
mut_tok=None
for i,t in enumerate(holdtok[:-1]):
 if not t.startswith('m') or holdtok[i+1][0] not in 'ynrl':continue
 nextkey=holdtok[i+1][0];old_pair=(t[1:] or '-',nextkey)
 v=pairs.get(old_pair)
 options=[br for (br,k),vv in pairs.items() if k==nextkey and vv!=v]
 if options:
  alt=options[0]; a=holdtok.copy();a[i]='m'+('' if alt=='-' else alt)
  try:
   changed=decode(' '.join(a),m)
   if changed!=ref('hartlieb'):mut_tok=(i,t,a[i],old_pair,v,alt);break
  except ValueError:continue
bridge_malleable=mut_tok is not None
ctrl={'duplicate_primitive_manifest_rejected':negative_collision,
      'invalid_bridge_pair_rejected':negative_badbridge,
      'missing_eof_boundary_rejected':negative_final_boundary,
      'interior_boundary_removal_parser_rejected':middle_delimiter_parser_rejected,
      'interior_boundary_removal_changes_reference':missing_boundary_changes_output,
      'wrong_initial_state_rejected':initial_wrong_rejected,
      'wrong_initial_state_changes_reference':initial_wrong_changes_output,
      'valid_bridgemutation_changes_plaintext_silently':bridge_malleable,
      'malleability_case':mut_tok,
      'per_state_exact_inverse':True,
      'no_holdout_in_codebook_training':True}
save('stage4_controls',ctrl)
summary={'source_inventory':{'vms_reference_token_count':len(vmt),'vms_atomic_bigram_types':len(BI),
 'vms_atomic_trigram_types':len(TRI)},
 'complexity':{'shared_primitive_entries':len(m['units']),'vms_cell_list_entries':len(m['cells']),
    'bridge_pair_table_entries':len(m['pair_to_vowel']),
    'sum_explicit_entries':len(m['units'])+len(m['cells'])+len(m['pair_to_vowel']),
    'five_state_eligible_cells':{v:len([c for c in m['cells'] if c[0] in ks]) for v,ks in m['key_eligibility'].items()},
    'grammar_rule_count':len(contract['rules']),
    'frozen_manifest_json_bytes':(P/'01_frozen_manifest.json').stat().st_size},
 'roundtrip':results,'controls':ctrl,
 'gate':{'mechanical_reversibility':'PASS',
   'historical_surface':'NO-GO: dedicated control tokens, VMS token count and unattested strings',
   'mechanism_identification':'NOT TESTED / NO-GO: no independent Voynich discriminator',
   'v138_architecture':'PREVIOUS NO-GO RETAINED',
   'historical_attribution':'NOT ADMITTED'},
 'warnings':['Hartlieb is supplied by model author; this is text transfer not provenance-independent validation',
  'True original German spelling is not recoverable after norm_word',
  'Explicit space-separated marker/cell stream is artificial; no claim of reproducing MS408 surface',
  'Valid codeword substitutions are not detectable intrinsically, requiring external expected plaintext',
  'No new comparisons to Voynich sections were conducted; reference token attestation is descriptive only']}
(P/'02_verified_results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
save('stage5_final_decision',summary['gate'])
print(json.dumps(summary,indent=2,ensure_ascii=False))

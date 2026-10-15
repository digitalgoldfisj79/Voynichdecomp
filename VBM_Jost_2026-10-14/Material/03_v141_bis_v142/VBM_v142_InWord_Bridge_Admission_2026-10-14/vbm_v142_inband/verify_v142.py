#!/usr/bin/env python3
"""Independent engineering checks and descriptive VMS B surface gate.
No fit/optimisation to the evaluation samples. VMS inventory is author-supplied in-sample.
"""
import json,collections,random,statistics,pickle,os,tempfile,sys
from pathlib import Path
P=Path(__file__).resolve().parent
SRC=Path(os.environ.get('VBM_SOURCE_DIR','/mnt/data/vbm_work/gen138'))
sys.path.insert(0,str(P))
from decode_v142_independent import decode,grammar
m=json.loads((P/'01_frozen_manifest.json').read_text());t,rev=grammar(m)

def atomic(name,obj):
 with tempfile.NamedTemporaryFile('wb',dir=P,delete=False) as f:
  pickle.dump(obj,f,protocol=5);f.flush();os.fsync(f.fileno());q=f.name
 os.replace(q,P/name)
def diagnostic(label):
 a=(P/(label+'.vbm')).read_text().split();src=(P/(label+'.reference.txt')).read_text().splitlines();out=decode(' '.join(a),m)
 assert out==src and len(out)==len(a)
 return {'word_count':len(a),'source_characters':sum(map(len,src)),'cipher_characters':sum(map(len,a)),
         'source_char_per_word':statistics.mean(map(len,src)), 'cipher_char_per_word':statistics.mean(map(len,a)),
         'cipher_char_median':statistics.median(map(len,a)),'cipher_95pct':sorted(map(len,a))[int(.95*len(a))],
         'types':len(set(a)), 'exact':True}
res={'engineering':{x:diagnostic(x) for x in ('train','hartlieb')}}
# Adversarial controls: positive arbitrary additions, negatives collision, prefix collision, vowel-marker collision,
# unknown glyph/word collision. Does not assume authentication against valid substitutions.
ctl={}
import copy
valid=' '.join((P/'hartlieb.vbm').read_text().split()[:120])
ctl['independent_roundtrip_hartlieb']=decode((P/'hartlieb.vbm').read_text(),m)==(P/'hartlieb.reference.txt').read_text().splitlines()
for name,modify in {
  'same_state_code_collision':lambda z:z['cell_tables']['i'].__setitem__(1,z['cell_tables']['i'][0]),
  'state_marker_collision':lambda z:z['source_vowel_to_glyph'].__setitem__('u','y'),
  'prefix_collision':lambda z:z['cell_tables']['i'].__setitem__(1,z['cell_tables']['i'][0]+'a')}.items():
 bad=copy.deepcopy(m);modify(bad)
 try:grammar(bad);ctl[name+'_detected']=False
 except ValueError:ctl[name+'_detected']=True
try:decode(valid+'$',m);ctl['invalid_glyph_rejected']=False
except ValueError:ctl['invalid_glyph_rejected']=True
# delete one ciphertext whitespace; lexical boundaries may be merged silently but comparison exposes error.
toks=(P/'hartlieb.vbm').read_text().split();joined=toks.copy();joined[20]=joined[20]+joined.pop(21)
try:
 mutant=decode(' '.join(joined),m)
 ctl['deleted_word_space_detected_intrinsically']=False
 ctl['deleted_word_space_changes_decoding']=(mutant!=(P/'hartlieb.reference.txt').read_text().splitlines())
except ValueError:
 ctl['deleted_word_space_detected_intrinsically']=True
 ctl['deleted_word_space_changes_decoding']=True
# Positive control: parser must handle arbitrary new composed strings (and spaces), not corpus-indexed dictionary.
cell_i=m['cell_tables']['i'][7];cell_e=m['cell_tables']['e'][17]
synthetic=cell_e+'i'+cell_i+'a'+'e'+cell_e
ctl['novel_valid_synthetic_word_decodable']=(decode(synthetic,m)==[m['unit_list'][17]+'i'+m['unit_list'][7]+'ae'+m['unit_list'][17]])
res['controls']=ctl
# Native reference atomisation. Uses author's *fixed* token atomizer exclusively for diagnostics.
sys.path.insert(0,str(SRC))
from panel_v126 import atoms
byfolio=collections.defaultdict(list)
for line in (SRC/'vms_B_REVA_atomstream.txt').read_text().splitlines():
 if '\t' not in line:continue
 k,sequence=line.split('\t',1)
 fol=k.split('.')[0]
 byfolio[fol].extend(sequence.split())
ref=[word for fo,words in byfolio.items() for word in words]
bi=set();tri=set()
for w in ref:
 a=atoms(w)
 bi.update(zip(a,a[1:]));tri.update(zip(a,a[1:],a[2:]))
def seam_stats(seq):
 lengths=list(map(len,seq));badbi=0;badtri=0;badbi_pos=0;badtri_pos=0;examples=[]
 for w in seq:
  a=atoms(w); bb=[x for x in zip(a,a[1:]) if x not in bi];tt=[x for x in zip(a,a[1:],a[2:]) if x not in tri]
  if bb:badbi+=1;badbi_pos+=len(bb)
  if tt:badtri+=1;badtri_pos+=len(tt)
  if (bb or tt) and len(examples)<7:examples.append({'token':w,'invalid_bigram':bb[:2],'invalid_trigram':tt[:2]})
 return {'count':len(seq),'mean_character_length':statistics.mean(lengths),'median':statistics.median(lengths),
  'pct95':sorted(lengths)[int(.95*len(seq))], 'unattested_bigram_token_count':badbi,
  'unattested_trigram_token_count':badtri, 'unattested_bigram_token_fraction':badbi/len(seq),
  'unattested_trigram_token_fraction':badtri/len(seq),
  'bad_bigram_positions':badbi_pos,'bad_trigram_positions':badtri_pos, 'examples':examples}
res['surface']={'reference_b':seam_stats(ref),'by_folio':{f:seam_stats(ws) for f,ws in byfolio.items()},
 'train':seam_stats((P/'train.vbm').read_text().split()),
 'hartlieb':seam_stats((P/'hartlieb.vbm').read_text().split()),
 'native_reference_folios':len(byfolio), 'inventory_in_sample':True}
res['surface']['reference_folio_mean_length_min_max']=[min(x['mean_character_length'] for x in res['surface']['by_folio'].values()),max(x['mean_character_length'] for x in res['surface']['by_folio'].values())]
# Independent *negative mechanism* control: shuffle within each real VMS line, preserving its tokens,
# its line edges and folio assignment: every token passes seam admissibility even though order destroyed.
rng=random.Random(142014)
shuffled=[];orig=[]
for ln in (SRC/'vms_B_REVA_atomstream.txt').read_text().splitlines():
 if '\t' not in ln:continue
 toks=ln.split('\t',1)[1].split()
 if len(toks)>2:
  mid=toks[1:-1].copy();rng.shuffle(mid);toks=[toks[0]]+mid+[toks[-1]]
 shuffled.extend(toks)
res['controls']['section_preserving_line_shuffle_surface_positive']=(seam_stats(shuffled)['unattested_trigram_token_count']==0 and len(shuffled)==len(ref))
res['controls']['word_order_is_not_tested_by_seam_gate']=True
# Explicit null on target metric cannot be obtained from shuffled token order, by mathematical construction.
res['surface']['seam_matched_shuffle_null_standard_deviation']=0.0
res['surface']['null_interpretation']='All within-line token permutations retain the exact token multiset, hence seam-count SD exactly zero; seam gate cannot discriminate any ordering mechanism.'
res['gates']={
 'exact_roundtrip':all(v['exact'] for v in res['engineering'].values()),
 'one_token_per_source_word':all(v['word_count']==len((P/(k+'.reference.txt')).read_text().splitlines()) for k,v in res['engineering'].items()),
 'prefix_free':True,
 'negative_control_sensitivity':all(ctl[k] for k in ['same_state_code_collision_detected','state_marker_collision_detected','prefix_collision_detected','invalid_glyph_rejected','deleted_word_space_changes_decoding','novel_valid_synthetic_word_decodable']),
 'seam_bigram_pass':all(res['surface'][k]['unattested_bigram_token_count']==0 for k in ('train','hartlieb')),
 'seam_trigram_pass':all(res['surface'][k]['unattested_trigram_token_count']==0 for k in ('train','hartlieb')),
 'historical_mechanism':'NOT_IDENTIFIED',
 'author_pair_vowel_mechanism':'NOT_IMPLEMENTED (explicitly relaxed)'
}
(P/'02_test_results.json').write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n')
atomic('stage3_roundtrip_and_controls.pkl',{'engineering':res['engineering'],'controls':ctl})
atomic('stage4_surface_and_gate.pkl',{'surface':res['surface'],'gates':res['gates']})
print(json.dumps({'engineering':res['engineering'],'controls':ctl,
 'train_surface':res['surface']['train'], 'hartlieb_surface':res['surface']['hartlieb'],
 'reference_mean':res['surface']['reference_b']['mean_character_length'],
 'folio_length_range':res['surface']['reference_folio_mean_length_min_max'],'gates':res['gates']},indent=2))

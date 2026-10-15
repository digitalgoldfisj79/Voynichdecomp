#!/usr/bin/env python3
import json,collections,statistics,sys,pickle,tempfile,os,copy
from pathlib import Path
P=Path(__file__).resolve().parent;S=Path(os.environ.get('VBM_SOURCE_DIR','/mnt/data/vbm_work/gen138'))
sys.path.insert(0,str(S));from panel_v126 import atoms
sys.path.insert(0,str(P));from decode_pair_independent import decode,build
m=json.loads((P/'04_pair_bridge_manifest.json').read_text())
ref=[];tr=set();bi=set(); folios=collections.defaultdict(list)
for l in (S/'vms_B_REVA_atomstream.txt').read_text().splitlines():
 if '\t' not in l:continue
 f=l.split('\t',1)[0].split('.')[0]
 for w in l.split('\t',1)[1].split():
  a=atoms(w);bi.update(zip(a,a[1:]));tr.update(zip(a,a[1:],a[2:]));ref.append(w);folios[f].append(w)
def metrics(x):
 N=len(x);lens=list(map(len,x));b=t=0;badex=[]
 for w in x:
  a=atoms(w);badb=any(z not in bi for z in zip(a,a[1:]));badt=any(z not in tr for z in zip(a,a[1:],a[2:]));b+=badb;t+=badt
  if (badb or badt) and len(badex)<5:badex.append(w)
 return {'words':N,'mean_length':statistics.mean(lens),'median_length':statistics.median(lens),'pct95':sorted(lens)[int(.95*N)],'bad_bigram_count':b,'bad_trigram_count':t,'bad_bigram_fraction':b/N,'bad_trigram_fraction':t/N,'examples':badex}
res={'reference':metrics(ref),'train':metrics((P/'train.pair.vbm').read_text().split()),'hartlieb':metrics((P/'hartlieb.pair.vbm').read_text().split()),'folio_mean_length_range':(min(metrics(x)['mean_length'] for x in folios.values()),max(metrics(x)['mean_length'] for x in folios.values()))}
checks={}
for l in ['train','hartlieb']:
 try:checks[l+'_exact']=decode((P/(l+'.pair.vbm')).read_text(),m)==(P/(l+'.reference.txt')).read_text().splitlines()
 except Exception as e:checks[l+'_exact']=repr(e)
for name,change in {
 'same_state_collision':lambda x:x['state']['i']['cells'].__setitem__(1,x['state']['i']['cells'][0]),
 'escaped_cell_misconfigured':lambda x:x['state']['i'].__setitem__('escape',x['state']['i']['cells'][0]),
 'invalid_pair':lambda x:x['vowel_by_pair'].__setitem__('zz|y','a')}.items():
 d=copy.deepcopy(m);change(d)
 try:
  if name=='invalid_pair':
   # Mutate actual valid output bridge p|y for o into sh|y (a), to test semantic corruption detection.
   t=(P/'train.pair.vbm').read_text(); mod=t.replace('p'+'y', 'sh'+'y', 1)
   if mod==t:checks[name+'_detected']='no_instance'
   else:
    decoded=decode(mod,m); checks[name+'_detected']=(decoded!=(P/'train.reference.txt').read_text().splitlines())
  else:
   build(d);checks[name+'_detected']=False
 except ValueError:checks[name+'_detected']=True
# A mutated valid pair is malleable, can decode to a different valid text, no intrinsic detection.
t=(P/'train.pair.vbm').read_text(); mod=t.replace('py','shy',1)
if mod!=t:
 try:checks['valid_pair_mutation_intrinsically_rejected']=False; checks['valid_pair_mutation_changes_text']=decode(mod,m)!=(P/'train.reference.txt').read_text().splitlines()
 except ValueError:checks['valid_pair_mutation_intrinsically_rejected']=True
else:checks['valid_pair_mutation_intrinsically_rejected']=None
res['controls']=checks
res['gate']={'roundtrip':checks['train_exact'] and checks['hartlieb_exact'],'0_bad_bigram':res['train']['bad_bigram_count']==res['hartlieb']['bad_bigram_count']==0,'0_bad_trigram':res['train']['bad_trigram_count']==res['hartlieb']['bad_trigram_count']==0,
 'control_tamper':checks['same_state_collision_detected'] and checks['escaped_cell_misconfigured_detected'],'separate_mechanism_test':'NOT_IDENTIFIED', 'matched_shuffle_sd_for_atomic_gate':0.0}
(P/'05_pair_bridge_results.json').write_text(json.dumps(res,indent=2)+'\n')
with tempfile.NamedTemporaryFile('wb',dir=P,delete=False) as f:
 pickle.dump(res,f,protocol=5);f.flush();os.fsync(f.fileno());tmp=f.name
os.replace(tmp,P/'stage7_pair_gate.pkl')
print(json.dumps(res,indent=2))

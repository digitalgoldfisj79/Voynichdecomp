#!/usr/bin/env python3
"""Follow-on exploratory shortcut sensitivity after initial four-shortcut results; not blind evidence."""
import sys, json, time, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_abbreviations as m
P=Path(__file__).resolve().parent.parent/'results'
CONFIGS={
 'zero':[],
 'two':['en','er'],
 'four':['en','er','ch','un'],
 'six':['en','er','ch','un','nd','ge'],
 'four_alt':['de','an','te','st']
}
RES={'disclaimer':'Development A/B reference used for code selection; Hartlieb has been repeatedly examined in earlier VBM rounds and is not a fresh blind. Exploratory grid added after initial 4-shortcut result. No cipher attribution.', 'configs':CONFIGS, 'scores':{}}
for fam,f in [('A','A_REVA_reference.txt'),('B','VMS_B_REVA_reference.txt')]:
 if len(sys.argv)>1 and fam not in sys.argv[1]:continue
 ref=m.target(f);su=m.prep(ref)
 for config,abbr in CONFIGS.items():
  if len(sys.argv)>2 and config!=sys.argv[2]:continue
  train, direct, escaped=m.construct_model(m.D,abbr,ref);test=m.tokenize(m.H,abbr)
  table={**direct,**escaped};calib=train[::40];valid=train[5::37]
  # two stochastic optimizers; select by held-out *development* validation
  chosen=[];start=time.time()
  for k in range(2):
   fitted,fitcost=m.optimise(table,calib,su,seed=202610144 + (0 if fam=='A' else 1000)+len(abbr)*37 + (51 if config=='four_alt' else 0)+k*137,iterations=700)
   validate=m.evaluate(valid,fitted,su)
   chosen.append((validate['objective'],k,fitted,validate,fitcost))
  score,rep,chosen_table,validate,fitcost=min(chosen,key=lambda z:z[0]);stats=m.evaluate(test,chosen_table,su,details=True)
  outputs=stats.pop('cipher_words');D={v:k for k,v in chosen_table.items()}
  assert len(D)==len(chosen_table)
  def decode(encoded):
   glyphs=m.ATOMS.findall(encoded)
   words=[];i=0
   while i<len(glyphs):
    if escaped and glyphs[i]=='g':
     assert i+1<len(glyphs)
     code=glyphs[i]+glyphs[i+1];i+=2
    else:code=glyphs[i];i+=1
    words.append(D[code])
   return ''.join(words)
  assert all(decode(w)==t for w,t in zip(outputs,m.H))
  data={'family':fam,'shortcut_set':abbr,'mapping':chosen_table,'direct_entries':len(direct),'escaped_entries':len(escaped),'table_entries':len(chosen_table),'development_validation':validate,'hartlieb_diagnostic':stats,'optimise_seconds':round(time.time()-start,3),'exact_recovery_n':len(m.H),'ops_per_word':sum(map(len,test))/len(test),'escape_calls_per_word':sum(x in escaped for row in test for x in row)/len(test),'selected_seed_index':rep,'training_objective':fitcost}
  RES['scores'][fam+'_'+config]=data
  print(fam,config, 'bad3',round(stats['bad_trigram_rate'],3),'avglen',round(stats['mean_length'],3),'long',round(stats['share_len_ge_10'],3),'dev val',round(score,3), 'time',round(time.time()-start,1), flush=True)

  save=P/'EXPLORATORY_RESULTS.json'
  if save.exists():
   old=json.loads(save.read_text());old['scores'].update(RES['scores']);RES=old
  save.write_text(json.dumps(RES,indent=2)+'\n')
print('done')

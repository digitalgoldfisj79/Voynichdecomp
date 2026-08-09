#!/usr/bin/env python3
import urllib.request
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/db711a6c3be6c742ae70c3a9df3fa463e9320553/experiments/bnf_m19_hmm_dev/dev_true_mapping_forward.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
ns={'__name__':'dev_true'}
exec(compile(src,'dev_true_mapping_forward.py','exec'),ns)

def main():
 lms,holds,meta=ns['ns']['load_sources']();base=ns['ns'];lang='latin';rep=0
 span=base['choose_span'](holds[lang],base['TRAIN']+base['HOLD'],(lang,rep));cipher,true,attempt=ns['gen'](span,lang,rep);ctra,cho=base['split_letters'](cipher,base['TRAIN']);symbols=[chr(65+i) for i in range(25)];Str=base['stats'](ctra.split(),symbols);how=cho.split();Sho=base['stats'](how,symbols)
 rows=[]
 for la in base['LANGS']:
  sc,m=base['optimize'](Str,base['induced'](lms[la]),('devfit',lang,rep,la));fw,n,bad=ns['forward_corpus'](how,m,lms[la]);acc=base['weighted_map_acc'](Sho,m,true);rows.append((la,fw,acc,sc));print('FIT_FORWARD',la,'forward',fw,'map_acc',acc,'pair_train',sc,flush=True)
 print('RANK',sorted(rows,key=lambda x:x[1],reverse=True),flush=True)
if __name__=='__main__':main()

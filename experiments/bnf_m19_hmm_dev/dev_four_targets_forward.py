#!/usr/bin/env python3
import urllib.request
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/4b23fc96beea9b84c8d7f8acfc12ac41ae05845e/experiments/bnf_m19_hmm_dev/dev_fitted_mapping_forward.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
ns={'__name__':'dev_fit'}
exec(compile(src,'dev_fitted_mapping_forward.py','exec'),ns)

def main():
 base=ns['ns']['ns'];gen=ns['ns']['gen'];forward=ns['ns']['forward_corpus'];lms,holds,meta=base['load_sources']()
 for lang in base['TARGETS']:
  rep=0;span=base['choose_span'](holds[lang],base['TRAIN']+base['HOLD'],(lang,rep));cipher,true,attempt=gen(span,lang,rep);ctra,cho=base['split_letters'](cipher,base['TRAIN']);symbols=[chr(65+i) for i in range(25)];Str=base['stats'](ctra.split(),symbols);how=cho.split();Sho=base['stats'](how,symbols);rows=[]
  for la in base['LANGS']:
   sc,m=base['optimize'](Str,base['induced'](lms[la]),('dev4',lang,rep,la));fw,n,bad=forward(how,m,lms[la]);acc=base['weighted_map_acc'](Sho,m,true);rows.append((la,fw,acc))
  rows.sort(key=lambda x:x[1],reverse=True);print('TARGET',lang,'TOP',rows[0][0],'MARGIN',rows[0][1]-rows[1][1],'TARGET_RANK',1+next(i for i,x in enumerate(rows) if x[0]==lang),'TARGET_ACC',next(x[2] for x in rows if x[0]==lang),'RANK',rows,flush=True)
if __name__=='__main__':main()

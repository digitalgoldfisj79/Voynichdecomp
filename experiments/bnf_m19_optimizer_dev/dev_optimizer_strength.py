#!/usr/bin/env python3
import urllib.request, json
import numpy as np
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/9fdec6ae1a9d630bdcb1b6a01c63e7bc63222a17/experiments/bnf_m19_hmm_v0_8/run_v08.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
ns={'__name__':'v08lib'}
exec(compile(src,'run_v08.py','exec'),ns)
b=ns['b']

def main():
    lms,pools,meta=ns['load_fresh'](); lang='french'
    span=b['choose_span'](pools[lang],b['TRAIN']+b['HOLD'],('v08qual',lang))
    cipher,true,attempt=ns['gen_control'](span,lang)
    ctra,cho=b['split_letters'](cipher,b['TRAIN']); symbols=[chr(65+i) for i in range(25)]
    trw=ctra.split(); how=cho.split(); Str=b['stats'](trw,symbols); Sho=b['stats'](how,symbols); comp=b['induced'](lms[lang])
    settings=[(14000,3),(24000,6),(36000,10)]
    out=[]
    for steps,restarts in settings:
        fits=[]
        for rep in range(2):
            sc,m=b['optimize'](Str,comp,('optimizer-dev',steps,restarts,rep),steps,restarts)
            fw,_,_,_=ns['forward_words'](how,m,symbols,lms[lang]);acc=ns['weighted_acc'](Sho,m,true)
            fits.append((sc,m,fw,acc))
        agr=ns['mapping_agreement'](Str['freq'],fits[0][1],fits[1][1])
        row={'steps':steps,'restarts':restarts,'fit1_acc':fits[0][3],'fit2_acc':fits[1][3],'agreement':agr,'fit1_forward':fits[0][2],'fit2_forward':fits[1][2],'fit1_pair':fits[0][0],'fit2_pair':fits[1][0]}
        out.append(row);print('SETTING',json.dumps(row,separators=(',',':')),flush=True)
    print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':main()

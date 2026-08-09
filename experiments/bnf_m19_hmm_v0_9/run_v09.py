#!/usr/bin/env python3
import urllib.request, json
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/9fdec6ae1a9d630bdcb1b6a01c63e7bc63222a17/experiments/bnf_m19_hmm_v0_8/run_v08.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
ns={'__name__':'v08lib'}
exec(compile(src,'run_v08.py','exec'),ns)
b=ns['b']

# v0.9 fresh partition: prior development {0,5} and v0.8 qualification {1,6} are excluded.
TRAIN_RES={3,4,8,9}; QUAL_RES={2,7}
def load_fresh_v09():
    lms={};pools={};meta={}
    for lang,u in b['LM_URLS'].items():
        ss=b['conllu'](b['fetch'](u));tr=[s for i,s in enumerate(ss) if i%10 in TRAIN_RES];qo=[s for i,s in enumerate(ss) if i%10 in QUAL_RES];lm=b['build_lm'](tr);lms[lang]=lm;pools[lang]=b['pool_text'](qo);meta[lang]={'sentences_total':len(ss),'train_sentences':len(tr),'qual_sentences':len(qo),'lm_letters':lm['letters'],'qual_letters':sum(c!=' ' for c in pools[lang])};print('FRESH_V09',lang,meta[lang],flush=True)
    return lms,pools,meta
ns['load_fresh']=load_fresh_v09

# Stronger convergence budget fixed prospectively by v0.9 protocol.
_old_opt=b['optimize']
def strong_optimize(S,comp,tag,steps=None,restarts=None):
    return _old_opt(S,comp,('v09strong',tag),24000,6)
b['optimize']=strong_optimize

# Fresh qualification span namespace while preserving exact generator law.
_old_choose=b['choose_span']
def choose_span_v09(pool,n,tag):
    if isinstance(tag,tuple) and len(tag)>=1 and tag[0]=='v08qual':
        tag=('v09qual',)+tuple(tag[1:])
    return _old_choose(pool,n,tag)
b['choose_span']=choose_span_v09

# New Voynich split namespace, otherwise identical v0.8 rule.
def split_vms_v09(data):
    pages=[]
    for f in data['pages']:
        w=b['extract_page'](data,f,'ZLZI')
        if w:pages.append((f,w,sum(map(len,w))))
    pages=sorted(pages,key=lambda p:b['seed']('M19HMMv09split',p[0]));nh=max(1,int(round(.2*len(pages))));hold=pages[:nh];train=pages[nh:];required=set(c for _,ws,_ in train for w in ws for c in w);cand=sorted(train,key=lambda p:b['seed']('M19HMMv09train',p[0]));sample=[];n=0;seen=set()
    for p in cand:
        sample.append(p);n+=p[2]
        for w in p[1]:seen.update(w)
        if n>=b['TRAIN'] and required.issubset(seen):break
    return sample,hold,pages,required
ns['split_vms']=split_vms_v09

ns['main']()

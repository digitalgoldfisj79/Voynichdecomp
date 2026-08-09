#!/usr/bin/env python3
import urllib.request, json
import numpy as np
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/185f0c55e910ae075dd945402f590586e1bd02cd/experiments/bnf_m19_hmm_v0_9/run_v09.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
# run_v09.py is itself a wrapper ending in an unconditional ns['main'](). Strip only that terminal invocation.
marker="ns['main']()"
pos=src.rfind(marker)
if pos<0: raise RuntimeError('parent main marker missing')
src=src[:pos]
lib={'__name__':'v09lib'}
exec(compile(src,'run_v09.py','exec'),lib)
b=lib['b']
RAW={'a':5,'b':22,'c':6,'d':4,'e':1,'f':16,'g':22,'h':3,'i':10,'j':20,'k':2,'l':12,'m':9,'n':23,'o':1,'p':7,'q':4,'r':24,'s':30,'t':8,'u':0,'v':28,'x':28,'y':5,'z':20}
SYMS=sorted(RAW); M=np.asarray([b['V2I'][RAW[s]] for s in SYMS],dtype=np.int16)
assert b['valid_map'](M)

def words_for(data,hold,tid):
    out=[]
    for f,_,_ in hold:
        if f in data['pages']:out.extend(b['extract_page'](data,f,tid))
    return out

def exact_coverage(words):
    ss=set(SYMS);tot=sum(map(len,words));known=sum(sum(c in ss for c in w) for w in words)
    return known/max(1,tot),known,tot

def main():
    lms,pools,meta=lib['load_fresh']();data=json.loads(b['fetch'](b['SLIM']));sample,hold,pages,required=lib['split_vms'](data)
    out={'heldout_folios':[f for f,_,_ in hold],'mapping':RAW,'results':{}}
    zw=words_for(data,hold,'ZLZI');zlex=lib['lexical_z'](zw,M,SYMS,lms['german'],('postsel','ZLZI','german'));out['results']['ZLZI']={'coverage':exact_coverage(zw)[0],'lexical':zlex};print('ZLZI',json.dumps(out['results']['ZLZI'],separators=(',',':')),flush=True)
    for tid in ['TTLI','VDRB']:
        words=words_for(data,hold,tid);cov,known,total=exact_coverage(words);rank=[]
        for la in b['LANGS']:
            sc,n,sk,fcov=lib['forward_words'](words,M,SYMS,lms[la]);rank.append((la,sc))
        rank.sort(key=lambda x:x[1],reverse=True);gscore=next(x[1] for x in rank if x[0]=='german');grank=1+next(i for i,x in enumerate(rank) if x[0]=='german');margin=(rank[0][1]-rank[1][1]) if rank[0][0]=='german' else None;lex=lib['lexical_z'](words,M,SYMS,lms['german'],('postsel',tid,'german'))
        r={'coverage':cov,'known_letters':known,'total_letters':total,'ranking':rank,'german_rank':grank,'german_score':gscore,'german_margin':margin,'lexical':lex};out['results'][tid]=r;print(tid,json.dumps(r,separators=(',',':')),flush=True)
    surv=out['results']['ZLZI']['lexical']['z']>=5
    for tid in ['TTLI','VDRB']:
        r=out['results'][tid];surv=surv and r['coverage']>=.90 and r['german_rank']==1 and r['german_margin'] is not None and r['german_margin']>=.03 and r['lexical']['z']>=3
    out['verdict']='SURVIVES TRANSCRIPTION DIAGNOSTIC' if surv else 'POST-SELECTION LEAD DOES NOT SURVIVE'
    print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':main()

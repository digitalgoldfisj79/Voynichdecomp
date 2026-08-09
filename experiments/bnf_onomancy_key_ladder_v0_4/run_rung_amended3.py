#!/usr/bin/env python3
import urllib.request, json
import numpy as np
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/bcee5b894996d9b4ae59ce72d6c59850c720a5e4/experiments/bnf_onomancy_key_ladder_v0_4/run_rung_amended2.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
# Load Amendments 001+002 without starting main, then add Amendment 003.
needle="ns['main']()"
pos=src.rfind(needle)
if pos<0: raise RuntimeError('amended2 main marker missing')
src=src[:pos]
env={'__name__':'rung_amended2_base'}
exec(compile(src,'run_rung_amended2.py','exec'),env)
ns=env['ns']

base_make_control=ns['make_control_cipher']
choose_letter_span=ns['choose_letter_span']
CAP2=ns['CAP2']
stable_seed=ns['stable_seed']
MODE=ns['MODE']
letters_in_text=ns['letters_in_text']
split_control_exact=ns['split_control_exact']

def make_piece_control_amended3(groups_eval,train_sample,plain_text,lang):
    needs={g:(sum(letters_in_text(t) for _,t in train_sample[g]),sum(letters_in_text(t) for _,t in d['hold'])) for g,d in groups_eval.items()}
    need=sum(a+b for a,b in needs.values())
    span,_=choose_letter_span(plain_text,need,('v04',MODE,lang,need))
    apos=[i for i,c in enumerate(span) if c!=' ']; cursor=0; out={}; covered=total=0
    for g in sorted(groups_eval):
        nt,nh=needs[g]; nn=nt+nh
        i0=apos[cursor]; i1=apos[cursor+nn-1]+1; seg=span[i0:i1].strip(); cursor+=nn
        seq,true,pa=base_make_control(seg,CAP2,('v04',MODE,lang,g))
        trseq,trpa,hseq,hpa=split_control_exact(seq,pa,nt)
        observed=sorted(set(int(x) for x in trseq if x>=0))
        old2new={old:i for i,old in enumerate(observed)}
        tr2=np.asarray([-1 if x<0 else old2new[int(x)] for x in trseq],dtype=np.int16)
        h2=[]
        for x in hseq:
            if x<0: h2.append(-1)
            else:
                total+=1
                if int(x) in old2new:
                    covered+=1; h2.append(old2new[int(x)])
                else: h2.append(-1)
        h2=np.asarray(h2,dtype=np.int16)
        true2=np.asarray([true[old] for old in observed],dtype=np.int16)
        out[g]={'train_seq':tr2,'hold_seq':h2,'train_pa':trpa,'hold_pa':hpa,'symbols':list(range(len(observed))),'true':true2}
    cov=covered/max(1,total)
    print('CONTROL_MAPPING_COVERAGE',MODE,lang,cov,covered,total,flush=True)
    if cov<.99:
        print('RESULT_JSON='+json.dumps({'protocol':'v0.4','mode':MODE,'control_language':lang,'control_mapping_coverage':cov,'verdict':'UNDERPOWERED: CONTROL UNSEEN HOMOPHONES'},separators=(',',':')),flush=True)
        raise SystemExit(0)
    return out

ns['make_piece_control']=make_piece_control_amended3
ns['main']()

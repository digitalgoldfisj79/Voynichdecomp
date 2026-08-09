#!/usr/bin/env python3
import sys, urllib.request, json
import numpy as np
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/8a221c63f0bb4dc8417235226a9fe8b1114a7869/experiments/bnf_onomancy_key_ladder_v0_4/run_rung.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
ns={'__name__':'rung_base'}
exec(compile(src,'run_rung.py','exec'),ns)

def pages_to_seq_amended(pages,symbols):
    s2i={s:i for i,s in enumerate(symbols)};arr=[]
    for _,t in pages:
        for c in t:
            if c.isspace():arr.append(-1)
            elif c in s2i:arr.append(s2i[c])
            else:arr.append(-1)  # hard break: train-unseen heldout label
        arr.append(-1)
    return np.asarray(arr,dtype=np.int16)

def build_actual_items_amended(info,train_sample):
    out={};covered=total=0
    for g,d in info.items():
        if not d['evaluable']:continue
        syms=sorted(set(c for _,t in train_sample[g] for c in t if not c.isspace()))
        sset=set(syms)
        for _,t in d['hold']:
            for c in t:
                if c.isspace():continue
                total+=1
                if c in sset:covered+=1
        trseq=pages_to_seq_amended(train_sample[g],syms);hseq=pages_to_seq_amended(d['hold'],syms)
        out[g]={'symbols':syms,'train_seq':trseq,'hold_seq':hseq,'train_pages':train_sample[g],'hold_pages':d['hold']}
    cov=covered/max(1,total)
    print('HOLD_MAPPING_COVERAGE',ns['MODE'],cov,covered,total,flush=True)
    if cov<.99:
        print('RESULT_JSON='+json.dumps({'protocol':'v0.4','mode':ns['MODE'],'hold_mapping_coverage':cov,'verdict':'UNDERPOWERED: UNSEEN HOLDOUT GLYPHS'},separators=(',',':')),flush=True)
        raise SystemExit(0)
    return out

ns['pages_to_seq']=pages_to_seq_amended
ns['build_actual_items']=build_actual_items_amended
# Functions look up globals in ns, so all downstream calls use the amended definitions.
ns['main']()

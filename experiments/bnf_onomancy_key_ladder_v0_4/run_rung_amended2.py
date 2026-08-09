#!/usr/bin/env python3
import sys, urllib.request, urllib.parse, json
import numpy as np
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/8a221c63f0bb4dc8417235226a9fe8b1114a7869/experiments/bnf_onomancy_key_ladder_v0_4/run_rung.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
ns={'__name__':'rung_base'}
exec(compile(src,'run_rung.py','exec'),ns)

# Amendment 001: train-only literal symbol inventory; train-unseen heldout labels are hard breaks.
def pages_to_seq_amended(pages,symbols):
    s2i={s:i for i,s in enumerate(symbols)};arr=[]
    for _,t in pages:
        for c in t:
            if c.isspace():arr.append(-1)
            elif c in s2i:arr.append(s2i[c])
            else:arr.append(-1)
        arr.append(-1)
    return np.asarray(arr,dtype=np.int16)

def build_actual_items_amended(info,train_sample):
    out={};covered=total=0
    for g,d in info.items():
        if not d['evaluable']:continue
        syms=sorted(set(c for _,t in train_sample[g] for c in t if not c.isspace()));sset=set(syms)
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

# Amendment 002: append independent Sefaria Hebrew known plaintext to the control pool only.
base_concat=ns['concat_norm']
SEF='https://storage.googleapis.com/sefaria-export/json/Halakhah/Mishneh Torah/Sefer Madda/Mishneh Torah, Torah Study/Hebrew/Torat Emet 363.json'
_sef_cache=None
def concat_norm_extended(sents):
    global _sef_cache
    x=base_concat(sents)
    probe=' '.join(sents[:8])
    is_hebrew=any('\u0590'<=c<='\u05ff' for c in probe)
    if is_hebrew:
        if _sef_cache is None:
            q=urllib.parse.quote(SEF,safe=':/?=&%')
            obj=json.loads(urllib.request.urlopen(q,timeout=90).read().decode('utf-8'))
            chunks=[]
            def walk(v):
                if isinstance(v,str):chunks.append(v)
                elif isinstance(v,list):
                    for y in v:walk(y)
            walk(obj.get('text',[]))
            _sef_cache=ns['norm'](' '.join(chunks))
            print('HEBREW_CONTROL_EXTENSION',sum(c!=' ' for c in _sef_cache),flush=True)
        if _sef_cache:x=(x+' '+_sef_cache).strip()
    return x

ns['pages_to_seq']=pages_to_seq_amended
ns['build_actual_items']=build_actual_items_amended
ns['concat_norm']=concat_norm_extended
ns['main']()

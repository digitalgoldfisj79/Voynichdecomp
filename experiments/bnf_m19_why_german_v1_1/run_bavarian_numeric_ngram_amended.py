#!/usr/bin/env python3
import urllib.request
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/9316220e481ac79de5b1b6fcb1aa26da0471105a/experiments/bnf_m19_why_german_v1_1/run_bavarian_numeric_ngram.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8');lib={'__name__':'bavnum_lib'};exec(compile(src,'run_bavarian_numeric_ngram.py','exec'),lib)
def emit_word_fixed(w,rng):
    out=[]
    for c in w:
        if c not in lib['b']['A2I']:continue
        vals=lib['b']['LETTER_VALS'][lib['b']['A2I'][c]]
        out.append(int(vals[int(rng.integers(0,len(vals)))]))
    return out
lib['emit_word']=emit_word_fixed
lib['main']()

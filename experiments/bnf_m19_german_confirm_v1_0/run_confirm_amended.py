#!/usr/bin/env python3
import urllib.request
from collections import Counter
import numpy as np
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/c7c50f74e1f1f88004a0f08ea379324a3d42c16d/experiments/bnf_m19_german_confirm_v1_0/run_confirm.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
lib={'__name__':'confirm_lib'}
exec(compile(src,'run_confirm.py','exec'),lib)
b=lib['b'];M=lib['M'];SYMS=lib['SYMS']

def lexical_fraction_types(words,mp,lm):
    cc=Counter(words);hit=tot=0
    for w,n in cc.items():
        d=b['viterbi'](w,mp,SYMS,lm)
        if d is None:continue
        tot+=n
        if d in lm['vocab']:hit+=n
    return hit/max(1,tot),hit,tot

def lexical_z256_fast(words,lm,tag):
    obs,hit,tot=lexical_fraction_types(words,M,lm);rng=np.random.default_rng(b['seed']('v10lex',tag));vals=[]
    for _ in range(256):
        x=M.copy();rng.shuffle(x);vals.append(lexical_fraction_types(words,x,lm)[0])
    mu=float(np.mean(vals));sd=float(np.std(vals,ddof=1));return {'fraction':obs,'hits':hit,'tokens':tot,'null_mean':mu,'null_sd':sd,'z':(obs-mu)/sd if sd>1e-15 else 0.0}

lib['lexical_z256']=lexical_z256_fast
lib['main']()

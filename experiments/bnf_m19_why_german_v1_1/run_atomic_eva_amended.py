#!/usr/bin/env python3
import urllib.request
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/72e4b7a197c3283c7f6067fcb89d1a1ff3bd6c81/experiments/bnf_m19_why_german_v1_1/run_atomic_eva.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
src=src.rsplit("if __name__=='__main__':main()",1)[0]
lib={'__name__':'atomic_lib'};exec(compile(src,'run_atomic_eva.py','exec'),lib)
_old=lib['choose_span']
def choose_span_amended(pool,n,tag):
    if isinstance(tag,tuple) and len(tag)>=2 and tag[0]=='qual' and tag[1]=='arabic':
        target=set(range(lib['b']['NV']))
        for k in range(1000):
            sp=_old(pool,n,('qual','arabic','full-repertoire',k))
            poss=set()
            for c in sp:
                if c==' ' or c not in lib['b']['A2I']:continue
                poss.update(lib['b']['V2I'][int(v)] for v in lib['b']['LETTER_VALS'][lib['b']['A2I'][c]])
            if poss==target:
                print('ARABIC_SPAN_REPERTOIRE_ATTEMPT',k,flush=True)
                return sp
        raise RuntimeError('no full-repertoire Arabic span found')
    return _old(pool,n,tag)
lib['choose_span']=choose_span_amended
lib['main']()

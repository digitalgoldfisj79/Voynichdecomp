#!/usr/bin/env python3
import urllib.request,re
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/87f12c085a9e4ada904184838166ebafbe75678d/experiments/bnf_m19_why_german_v1_1/run_transcription_crosswalk.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
lib={'__name__':'crosswalk_lib'};exec(compile(src,'run_transcription_crosswalk.py','exec'),lib)
def pairs_from_cigar_fixed(query,target,cigar):
    qi=ti=0;out=[]
    for n,op in re.findall(r'(\d+)([=XID])',cigar or ''):
        n=int(n)
        if op in '=X':
            for k in range(n):out.append((query[qi+k],target[ti+k]))
            qi+=n;ti+=n
        elif op=='I': # Edlib: insertion in query relative to target; consumes query
            qi+=n
        elif op=='D': # deletion from query / target-only run; consumes target
            ti+=n
    return out
lib['pairs_from_cigar']=pairs_from_cigar_fixed
lib['main']()

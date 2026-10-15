#!/usr/bin/env python3
"""Independent implementation of v143: reconstruct source/carry from ciphertext, not encoder ledger."""
import json,hashlib,sys,collections
from pathlib import Path
STATES=['-','a','e','i','o','u']

def last_vowel(word):
    for c in reversed(word):
        if c in 'aeiou':return c
    return '-'

def schedule(i,spec):
    seed=spec['statefree_seed'];weights=spec['statefree_weights_from_development']
    u=int.from_bytes(hashlib.blake2b((seed+'|'+str(i)).encode(),digest_size=8).digest(),'big')/2**64
    for row in weights:
        u-=row['weight']
        if u<0:return row['state']
    return weights[-1]['state']

def construct(manifest):
    spec=manifest['carry_spec'];assert spec['carry_states']==STATES
    vs={};unit=manifest['unit_list'];pairs={tuple(k.split('|')):v for k,v in manifest['vowel_by_pair'].items()}
    if len(set(unit))!=len(unit) or unit[0]!='':raise ValueError('invalid shared units')
    for v,x in manifest['state'].items():
        codes=x['cells'];D=x['direct'];esc=x['escape'];
        if len(set(codes))!=len(codes):raise ValueError('duplicate codecell')
        if any(y.startswith(z) for j,y in enumerate(codes) for k,z in enumerate(codes) if j!=k):
            raise ValueError('prefix collision')
        groups=collections.defaultdict(list)
        for c in codes:groups[(c[0],len(c))].append(c)
        inverse={}
        for ci,s in enumerate(STATES):
            dec={}
            for arr in groups.values():
                n=len(arr)
                for j,c in enumerate(arr):dec[arr[(j+ci)%n]]=c
            if len(dec)!=len(codes):raise ValueError('noninjective permutation')
            inverse[s]=dec
        # Trie parsing for prefixfree cells
        trie={}
        for c in codes:
            node=trie
            for char in c:node=node.setdefault(char,{})
            node['END']=c
        vs[v]={'pos':{c:i for i,c in enumerate(codes)}, 'inv':inverse, 'trie':trie,'direct':D,'escape':esc, 'codes':codes}
    return unit,pairs,vs

def decode(text,manifest,arm):
    if arm not in ('carry','statefree','baseline'):raise ValueError('unknown arm')
    units,pairs,vs=construct(manifest)
    out=[];carry='-'
    for word_index,w in enumerate(text.split()):
        q=carry if arm=='carry' else schedule(word_index,manifest['carry_spec']) if arm=='statefree' else '-'
        i=0;v='e';buffer='';pending=False;epsilon=False
        def take(pos,vowel):
            node=vs[vowel]['trie']; j=pos
            while j<len(w) and w[j] in node:
                node=node[w[j]];j+=1
                if 'END' in node:return node['END'],j
            raise ValueError(f'unknown codecell at word {word_index}, {pos}, {vowel}')
        while i<len(w):
            if w[i] not in 'ynr':
                if w.startswith(('ch','sh'),i):bridge=w[i:i+2];i+=2
                elif w[i] in 'odapl':bridge=w[i];i+=1
                else:raise ValueError('invalid bridge')
                if i>=len(w) or w[i] not in 'ynr':raise ValueError('dangling bridge')
                v=pairs.get((bridge,w[i]))
                if v is None:raise ValueError('non-admitted vowel pair')
                buffer+=v;pending=True;epsilon=False
            else:
                if epsilon:raise ValueError('continuation after epsilon')
                native,i=take(i,v)
                c=native if arm=='baseline' else vs[v]['inv'][q][native]
                pos=vs[v]['pos'];idx=pos[c];D=vs[v]['direct']; esc=vs[v]['escape']
                if esc is not None and c==esc:
                    code2,i=take(i,v)
                    canonical2=code2 if arm=='baseline' else vs[v]['inv'][q][code2]
                    idx=D+pos[canonical2]
                elif idx>=D:raise ValueError('escape required')
                if idx>=len(units):raise ValueError('invalid escape slot')
                chunk=units[idx]
                if chunk=='':
                    if not pending:raise ValueError('epsilon outside vowel')
                    epsilon=True
                else:buffer+=chunk
                pending=False
        if not buffer or pending:raise ValueError('invalid or empty decoded word')
        out.append(buffer)
        carry=last_vowel(buffer)
    return out

if __name__=='__main__':
    if len(sys.argv)!=5:raise SystemExit('usage MANIFEST CIPHERTEXT ARM OUT')
    m=json.loads(Path(sys.argv[1]).read_text())
    plain=decode(Path(sys.argv[2]).read_text(),m,sys.argv[3]);Path(sys.argv[4]).write_text('\n'.join(plain)+'\n')
    print(len(plain),'words')

#!/usr/bin/env python3
"""Frozen v142b to cross-boundary permutation v143; source used only to choose previous-word state."""
from pathlib import Path
import hashlib, json, os, pickle, tempfile, collections
P=Path(__file__).resolve().parent
B=P/'vbm_v142_inband'
m=json.loads((B/'04_pair_bridge_manifest.json').read_text()); contract=json.loads((P/'00_FROZEN_CONTRACT.json').read_text())
STATES=contract['permutation_order']; seeds=contract['statefree_schedule_seed']
assert STATES==['-','a','e','i','o','u']

def persist(name,obj):
    p=P/name; fd,t=tempfile.mkstemp(dir=P, prefix='.checkpoint-')
    with os.fdopen(fd,'wb') as f:
        pickle.dump(obj,f,protocol=5);f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def carry(word):
    return next((c for c in word[::-1] if c in 'aeiou'),'-')

maps={}
for vowel,info in m['state'].items():
    groups=collections.defaultdict(list)
    for cell in info['cells']:groups[(cell[0],len(cell))].append(cell)
    forward={}
    for state in STATES:
        shift=STATES.index(state);mapping={}
        for clist in groups.values():
            n=len(clist)
            for i,c in enumerate(clist):mapping[c]=clist[(i+shift)%n]
        assert set(mapping)==set(mapping.values())==set(info['cells'])
        assert all(k[0]==v[0] and len(k)==len(v) for k,v in mapping.items())
        forward[state]=mapping
    maps[vowel]=forward

# Unambiguous v142b cell+bridge parsing, extracting native cells for remapping.
by_v={v:tuple(z['cells']) for v,z in m['state'].items()}
bridge_pairs={tuple(k.split('|')):v for k,v in m['vowel_by_pair'].items()}
def transcode(word, choice):
    i=0;st='e';result=[]; ncell=0;nchange=0
    while i<len(word):
        if word[i] not in 'ynr':
            if word.startswith(('ch','sh'),i): br=word[i:i+2];i+=2
            else: br=word[i];i+=1
            if i==len(word):raise ValueError('unpaired bridge')
            st=bridge_pairs[(br,word[i])]
            result.append(br)
        else:
            poss=[s for s in by_v[st] if word.startswith(s,i)]
            if len(poss)!=1:raise ValueError(f'nonunique cell at {i}')
            c=poss[0];d=maps[st][choice][c]
            result.append(d);ncell+=1;nchange+=(c!=d);i+=len(c)
    return ''.join(result),ncell,nchange

train=(B/'train.reference.txt').read_text().splitlines()
counts=collections.Counter('-' if i==0 else carry(train[i-1]) for i in range(len(train)))
# distribution learned on development only (this is not the holdout)
total=len(train); weights=[counts[s]/total for s in STATES]

def pseudo_state(i):
    h=hashlib.blake2b((seeds+'|'+str(i)).encode(),digest_size=8).digest()
    u=int.from_bytes(h,'big')/(1<<64)
    for s,p in zip(STATES,weights):
        u-=p
        if u<0:return s
    return STATES[-1]

def run(label):
    words=(B/(label+'.reference.txt')).read_text().splitlines()
    baseline=(B/(label+'.pair.vbm')).read_text().split()
    assert len(words)==len(baseline)
    arms={'carry':[],'statefree':[]}; change=collections.Counter(); cells=collections.Counter();chooser=collections.Counter()
    for i,(w,old) in enumerate(zip(words,baseline)):
        last='-' if i==0 else carry(words[i-1]); schedule=pseudo_state(i)
        for arm,s in [('carry',last),('statefree',schedule)]:
            c,ncell,nchange=transcode(old,s)
            arms[arm].append(c);change[arm]+=nchange;cells[arm]+=ncell;chooser[(arm,s)]+=1
            assert len(c)==len(old), (arm,i)
    for arm,outputs in arms.items():
        f=P/f'{label}.{arm}.vbm'
        f.write_text(' '.join(outputs)+'\n')
    res={'label':label,'words':len(words),'states':dict((s,{'carry':chooser['carry',s],'statefree':chooser['statefree',s]}) for s in STATES),
         'changed_cells':dict(change),'total_cells':dict(cells),'per_word_length_equal':all(len(a)==len(b)==len(c) for a,b,c in zip(baseline,arms['carry'],arms['statefree'])),
         'base_sha256':hashlib.sha256((B/(label+'.pair.vbm')).read_bytes()).hexdigest(),
         'carry_sha256':hashlib.sha256((P/f'{label}.carry.vbm').read_bytes()).hexdigest(),
         'statefree_sha256':hashlib.sha256((P/f'{label}.statefree.vbm').read_bytes()).hexdigest()}
    persist('stage1_'+label+'.pkl',res)
    return res

# frozen manifest: only weights and original manifest; no expanded state-cell matrices
new=dict(m);new['format']='vbm143_cross_word_permuted_pair';new['carry_spec']={
    'carry_states':STATES,'true_carry':'last vowel of immediately previously decoded plaintext word, else -',
    'permutation':'within each vowel, first-cell-letter, cell-length group; rotate by carry index',
    'no_separate_state_markers':True,
    'statefree_seed':seeds,
    'statefree_weights_from_development':[{'state':s,'weight':counts[s]/total,'count':counts[s]} for s in STATES],
    'weight_source':'development-only; not refit on Hartlieb',
    'baseline_manifest_sha256':hashlib.sha256((B/'04_pair_bridge_manifest.json').read_bytes()).hexdigest()}
(P/'01_v143_frozen_manifest.json').write_text(json.dumps(new,indent=2,sort_keys=True)+'\n')
persist('stage0_v143_design.pkl',{'contract_sha256':hashlib.sha256((P/'00_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),
                                   'manifest_sha256':hashlib.sha256((P/'01_v143_frozen_manifest.json').read_bytes()).hexdigest(), 'weights':weights})
print(json.dumps({'train':run('train'),'hartlieb':run('hartlieb')},indent=2))

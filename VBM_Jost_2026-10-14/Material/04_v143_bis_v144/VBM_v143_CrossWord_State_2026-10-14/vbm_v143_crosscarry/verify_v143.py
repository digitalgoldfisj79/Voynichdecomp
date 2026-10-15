#!/usr/bin/env python3
import collections,copy,hashlib,json,math,os,pickle,random,statistics,tempfile
from pathlib import Path
from decode_v143_independent import decode,construct,last_vowel,schedule
P=Path(__file__).resolve().parent; B=P/'vbm_v142_inband'
m=json.loads((P/'01_v143_frozen_manifest.json').read_text())
contracts=json.loads((P/'00_FROZEN_CONTRACT.json').read_text())
STATES=contracts['permutation_order']

def save(obj,name):
    fd,tmp=tempfile.mkstemp(dir=P,prefix='.atomic-')
    with os.fdopen(fd,'wb') as f:
        pickle.dump(obj,f,protocol=5);f.flush();os.fsync(f.fileno())
    os.replace(tmp,P/name)

def get_codecell_pair(can,alter):
    # First code-cell with an equivalence class >= 6; frozen in advance of holdout evaluation.
    i=j=0;v='e'
    while i<len(can) and j<len(alter):
        if can[i] not in 'ynr':
            def bridge(w,pos):
                return w[pos:pos+2] if w.startswith(('ch','sh'),pos) else w[pos]
            bc=bridge(can,i);ba=bridge(alter,j)
            if bc!=ba:raise ValueError('changed vowel bridge')
            i+=len(bc);j+=len(ba)
            v=m['vowel_by_pair'][bc+'|'+can[i]]
        else:
            cs=m['state'][v]['cells']
            t=[s for s in cs if can.startswith(s,i)]
            u=[s for s in cs if alter.startswith(s,j)]
            if len(t)!=1 or len(u)!=1:raise ValueError('bad parsing')
            c,a=t[0],u[0]
            if len(c)!=len(a) or c[0]!=a[0]:raise ValueError('key/length not preserved')
            pool=[s for s in cs if s[0]==c[0] and len(s)==len(c)]
            i+=len(c);j+=len(a)
            if len(pool)>=6:
                q=(pool.index(a)-pool.index(c))%len(pool)
                if q>=6:raise ValueError('undocumented permutation')
                return STATES[q],pool[0]
    return None,None

def source_aware_panel(label,arms):
    ref=(B/f'{label}.reference.txt').read_text().splitlines()
    canonical=(B/f'{label}.pair.vbm').read_text().split()
    assert len(ref)==len(canonical)
    events={}
    for arm in arms:
        text=(P/f'{label}.{arm}.vbm').read_text().split()
        assert len(text)==len(ref)
        a=[]
        for i,(w,x,y) in enumerate(zip(ref,canonical,text)):
            state,_=get_codecell_pair(x,y)
            if state is not None:
                prev='-' if i==0 else last_vowel(ref[i-1]);a.append((w,state,prev,i))
        events[arm]=a
    return events

def test_paired(events,rep=300,seed=17682):
    # Conditional permutation: shuffle observed assignment among *identical current plaintext words* only.
    # This preserves exact current-word features, all q marginals per type and corpus positions.
    # For once-only words, an exact-word permutation cannot change anything, hence we exclude them.
    group=collections.defaultdict(list)
    for i,(word,q,p,idx) in enumerate(events):group[word].append(i)
    eligible=[ix for v in group.values() if len(v)>=2 for ix in v]
    eligible.sort(); idx_map={old:i for i,old in enumerate(eligible)}
    G=[[idx_map[j] for j in v if j in idx_map] for v in group.values() if len(v)>=2]
    qs=[events[ix][1] for ix in eligible];ps=[events[ix][2] for ix in eligible]
    obs=sum(q==p for q,p in zip(qs,ps))/len(qs)
    rng=random.Random(seed);nulls=[]
    for _ in range(rep):
        new=qs[:]
        for g in G:
            vals=[qs[j] for j in g];rng.shuffle(vals)
            for j,v in zip(g,vals):new[j]=v
        nulls.append(sum(q==p for q,p in zip(new,ps))/len(qs))
    avg=statistics.mean(nulls);sd=statistics.stdev(nulls)
    return {'eligible_events':len(qs),'total_decodable_events':len(events),'exact_word_groups':len(G),'observed_accuracy':obs,
            'conditional_null_mean':avg,'null_sd':sd,'effect_accuracy':obs-avg,'z':(obs-avg)/sd if sd else None,
            'null_max':max(nulls),'permutations':rep, 'seed':seed}

# Candidate intrinsic ciphertext fit: all arms occupy EXACTLY the same formal language.
# Model cannot be identified by syntactic support alone.
res={'manifest_sha256':hashlib.sha256((P/'01_v143_frozen_manifest.json').read_bytes()).hexdigest(),
     'contract_sha256':hashlib.sha256((P/'00_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),
     'engineering':{},'paired_tests':{},'synthetic_controls':{},'interpretation':{}}
for label in ['train','hartlieb']:
    ref=(B/f'{label}.reference.txt').read_text().splitlines()
    words0=(B/f'{label}.pair.vbm').read_text().split()
    d={'word_count':len(ref),'baseline_distinct':len(set(words0)),
       'baseline_hapax_types':sum(f==1 for f in collections.Counter(words0).values()),'arms':{}}
    variants={'baseline':words0}
    for arm in ['carry','statefree']:
        ww=(P/f'{label}.{arm}.vbm').read_text().split()
        variants[arm]=ww
    for arm,ww in variants.items():
        inp=' '.join(ww)+'\n';got=decode(inp,m,arm)
        assert got==ref, (arm,label,'roundtrip failed')
        freq=collections.Counter(ww)
        d['arms'][arm]={'roundtrip':True,'exact_words':len(got),'distinct_words':len(freq),
                         'hapax_types':sum(f==1 for f in freq.values()),'characters':sum(map(len,ww)),
                         'pct_types_hapax':sum(f==1 for f in freq.values())/len(freq)}
    d['identical_word_lengths']=all(len(a)==len(b)==len(c) for a,b,c in zip(*variants.values()))
    assert d['identical_word_lengths']
    d['pairwise_differing_word_fractions']={
       x+'__'+y:sum(a!=b for a,b in zip(variants[x],variants[y]))/len(ref)
       for x,y in [('baseline','carry'),('baseline','statefree'),('carry','statefree')]}
    d['same_cell_inventory_and_lengths']=True # full-word grammar admissibility can nevertheless differ because epsilon is positional
    res['engineering'][label]=d
    if label != 'hartlieb': continue
    events=source_aware_panel(label,['carry','statefree'])
    for arm,a in events.items():
        res['paired_tests'][f'{label}_{arm}']=test_paired(a,rep=300,seed=17682)
        expected=lambda i: ('-' if i==0 else last_vowel(ref[i-1])) if arm=='carry' else schedule(i,m['carry_spec'])
        assert all(q==expected(i) for w,q,p,i in a),('wrong encoder state',arm,label)
    save(d,'stage2_decode_'+label+'.pkl')
    save({k:test_paired(v,rep=300,seed=17682) for k,v in events.items()},'stage3_paired_'+label+'.pkl')

# adversarial tests: out-of-set bridge and duplicate table cannot silently pass; carry manipulation does not guarantee rejection.
dummy=copy.deepcopy(m);dummy['state']['i']['cells'][1]=dummy['state']['i']['cells'][0]
try:construct(dummy);res['synthetic_controls']['collision_rejected']=False
except ValueError:res['synthetic_controls']['collision_rejected']=True
try:
    example=(P/'hartlieb.carry.vbm').read_text();t='X'+example[1:]
    decode(t,m,'carry');res['synthetic_controls']['invalid_letter_rejected']=False
except ValueError:res['synthetic_controls']['invalid_letter_rejected']=True
# Decoder can read same surface tokens under all branches. This is a *formal* parser ambiguity,
# not a claim the wrong decoded German is linguistically sensible.
for label in ['train','hartlieb']:
    ctext=(P/f'{label}.carry.vbm').read_text()
    a=decode(ctext,m,'carry')
    alt={}
    for wrong in ['baseline','statefree']:
        try:
            z=decode(ctext,m,wrong)
            alt[wrong]={'accepted':True,'incorrect_words':sum(x!=y for x,y in zip(a,z))}
        except ValueError as e:
            alt[wrong]={'accepted':False,'error':str(e)[:160]}
    res['synthetic_controls'][label+'_cross_decode']=alt
res['interpretation']={'source_aware':'Paired controls measure an engineered relationship and are not evidence about unlabeled Voynich.',
 'output_only_identifiability':'All arms use the same codecell glyph inventory and output lengths but wrong-state decoders may reject on epsilon/boundary syntax. With an independently justified codebook, that can distinguish particular grammars; without a historically grounded codebook, an autonomous state generator can reproduce the full ciphertext law.',
 'vms_surface':'Inherited V142b seam NO-GO; no Voynich mechanism test admitted or run.',
 'full_historical_attribution':'NOT IDENTIFIED'}
(P/'02_results.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
save(res,'stage4_final_audit.pkl')
print(json.dumps(res,indent=2))

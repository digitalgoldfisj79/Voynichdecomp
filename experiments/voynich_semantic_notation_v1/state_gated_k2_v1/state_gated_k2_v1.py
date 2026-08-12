#!/usr/bin/env python3
import argparse, hashlib, json, math, random, re, statistics, urllib.request
from collections import Counter, defaultdict

NS='VSN_STATE_GATED_K2_V1'
LEX_URL='https://raw.githubusercontent.com/sjgallagher2/PyWORDS/master/pywords/data/lingualatina_voclist.txt'
LEX_SHA256='5a139a6e7a3b9bfe9ef0b0e98e5178fb1c42be66dc3034c3f6f5e3d91b099b9c'
V=set('aeiouy'); D={'ae','au','oe','ei','eu','ui'}; M=set('bcdgptfk'); L=set('lr')
WIDTH_PAIRS=[(1,1),(1,2),(2,1),(2,2),(2,3),(3,2),(3,3),(4,4)]
LINE_WIDTHS=[1,2]
DISCOVERY_SEEDS=[2026081201+i for i in range(4)]
HOLDOUT_SEEDS=[2026081301+i for i in range(20)]
TOLS={'pair_log':math.log(1.25),'edit_tv':0.08,'line_enrich':0.25,'hnext':0.35,'rml':0.10,'mean_len':0.75}

def h_int(*parts):
    b=('::'.join([NS]+[str(x) for x in parts])).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8],'big')

def cat(kind,syll): return h_int('cat',kind,syll)%4

def target_cat(section,kind,slot): return h_int('section',section,kind,slot)%4

def allowed(t,w): return {(t+i)%4 for i in range(w)}

def fs(w):
    w=re.sub('[^a-z]','',w.lower().replace('j','i')); ns=[]; i=0
    while i<len(w):
        if w[i] in V:
            if i+1<len(w) and w[i:i+2] in D: ns.append((i,i+2)); i+=2
            else: ns.append((i,i+1)); i+=1
        else: i+=1
    if not ns:return w
    if len(ns)==1:return w
    e=ns[0][1]; s=ns[1][0]; cl=w[e:s]
    if len(cl)<=1:return w[:e]
    cut=s-2 if cl[-2] in M and cl[-1] in L else s-1
    return w[:cut]

def acquire_lexicon():
    raw=urllib.request.urlopen(LEX_URL,timeout=60).read()
    got=hashlib.sha256(raw).hexdigest()
    if got!=LEX_SHA256: raise RuntimeError(('lexicon_sha_mismatch',got,LEX_SHA256))
    text=raw.decode('utf-8')
    words=sorted(set(w.strip().lower() for w in text.splitlines()
                     if re.fullmatch('[A-Za-z]+',w.strip()) and len(w.strip())>=2
                     and any(c in V for c in w.strip().lower())))
    syll=[fs(w) for w in words]
    assert [fs(x) for x in ['tripode','pepo','corvus','vetula']]==['tri','pe','cor','ve']
    return words,syll,raw

def entropy(vals):
    c=Counter(vals); n=sum(c.values())
    return -sum(v/n*math.log2(v/n) for v in c.values()) if n else 0.0

def surface_metrics(toks):
    lens=[len(t) for t in toks]; chars=[c for t in toks for c in t]
    byp=defaultdict(list); byr=defaultdict(list); bg=[]; bp=defaultdict(list)
    for t in toks:
        for i,c in enumerate(t): byp[i+1].append(c); byr[len(t)-i].append(c)
        for a,b in zip(t,t[1:]): bg.append((a,b)); bp[a].append(b)
    nc=sum(lens)
    h_abs=sum(len(v)/nc*entropy(v) for v in byp.values())
    h_right=sum(len(v)/nc*entropy(v) for v in byr.values())
    hnext=sum(len(v)/len(bg)*entropy(v) for v in bp.values()) if bg else 0.0
    return {'mean_len':sum(lens)/len(lens),'hnext':hnext,'rml':h_right-h_abs,
            'h_abs':h_abs,'h_right':h_right,'h_char':entropy(chars)}

def edit_pairs(toks):
    S=set(toks); pairs=set()
    for w in toks:
        for i in range(len(w)):
            d=w[:i]+w[i+1:]
            if d in S:pairs.add(tuple(sorted((w,d))))
    B=defaultdict(list)
    for w in toks:
        for i in range(len(w)): B[(len(w),i,w[:i],w[i+1:])].append(w)
    for ws in B.values():
        if len(ws)>1:
            ws=sorted(set(ws))
            for i in range(len(ws)):
                for j in range(i+1,len(ws)): pairs.add((ws[i],ws[j]))
    loc=Counter()
    for a,b in pairs:
        if len(a)==len(b):
            k=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
            p='prefix' if k==0 else ('suffix' if k==len(a)-1 else 'internal')
        else:
            long,short=(a,b) if len(a)>len(b) else (b,a)
            poss=[i for i in range(len(long)) if long[:i]+long[i+1:]==short]
            pcs=[('prefix' if i==0 else ('suffix' if i==len(long)-1 else 'internal')) for i in poss]
            p=pcs[0] if pcs and all(x==pcs[0] for x in pcs) else 'internal'
        loc[p]+=1
    n=len(pairs)
    return pairs, {'pairs':n,'prefix':loc['prefix']/n if n else 0.0,
                   'internal':loc['internal']/n if n else 0.0,
                   'suffix':loc['suffix']/n if n else 0.0}

def config_grid():
    out=[{'family':'BASE','w1':4,'w2':4,'line_w':4}]
    for w1,w2 in WIDTH_PAIRS: out.append({'family':'DOMAIN','w1':w1,'w2':w2,'line_w':4})
    for w1,w2 in WIDTH_PAIRS: out.append({'family':'STATE','w1':w1,'w2':w2,'line_w':4})
    for w1,w2 in WIDTH_PAIRS:
        for lw in LINE_WIDTHS: out.append({'family':'LINE','w1':w1,'w2':w2,'line_w':lw})
    for c in out: c['id']=f"{c['family']}-w{c['w1']}{c['w2']}-l{c['line_w']}"
    return out

def pools_for(section,cfg,sylls):
    if cfg['family']=='BASE': return list(sylls),list(sylls)
    if cfg['family']=='DOMAIN':
        a1=allowed(target_cat(section,'domain',1),cfg['w1'])
        a2=allowed(target_cat(section,'domain',2),cfg['w2'])
        return [s for s in sylls if cat('domain',s) in a1],[s for s in sylls if cat('domain',s) in a2]
    a1=allowed(target_cat(section,'class',1),cfg['w1'])
    a2=allowed(target_cat(section,'degree',2),cfg['w2'])
    return [s for s in sylls if cat('class',s) in a1],[s for s in sylls if cat('degree',s) in a2]

def generate_inventory(section,n,cfg,sylls,seed):
    p1,p2=pools_for(section,cfg,sylls)
    if not p1 or not p2:return None
    r=random.Random(h_int('inventory',seed,section,cfg['id']))
    out={}; attempts=0
    while len(out)<n and attempts<5000000:
        s1=r.choice(p1); s2=r.choice(p2); surf=s1+s2
        if surf not in out: out[surf]=(s1,s2)
        attempts+=1
    if len(out)<n:return None
    return out,attempts

def expand_hist(hist):
    arr=[]
    for k,v in sorted(((int(k),int(v)) for k,v in hist.items())): arr.extend([k]*v)
    return arr

def simulate_lines(section,hist,cfg,prov,pairset,seed):
    lengths=expand_hist(hist)
    if not lengths:return {'line_enrich':None,'pair_rate':None,'baseline':None,'hits':0,'opp':0,'n_tokens':0}
    types=sorted(prov); lines=[]
    for idx,n in enumerate(lengths):
        r=random.Random(h_int('line',seed,section,cfg['id'],idx,n))
        pool=types
        if cfg['family']=='LINE':
            t=h_int('line_state',section,idx)%4; A=allowed(t,cfg['line_w'])
            pool=[x for x in types if cat('line1',prov[x][0]) in A and cat('line2',prov[x][1]) in A]
            if not pool:return None
        lines.append([r.choice(pool) for _ in range(n)])
    f=Counter(x for line in lines for x in line); N=sum(f.values())
    base_num=0.0
    for a,b in pairset: base_num += 2.0*f.get(a,0)*f.get(b,0)
    baseline=base_num/(N*(N-1)) if N>1 else 0.0
    hits=opp=0
    for line in lines:
        for i in range(len(line)):
            for j in range(i+1,len(line)):
                opp+=1
                if tuple(sorted((line[i],line[j]))) in pairset:hits+=1
    rate=hits/opp if opp else 0.0
    enrich=rate/baseline if baseline>0 else None
    return {'line_enrich':enrich,'pair_rate':rate,'baseline':baseline,'hits':hits,'opp':opp,'n_tokens':N}

def tv(a,b): return 0.5*sum(abs(a[k]-b[k]) for k in ['prefix','internal','suffix'])

def evaluate_section(section,target,cfg,sylls,seed):
    g=generate_inventory(section,target['n_types'],cfg,sylls,seed)
    if g is None:return {'valid':False,'section':section,'config':cfg['id'],'seed':seed,'reason':'inventory_capacity'}
    prov,attempts=g; toks=sorted(prov); ps,em=edit_pairs(toks); sm=surface_metrics(toks)
    lm=simulate_lines(section,target.get('hist') or {},cfg,prov,ps,seed)
    if lm is None:return {'valid':False,'section':section,'config':cfg['id'],'seed':seed,'reason':'line_pool_empty'}
    pair_ratio=em['pairs']/target['n_pairs'] if target['n_pairs'] else None
    edit_tv=tv(em,target)
    d={'valid':True,'section':section,'config':cfg['id'],'seed':seed,'attempts':attempts,
       **em,**sm,**lm,'pair_ratio':pair_ratio,'edit_tv':edit_tv}
    if target.get('line_enrichment') is not None and lm['line_enrich'] is not None:
        errs=[math.log(max(pair_ratio,1e-12))/TOLS['pair_log'],edit_tv/TOLS['edit_tv'],
              (lm['line_enrich']-target['line_enrichment'])/TOLS['line_enrich'],
              (sm['hnext']-target['hnext'])/TOLS['hnext'],
              (sm['rml']-target['rml'])/TOLS['rml'],
              (sm['mean_len']-target['mean_type_len'])/TOLS['mean_len']]
    else:
        errs=[math.log(max(pair_ratio,1e-12))/TOLS['pair_log'],edit_tv/TOLS['edit_tv'],
              (sm['hnext']-target['hnext'])/TOLS['hnext'],
              (sm['rml']-target['rml'])/TOLS['rml'],
              (sm['mean_len']-target['mean_type_len'])/TOLS['mean_len']]
    d['loss']=sum(x*x for x in errs)/len(errs)
    d['target']={k:target.get(k) for k in ['n_pairs','prefix','internal','suffix','mean_type_len','hnext','rml','line_enrichment']}
    return d

def summarize_config(rows,sections):
    vr=[r for r in rows if r['valid']]
    if len(vr)!=len(sections)*len(DISCOVERY_SEEDS): return None
    per_sec={}
    for s in sections:
        xs=[r for r in vr if r['section']==s]
        per_sec[s]={'mean_loss':statistics.mean(r['loss'] for r in xs),'median_loss':statistics.median(r['loss'] for r in xs)}
    return {'mean_loss':statistics.mean(r['loss'] for r in vr),'median_loss':statistics.median(r['loss'] for r in vr),'per_section':per_sec}

def discovery(targets,sylls):
    sections=targets['discovery_sections']; allrows=[]; summaries=[]
    for cfg in config_grid():
        rows=[]
        for seed in DISCOVERY_SEEDS:
            for s in sections: rows.append(evaluate_section(s,targets['sections'][s],cfg,sylls,seed))
        allrows.extend(rows); sm=summarize_config(rows,sections)
        if sm:summaries.append({'config':cfg,**sm})
    summaries.sort(key=lambda x:(x['mean_loss'],['BASE','DOMAIN','STATE','LINE'].index(x['config']['family'])))
    base=next(x for x in summaries if x['config']['family']=='BASE')
    winner=summaries[0]
    tied=[x for x in summaries if x['mean_loss']<=winner['mean_loss']*1.02]
    tied.sort(key=lambda x:(['BASE','DOMAIN','STATE','LINE'].index(x['config']['family']),-(x['config']['w1']+x['config']['w2']+x['config']['line_w']),x['mean_loss']))
    winner=tied[0]
    improves=sum(winner['per_section'][s]['mean_loss']<base['per_section'][s]['mean_loss'] for s in sections)
    rel_improve=(base['mean_loss']-winner['mean_loss'])/base['mean_loss'] if base['mean_loss'] else 0.0
    unlock=(winner['config']['family']!='BASE' and rel_improve>=0.20 and improves>=3)
    return {'phase':'discovery','winner':winner,'base':base,'relative_improvement':rel_improve,
            'sections_improved':improves,'holdout_unlocked':unlock,'summaries':summaries,'rows':allrows}

def pass_section(r):
    t=r['target']; checks={
      'pair_ratio':0.8<=r['pair_ratio']<=1.25,
      'edit_tv':r['edit_tv']<=0.08,
      'line_enrich':(t['line_enrichment'] is None) or (r['line_enrich'] is not None and abs(r['line_enrich']-t['line_enrichment'])<=0.25),
      'hnext':abs(r['hnext']-t['hnext'])<=0.35,
      'rml':r['rml']<0 and abs(r['rml']-t['rml'])<=0.10,
      'mean_len':abs(r['mean_len']-t['mean_type_len'])<=0.75,
    }
    return checks,all(checks.values())

def holdout(targets,sylls,cfg):
    sections=targets['holdout_sections']; rows=[]
    for seed in HOLDOUT_SEEDS:
        for s in sections:
            r=evaluate_section(s,targets['sections'][s],cfg,sylls,seed)
            if r['valid']:r['checks'],r['all_pass']=pass_section(r)
            rows.append(r)
    summary={}
    for s in sections:
        xs=[r for r in rows if r['section']==s and r['valid']]
        med={k:statistics.median(r[k] for r in xs) for k in ['pair_ratio','edit_tv','line_enrich','hnext','rml','mean_len']}
        proxy={**xs[0],**med}; checks,ok=pass_section(proxy)
        summary[s]={'median':med,'median_checks':checks,'median_pass':ok,'seed_all_pass':sum(r['all_pass'] for r in xs),'n':len(xs)}
    overall=all(summary[s]['median_pass'] for s in sections)
    return {'phase':'holdout','config':cfg,'overall_pass':overall,'summary':summary,'rows':rows}

def diagnostics(targets,sylls,cfg):
    out=[]
    for s in targets.get('diagnostic_sections',[]):
        for seed in HOLDOUT_SEEDS[:4]:out.append(evaluate_section(s,targets['sections'][s],cfg,sylls,seed))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['discovery','holdout'],required=True)
    ap.add_argument('--targets-url',required=True); ap.add_argument('--config-json',default=None)
    args=ap.parse_args()
    rawt=urllib.request.urlopen(args.targets_url,timeout=60).read(); targets=json.loads(rawt)
    words,sylls,raw=acquire_lexicon()
    meta={'namespace':NS,'lex_sha256':hashlib.sha256(raw).hexdigest(),'lex_words':len(words),'unique_syllables':len(set(sylls)),
          'targets_sha256':hashlib.sha256(rawt).hexdigest(),'phase':args.phase}
    if args.phase=='discovery':result=discovery(targets,sylls)
    else:
        if not args.config_json:raise SystemExit('--config-json required for holdout')
        cfg=json.loads(args.config_json); result=holdout(targets,sylls,cfg); result['diagnostics']=diagnostics(targets,sylls,cfg)
    print(json.dumps({'meta':meta,'result':result},sort_keys=True))
if __name__=='__main__':main()

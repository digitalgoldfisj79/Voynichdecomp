#!/usr/bin/env python3
import hashlib, json, math, random, statistics
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

SEED=20260815
NFOLD=5
NREP=30
CARRIERS=['daiin','ol','chedy','aiin','cshedy','chol','or','ar','chey','dar','qokeey','qokeedy','cshey','qokain','qokedy','dy','qokaiin','al','dal','chor']
EMPTY_MARKERS={'','∅',None}

@lru_cache(maxsize=1000000)
def is_ed1(a,b):
    if a==b:return False
    la,lb=len(a),len(b)
    if abs(la-lb)>1:return False
    if la==lb:return sum(x!=y for x,y in zip(a,b))==1
    if la>lb:a,b=b,a;la,lb=lb,la
    i=j=d=0
    while i<la and j<lb:
        if a[i]==b[j]:i+=1;j+=1
        else:
            d+=1;j+=1
            if d>1:return False
    return True

def lenbin(n):
    if n<=6:return 'S'
    if n<=9:return 'L7_9'
    if n<=12:return 'L10_12'
    if n<=16:return 'L13_16'
    return 'L17P'

def posclass(i,n):
    if n<5:return 'START' if i<min(2,n) else 'END'
    if i<2:return 'START'
    if i>=n-2:return 'END'
    return 'MID'

def groups(n,null):
    if null=='N0':return [list(range(n))] if n else []
    if null=='N1':
        if n<5:
            return [g for g in (list(range(min(2,n))),list(range(min(2,n),n))) if g]
        return [list(range(0,2)),list(range(2,n-2)),list(range(n-2,n))]
    raise ValueError(null)

def p70_state(r):return 1 if bool(r['empty_core']) else 0

def sig(r):return (r.get('prefix',''),r.get('gallows',''),r.get('core',''),r.get('suffix',''))

def stable_hash(s):return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)

def load_data():
    obj=json.load(open('enriched_records.json',encoding='utf-8'))
    by=defaultdict(list)
    for r in obj['records']:
        rr=dict(r); rr['_state']=p70_state(rr); rr['_sig']=sig(rr)
        by[(rr['folio'],int(rr['line_no']))].append(rr)
    p0=set()
    slim=json.load(open('voynich_transcriptions_slim.json',encoding='utf-8'))
    for page,pd in slim['pages'].items():
        for lid,rec in pd.items():
            u=rec.get('u','')
            if len(u)>=2 and u[1:]=='P0':
                try:p0.add((page,int(lid)))
                except:pass
    lines=[]
    for key,rr in by.items():
        rr.sort(key=lambda x:int(x['pos']))
        section=rr[0]['section'] if rr else 'UNKNOWN'
        lines.append({'folio':key[0],'line_no':key[1],'section':section,'is_p0':key in p0,'items':rr})
    lines.sort(key=lambda x:(stable_hash(x['folio']),x['line_no']))
    # dominant section and stratified deterministic folds
    sec_counts=defaultdict(Counter)
    for L in lines:sec_counts[L['folio']][L['section']]+=len(L['items'])
    dom={f:sorted(c.items(),key=lambda kv:(-kv[1],kv[0]))[0][0] for f,c in sec_counts.items()}
    fold={}
    bysec=defaultdict(list)
    for f,s in dom.items():bysec[s].append(f)
    for s,fs in bysec.items():
        fs=sorted(fs,key=lambda f:stable_hash(f'{SEED}|{f}'))
        for j,f in enumerate(fs):fold[f]=j%NFOLD
    return lines,fold,dom

class Dist:
    def __init__(self):self.c=defaultdict(Counter)
    def add(self,key,val,n=1):self.c[key][val]+=n
    def draw(self,keys,rng):
        for key in keys:
            cc=self.c.get(key)
            if cc:
                vals=list(cc); w=[cc[x] for x in vals]; return rng.choices(vals,weights=w,k=1)[0],key
        raise RuntimeError('No backoff distribution')

def train_model(lines,train_folios,model):
    tok=Dist(); state=Dist(); sg=Dist(); token_state={}; sigtok=defaultdict(Counter); section_seen=Counter()
    for L in lines:
        if L['folio'] not in train_folios:continue
        n=len(L['items']); lb=lenbin(n); sec=L['section']; section_seen[sec]+=len(L['items'])
        for i,r in enumerate(L['items']):
            pc=posclass(i,n); t=r['token']; st=r['_state']; s=r['_sig']; token_state[t]=st
            tok.add(('global',),t); tok.add(('sec',sec),t); tok.add(('sec_lb',sec,lb),t); tok.add(('sec_pc',sec,pc),t); tok.add(('sec_lb_pc',sec,lb,pc),t)
            state.add(('global',),st); state.add(('sec',sec),st); state.add(('sec_pc',sec,pc),st); state.add(('sec_lb_pc',sec,lb,pc),st)
            sg.add(('state',st),s); sg.add(('sec_state',sec,st),s); sg.add(('sec_pc_state',sec,pc,st),s); sg.add(('sec_lb_pc_state',sec,lb,pc,st),s)
            sigtok[s][t]+=1
    canon={s:sorted(c.items(),key=lambda kv:(-kv[1],kv[0]))[0][0] for s,c in sigtok.items()}
    return {'tok':tok,'state':state,'sig':sg,'token_state':token_state,'canon':canon,'section_seen':section_seen,'model':model}

def generate_line(L,fit,rng,diag):
    n=len(L['items']); sec=L['section']; lb=lenbin(n); out=[]
    if not fit['section_seen'][sec]:diag['unseen_section_tokens']+=n
    for i in range(n):
        pc=posclass(i,n); m=fit['model']
        if m=='M0':
            t,k=fit['tok'].draw([('sec_lb',sec,lb),('sec',sec),('global',)],rng); st=fit['token_state'][t]
        elif m=='M1':
            t,k=fit['tok'].draw([('sec_lb_pc',sec,lb,pc),('sec_pc',sec,pc),('sec_lb',sec,lb),('sec',sec),('global',)],rng); st=fit['token_state'][t]
        elif m=='M2':
            st,ks=fit['state'].draw([('sec_lb_pc',sec,lb,pc),('sec_pc',sec,pc),('sec',sec),('global',)],rng)
            s,kg=fit['sig'].draw([('sec_lb_pc_state',sec,lb,pc,st),('sec_pc_state',sec,pc,st),('sec_state',sec,st),('state',st)],rng)
            t=fit['canon'][s]
        else:raise ValueError(m)
        out.append({'token':t,'state':int(st)})
    return {'folio':L['folio'],'line_no':L['line_no'],'section':sec,'is_p0':L['is_p0'],'items':out}

def build_oof(lines,folds,model,rep):
    folios=set(folds); out=[]; diag=Counter()
    for f in range(NFOLD):
        test={x for x in folios if folds[x]==f}; train=folios-test
        fit=train_model(lines,train,model)
        rng=random.Random(SEED + rep*100003 + {'M0':1000,'M1':2000,'M2':3000}[model] + f*101)
        for L in lines:
            if L['folio'] in test:out.append(generate_line(L,fit,rng,diag))
    out.sort(key=lambda x:(stable_hash(x['folio']),x['line_no']))
    return out,dict(diag)

def pair_group_multiplicity(n,null,pairs):
    gs=groups(n,null); gid={p:g for g,G in enumerate(gs) for p in G}; c=Counter()
    for p,q in pairs:c[(gid[p],gid[q])]+=1
    return gs,c

def pair_expect(items, G, H, classifier, same):
    tot=0; c=Counter()
    for i in G:
        for j in H:
            if same and i==j:continue
            tot+=1
            for k in classifier(items[i],items[j]):c[k]+=1
    if tot==0:return {}
    return {k:v/tot for k,v in c.items()}

def ed_classifier(a,b):
    ta,tb=a['token'],b['token']
    if not is_ed1(ta,tb):return []
    keys=[]; m=(len(ta)+len(tb))/2
    keys.append('len_short' if m<=4 else ('len_mid' if m<=6 else 'len_long'))
    if len(tb)>len(ta):keys.append('accretion')
    elif len(tb)<len(ta):keys.append('reduction')
    else:
        keys.append('substitution')
        idx=next((i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y),None)
        if idx is not None:keys.append('sub_first' if idx<len(ta)/2 else 'sub_second')
    sa,sb=a.get('state',-1),b.get('state',-1)
    if sa>=0 and sb>=0:
        keys.append('state_both_empty' if sa==1 and sb==1 else ('state_both_nonempty' if sa==0 and sb==0 else 'state_mixed'))
    return keys

def e2_classifier(a,b):
    if a['token']!=b['token']:return []
    st=a.get('state',-1)
    return ['e2_empty' if st==1 else 'e2_nonempty'] if st>=0 else []

def expected_counts(items,null,pairs,classifier):
    n=len(items)
    if not pairs:return Counter()
    gs,mult=pair_group_multiplicity(n,null,pairs); out=Counter()
    cache={}
    for (ga,gb),mm in mult.items():
        key=(ga,gb)
        if key not in cache:cache[key]=pair_expect(items,gs[ga],gs[gb],classifier,ga==gb)
        for k,p in cache[key].items():out[k]+=mm*p
    return out

def actual_counts(items,pairs,classifier):
    c=Counter()
    for i,j in pairs:
        for k in classifier(items[i],items[j]):c[k]+=1
    return c

def entropy_fraction(vals):
    n=len(vals)
    if n<2:return 0.0
    c=Counter(vals); H=-sum((v/n)*math.log2(v/n) for v in c.values()); return H/math.log2(n)

def score(lines):
    obs_all=Counter(); null_all=Counter(); p0_obs_wh=Counter(); p0_obs_int_ed=Counter(); p0_obs_int_e2=Counter()
    p0_n0_wh=Counter(); p0_n1_wh=Counter(); p0_n1_int_ed=Counter(); p0_n1_int_e2=Counter(); bvals=[]; bt=0
    p0_tokens=0
    for L in lines:
        items=L['items']; n=len(items)
        adj=[(i,i+1) for i in range(max(0,n-1))]; lag=[(i,i+2) for i in range(max(0,n-2))]
        a=actual_counts(items,adj,ed_classifier); obs_all.update(a); null_all.update(expected_counts(items,'N0',adj,ed_classifier))
        for i,j in lag:
            if items[i]['token']==items[j]['token'] and items[i]['token'] in CARRIERS:
                bvals.append(items[i+1]['token']); bt+=1
        if L['is_p0']:
            p0_tokens+=n
            p0_obs_wh.update(actual_counts(items,adj,ed_classifier)); p0_obs_wh.update(actual_counts(items,lag,e2_classifier))
            p0_n0_wh.update(expected_counts(items,'N0',adj,ed_classifier)); p0_n0_wh.update(expected_counts(items,'N0',lag,e2_classifier))
            p0_n1_wh.update(expected_counts(items,'N1',adj,ed_classifier)); p0_n1_wh.update(expected_counts(items,'N1',lag,e2_classifier))
            if n>=6:
                ai=[(i,i+1) for i in range(2,n-3)]
                p0_obs_int_ed.update(actual_counts(items,ai,ed_classifier)); p0_n1_int_ed.update(expected_counts(items,'N1',ai,ed_classifier))
            if n>=7:
                li=[(i,i+2) for i in range(2,n-4)]
                p0_obs_int_e2.update(actual_counts(items,li,e2_classifier)); p0_n1_int_e2.update(expected_counts(items,'N1',li,e2_classifier))
    def R(o,n,k):return float(o[k]/n[k]) if n[k]>0 else None
    ratios={
      'F1_e2_empty_N0':R(p0_obs_wh,p0_n0_wh,'e2_empty'),
      'F1_e2_empty_N1':R(p0_obs_wh,p0_n1_wh,'e2_empty'),
      'F1_e2_empty_N3':R(p0_obs_int_e2,p0_n1_int_e2,'e2_empty'),
      'F2_be_N0':R(p0_obs_wh,p0_n0_wh,'state_both_empty'),
      'F2_bn_N0':R(p0_obs_wh,p0_n0_wh,'state_both_nonempty'),
      'F2_mix_N0':R(p0_obs_wh,p0_n0_wh,'state_mixed'),
      'F2_be_N1':R(p0_obs_wh,p0_n1_wh,'state_both_empty'),
      'F2_bn_N1':R(p0_obs_wh,p0_n1_wh,'state_both_nonempty'),
      'F2_mix_N1':R(p0_obs_wh,p0_n1_wh,'state_mixed'),
      'F2_be_N3':R(p0_obs_int_ed,p0_n1_int_ed,'state_both_empty'),
      'F2_bn_N3':R(p0_obs_int_ed,p0_n1_int_ed,'state_both_nonempty'),
      'F2_mix_N3':R(p0_obs_int_ed,p0_n1_int_ed,'state_mixed'),
      'F3_short':R(obs_all,null_all,'len_short'),'F3_mid':R(obs_all,null_all,'len_mid'),'F3_long':R(obs_all,null_all,'len_long')}
    obs_ar=(obs_all['accretion']/obs_all['reduction']) if obs_all['reduction'] else None
    nul_ar=(null_all['accretion']/null_all['reduction']) if null_all['reduction'] else None
    obs_ss=(obs_all['sub_first']/obs_all['sub_second']) if obs_all['sub_second'] else None
    nul_ss=(null_all['sub_first']/null_all['sub_second']) if null_all['sub_second'] else None
    dirdev={
      'acc_red_obs':obs_ar,'acc_red_null':nul_ar,'acc_red_abs_logdev':abs(math.log(obs_ar/nul_ar)) if obs_ar and nul_ar else None,
      'subsite_obs':obs_ss,'subsite_null':nul_ss,'subsite_abs_logdev':abs(math.log(obs_ss/nul_ss)) if obs_ss and nul_ss else None}
    F={}
    q=ratios
    F['F1']=bool(q['F1_e2_empty_N0'] is not None and 1.10<=q['F1_e2_empty_N0']<=1.32 and q['F1_e2_empty_N1']<1.12 and q['F1_e2_empty_N3']<1.12)
    F['F2']=bool(q['F2_be_N0']>=1.10 and q['F2_bn_N0']>=1.15 and max(q['F2_be_N1'],q['F2_bn_N1'])>=1.07 and q['F2_mix_N1']<1.10 and q['F2_be_N3']<1.10 and q['F2_bn_N3']<1.10)
    F['F3']=bool(q['F3_long']>=1.15 and q['F3_short']>=1.10 and q['F3_long']>q['F3_mid'])
    F['F4']=bool(dirdev['acc_red_abs_logdev'] is not None and dirdev['subsite_abs_logdev'] is not None and dirdev['acc_red_abs_logdev']<.15 and dirdev['subsite_abs_logdev']<.15)
    bf=entropy_fraction(bvals); F['F5']=bool(bt>=50 and bf>=.80)
    return {'ratios':ratios,'direction':dirdev,'bslot':{'n':bt,'entropy_fraction':bf,'distinct_B':len(set(bvals))},'fingerprints':F,
      'counts':{'all_ed_len':{k:int(obs_all['len_'+k]) for k in ('short','mid','long')},'p0_tokens':p0_tokens}}

def median(xs):return statistics.median(xs)
def quant(xs,p):
    a=sorted(xs); i=max(0,min(len(a)-1,math.ceil(p*len(a))-1)); return a[i]

def aggregate(model_rows,real):
    keys=list(real['ratios']); med={}; spread={}
    for k in keys:
        vals=[r['ratios'][k] for r in model_rows if r['ratios'][k] is not None]
        med[k]=median(vals); spread[k]={'p10':quant(vals,.10),'p90':quant(vals,.90)}
    dkeys=[k for k in keys if k.startswith(('F1_','F2_','F3_')) and real['ratios'][k] and med[k]]
    rmse=math.sqrt(sum((math.log(med[k])-math.log(real['ratios'][k]))**2 for k in dkeys)/len(dkeys))
    dmed={k:median([r['direction'][k] for r in model_rows if r['direction'][k] is not None]) for k in ('acc_red_abs_logdev','subsite_abs_logdev')}
    bmed={'n':median([r['bslot']['n'] for r in model_rows]),'entropy_fraction':median([r['bslot']['entropy_fraction'] for r in model_rows])}
    # apply gates to median summary
    q=med; F={}
    F['F1']=bool(1.10<=q['F1_e2_empty_N0']<=1.32 and q['F1_e2_empty_N1']<1.12 and q['F1_e2_empty_N3']<1.12)
    F['F2']=bool(q['F2_be_N0']>=1.10 and q['F2_bn_N0']>=1.15 and max(q['F2_be_N1'],q['F2_bn_N1'])>=1.07 and q['F2_mix_N1']<1.10 and q['F2_be_N3']<1.10 and q['F2_bn_N3']<1.10)
    F['F3']=bool(q['F3_long']>=1.15 and q['F3_short']>=1.10 and q['F3_long']>q['F3_mid'])
    F['F4']=bool(dmed['acc_red_abs_logdev']<.15 and dmed['subsite_abs_logdev']<.15)
    F['F5']=bool(bmed['n']>=50 and bmed['entropy_fraction']>=.80)
    n=sum(F.values()); verdict='QUALIFIED_MEMORYLESS_ARCHITECTURE' if n==5 else ('PARTIAL_MEMORYLESS_ARCHITECTURE' if n==4 else 'FAILED_MEMORYLESS_ARCHITECTURE')
    passfrac={k:sum(r['fingerprints'][k] for r in model_rows)/len(model_rows) for k in F}
    return {'median_ratios':med,'ratio_p10_p90':spread,'median_direction':dmed,'median_bslot':bmed,'fingerprints':F,'replicate_pass_fraction':passfrac,'target_logratio_rmse':rmse,'n_pass':n,'verdict':verdict}

def validate(real):
    errs=[]
    if real['counts']['all_ed_len']!={'short':598,'mid':501,'long':146}:errs.append(f"ABC counts {real['counts']['all_ed_len']}")
    if real['counts']['p0_tokens']!=33200:errs.append(f"P0 tokens {real['counts']['p0_tokens']}")
    target={'F1_e2_empty_N0':1.212,'F1_e2_empty_N1':1.079,'F1_e2_empty_N3':1.084,
            'F2_be_N0':1.157,'F2_bn_N0':1.372,'F2_be_N1':1.097,'F2_bn_N1':1.214,'F2_be_N3':1.049,'F2_bn_N3':1.037}
    for k,v in target.items():
        tol=.03 if k.startswith('F1') else .04
        if real['ratios'][k] is None or abs(real['ratios'][k]-v)>tol:errs.append(f'{k}={real["ratios"][k]} vs {v}')
    return {'pass':not errs,'errors':errs}

def main():
    lines,folds,dom=load_data()
    # convert real records to compact scoring form
    real_lines=[]
    for L in lines:
        real_lines.append({'folio':L['folio'],'line_no':L['line_no'],'section':L['section'],'is_p0':L['is_p0'],'items':[{'token':r['token'],'state':r['_state']} for r in L['items']]})
    real=score(real_lines); val=validate(real)
    out={'metadata':{'seed':SEED,'folds':NFOLD,'replicates':NREP,'folios':len(set(folds)),'lines':len(lines),'tokens':sum(len(L['items']) for L in lines),'models':['M0','M1','M2']},'validation':val,'real':real,'fold_counts':dict(Counter(folds.values())),'models':{}}
    if not val['pass']:
        out['status']='ABORTED_VALIDATION_FAILED'
    else:
        out['status']='COMPLETE'
        for model in ('M0','M1','M2'):
            rows=[]; diags=[]
            for rep in range(NREP):
                gen,diag=build_oof(lines,folds,model,rep); rows.append(score(gen)); diags.append(diag)
            out['models'][model]={'aggregate':aggregate(rows,real),'replicates':rows,'diagnostics':{'median_unseen_section_tokens':median([d.get('unseen_section_tokens',0) for d in diags])}}
    outdir=Path('results/position_conditioned_p70_emission_v0_1'); outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'RESULTS_20260815.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    md=['# Position-conditioned P70 emission v0.1 — results','',f"Status: **{out['status']}**. Validation: **{'PASS' if val['pass'] else 'FAIL'}**.",f"Corpus: {out['metadata']['tokens']:,} tokens, {out['metadata']['lines']:,} lines, {out['metadata']['folios']} folios; 5 held-out folds; 30 complete OOF replicates/model.",'']
    if not val['pass']:
        md+=['Validation errors:']+[f'- {e}' for e in val['errors']]
    else:
        md+=['## Real-data exact-null anchor','|signature|ratio|','|---|---:|']
        for k,v in real['ratios'].items():md.append(f'|{k}|{v:.3f}|')
        md+=['','## Model decisions','|model|F1|F2|F3|F4|F5|passes|target RMSE|verdict|','|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for m in ('M0','M1','M2'):
            a=out['models'][m]['aggregate']; f=a['fingerprints']; md.append(f"|{m}|{'PASS' if f['F1'] else 'FAIL'}|{'PASS' if f['F2'] else 'FAIL'}|{'PASS' if f['F3'] else 'FAIL'}|{'PASS' if f['F4'] else 'FAIL'}|{'PASS' if f['F5'] else 'FAIL'}|{a['n_pass']}/5|{a['target_logratio_rmse']:.4f}|**{a['verdict']}**|")
        md+=['','## Median generated ratios','|signature|real|M0|M1|M2|','|---|---:|---:|---:|---:|']
        for k,v in real['ratios'].items():md.append(f"|{k}|{v:.3f}|{out['models']['M0']['aggregate']['median_ratios'][k]:.3f}|{out['models']['M1']['aggregate']['median_ratios'][k]:.3f}|{out['models']['M2']['aggregate']['median_ratios'][k]:.3f}|")
        md+=['','## Direction and ABA freedom','|model|acc/red logdev|subsite logdev|ABA n|H(B)/max|','|---|---:|---:|---:|---:|']
        for m in ('M0','M1','M2'):
            a=out['models'][m]['aggregate']; md.append(f"|{m}|{a['median_direction']['acc_red_abs_logdev']:.3f}|{a['median_direction']['subsite_abs_logdev']:.3f}|{a['median_bslot']['n']:.1f}|{a['median_bslot']['entropy_fraction']:.3f}|")
        a=out['models']['M2']['aggregate']; failed=[k for k,v in a['fingerprints'].items() if not v]
        md+=['',f"## Primary M2 verdict: **{a['verdict']}**",f"Failed fingerprints: {', '.join(failed) if failed else 'none'}."]
        if not failed:md.append('Per preregistration: do not add a local kernel for these signatures.')
        else:md.append('Per preregistration: only the smallest failed fingerprint(s) may motivate a subsequent local mechanism; generic generator expansion is disallowed.')
    (outdir/'RESULTS_20260815.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__':main()

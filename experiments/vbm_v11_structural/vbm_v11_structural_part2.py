def branch_B(segments):
    occ=nucleus_occurrences(segments);cnt=collections.Counter(o['nucleus'] for o in occ);elig=sorted([n for n,c in cnt.items() if c>=20])
    bc=collections.Counter(x for o in occ for x in (o['prev'],o['next']) if x!='EDGE');topb=[x for x,_ in bc.most_common(64)]
    def one(half,tag,NNULL=10000):
        oo=occ if half is None else nucleus_occurrences(segments,half)
        X=nucleus_context_matrix(oo,elig,topb);D=js_matrix(X);ix={n:i for i,n in enumerate(elig)}
        groups=collections.defaultdict(list)
        for n in elig:groups[eskel(n)].append(n)
        pairs=[]
        for g,ls in groups.items():
            for i in range(len(ls)):
                for j in range(i+1,len(ls)):
                    if ecount(ls[i])!=ecount(ls[j]):pairs.append((ls[i],ls[j]))
        if not pairs:return {'pairs':0,'obs':None,'z':None,'p':1.0}
        obs=float(np.median([D[ix[a],ix[b]] for a,b in pairs]))
        freqs=np.array([cnt[n] for n in elig],float);qs=np.quantile(freqs,np.linspace(0,1,11))
        dec={n:min(9,max(0,int(np.searchsorted(qs,cnt[n],side='right')-2))) for n in elig}
        pools={}
        for a,b in pairs:
            cand=[c for c in elig if eskel(c)!=eskel(a) and dec[c]==dec[b] and abs(len(c)-len(b))<=1 and c[0]==b[0]]
            if not cand:cand=[c for c in elig if eskel(c)!=eskel(a) and dec[c]==dec[b] and abs(len(c)-len(b))<=1]
            if not cand:cand=[c for c in elig if eskel(c)!=eskel(a) and dec[c]==dec[b]]
            pools[(a,b)]=cand or [c for c in elig if eskel(c)!=eskel(a)]
        rng=np.random.default_rng(seed(NS,'B_NULL',tag));null=np.empty(NNULL,float)
        for r in range(NNULL):
            vals=[]
            for a,b in pairs:
                c=pools[(a,b)][int(rng.integers(len(pools[(a,b)])))]
                vals.append(D[ix[a],ix[c]])
            null[r]=np.median(vals)
        sd=float(np.std(null,ddof=1));z=(float(np.mean(null))-obs)/sd if sd>0 else 0.0
        p=float((1+np.sum(null<=obs))/(NNULL+1))
        return {'pairs':len(pairs),'obs_median_js':obs,'null_mean':float(np.mean(null)),'null_sd':sd,'z':z,'p':p}
    full=one(None,'FULL');ha=one('A','A');hb=one('B','B')
    gate=bool(full['pairs']>=20 and full['z'] is not None and full['z']>=2.5 and full['p']<=.01 and ha['z'] is not None and hb['z'] is not None and ha['z']>=1.5 and hb['z']>=1.5)
    return {'eligible_nuclei':len(elig),'full':full,'half_A':ha,'half_B':hb,'gate':gate,
            'verdict':'B_E_LADDER_COMPOSITIONALITY_SUPPORTED' if gate else 'B_NO_E_LADDER_COMPOSITIONALITY'}

def context_bucket(n,top):
    if not n or n=='EMPTY':return 'EMPTY'
    return n if n in top else 'OTHER'

def dirichlet_probs(counts,keys,alpha=.5):
    total=sum(counts.values())+alpha*len(keys)
    return {k:(counts.get(k,0)+alpha)/total for k in keys}

def branch_C(segments):
    train=bridge_occurrences(segments,'TRAIN');hold=bridge_occurrences(segments,'HOLD')
    nc=collections.Counter(x for o in train for x in (o['leftN'],o['rightN']) if x!='EMPTY');top=[x for x,_ in nc.most_common(32)]
    keys=top+['OTHER','EMPTY']
    def side_eval(side,lmap_override=None):
        pooled=collections.Counter();pair=collections.defaultdict(collections.Counter);rc=collections.defaultdict(collections.Counter);lc=collections.defaultdict(collections.Counter)
        def halves(b):
            r,l=b.split('|',1);return r,(lmap_override or {}).get(b,l)
        for o in train:
            c=context_bucket(o[side],top);b=o['bridge'];r,l=halves(b);pooled[c]+=1;pair[b][c]+=1;rc[r][c]+=1;lc[l][c]+=1
        pp=dirichlet_probs(pooled,keys)
        def probs(counter):return dirichlet_probs(counter,keys)
        cacheP={b:probs(v) for b,v in pair.items()};cacheR={r:probs(v) for r,v in rc.items()};cacheL={l:probs(v) for l,v in lc.items()}
        ll0=llp=lla=0.;n=0
        for o in hold:
            c=context_bucket(o[side],top);b=o['bridge'];r,l=halves(b);n+=1
            ll0+=math.log(pp[c]);llp+=math.log(cacheP.get(b,pp)[c])
            pr=cacheR.get(r,pp);pl=cacheL.get(l,pp);raw=np.array([pr[k]*pl[k]/pp[k] for k in keys]);raw/=raw.sum();lla+=math.log(raw[keys.index(c)])
        return {'M0':ll0/max(1,n),'MPAIR':llp/max(1,n),'MADD':lla/max(1,n),'n':n}
    results={};gate_sides=[]
    freq=collections.Counter(o['bridge'] for o in train);bs=sorted(freq,key=str);vals=np.array([freq[b] for b in bs],float);qs=np.quantile(vals,np.linspace(0,1,11));dec={b:min(9,max(0,int(np.searchsorted(qs,freq[b],side='right')-2))) for b in bs}
    for side in ['leftN','rightN']:
        obs=side_eval(side);pg=obs['MPAIR']-obs['M0'];ag=obs['MADD']-obs['M0'];ratio=ag/pg if pg>0 else float('-inf')
        null=[]
        for r0 in range(100):
            rng=np.random.default_rng(seed(NS,'C_NULL',side,r0));mp={}
            for d in range(10):
                group=[b for b in bs if dec[b]==d];Ls=[b.split('|',1)[1] for b in group];rng.shuffle(Ls)
                for b,l in zip(group,Ls):mp[b]=l
            z=side_eval(side,mp);null.append(z['MADD']-z['M0'])
        p=(1+sum(x>=ag for x in null))/(len(null)+1);ok=(pg>0 and ratio>=.80 and p<=.01);gate_sides.append(ok)
        results[side]={'scores':obs,'pair_gain':pg,'additive_gain':ag,'factorisation_ratio':ratio,'null_p':p,'null_p99':float(np.quantile(null,.99)),'side_gate':ok}
    gate=all(gate_sides)
    return {'sides':results,'gate':gate,'verdict':'C_BRIDGE_HALVES_FACTORISE' if gate else 'C_NO_STRONG_HALF_FACTORISATION'}

def runlen_models(v10):
    models={}
    for lang in ['DE','IT']:
        la={'DE':'de','IT':'it'}[lang];txt=v10['bank'](la,'V11_RUNLEN',600000)
        runs,_=v10['decomp'](txt);seq=[min(5,len(r)) for r in runs]
        cnt=np.full((6,6),.5,float);start=np.full(6,.5,float)
        if seq:start[seq[0]]+=1
        for a,b in zip(seq,seq[1:]):cnt[a,b]+=1
        cnt/=cnt.sum(1,keepdims=True);start/=start.sum();models[lang]=(start,cnt)
    return models

def rule_len(n,r):
    if not n:return 0
    ec=n.count('e');em=max([len(x) for x in re.findall(r'e+',n)] or [0]);L=len(n);term=int(n[-1] in 'dyrl')
    if r=='D1':return 1+min(4,ec)
    if r=='D2':return 1+min(4,em)
    if r=='D3':return min(5,L)
    if r=='D4':return min(5,1+ec+term)
    if r=='D5':return min(5,max(1,L-ec+1))
    raise KeyError(r)

def score_run_segments(segments,split,mapping,model):
    start,trans=model;ll=0.;n=0
    for s in segments:
        if s['split']!=split:continue
        seq=[0 if not x else mapping[x] for x in s['nuclei']]
        if not seq:continue
        ll+=math.log(start[seq[0]]);n+=1
        for a,b in zip(seq,seq[1:]):ll+=math.log(trans[a,b]);n+=1
    return ll/max(1,n)

def branch_D(segments,v10):
    rules=['D1','D2','D3','D4','D5'];models=runlen_models(v10)
    allnts=sorted({n for s in segments for n in s['nuclei'] if n})
    nts=sorted({n for s in segments if s['split']=='TRAIN' for n in s['nuclei'] if n});cnt=collections.Counter(n for s in segments if s['split']=='TRAIN' for n in s['nuclei'] if n)
    maps={r:{n:rule_len(n,r) for n in allnts} for r in rules}
    scores={split:{lang:{r:score_run_segments(segments,split,maps[r],models[lang]) for r in rules} for lang in ['DE','IT']} for split in ['TRAIN','HOLD']}
    def combined(split,r):return .5*(scores[split]['DE'][r]+scores[split]['IT'][r])
    winT=max(rules,key=lambda r:(combined('TRAIN',r),r));winH=max(rules,key=lambda r:(combined('HOLD',r),r))
    vals=np.array([cnt[n] for n in nts],float);qs=np.quantile(vals,np.linspace(0,1,11));dec={n:min(9,max(0,int(np.searchsorted(qs,cnt[n],side='right')-2))) for n in nts};groups={d:[n for n in nts if dec[n]==d] for d in range(10)}
    nullmax={lang:np.empty(1000,float) for lang in ['DE','IT']}
    for q in range(1000):
        rng=np.random.default_rng(seed(NS,'D_NULL',q));nullscores={lang:[] for lang in ['DE','IT']}
        for r in rules:
            mm=maps[r].copy()
            for d,g in groups.items():
                if len(g)>1:
                    lens=[mm[n] for n in g];rng.shuffle(lens)
                    for n,v in zip(g,lens):mm[n]=v
            for lang in ['DE','IT']:nullscores[lang].append(score_run_segments(segments,'HOLD',mm,models[lang]))
        for lang in ['DE','IT']:nullmax[lang][q]=max(nullscores[lang])
    p={lang:float((1+np.sum(nullmax[lang]>=scores['HOLD'][lang][winH]))/1001) for lang in ['DE','IT']}
    gate=bool(winT==winH and all(p[la]<=.01 for la in ['DE','IT']))
    return {'scores':scores,'winner_train':winT,'winner_hold':winH,'hold_familywise_p':p,
            'null_p99':{la:float(np.quantile(nullmax[la],.99)) for la in ['DE','IT']},'gate':gate,
            'verdict':'D_SIMPLE_LENGTH_RULE_SURVIVES' if gate else 'D_NO_SIMPLE_MORPHOLOGY_LENGTH_RULE'}

def internal_bridge_model(segments):
    c=collections.Counter(b for s in segments if s['split']=='TRAIN' for b in s['bridges']);tot=sum(c.values());V=len(c)+1;den=tot+.5*V
    return c,{b:math.log((n+.5)/den) for b,n in c.items()},math.log(.5/den)

def edge_metrics(pairs,c,lp,unseen):
    if not pairs:return {'n':0,'support':float('nan'),'mean_logp':float('nan')}
    return {'n':len(pairs),'support':sum(p in c for p in pairs)/len(pairs),'mean_logp':float(np.mean([lp.get(p,unseen) for p in pairs]))}

def line_pairs(lines,split,cyclic=False):
    by=collections.defaultdict(list)
    for L in lines:
        if L['split']==split:by[L['folio']].append(L)
    pairs=[];groups=[]
    for fid,ls in by.items():
        ls=sorted(ls,key=lambda x:line_key(x['line']))
        if cyclic:
            p=[L['end_right']+'|'+L['start_left'] for L in ls];pairs.extend(p);groups.append((fid,[L['end_right'] for L in ls],[L['start_left'] for L in ls]))
        else:
            for a,b in zip(ls,ls[1:]):pairs.append(a['end_right']+'|'+b['start_left'])
            if len(ls)>1:groups.append((fid,[L['end_right'] for L in ls[:-1]],[L['start_left'] for L in ls[1:]]))
    return pairs,groups

def edge_null(groups,c,lp,unseen,nnull,tag):
    sup=np.empty(nnull);ll=np.empty(nnull);rng=np.random.default_rng(seed(NS,'E_NULL',tag))
    for r in range(nnull):
        ps=[]
        for fid,ends,starts in groups:
            st=list(starts);rng.shuffle(st);ps.extend([a+'|'+b for a,b in zip(ends,st)])
        m=edge_metrics(ps,c,lp,unseen);sup[r]=m['support'];ll[r]=m['mean_logp']
    return sup,ll

def branch_E(segments,lines):
    c,lp,unseen=internal_bridge_model(segments)
    tr,grp=line_pairs(lines,'TRAIN',False);obs=edge_metrics(tr,c,lp,unseen);ns,nl=edge_null(grp,c,lp,unseen,10000,'TRAIN')
    pS=(1+np.sum(ns>=obs['support']))/10001;pL=(1+np.sum(nl>=obs['mean_logp']))/10001
    ho,hg=line_pairs(lines,'HOLD',False);hob=edge_metrics(ho,c,lp,unseen);hs,hl=edge_null(hg,c,lp,unseen,2000,'HOLD')
    rep=(hob['support']>float(np.median(hs)) and hob['mean_logp']>float(np.median(hl)))
    gate=bool(pS<=.01 and pL<=.01 and rep)
    cy,cg=line_pairs(lines,'TRAIN',True);cym=edge_metrics(cy,c,lp,unseen);cs,cl=edge_null(cg,c,lp,unseen,5000,'CYCLIC')
    return {'sequential_train':obs,'train_support_p':float(pS),'train_logp_p':float(pL),
            'train_null_support_p99':float(np.quantile(ns,.99)),'train_null_logp_p99':float(np.quantile(nl,.99)),
            'sequential_hold':hob,'hold_null_support_median':float(np.median(hs)),'hold_null_logp_median':float(np.median(hl)),'hold_replication_direction':rep,
            'cyclic_exploratory':cym,'cyclic_support_p':float((1+np.sum(cs>=cym['support']))/(len(cs)+1)),
            'cyclic_logp_p':float((1+np.sum(cl>=cym['mean_logp']))/(len(cl)+1)),
            'gate':gate,'verdict':'E_SEQUENTIAL_CLOSURE_SUPPORTED' if gate else 'E_NO_SEQUENTIAL_CLOSURE_EVIDENCE'}

def main():
    data=get_json(DATA_URL);segments,lines=build_corpus(data);v10=load_v10()
    meta={'protocol':'VBM_V11_STRUCTURAL_CONSTRAINTS_PROTOCOL.md','segments':len(segments),'lines':len(lines),
          'train_segments':sum(s['split']=='TRAIN' for s in segments),'hold_segments':sum(s['split']=='HOLD' for s in segments)}
    print('V11_META='+json.dumps(meta,sort_keys=True),flush=True)
    out={}
    for name,fn in [('A',lambda:branch_A(segments,v10)),('B',lambda:branch_B(segments)),('C',lambda:branch_C(segments)),('D',lambda:branch_D(segments,v10)),('E',lambda:branch_E(segments,lines))]:
        print(f'V11_START_{name}',flush=True);z=fn();out[name]=z;print(f'V11_RESULT_{name}='+json.dumps(z,sort_keys=True),flush=True)
    positives=sum(bool(out[k]['gate']) for k in out)
    if positives==0:decision='V11_NO_NEW_IDENTIFYING_STRUCTURE'
    elif positives==1:decision='V11_ONE_STRUCTURAL_CONSTRAINT_REQUIRES_INDEPENDENT_MODEL'
    else:decision='V11_MULTIPLE_STRUCTURAL_CONSTRAINTS_JUSTIFY_V12_SYNTHETIC_MODEL'
    final={'positive_primary_branches':positives,'decision':decision,'branch_verdicts':{k:out[k]['verdict'] for k in out},'voynich_plaintext_opened':False,'gpu_used':False}
    print('VBM_V11_FINAL_RESULT='+json.dumps(final,sort_keys=True),flush=True)

if __name__=='__main__':main()

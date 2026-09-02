#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "scipy>=1.13,<2", "scikit-learn>=1.5,<2"]
# ///
from __future__ import annotations
import collections, hashlib, json, re, urllib.request
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

NS='VBMV13EOP20260902'
BRANCH='experiment/vbm-v13-e-operator-geometry-20260902'
V11_URL=f'https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/{BRANCH}/experiments/vbm_v11_structural/vbm_v11_structural_part1.py'
V12_URL=f'https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/{BRANCH}/experiments/vbm_v12_compositional/vbm_v12_compositional_runner.py'
UA={'User-Agent':'VBMV13EOperator/2026-09-02'}
KGRID=tuple(range(2,17)); ALPHA=.5; NNULL=10000

def seed(*xs):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,(NS,)+xs)).encode()).digest()[:8],'big') & 0x7fffffff

def get_text(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=120) as r:return r.read().decode('utf-8')

def load_remote(url,name):
    ns={'__name__':name}; exec(compile(get_text(url),url,'exec'),ns); return ns

def eskel(s): return re.sub(r'e+','E',s)
def ecount(s): return s.count('e')

def split_bridge(b):
    r,l=b.split('|',1); return r,l

def actual_occurrences(segments,split):
    out=[]
    for seg in segments:
        if seg['split']!=split: continue
        nuc=seg['nuclei']; br=seg['bridges']
        for i,n in enumerate(nuc):
            if not n: continue
            if i>0: pr,pl=split_bridge(br[i-1])
            else: pr=pl='EDGE'
            if i<len(br): nr,nl=split_bridge(br[i])
            else: nr=nl='EDGE'
            out.append((n,(pr,pl,nr,nl)))
    return out

def synthetic_occurrences(lines,stage):
    out=[]
    for ns,bs in lines:
        for i,n in enumerate(ns):
            if i>0:
                b=int(bs[i-1]); pr=b//stage.L; pl=b%stage.L
            else: pr=pl='EDGE'
            if i<len(bs):
                b=int(bs[i]); nr=b//stage.L; nl=b%stage.L
            else: nr=nl='EDGE'
            out.append((int(n),(pr,pl,nr,nl)))
    return out

def build_features(train_occ,hold_occ,eligible):
    vocabs=[]
    for q in range(4):
        c=collections.Counter(v[q] for _,v in train_occ if v[q]!='EDGE')
        vocabs.append([x for x,_ in c.most_common(32)])
    def one(occ):
        by=collections.defaultdict(list)
        for t,v in occ:
            if t in eligible: by[t].append(v)
        X=[]
        for t in eligible:
            parts=[]; vals=by.get(t,[])
            for q in range(4):
                vocab=vocabs[q]; idx={x:i for i,x in enumerate(vocab)}; arr=np.full(len(vocab)+2,ALPHA,float)
                for v in vals:
                    z=v[q]; j=len(vocab)+1 if z=='EDGE' else idx.get(z,len(vocab)); arr[j]+=1
                arr/=arr.sum(); parts.append(np.sqrt(arr))
            X.append(np.concatenate(parts))
        return np.vstack(X)
    return one(train_occ),one(hold_occ)

def type_counts(occ): return collections.Counter(t for t,_ in occ)

def ladder_steps(eligible,traincnt,skeleton_of,m_of):
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for t in eligible: groups[skeleton_of(t)][m_of(t)].append(t)
    steps=[]; chains=[]
    for sk,levels in groups.items():
        lev=sorted(levels)
        def ranked(m): return sorted(levels[m],key=lambda x:(-traincnt[x],str(x)))
        for m in lev:
            if m+1 in levels:
                a=ranked(m);b=ranked(m+1)
                for x,y in zip(a,b): steps.append((x,y,m,traincnt[y],sk))
            if m+1 in levels and m+2 in levels:
                a=ranked(m);b=ranked(m+1);c=ranked(m+2)
                for x,y,z in zip(a,b,c): chains.append((x,y,z,m,traincnt[z],sk))
    return steps,chains

def norm_acc(acc,k): return (acc-1.0/k)/(1.0-1.0/k)

def fit_k_models(Xtr,Xh,eligible):
    ix={t:i for i,t in enumerate(eligible)}; models={}
    for k in KGRID:
        if k>=len(eligible): continue
        km=KMeans(n_clusters=k,n_init=64,random_state=seed('KMEANS',k)).fit(Xtr)
        models[k]={'lt':km.labels_.astype(int),'lh':km.predict(Xh).astype(int)}
    return models,ix

def permutation_from_steps(labels,steps,ix,k):
    C=np.zeros((k,k),dtype=int)
    for a,b,*_ in steps:C[labels[ix[a]],labels[ix[b]]]+=1
    r,c=linear_sum_assignment(-C); P=np.arange(k,dtype=int);P[r]=c
    acc=float(np.mean([P[labels[ix[a]]]==labels[ix[b]] for a,b,*_ in steps])) if steps else float('nan')
    return P,acc

def select_operator(models,steps,ix):
    rows=[]
    for k,M in models.items():
        P,acc=permutation_from_steps(M['lt'],steps,ix,k);rows.append((norm_acc(acc,k),-k,k,P,acc))
    if not rows:return None
    best=max(rows,key=lambda x:(x[0],x[1]));_,_,k,P,ta=best;M=models[k]
    ha=float(np.mean([P[M['lh'][ix[a]]]==M['lh'][ix[b]] for a,b,*_ in steps])) if steps else float('nan')
    return {'k':k,'P':P,'train_acc':ta,'train_nacc':norm_acc(ta,k),'hold_acc':ha,'hold_nacc':norm_acc(ha,k)}

def tertiles(vals):
    return tuple(np.quantile(np.asarray(vals,float),[1/3,2/3]).tolist()) if vals else (0,0)
def tert(v,q): return 0 if v<=q[0] else (1 if v<=q[1] else 2)

def shuffle_targets(steps,rng):
    q=tertiles([x[3] for x in steps]); groups=collections.defaultdict(list); out=[list(x) for x in steps]
    for i,x in enumerate(steps):groups[(x[2],tert(x[3],q))].append(i)
    for inds in groups.values():
        tg=[out[i][1] for i in inds];rng.shuffle(tg)
        for i,z in zip(inds,tg):out[i][1]=z
    return [tuple(x) for x in out]

def cycles_of(P):
    seen=set();cy=[]
    for i in range(len(P)):
        if i in seen:continue
        cur=[];x=i
        while x not in seen:
            seen.add(x);cur.append(int(x));x=int(P[x])
        cy.append(cur)
    return sorted(cy,key=lambda c:(-len(c),c))

def observed_operator(train_occ,hold_occ,eligible,steps,chains,do_null=True):
    Xtr,Xh=build_features(train_occ,hold_occ,eligible);models,ix=fit_k_models(Xtr,Xh,eligible);op=select_operator(models,steps,ix)
    if op is None:return {'steps':len(steps),'chains':len(chains),'error':'no_models'}
    out={'steps':len(steps),'chains':len(chains),'K':op['k'],'train_acc':op['train_acc'],'train_nacc':op['train_nacc'],'hold_acc':op['hold_acc'],'hold_nacc':op['hold_nacc'],'cycles':cycles_of(op['P'])}
    if do_null:
        rng=np.random.default_rng(seed('A_NULL'));null=np.empty(NNULL,float)
        for r in range(NNULL):
            z=select_operator(models,shuffle_targets(steps,rng),ix);null[r]=z['hold_acc']
        out['null_p99']=float(np.quantile(null,.99));out['p']=float((1+np.sum(null>=op['hold_acc']))/(NNULL+1));out['gate']=bool(len(steps)>=15 and op['hold_nacc']>=.50 and op['hold_acc']>out['null_p99'] and out['p']<=.01);out['verdict']='A_SHARED_E_PERMUTATION_SUPPORTED' if out['gate'] else 'A_NO_SHARED_E_PERMUTATION'
    if len(chains)<5:out['B']={'chains':len(chains),'gate':False,'verdict':'B_UNDERPOWERED_TWO_STEP'}
    else:
        M=models[op['k']];P2=op['P'][op['P']];acc=float(np.mean([P2[M['lh'][ix[a]]]==M['lh'][ix[c]] for a,_b,c,*_ in chains]));B={'chains':len(chains),'hold_acc':acc}
        if do_null:
            q=tertiles([x[4] for x in chains]);rng2=np.random.default_rng(seed('B_NULL'));nul=np.empty(NNULL,float);base=[list(x) for x in chains];groups=collections.defaultdict(list)
            for i,x in enumerate(chains):groups[(x[3],tert(x[4],q))].append(i)
            for r in range(NNULL):
                cc=[x[:] for x in base]
                for inds in groups.values():
                    tg=[cc[i][2] for i in inds];rng2.shuffle(tg)
                    for i,z in zip(inds,tg):cc[i][2]=z
                nul[r]=np.mean([P2[M['lh'][ix[a]]]==M['lh'][ix[c]] for a,_b,c,*_ in cc])
            B['null_p99']=float(np.quantile(nul,.99));B['p']=float((1+np.sum(nul>=acc))/(NNULL+1));B['gate']=bool(out.get('gate',False) and acc>=.60 and acc>B['null_p99'] and B['p']<=.01);B['verdict']='B_ITERATED_E_OPERATOR_SUPPORTED' if B['gate'] else 'B_NO_ITERATED_E_OPERATOR'
        out['B']=B
    return out

def synthetic_case(v12,fam,rep,mode):
    st=v12['STAGE_A'];lines,_latent,_truth=v12['generate'](st,fam,rep,mode);cut=int(.8*len(lines));tr=synthetic_occurrences(lines[:cut],st);ho=synthetic_occurrences(lines[cut:],st);ct=type_counts(tr);ch=type_counts(ho);eligible=sorted([t for t in ct if ct[t]>=20 and ch[t]>=5]);sk=lambda t:int(t)//st.E;mm=lambda t:int(t)%st.E;steps,chains=ladder_steps(eligible,ct,sk,mm);return observed_operator(tr,ho,eligible,steps,chains,False)

def actual_case(v11):
    data=v11['get_json'](v11['DATA_URL']);segments,_=v11['build_corpus'](data);tr=actual_occurrences(segments,'TRAIN');ho=actual_occurrences(segments,'HOLD');ct=type_counts(tr);ch=type_counts(ho);eligible=sorted([t for t in ct if ct[t]>=20 and ch[t]>=5]);steps,chains=ladder_steps(eligible,ct,eskel,ecount);return observed_operator(tr,ho,eligible,steps,chains,True),{'eligible':len(eligible),'train_occ':len(tr),'hold_occ':len(ho)}

def main():
    v11=load_remote(V11_URL,'v11base');v12=load_remote(V12_URL,'v12base');synth=[]
    for fam in ['PEAKED','MODERATE']:
        for rep in range(3):
            pos=synthetic_case(v12,fam,rep,'POS');neg=synthetic_case(v12,fam,rep,'NUC_BROKEN');row={'family':fam,'rep':rep,'POS':pos,'NUC_BROKEN':neg};synth.append(row);print('V13_SYNTH='+json.dumps(row,sort_keys=True),flush=True)
    pos=[r['POS'] for r in synth];neg=[r['NUC_BROKEN'] for r in synth];medgap=float(np.median([r['hold_nacc'] for r in pos])-np.median([r['hold_nacc'] for r in neg]));cal={'pos_nacc_ge_050':sum(r['hold_nacc']>=.50 for r in pos),'median_pos_nacc':float(np.median([r['hold_nacc'] for r in pos])),'median_neg_nacc':float(np.median([r['hold_nacc'] for r in neg])),'median_gap':medgap,'paired_raw_wins':sum(p['hold_acc']>n['hold_acc'] for p,n in zip(pos,neg))};cal['qualified']=bool(cal['pos_nacc_ge_050']>=5 and medgap>=.25 and cal['paired_raw_wins']>=5);print('V13_CALIBRATION='+json.dumps(cal,sort_keys=True),flush=True)
    voy,meta=actual_case(v11);print('V13_VOYNICH='+json.dumps({'meta':meta,'result':voy},sort_keys=True),flush=True)
    if not cal['qualified']:ver='V13_METHOD_NOT_QUALIFIED'
    elif not voy.get('gate',False):ver='V13_E_LADDER_SIMILARITY_WITHOUT_GLOBAL_OPERATOR'
    elif not voy.get('B',{}).get('gate',False):ver='V13_ONE_STEP_OPERATOR_SIGNAL_ONLY'
    else:ver='V13_SHARED_ITERATED_E_OPERATOR_SUPPORTED'
    print('VBM_V13_FINAL_RESULT='+json.dumps({'verdict':ver,'calibration':cal,'voynich':voy,'meta':meta,'plaintext_opened':False,'gpu_used':False},sort_keys=True),flush=True)
if __name__=='__main__':main()

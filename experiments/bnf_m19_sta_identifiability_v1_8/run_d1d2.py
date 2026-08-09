#!/usr/bin/env python3
import urllib.request, hashlib, math, json
import numpy as np

Q3_URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/15e1cfa0e37119907d6a99ba6b2e2be1c4730fa6/experiments/bnf_m19_sta_hierarchy_v1_7/run_v17_q3_qual.py'
q3=urllib.request.urlopen(Q3_URL,timeout=120).read().decode()
prefix=q3.split("K=int(os.environ.get('M19_K','0'))")[0]
ns={'__name__':'v18diag'}
exec(compile(prefix,'run_v17_q3_qual.py','exec'),ns)
b=ns['b']; qseed=ns['qseed']
K=22; LA='arabic'

lms,lmmeta=b['build_lms'](); comps={la:b['ns']['induced'](lms[la]) for la in b['LANGS']}; pools,poolmeta=b['control_pools']()
tr,ho,syms,true,attempt=b['gen_control'](pools[LA],LA,K); S=b['stats'](tr,syms); H=b['stats'](ho,syms); comp=comps[LA]

s1,m1=b['optimize'](S,comp,('qual',K,LA,1),K); s2,m2=b['optimize'](S,comp,('qual',K,LA,2),K)
fit=m1 if s1>=s2 else m2
acc=b['map_acc'](H['freq'],fit,true); agr=b['agreement'](S['freq'],m1,m2)
print('REPRO',json.dumps({'attempt':attempt,'acc':acc,'agreement':agr,'s1':s1,'s2':s2,'true_score':b['score_num'](S,true,comp),'fit_score':b['score_num'](S,fit,comp)},separators=(',',':')),flush=True)
if abs(acc-0.7632051282051282)>1e-12 or abs(agr-1.0)>1e-12:
    raise RuntimeError(('Q3 reproduction mismatch',acc,agr))

# D1 stronger optimizer: exact full rescoring, 24 x 100k, exhaustive legal local polish.
def strong_optimize(S,comp):
    best=(-1e100,None,None)
    for rr in range(24):
        rng=np.random.default_rng(qseed('v18-strong',rr)); m=b['init_map'](K,rng); s=b['score_num'](S,m,comp)
        ds=[]
        for _ in range(64):
            x,ch=b['proposal'](m,rng); ds.append(abs(b['score_num'](S,x,comp)-s))
        t0=max(1e-6,float(np.median(ds))*4); local=(s,m.copy())
        for k in range(100000):
            frac=k/99999.0; temp=max(1e-8,t0*(0.003**frac)); x,ch=b['proposal'](m,rng); s2=b['score_num'](S,x,comp); d=s2-s
            if d>=0 or rng.random()<math.exp(max(-60,d/temp)):
                m=x;s=s2
                if s>local[0]: local=(s,m.copy())
        m=local[1].copy(); s=b['score_num'](S,m,comp)
        improved=True; rounds=0
        while improved and rounds<30:
            improved=False; rounds+=1; bd=1e-14; bx=None; cnt=np.bincount(m,minlength=b['NV'])
            for a in range(K):
                for c in range(a+1,K):
                    if m[a]==m[c]: continue
                    x=m.copy(); x[a],x[c]=x[c],x[a]; s2=b['score_num'](S,x,comp); d=s2-s
                    if d>bd: bd=d; bx=x
            if np.any(cnt==2) and np.any(cnt==1):
                for sv in np.flatnonzero(cnt==2):
                    for dv in np.flatnonzero(cnt==1):
                        for i in np.flatnonzero(m==sv):
                            x=m.copy();x[i]=dv;s2=b['score_num'](S,x,comp);d=s2-s
                            if d>bd:bd=d;bx=x
            if bx is not None:
                m=bx;s=b['score_num'](S,m,comp);improved=True
        a=b['map_acc'](H['freq'],m,true)
        print('STRONG_RESTART',rr,'score',s,'acc',a,'rounds',rounds,flush=True)
        if s>best[0]:best=(s,m.copy(),rr)
    return best

ss,strong,strong_rr=strong_optimize(S,comp)

# Surface-level mismatch audit.
mis=[]
for i,sym in enumerate(syms):
    if int(true[i])!=int(fit[i]):
        mis.append({'surface':sym,'index':i,'true_i':int(true[i]),'fit_i':int(fit[i]),'true_value':int(b['VALUES'][int(true[i])]),'fit_value':int(b['VALUES'][int(fit[i])]),'train_freq':int(S['freq'][i]),'hold_freq':int(H['freq'][i])})

# D2 state signatures from frozen induced Arabic model.
lt,ls,le=comp
sig=np.concatenate([lt,lt.T,ls[:,None],le[:,None]],axis=1).astype(float)
mu=sig.mean(0); sd=sig.std(0); sd[sd<1e-12]=1.0; z=(sig-mu)/sd
pairs=[]
for i in range(b['NV']):
    for j in range(i+1,b['NV']):
        dist=float(np.linalg.norm(z[i]-z[j])); cos=float(np.dot(z[i],z[j])/(max(1e-15,np.linalg.norm(z[i])*np.linalg.norm(z[j]))));pairs.append((dist,i,j,cos))
pairs.sort(); prank={(i,j):r+1 for r,(_,i,j,_) in enumerate(pairs)}; pdist={(i,j):(d,c) for d,i,j,c in pairs}
mis_pairs=[]
for r in mis:
    i,j=sorted((r['true_i'],r['fit_i'])); d,c=pdist[(i,j)]; rank=prank[(i,j)]; pct=100*rank/len(pairs)
    mis_pairs.append({**r,'state_pair':[i,j],'distance':d,'cosine':c,'pair_rank':rank,'rank_percent':pct})

autos=[]
for i in range(b['NV']):
    for j in range(i+1,b['NV']):
        p=np.arange(b['NV']);p[i],p[j]=p[j],p[i]
        err=max(float(np.max(np.abs(lt-lt[np.ix_(p,p)]))),float(np.max(np.abs(ls-ls[p]))),float(np.max(np.abs(le-le[p]))))
        if err<=1e-12:autos.append({'pair':[i,j],'values':[int(b['VALUES'][i]),int(b['VALUES'][j])],'maxerr':err})

out={
 'reproduction':{'attempt':attempt,'mapping_acc':acc,'fit_agreement':agr,'true_score':b['score_num'](S,true,comp),'q3_fit_score':b['score_num'](S,fit,comp)},
 'd1':{'strong_score':ss,'strong_restart':strong_rr,'strong_acc':b['map_acc'](H['freq'],strong,true),'strong_vs_q3':ss-b['score_num'](S,fit,comp),'true_vs_q3':b['score_num'](S,true,comp)-b['score_num'](S,fit,comp),'true_vs_strong':b['score_num'](S,true,comp)-ss,'classification':'OPTIMIZER MISS' if (b['map_acc'](H['freq'],strong,true]>=0.999999 or ss-b['score_num'](S,fit,comp)>=1e-5) else 'NO MATERIAL OPTIMIZER MISS'},
 'maps':{'symbols':syms,'true_indices':[int(x) for x in true],'q3_fit_indices':[int(x) for x in fit],'strong_indices':[int(x) for x in strong],'mismatches':mis},
 'd2':{'closest_pairs':[{'rank':r+1,'indices':[i,j],'values':[int(b['VALUES'][i]),int(b['VALUES'][j])],'distance':d,'cosine':c} for r,(d,i,j,c) in enumerate(pairs[:20])],'mismatch_pair_geometry':mis_pairs,'exact_single_swap_automorphisms':autos}
}
print('V18_D1D2_JSON='+json.dumps(out,separators=(',',':')),flush=True)

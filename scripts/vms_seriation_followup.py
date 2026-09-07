#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math
from collections import Counter
from pathlib import Path
import numpy as np

from vms_blind_bifolium_seriation import (
    CORPORA, CODES, FAMILIES, fetch_text, parse_ivtff, unit_nodes,
    build_spaces, js_matrix, mutual_knn, consensus, meta_signature, parse_layout
)

OUT=Path('artifacts/vms_seriation_followup_v01'); OUT.mkdir(parents=True,exist_ok=True)
PROTO='vms_seriation_followup_20260907_v01'
EDGE=('q01_b3_6','q03_b17_24')
Q9='q09_b67_68'; Q10='q10_b69_70'; Q11='q11_b71_72'
Q20_PATH=('q20_b104_115','q20_b106_113','q20_b107_112','q20_b108_111','q20_b103_116')
Q20_INSERT='q20_b105_114'


def load_all():
    bodies={}; data={}; nodes_ref=None
    for code,(url,sha) in CORPORA.items():
        body=fetch_text(url,sha); bodies[code]=body; d=parse_ivtff(body); data[code]=d
        nodes=unit_nodes(d[0])
        if nodes_ref is None: nodes_ref=nodes
        else:
            common=set(x[0] for x in nodes_ref)&set(x[0] for x in nodes)
            nodes_ref=[x for x in nodes_ref if x[0] in common]
    return bodies,data,nodes_ref

def graph_for(corpus_tokens, mode):
    ce={}
    for code,toks0 in corpus_tokens.items():
        if mode=='DROP_M_TOKEN': toks=[[t for t in xs if 'm' not in t] for xs in toks0]
        elif mode=='MASK_M': toks=[[t.replace('m','x') for t in xs] for xs in toks0]
        else: toks=[list(xs) for xs in toks0]
        sp,fx=build_spaces(toks)
        for fam in FAMILIES:
            X=sp[fam]
            if mode=='DROP_M_C3' and fam=='C3':
                keep=[i for i,g in enumerate(fx['c3v']) if 'm' not in g]
                X=X[:,keep] if keep else X
            ce[(code,fam)]=mutual_knn(js_matrix(X),2)
    return ce,consensus(ce)

def edge_support(ce,e):
    ch=[f'{c}:{f}' for (c,f),es in ce.items() if e in es]
    return {'support':len(ch),'channels':sorted(ch),'families':sorted(set(x.split(':')[1] for x in ch)),'codes':sorted(set(x.split(':')[0] for x in ch))}

def heldout_mrare(data,nodes,e):
    # averaged rare Jaccard after excluding every m-containing rare type
    n=len(nodes); mat=np.zeros((n,n))
    for code in CODES:
        ft=data[code][0]; gc=Counter(t for xs in ft.values() for t in xs)
        rare={t for t,c in gc.items() if 2<=c<=5 and 'm' not in t}
        S=[set(t for t in ft[a]+ft[b] if t in rare) for uid,a,b in nodes]
        for i in range(n):
            for j in range(i+1,n):
                z=len(S[i]&S[j])/len(S[i]|S[j]) if S[i]|S[j] else 0.0
                mat[i,j]+=z; mat[j,i]+=z
    mat/=len(CODES)
    fm=data['ZL'][1]; sig=[meta_signature(u,fm) for u in nodes]
    cls=tuple(sorted((repr(sig[e[0]]),repr(sig[e[1]]))))
    alts=[]
    for i in range(n):
        for j in range(i+1,n):
            if (i,j)==e: continue
            if tuple(sorted((repr(sig[i]),repr(sig[j]))))==cls: alts.append(mat[i,j])
    a=np.asarray(alts,float); obs=float(mat[e]); sd=float(a.std(ddof=1)) if len(a)>1 else 0.0
    z=(obs-float(a.mean()))/sd if len(a)>=5 and sd>0 else None
    return {'observed':obs,'n_alt':len(alts),'null_mean':float(a.mean()) if len(a) else None,'null_sd':sd if len(a) else None,'effect':obs-float(a.mean()) if len(a) else None,'effect_over_null_sd':z,'pass':bool(z is not None and z>=2)}

def heldout_mats(bodies,data,nodes):
    n=len(nodes); rare=np.zeros((n,n)); layout=np.zeros((n,n))
    for code in CODES:
        ft=data[code][0]; gc=Counter(t for xs in ft.values() for t in xs); rv={t for t,c in gc.items() if 2<=c<=5}
        S=[set(t for t in ft[a]+ft[b] if t in rv) for uid,a,b in nodes]
        lay=parse_layout(bodies[code]); V=[]
        for uid,a,b in nodes:
            h=np.zeros(21)
            for z in lay.get(a,[])+lay.get(b,[]): h[min(z,21)-1]+=1
            V.append(h)
        LD=js_matrix(np.asarray(V))
        for i in range(n):
            for j in range(i+1,n):
                x=len(S[i]&S[j])/len(S[i]|S[j]) if S[i]|S[j] else 0.0
                rare[i,j]+=x; rare[j,i]+=x
        layout+=LD
    return rare/len(CODES),layout/len(CODES)

def zcmp(obs,vals,higher=True):
    a=np.asarray(vals,float); m=float(a.mean()); sd=float(a.std(ddof=1))
    effect=(obs-m) if higher else (m-obs)
    z=effect/sd if sd else float('nan')
    if higher: p=float((1+np.sum(a>=obs))/(len(a)+1))
    else: p=float((1+np.sum(a<=obs))/(len(a)+1))
    return {'observed':float(obs),'null_mean':m,'null_sd':sd,'favorable_effect':float(effect),'favorable_effect_over_null_sd':float(z),'p_one_sided':p,'n_null':len(a)}

def path_score(path,M,higher):
    vals=[M[path[i],path[i+1]] for i in range(len(path)-1)]
    return float(np.mean(vals))

def unique_paths(ids):
    # reversal-equivalent paths
    out=[]
    for p in itertools.permutations(ids):
        if p[0] < p[-1]: out.append(p)
    return out

def main():
    bodies,data,nodes=load_all(); idx={u[0]:i for i,u in enumerate(nodes)}
    corpus_tokens={code:[data[code][0][a]+data[code][0][b] for uid,a,b in nodes] for code in CODES}

    # F1
    target=tuple(sorted((idx[EDGE[0]],idx[EDGE[1]])))
    ab={}
    for mode in ('DROP_M_TOKEN','MASK_M','DROP_M_C3'):
        ce,con=graph_for(corpus_tokens,mode); s=edge_support(ce,target); s['consensus_candidate']=target in con; ab[mode]=s
    if (ab['DROP_M_TOKEN']['consensus_candidate'] and ab['MASK_M']['consensus_candidate'] and
        (ab['DROP_M_TOKEN']['support']>=8 or ab['MASK_M']['support']>=8)):
        cls='SURVIVES_M_ABLATION'
    elif not ab['DROP_M_TOKEN']['consensus_candidate'] and not ab['MASK_M']['consensus_candidate']:
        cls='M_SUBTYPE_DEPENDENT'
    else: cls='PARTIALLY_M_DEPENDENT'
    mrare=heldout_mrare(data,nodes,target)
    F1={'edge':EDGE,'ablations':ab,'classification':cls,'m_independent_rare':mrare,
        'm_independent_confirmation':bool(cls=='SURVIVES_M_ABLATION' and mrare['pass'])}

    rare,layout=heldout_mats(bodies,data,nodes)
    fm=data['ZL'][1]; sig=[meta_signature(u,fm) for u in nodes]

    # F2 q10 bridge between q9 and q11, held-out only
    a,b,x=idx[Q9],idx[Q11],idx[Q10]
    qsig=sig[x]; alternatives=[i for i in range(len(nodes)) if i not in (a,b,x) and sig[i]==qsig]
    relaxed=False
    if len(alternatives)<8:
        alternatives=[i for i in range(len(nodes)) if i not in (a,b,x)]; relaxed=True
    robs=float((rare[a,x]+rare[x,b])/2); lobs=float((layout[a,x]+layout[x,b])/2)
    rnull=[float((rare[a,i]+rare[i,b])/2) for i in alternatives]
    lnull=[float((layout[a,i]+layout[i,b])/2) for i in alternatives]
    rz=zcmp(robs,rnull,True); lz=zcmp(lobs,lnull,False)
    bridge_pass=bool(rz['favorable_effect']>0 and lz['favorable_effect']>0 and max(rz['favorable_effect_over_null_sd'],lz['favorable_effect_over_null_sd'])>=2)
    F2={'endpoints':[Q9,Q11],'middle':Q10,'metadata_relaxed':relaxed,'n_alternative_middles':len(alternatives),'RARE':rz,'LAYOUT':lz,
        'classification':'HELDOUT_BRIDGE_SUPPORT' if bridge_pass else 'UNRESOLVED'}

    # F3 exact path test
    ids=[idx[u] for u in Q20_PATH]; obs=tuple(ids); paths=unique_paths(ids)
    rscore=path_score(obs,rare,True); lscore=path_score(obs,layout,False)
    rvals=[path_score(p,rare,True) for p in paths]; lvals=[path_score(p,layout,False) for p in paths]
    rr=zcmp(rscore,rvals,True); ll=zcmp(lscore,lvals,False); comb=(rr['favorable_effect_over_null_sd']+ll['favorable_effect_over_null_sd'])/2
    f3pass=bool(rr['favorable_effect']>0 and ll['favorable_effect']>0 and min(rr['p_one_sided'],ll['p_one_sided'])<=.05 and comb>=2)
    # exact favorable ranks: 1 best
    rrank=1+sum(v>rscore for v in rvals); lrank=1+sum(v<lscore for v in lvals)
    F3={'path':Q20_PATH,'reversal_equivalent':True,'n_exact_paths':len(paths),'RARE':rr,'RARE_rank_best1':rrank,'LAYOUT':ll,'LAYOUT_rank_best1':lrank,'combined_mean_favorable_z':comb,
        'classification':'HELDOUT_Q20_PATH_SUPPORT' if f3pass else 'UNRESOLVED'}

    # F4 only if F3 passes
    F4={'run':False,'classification':'NOT_OPENED_BECAUSE_F3_FAILED'}
    if f3pass:
        y=idx[Q20_INSERT]; base=list(ids); opts=[]
        for pos in range(6):
            p=tuple(base[:pos]+[y]+base[pos:]); opts.append((pos,p,path_score(p,rare,True),path_score(p,layout,False)))
        best_r=max(opts,key=lambda z:z[2]); best_l=min(opts,key=lambda z:z[3]); same=best_r[0]==best_l[0]
        # combined standardized favorable score across six insertion options
        rv=np.asarray([z[2] for z in opts]); lv=np.asarray([z[3] for z in opts]); rzv=(rv-rv.mean())/(rv.std(ddof=1) or 1); lzv=(lv.mean()-lv)/(lv.std(ddof=1) or 1); cv=(rzv+lzv)/2
        best=int(np.argmax(cv)); pass4=bool(same and best==best_r[0] and cv[best]>=2)
        F4={'run':True,'insert_unit':Q20_INSERT,'options':[{'position':p,'rare':r,'layout':l,'combined_z':float(cv[k])} for k,(p,_,r,l) in enumerate(opts)],
            'rare_best_position':best_r[0],'layout_best_position':best_l[0],'combined_best_position':best,'combined_best_z':float(cv[best]),
            'classification':'INSERTION_POSITION_SUPPORTED' if pass4 else 'UNRESOLVED'}

    out={'protocol':PROTO,'F1':F1,'F2':F2,'F3':F3,'F4':F4}
    (OUT/'followup.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    lines=['# VMS seriation follow-up v0.1','',
           f"F1 {EDGE[0]} ↔ {EDGE[1]}: **{cls}**; m-independent rare z={mrare['effect_over_null_sd']}",
           f"F2 Q9–Q10–Q11 held-out bridge: **{F2['classification']}**; RARE {rz['favorable_effect_over_null_sd']:.3f} SD, LAYOUT {lz['favorable_effect_over_null_sd']:.3f} SD.",
           f"F3 Q20 path: **{F3['classification']}**; RARE rank {rrank}/60, z={rr['favorable_effect_over_null_sd']:.3f}; LAYOUT rank {lrank}/60, z={ll['favorable_effect_over_null_sd']:.3f}; combined={comb:.3f}.",
           f"F4 insertion: **{F4['classification']}**.",'',
           'No result here licenses chronological direction or a unique original order.']
    (OUT/'FOLLOWUP_CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'F1':cls,'F1_mrare_z':mrare['effect_over_null_sd'],'F2':F2['classification'],'F2_rare_z':rz['favorable_effect_over_null_sd'],'F2_layout_z':lz['favorable_effect_over_null_sd'],'F3':F3['classification'],'F3_rare_rank':rrank,'F3_layout_rank':lrank,'F3_combined_z':comb,'F4':F4['classification']},sort_keys=True),flush=True)

if __name__=='__main__': main()

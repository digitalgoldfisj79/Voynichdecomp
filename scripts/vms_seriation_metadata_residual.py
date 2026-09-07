#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from vms_blind_bifolium_seriation import CORPORA,CODES,FAMILIES,fetch_text,parse_ivtff,unit_nodes,build_spaces,js_matrix,mutual_knn,consensus,meta_signature

OUT=Path('artifacts/vms_seriation_metadata_residual_v01'); OUT.mkdir(parents=True,exist_ok=True)
PAIRS=[('q09_b67_68','q10_b69_70'),('q10_b69_70','q11_b71_72')]

def node_meta(nodes,fm):
    out=[]
    for u in nodes:
        s=meta_signature(u,fm)
        out.append({'L':'/'.join(s[0]) if s[0] else 'NA','H':'/'.join(s[1]) if s[1] else 'NA','I':'/'.join(s[2]) if s[2] else 'NA'})
    return out

def design(meta):
    cats={k:sorted(set(m[k] for m in meta)) for k in ('L','H','I')}
    col=[]
    for k in ('L','H','I'):
        for c in cats[k][1:]: col.append((k,c))
    rows=[]; ij=[]
    n=len(meta)
    for i in range(n):
        for j in range(i+1,n):
            x=[1.0]
            for k,c in col: x.append(float(meta[i][k]==c)+float(meta[j][k]==c))
            x += [float(meta[i]['L']==meta[j]['L']),float(meta[i]['H']==meta[j]['H']),float(meta[i]['I']==meta[j]['I'])]
            rows.append(x); ij.append((i,j))
    return np.asarray(rows,float),ij

def residual_matrix(D,X,ij):
    y=np.asarray([D[i,j] for i,j in ij],float)
    beta=np.linalg.pinv(X)@y; r=y-X@beta
    R=np.zeros_like(D)
    for v,(i,j) in zip(r,ij): R[i,j]=R[j,i]=v
    np.fill_diagonal(R,-1e9) # excluded explicitly by mutual_knn anyway
    return R

def support(ce,e): return sum(e in es for es in ce.values())

def main():
    data={}; nodes_ref=None
    for code,(url,sha) in CORPORA.items():
        d=parse_ivtff(fetch_text(url,sha)); data[code]=d; n=unit_nodes(d[0])
        if nodes_ref is None: nodes_ref=n
        else:
            common=set(x[0] for x in nodes_ref)&set(x[0] for x in n); nodes_ref=[x for x in nodes_ref if x[0] in common]
    nodes=nodes_ref; idx={u[0]:i for i,u in enumerate(nodes)}; meta=node_meta(nodes,data['ZL'][1]); X,ij=design(meta)
    raw={}; resid={}
    for code in CODES:
        toks=[data[code][0][a]+data[code][0][b] for uid,a,b in nodes]; sp,_=build_spaces(toks)
        for fam in FAMILIES:
            D=js_matrix(sp[fam]); raw[(code,fam)]=mutual_knn(D,2); resid[(code,fam)]=mutual_knn(residual_matrix(D,X,ij),2)
    cr=consensus(raw); cx=consensus(resid); rows=[]
    for a,b in PAIRS:
        e=tuple(sorted((idx[a],idx[b])))
        rows.append({'units':[a,b],'raw_support':support(raw,e),'raw_consensus':e in cr,'residual_support':support(resid,e),'residual_consensus':e in cx,
                     'residual_channels':sorted([f'{c}:{f}' for (c,f),es in resid.items() if e in es])})
    n=sum(r['residual_consensus'] for r in rows); cls='SURVIVES_METADATA_RESIDUALISATION' if n==2 else ('PARTIAL' if n==1 else 'METADATA_DEPENDENT')
    out={'protocol':'vms_seriation_metadata_residual_20260907_v01','classification':cls,'design_columns':X.shape[1],'n_pair_rows':X.shape[0],'edges':rows,'metadata':[{'unit':u[0],**m} for u,m in zip(nodes,meta)]}
    (OUT/'metadata_residual.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    (OUT/'CLOSEOUT.md').write_text('# Metadata-residual Q9-Q10-Q11 falsification\n\n'+f'Classification: **{cls}**.\n\n'+ '\n'.join(f"- {r['units'][0]} ↔ {r['units'][1]}: raw {r['raw_support']}/12; residual {r['residual_support']}/12; consensus={r['residual_consensus']}" for r in rows)+'\n')
    print(json.dumps({'classification':cls,'edges':rows},sort_keys=True),flush=True)
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Stage-F text robustness + held-out validation for frozen residual edges.

Implements `vms_edge_forensics_orientation_20260907_v01` F1/F2.
Candidate set and thresholds were committed before this runner.
No physical evidence is read here.
"""
from __future__ import annotations
import json, math
from collections import Counter
from pathlib import Path
import numpy as np

from vms_blind_bifolium_seriation import (
    CORPORA,CODES,FAMILIES,fetch_text,parse_ivtff,unit_nodes,build_spaces,js_matrix,
    mutual_knn,consensus,meta_signature,parse_layout
)
from vms_seriation_metadata_residual import node_meta,design,residual_matrix

PROTOCOL='vms_edge_forensics_orientation_20260907_v01'
OUT=Path('artifacts/vms_edge_forensics_stage_f_text_v01'); OUT.mkdir(parents=True,exist_ok=True)
CANDS=[
 ('E1','q01_b3_6','q03_b17_24'),
 ('E2','q06_b42_47','q01_b1_8'),
 ('E3','q05_b36_37','q03_b19_22'),
 ('E4','q13_b76_83','q13_b77_82'),
 ('E5','q13_b75_84','q13_b78_81'),
]
MIN_ALT=20

def rare_jaccard(a,b): return len(a&b)/len(a|b) if a|b else 0.0

def emp_upper(obs,a):
    a=np.asarray(a,float); return float((1+np.sum(a>=obs))/(len(a)+1))
def emp_lower(obs,a):
    a=np.asarray(a,float); return float((1+np.sum(a<=obs))/(len(a)+1))

def zscore(obs,a,closer='high'):
    a=np.asarray(a,float); m=float(a.mean()); sd=float(a.std(ddof=1)) if len(a)>1 else 0.0
    if sd==0: return {'observed':float(obs),'null_mean':m,'null_sd':0.0,'effect':float(obs-m),'z':None,'p':None}
    if closer=='high': z=(obs-m)/sd; p=emp_upper(obs,a)
    else: z=(m-obs)/sd; p=emp_lower(obs,a)
    return {'observed':float(obs),'null_mean':m,'null_sd':sd,'effect':float(obs-m),'z':float(z),'p':p}

def node_sig(unit,fm,keys):
    # unit metadata values, stable set representation.
    uid,a,b=unit; out=[]
    for k in keys:
        vals=tuple(sorted(set(v for v in (fm.get(a,{}).get(k),fm.get(b,{}).get(k)) if v is not None)))
        out.append(vals)
    return tuple(out)

def pair_class(i,j,sigs): return tuple(sorted((repr(sigs[i]),repr(sigs[j]))))

def choose_null(i,j,n,fm,nodes):
    # Frozen hierarchy A-D; pairs sharing candidate endpoints are excluded.
    levels=[('A_LHI',('L','H','I')),('B_LH',('L','H')),('C_LI',('L','I'))]
    allpairs=[(a,b) for a in range(n) for b in range(a+1,n) if a not in (i,j) and b not in (i,j)]
    for name,keys in levels:
        sigs=[node_sig(u,fm,keys) for u in nodes]; cls=pair_class(i,j,sigs)
        alts=[p for p in allpairs if pair_class(p[0],p[1],sigs)==cls]
        if len(alts)>=MIN_ALT: return name,keys,alts
    return 'D_ALL',(),allpairs

def main():
    bodies={}; data={}; nodes_ref=None
    for code,(url,sha) in CORPORA.items():
        body=fetch_text(url,sha); bodies[code]=body; d=parse_ivtff(body); data[code]=d
        nn=unit_nodes(d[0])
        if nodes_ref is None: nodes_ref=nn
        else:
            common=set(x[0] for x in nodes_ref)&set(x[0] for x in nn)
            nodes_ref=[x for x in nodes_ref if x[0] in common]
    nodes=nodes_ref; n=len(nodes); idx={u[0]:i for i,u in enumerate(nodes)}
    X,ij=design(node_meta(nodes,data['ZL'][1]))

    # Discovery residual channels, already frozen by parent protocol.
    ce={}
    for code in CODES:
        ft=data[code][0]; toks=[ft[a]+ft[b] for uid,a,b in nodes]; sp,_=build_spaces(toks)
        for fam in FAMILIES:
            D=js_matrix(sp[fam]); ce[(code,fam)]=mutual_knn(residual_matrix(D,X,ij),2)
    fullcon=consensus(ce)

    # Heldout matrices averaged across transcriptions.
    rare_mat=np.zeros((n,n)); layout_mat=np.zeros((n,n))
    for code in CODES:
        ft=data[code][0]; gc=Counter(t for toks in ft.values() for t in toks); rare={t for t,c in gc.items() if 2<=c<=5}
        sets=[set(t for t in ft[a]+ft[b] if t in rare) for uid,a,b in nodes]
        lay=parse_layout(bodies[code]); LV=[]
        for uid,a,b in nodes:
            h=np.zeros(21)
            for z in lay.get(a,[])+lay.get(b,[]): h[min(z,21)-1]+=1
            LV.append(h)
        LD=js_matrix(np.asarray(LV))
        for i in range(n):
            for j in range(i+1,n):
                r=rare_jaccard(sets[i],sets[j]); rare_mat[i,j]+=r; rare_mat[j,i]+=r
        layout_mat += LD
    rare_mat/=len(CODES); layout_mat/=len(CODES)

    rows=[]
    for eid,a,b in CANDS:
        e=tuple(sorted((idx[a],idx[b])))
        rawch=sorted([f'{c}:{f}' for (c,f),es in ce.items() if e in es])
        # Delete one family with thresholds unchanged.
        fd={}
        for drop in FAMILIES:
            sub={k:v for k,v in ce.items() if k[1]!=drop}; fd[drop]=bool(e in consensus(sub))
        # Delete one transcription with thresholds unchanged.
        cd={}
        for drop in CODES:
            sub={k:v for k,v in ce.items() if k[0]!=drop}; cd[drop]=bool(e in consensus(sub))
        text_robust=(sum(fd.values())>=2 and sum(cd.values())>=3)

        level,keys,alts=choose_null(e[0],e[1],n,data['ZL'][1],nodes)
        rr=[rare_mat[p] for p in alts]; ll=[layout_mat[p] for p in alts]
        rz=zscore(rare_mat[e],rr,'high'); lz=zscore(layout_mat[e],ll,'low')
        rpass=bool(rz['z'] is not None and rz['z']>=2 and rz['p'] is not None and rz['p']<=.01)
        lpass=bool(lz['z'] is not None and lz['z']>=2 and lz['p'] is not None and lz['p']<=.01)
        rows.append({
          'edge_id':eid,'unit_a':a,'unit_b':b,'residual_support':len(rawch),'residual_channels':rawch,'full_residual_consensus':e in fullcon,
          'family_deletion':fd,'n_family_deletions_pass':sum(fd.values()),
          'transcription_deletion':cd,'n_transcription_deletions_pass':sum(cd.values()),
          'text_robust':text_robust,
          'heldout_null_level':level,'heldout_null_keys':list(keys),'n_heldout_alternatives':len(alts),
          'rare':rz,'layout':lz,'rare_pass':rpass,'layout_pass':lpass,'heldout_pass':bool(rpass or lpass)
        })

    out={'protocol':PROTOCOL,'stage':'F1_F2','candidate_count':len(rows),'minimum_matched_alternatives':MIN_ALT,'edges':rows,
         'rules':{'TEXT_ROBUST':'family deletions >=2/3 AND transcription deletions >=3/4; no threshold retuning',
                  'HELDOUT_PASS':'RARE or LAYOUT: closeness z>=2 AND empirical one-sided p<=.01; first null hierarchy level with >=20 alternatives'}}
    (OUT/'stage_f_text.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    lines=['# Stage F1/F2 closeout','', '## RETRACTIONS / BOUNDS','- These tests can validate a text-state edge, not physical adjacency or direction.','- Physical Stage F remains unopened by this script.','']
    for r in rows:
        lines.append(f"## {r['edge_id']} {r['unit_a']} ↔ {r['unit_b']}")
        lines.append(f"- residual support: {r['residual_support']}/12; TEXT_ROBUST={r['text_robust']} (family deletion {r['n_family_deletions_pass']}/3; transcription deletion {r['n_transcription_deletions_pass']}/4)")
        lines.append(f"- heldout null: {r['heldout_null_level']} n={r['n_heldout_alternatives']}")
        lines.append(f"- RARE: effect {r['rare']['effect']:.6g}, null SD {r['rare']['null_sd']:.6g}, z={r['rare']['z']}, p={r['rare']['p']}; pass={r['rare_pass']}")
        lines.append(f"- LAYOUT: raw effect {r['layout']['effect']:.6g}, null SD {r['layout']['null_sd']:.6g}, closeness z={r['layout']['z']}, p={r['layout']['p']}; pass={r['layout_pass']}")
        lines.append(f"- HELDOUT_PASS={r['heldout_pass']}")
        lines.append('')
    (OUT/'CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'protocol':PROTOCOL,'edges':[{'edge_id':r['edge_id'],'text_robust':r['text_robust'],'heldout_pass':r['heldout_pass'],'rare_z':r['rare']['z'],'layout_z':r['layout']['z']} for r in rows]},sort_keys=True),flush=True)

if __name__=='__main__': main()

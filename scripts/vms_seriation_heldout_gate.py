#!/usr/bin/env python3
"""Held-out validator for blind bifolium seriation v0.1.

FROZEN BEFORE TARGET EDGE INSPECTION under protocol addendum commit
1f032f5c72a1e94d98a3a4610a2418359a064d24.

Recomputes the blind discovery edge set, then opens only reveal-stage metadata
and two feature families that were never used for edge selection:
  RARE   rare-word Jaccard (higher = closer)
  LAYOUT line-length distribution JS distance (lower = closer)

For each candidate, compare the held-out score against all alternative node
pairs in the exact same unordered (L-set,H-set,I-set) metadata-pair class.
A family resolves the edge only when >=5 matched alternatives and non-zero SD.
Pass = RARE z>=2 or LAYOUT closeness z>=2. Present quire/current adjacency are
not used as validation.
"""
from __future__ import annotations
import json, math
from collections import Counter
from pathlib import Path
import numpy as np
from vms_blind_bifolium_seriation import (
    CORPORA, fetch_text, parse_ivtff, unit_nodes, channel_matrices,
    channel_edges_from_mats, consensus, meta_signature, parse_layout,
    js_matrix, CODES
)

OUT=Path('artifacts/vms_seriation_heldout_gate_v01'); OUT.mkdir(parents=True,exist_ok=True)

def rare_jaccard(A,B): return len(A&B)/len(A|B) if A|B else 0.0

def main():
    all_data={}; bodies={}; nodes_ref=None
    for code,(url,sha) in CORPORA.items():
        body=fetch_text(url,sha); bodies[code]=body; data=parse_ivtff(body); all_data[code]=data
        nodes=unit_nodes(data[0])
        if nodes_ref is None: nodes_ref=nodes
        else:
            common=set(x[0] for x in nodes_ref)&set(x[0] for x in nodes)
            nodes_ref=[x for x in nodes_ref if x[0] in common]
    nodes=nodes_ref
    corpora_nodes={code:[all_data[code][0][a]+all_data[code][0][b] for uid,a,b in nodes] for code in CODES}
    mats,spaces,defs=channel_matrices(corpora_nodes); ce=channel_edges_from_mats(mats); cands=consensus(ce)

    # Held-out pair matrices, averaged across transcriptions.
    n=len(nodes); rare_mat=np.zeros((n,n)); layout_mat=np.zeros((n,n))
    for code in CODES:
        ft=all_data[code][0]; gc=Counter(t for toks in ft.values() for t in toks); rare={t for t,c in gc.items() if 2<=c<=5}
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
        layout_mat+=LD
    rare_mat/=len(CODES); layout_mat/=len(CODES)

    fm=all_data['ZL'][1]; sig=[meta_signature(u,fm) for u in nodes]
    pairclass={}
    for i in range(n):
        for j in range(i+1,n): pairclass[(i,j)]=tuple(sorted((repr(sig[i]),repr(sig[j]))))

    rows=[]
    for e,v in sorted(cands.items()):
        i,j=e; cls=pairclass[e]
        alts=[p for p,c in pairclass.items() if p!=e and c==cls]
        rr=np.asarray([rare_mat[p] for p in alts],float); ll=np.asarray([layout_mat[p] for p in alts],float)
        rz=lz=None
        if len(rr)>=5 and rr.std(ddof=1)>0: rz=float((rare_mat[e]-rr.mean())/rr.std(ddof=1))
        if len(ll)>=5 and ll.std(ddof=1)>0: lz=float((ll.mean()-layout_mat[e])/ll.std(ddof=1)) # positive means closer
        rpass=rz is not None and rz>=2; lpass=lz is not None and lz>=2
        rows.append({'edge_index':[i,j],'unit_a':nodes[i][0],'unit_b':nodes[j][0],'support':v['support'],'n_matched_alternatives':len(alts),
                     'rare_jaccard':float(rare_mat[e]),'rare_z':rz,'layout_js':float(layout_mat[e]),'layout_closeness_z':lz,
                     'rare_pass':rpass,'layout_pass':lpass,'heldout_pass':bool(rpass or lpass)})
    out={'protocol':'vms_blind_bifolium_seriation_20260907_v01','addendum_commit':'1f032f5c72a1e94d98a3a4610a2418359a064d24','n_candidates':len(cands),'edges':rows,
         'rule':'heldout_pass iff RARE z>=2 or LAYOUT closeness z>=2 within exact revealed metadata-pair class; >=5 alternatives required'}
    (OUT/'heldout_gate.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    passes=[r for r in rows if r['heldout_pass']]
    lines=['# Blind seriation held-out gate','',f'Candidates: {len(rows)}; held-out passes: **{len(passes)}**.','']
    for r in passes: lines.append(f"- {r['unit_a']} ↔ {r['unit_b']}: support={r['support']}/12; rare_z={r['rare_z']}; layout_z={r['layout_closeness_z']}")
    if not passes: lines.append('No candidate edge passes the frozen independent held-out gate.')
    (OUT/'HELDOUT_CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'n_candidates':len(rows),'n_heldout_pass':len(passes)},sort_keys=True),flush=True)
if __name__=='__main__': main()

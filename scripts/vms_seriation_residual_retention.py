#!/usr/bin/env python3
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from vms_blind_bifolium_seriation import CORPORA,CODES,FAMILIES,fetch_text,parse_ivtff,unit_nodes,build_spaces,js_matrix,mutual_knn,consensus,meta_signature
from vms_seriation_metadata_residual import node_meta,design,residual_matrix

OUT=Path('artifacts/vms_seriation_residual_retention_v01'); OUT.mkdir(parents=True,exist_ok=True)
STABLE=[
('q19_b99_102','q19_b100_101'),('q19_b99_102','q15_b88_89'),('q19_b100_101','q15_b88_89'),
('q20_b106_113','q20_b107_112'),('q20_b106_113','q20_b104_115'),('q20_b107_112','q20_b108_111'),('q20_b103_116','q20_b108_111'),
('q09_b67_68','q10_b69_70'),('q10_b69_70','q11_b71_72'),('q07_b50_55','q05_b33_40'),('q13_b76_83','q13_b77_82'),
('q06_b42_47','q01_b1_8'),('q01_b3_6','q03_b17_24'),('q05_b36_37','q03_b19_22'),('q13_b75_84','q13_b79_80'),('q13_b75_84','q13_b78_81')]

def main():
    data={}; nodes_ref=None
    for code,(url,sha) in CORPORA.items():
        d=parse_ivtff(fetch_text(url,sha)); data[code]=d; n=unit_nodes(d[0])
        if nodes_ref is None:nodes_ref=n
        else:
            c=set(x[0] for x in nodes_ref)&set(x[0] for x in n); nodes_ref=[x for x in nodes_ref if x[0] in c]
    nodes=nodes_ref; idx={u[0]:i for i,u in enumerate(nodes)}; X,ij=design(node_meta(nodes,data['ZL'][1])); ce={}
    for code in CODES:
        toks=[data[code][0][a]+data[code][0][b] for uid,a,b in nodes]; sp,_=build_spaces(toks)
        for fam in FAMILIES:
            D=js_matrix(sp[fam]); ce[(code,fam)]=mutual_knn(residual_matrix(D,X,ij),2)
    con=consensus(ce); rows=[]; retained=set()
    for a,b in STABLE:
        e=tuple(sorted((idx[a],idx[b]))); ch=[(c,f) for (c,f),es in ce.items() if e in es]; fams={f for c,f in ch}; codes={c for c,f in ch}; keep=e in con
        if keep:retained.add(tuple(sorted((a,b))))
        rows.append({'units':[a,b],'support':len(ch),'families':sorted(fams),'codes':sorted(codes),'retained_consensus':keep})
    # retained components
    adj=defaultdict(set)
    for a,b in retained:adj[a].add(b);adj[b].add(a)
    seen=set(); comps=[]
    for x in list(adj):
        if x in seen:continue
        q=[x];seen.add(x);ns=[]
        while q:
            u=q.pop();ns.append(u)
            for v in adj[u]:
                if v not in seen:seen.add(v);q.append(v)
        es=sorted([list(e) for e in retained if e[0] in ns and e[1] in ns]); deg={u:len(adj[u]) for u in ns}; seed=len(ns)>=3 and all(d<=2 for d in deg.values()) and len(es)>=2
        comps.append({'nodes':sorted(ns),'edges':es,'degrees':deg,'ordering_seed_component':seed})
    out={'protocol':'vms_seriation_residual_retention_20260907_v01','n_original_stable':len(STABLE),'n_retained':len(retained),'edges':rows,'components':comps,'ordering_seed_components':[c for c in comps if c['ordering_seed_component']]}
    (OUT/'retention.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    lines=['# Metadata-residual retention of frozen blind edges','',f'Original stable edges: {len(STABLE)}; retained consensus: **{len(retained)}**.','']
    for r in rows:lines.append(f"- {r['units'][0]} ↔ {r['units'][1]}: {r['support']}/12; retained={r['retained_consensus']}")
    lines+=['',f"Ordering-seed components: **{len(out['ordering_seed_components'])}**."]
    for c in out['ordering_seed_components']:lines.append('- '+' — '.join(c['nodes']))
    (OUT/'CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'n_retained':len(retained),'ordering_seed_components':out['ordering_seed_components']},sort_keys=True),flush=True)
if __name__=='__main__':main()

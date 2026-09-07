#!/usr/bin/env python3
"""Blind bifolium neighbourhood seriation v0.1.

Implements protocol vms_blind_bifolium_seriation_20260907_v01.
Target optimisation never sees present folio order, quire position, Currier,
hand, section, or physical-edge evidence. It works on opaque node IDs only.

Output separation:
  blind_graph.json          opaque target candidates + stability/nulls
  calibration.json          known-answer folio matching evaluation
  reveal_validation.json    folio/meta reveal after target candidates frozen
  closeout.md               licensed interpretation
"""
from __future__ import annotations
import hashlib, json, math, os, random, re
from collections import Counter, defaultdict, deque
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist, squareform

from vms_topology_marginalization import (
    CORPORA, UNITS, SEED, fetch_text, parse_ivtff, clean_tokens, LINE_RE, base_folio
)

PROTOCOL='vms_blind_bifolium_seriation_20260907_v01'
OUT=Path('artifacts/vms_blind_bifolium_seriation_v01'); OUT.mkdir(parents=True,exist_ok=True)
SALT='vms-seriation-v01-20260907-frozen'
KNN=2; BOOT=128; N_NULL=5000; CAL_NULL=5000
FAMILIES=('LEX','C3','SHAPE'); CODES=tuple(CORPORA)


def opaque(uid): return hashlib.sha256((SALT+'|'+uid).encode()).hexdigest()[:12]

def js_matrix(X):
    X=np.asarray(X,float)+1e-12
    X/=X.sum(axis=1,keepdims=True)
    return squareform(pdist(X,metric='jensenshannon'))

def mutual_knn(D,k=2):
    n=D.shape[0]; nbr=[]
    for i in range(n):
        order=np.argsort(D[i]); nbr.append(set(int(x) for x in order if x!=i) and set([int(x) for x in order if x!=i][:k]))
    e=set()
    for i in range(n):
        for j in nbr[i]:
            if i in nbr[j]: e.add((min(i,j),max(i,j)))
    return e

def consensus(channel_edges):
    pairs=defaultdict(list)
    for (code,fam),es in channel_edges.items():
        for e in es: pairs[e].append((code,fam))
    out={}
    for e,ch in pairs.items():
        fams={f for c,f in ch}; codes={c for c,f in ch}
        if len(ch)>=6 and len(fams)>=2 and len(codes)>=3:
            out[e]={'support':len(ch),'families':sorted(fams),'codes':sorted(codes),'channels':sorted([f'{c}:{f}' for c,f in ch])}
    return out

def largest_component(edges,n):
    adj=defaultdict(set)
    for a,b in edges: adj[a].add(b); adj[b].add(a)
    seen=set(); best=0
    for x in range(n):
        if x in seen or x not in adj: continue
        q=[x]; seen.add(x); z=0
        while q:
            u=q.pop(); z+=1
            for v in adj[u]:
                if v not in seen: seen.add(v); q.append(v)
        best=max(best,z)
    return best

def unit_nodes(ft):
    rows=[]
    for uid,q,a,b in UNITS:
        if a in ft and b in ft and ft[a] and ft[b]: rows.append((uid,a,b))
    # Deliberately hash-sort: no folio/quire ordering enters optimiser index.
    rows.sort(key=lambda r: opaque(r[0]))
    return rows

def global_vocab(nodes_tokens,min_count=5):
    c=Counter(t for toks in nodes_tokens for t in toks)
    return sorted([t for t,n in c.items() if n>=min_count])

def char3s(tok):
    s='^'+tok+'$'
    return [s[i:i+3] for i in range(max(0,len(s)-2))]

def build_spaces(nodes_tokens, fixed=None):
    if fixed is None:
        lexv=global_vocab(nodes_tokens,5)
        cc=Counter(g for toks in nodes_tokens for t in toks for g in char3s(t))
        c3v=sorted([g for g,n in cc.items() if n>=20])
        chars=sorted(set(''.join(t for toks in nodes_tokens for t in toks)))
        fixed={'lexv':lexv,'c3v':c3v,'chars':chars}
    else:
        lexv=fixed['lexv']; c3v=fixed['c3v']; chars=fixed['chars']
    li={t:i for i,t in enumerate(lexv)}; gi={g:i for i,g in enumerate(c3v)}; ci={c:i for i,c in enumerate(chars)}
    L=[]; C=[]; S=[]
    for toks in nodes_tokens:
        x=np.zeros(len(lexv)); y=np.zeros(len(c3v));
        # shape: len 1..12+, initials, terminals
        z=np.zeros(13+2*len(chars))
        for t in toks:
            if t in li: x[li[t]]+=1
            for g in char3s(t):
                if g in gi: y[gi[g]]+=1
            z[min(len(t),13)-1]+=1
            if t and t[0] in ci: z[13+ci[t[0]]]+=1
            if t and t[-1] in ci: z[13+len(chars)+ci[t[-1]]]+=1
        L.append(x); C.append(y); S.append(z)
    return {'LEX':np.asarray(L),'C3':np.asarray(C),'SHAPE':np.asarray(S)},fixed

def channel_matrices(corpora_nodes):
    mats={}; spaces={}; defs={}
    for code,toks in corpora_nodes.items():
        sp,fx=build_spaces(toks); spaces[code]=sp; defs[code]=fx
        for fam in FAMILIES: mats[(code,fam)]=js_matrix(sp[fam])
    return mats,spaces,defs

def channel_edges_from_mats(mats): return {k:mutual_knn(v,KNN) for k,v in mats.items()}

def accidental_null(channel_edges,n,rng):
    lex={k:e for k,e in channel_edges.items() if k[1]=='LEX'}
    c3={k:e for k,e in channel_edges.items() if k[1]=='C3'}
    shape={k:e for k,e in channel_edges.items() if k[1]=='SHAPE'}
    counts=[]; comps=[]
    for _ in range(N_NULL):
        p3=list(range(n)); ps=list(range(n)); rng.shuffle(p3); rng.shuffle(ps)
        ce=dict(lex)
        for k,es in c3.items(): ce[k]={tuple(sorted((p3[a],p3[b]))) for a,b in es}
        for k,es in shape.items(): ce[k]={tuple(sorted((ps[a],ps[b]))) for a,b in es}
        con=consensus(ce); counts.append(len(con)); comps.append(largest_component(con,n))
    return counts,comps

def null_summary(obs,vals):
    a=np.asarray(vals,float); m=float(a.mean()); sd=float(a.std(ddof=1)); z=(obs-m)/sd if sd else float('nan')
    p=float((1+np.sum(a>=obs))/(len(a)+1))
    return {'observed':obs,'null_mean':m,'null_sd':sd,'effect':obs-m,'effect_over_null_sd':z,'p_upper':p,'n_null':len(vals)}

def bootstrap_candidates(corpora_nodes,defs,cands,rng):
    hit=Counter(); n=len(next(iter(corpora_nodes.values())))
    for r in range(BOOT):
        ce={}
        for code,tokslist in corpora_nodes.items():
            bt=[]
            for toks in tokslist:
                # resample same token count; each corpus/node independently.
                idx=[rng.randrange(len(toks)) for _ in range(len(toks))] if toks else []
                bt.append([toks[i] for i in idx])
            sp,_=build_spaces(bt,defs[code])
            for fam in FAMILIES: ce[(code,fam)]=mutual_knn(js_matrix(sp[fam]),KNN)
        con=consensus(ce)
        for e in cands:
            if e in con: hit[e]+=1
    return {e:hit[e]/BOOT for e in cands}

def strict_calibration(all_data):
    # Use ZL metadata only to define exchangeable strata; labels are used solely
    # to evaluate known physical conjoint partners after blind feature matching.
    ft0,fm0,_,_=all_data['ZL']
    strata=defaultdict(list); truth=set()
    for uid,q,a,b in UNITS:
        if a not in ft0 or b not in ft0 or not ft0[a] or not ft0[b]: continue
        ka=(fm0.get(a,{}).get('L'),fm0.get(a,{}).get('H'),fm0.get(a,{}).get('I'))
        kb=(fm0.get(b,{}).get('L'),fm0.get(b,{}).get('H'),fm0.get(b,{}).get('I'))
        if ka==kb and all(x is not None for x in ka):
            sk=(q,)+ka; strata[sk].extend([a,b]); truth.add(tuple(sorted((a,b))))
    strata={k:sorted(set(v)) for k,v in strata.items() if len(set(v))>=4}
    folios=sorted({f for xs in strata.values() for f in xs}, key=lambda f: hashlib.sha256((SALT+'|cal|'+str(f)).encode()).hexdigest())
    fi={f:i for i,f in enumerate(folios)}
    # Build all feature matrices over the calibration folios, then nearest-neighbour only within own stratum.
    ce={}
    for code,(ft,fm,pt,pm) in all_data.items():
        toks=[ft.get(f,[]) for f in folios]; sp,_=build_spaces(toks)
        full={fam:js_matrix(sp[fam]) for fam in FAMILIES}
        for fam,D in full.items():
            es=set()
            for sk,xs in strata.items():
                ids=[fi[f] for f in xs if f in fi]
                for i in ids:
                    order=sorted((j for j in ids if j!=i), key=lambda j:D[i,j])[:KNN]
                    for j in order:
                        # reciprocal tested below by checking ranks both directions
                        rev=sorted((u for u in ids if u!=j), key=lambda u:D[j,u])[:KNN]
                        if i in rev: es.add(tuple(sorted((i,j))))
            ce[(code,fam)]=es
    con=consensus(ce); pred={tuple(sorted((folios[a],folios[b]))) for a,b in con}
    truth_e={e for e in truth if e[0] in fi and e[1] in fi and any(e[0] in xs and e[1] in xs for xs in strata.values())}
    overlap=len(pred & truth_e); precision=overlap/len(pred) if pred else 0.0; recall=overlap/len(truth_e) if truth_e else 0.0
    # Matched truth-label null: random perfect matchings inside the same strata.
    rng=random.Random(SEED+777); ov=[]
    for _ in range(CAL_NULL):
        fake=set()
        for sk,xs0 in strata.items():
            xs=xs0[:]; rng.shuffle(xs)
            fake|={tuple(sorted((xs[i],xs[i+1]))) for i in range(0,len(xs),2)}
        ov.append(len(pred & fake))
    ns=null_summary(overlap,ov)
    qualified=bool(math.isfinite(ns['effect_over_null_sd']) and ns['effect_over_null_sd']>=2 and overlap>0)
    return {'n_folios':len(folios),'n_strata':len(strata),'n_truth_edges':len(truth_e),'n_predicted_edges':len(pred),'overlap':overlap,'precision':precision,'recall':recall,'overlap_null':ns,'qualified':qualified,
            'criterion':'known-answer overlap >=2 matched-null SD; otherwise serial neighbourhood inference not qualified'}

def parse_layout(body):
    lines=defaultdict(list)
    for raw in body.splitlines():
        m=LINE_RE.match(raw)
        if not m: continue
        page,txt=m.groups(); toks=clean_tokens(txt)
        if toks: lines[base_folio(page)].append(len(toks))
    return lines

def heldout_scores(all_bodies,all_data,nodes,cands):
    # Scores are calculated only after candidate set has frozen.
    rare_by_code={}; layout_by_code={}
    for code,(ft,fm,pt,pm) in all_data.items():
        gc=Counter(t for toks in ft.values() for t in toks); rare={t for t,n in gc.items() if 2<=n<=5}
        rare_sets=[]
        lay=parse_layout(all_bodies[code]); lay_vec=[]
        for uid,a,b in nodes:
            rare_sets.append(set(t for t in ft[a]+ft[b] if t in rare))
            h=np.zeros(21)
            for z in lay.get(a,[])+lay.get(b,[]): h[min(z,21)-1]+=1
            lay_vec.append(h)
        rare_by_code[code]=rare_sets; layout_by_code[code]=js_matrix(np.asarray(lay_vec))
    out={}
    for e in cands:
        a,b=e; rz=[]; ld=[]
        for code in CODES:
            A=rare_by_code[code][a]; B=rare_by_code[code][b]
            rz.append(len(A&B)/len(A|B) if A|B else 0.0)
            ld.append(float(layout_by_code[code][a,b]))
        out[e]={'rare_jaccard_mean':float(np.mean(rz)),'layout_js_mean':float(np.mean(ld))}
    return out

def meta_signature(unit,fm):
    uid,a,b=unit
    sig=[]
    for k in ('L','H','I'):
        vals=tuple(sorted(set(v for v in (fm.get(a,{}).get(k),fm.get(b,{}).get(k)) if v is not None)))
        sig.append(vals)
    return tuple(sig)

def components(edges,n):
    adj=defaultdict(set)
    for a,b in edges: adj[a].add(b); adj[b].add(a)
    seen=set(); comps=[]
    for x in range(n):
        if x in seen or x not in adj: continue
        q=[x]; seen.add(x); nodes=[]
        while q:
            u=q.pop(); nodes.append(u)
            for v in adj[u]:
                if v not in seen: seen.add(v); q.append(v)
        ee=sorted([e for e in edges if e[0] in nodes and e[1] in nodes])
        deg={u:len(adj[u]) for u in nodes}; pathlike=all(d<=2 for d in deg.values())
        comps.append({'nodes':sorted(nodes),'edges':ee,'degrees':deg,'pathlike':pathlike})
    return comps

def main():
    rng=random.Random(SEED+333)
    all_data={}; bodies={}; nodes_ref=None; corpora_nodes={}
    for code,(url,sha) in CORPORA.items():
        body=fetch_text(url,sha); bodies[code]=body; data=parse_ivtff(body); all_data[code]=data
        ft=data[0]; nodes=unit_nodes(ft)
        if nodes_ref is None: nodes_ref=nodes
        else:
            # all target units must map identically; drop only if a corpus truly lacks text.
            ids0=[x[0] for x in nodes_ref]; ids=[x[0] for x in nodes]
            common=set(ids0)&set(ids); nodes_ref=[x for x in nodes_ref if x[0] in common]
    # Rebuild aligned opaque target token arrays on final common unit set.
    nodes=nodes_ref; blind=[opaque(u[0]) for u in nodes]
    for code,(ft,fm,pt,pm) in all_data.items(): corpora_nodes[code]=[ft[a]+ft[b] for uid,a,b in nodes]

    # Known-answer calibration is evaluated before target interpretation.
    cal=strict_calibration(all_data)
    (OUT/'calibration.json').write_text(json.dumps(cal,indent=2,sort_keys=True))

    mats,spaces,defs=channel_matrices(corpora_nodes); ce=channel_edges_from_mats(mats); con=consensus(ce)
    # Freeze opaque candidate set before any reveal-stage metadata use.
    boot=bootstrap_candidates(corpora_nodes,defs,con,rng) if con else {}
    stable={e:v for e,v in con.items() if boot.get(e,0)>=0.60}
    null_counts,null_comps=accidental_null(ce,len(nodes),rng)
    blind_edges=[]
    for e,v in sorted(con.items()):
        blind_edges.append({'a':blind[e[0]],'b':blind[e[1]],'support':v['support'],'families':v['families'],'codes':v['codes'],'channels':v['channels'],'bootstrap_consensus':boot.get(e,0.0),'stable':boot.get(e,0.0)>=0.60})
    blind_out={'protocol':PROTOCOL,'n_nodes':len(nodes),'node_ids':blind,'candidate_edges':blind_edges,'n_candidates':len(con),'n_stable':len(stable),
               'candidate_count_null':null_summary(len(con),null_counts),'largest_component_null':null_summary(largest_component(con,len(nodes)),null_comps),
               'known_answer_calibration_qualified':cal['qualified'],'target_interpretation_licensed':bool(cal['qualified'] and null_summary(len(con),null_counts)['effect_over_null_sd']>=2),
               'leakage_statement':'target graph computed only from opaque IDs + LEX/C3/SHAPE; present order/quire position/L/H/I/physical evidence unopened'}
    (OUT/'blind_graph.json').write_text(json.dumps(blind_out,indent=2,sort_keys=True))

    # ---- REVEAL STAGE starts only after blind_graph is materialised ----
    fm=all_data['ZL'][1]; held=heldout_scores(bodies,all_data,nodes,con)
    sig=[meta_signature(u,fm) for u in nodes]
    revealed=[]
    for e,v in sorted(con.items()):
        a,b=e; ua,ub=nodes[a],nodes[b]
        # Candidate support versus alternative pairs with same revealed metadata-pair class.
        cls=tuple(sorted((repr(sig[a]),repr(sig[b])))); alt=[]
        for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                if (i,j)==e: continue
                if tuple(sorted((repr(sig[i]),repr(sig[j]))))==cls:
                    alt.append(len([1 for es in ce.values() if (i,j) in es]))
        az=None
        if len(alt)>=5 and np.std(alt,ddof=1)>0: az=(v['support']-float(np.mean(alt)))/float(np.std(alt,ddof=1))
        # Present adjacency only revealed here; arithmetic is validation, never optimisation.
        fols_a=sorted((ua[1],ua[2])); fols_b=sorted((ub[1],ub[2]))
        present_adj=min(abs(x-y) for x in fols_a for y in fols_b)==1
        same_q=next(q for uid,q,x,y in UNITS if uid==ua[0])==next(q for uid,q,x,y in UNITS if uid==ub[0])
        h=held[e]
        promoted=bool(blind_out['target_interpretation_licensed'] and boot.get(e,0)>=0.60 and az is not None and az>=2)
        revealed.append({'opaque_a':blind[a],'opaque_b':blind[b],'unit_a':ua[0],'unit_b':ub[0],'folios_a':fols_a,'folios_b':fols_b,
                         'support':v['support'],'bootstrap_consensus':boot.get(e,0.0),'meta_a':{'L':sig[a][0],'H':sig[a][1],'I':sig[a][2]},'meta_b':{'L':sig[b][0],'H':sig[b][1],'I':sig[b][2]},
                         'same_current_quire':same_q,'present_folio_adjacency_exists':present_adj,'matched_meta_support_z':az,'matched_meta_alt_n':len(alt),
                         'rare_jaccard_mean_heldout':h['rare_jaccard_mean'],'layout_js_mean_heldout':h['layout_js_mean'],'candidate_production_neighbour':promoted})
    stable_edges=set(e for e in stable)
    comp=components(stable_edges,len(nodes))
    comp_reveal=[]
    for c in comp:
        comp_reveal.append({'units':[nodes[i][0] for i in c['nodes']], 'folios':[[nodes[i][1],nodes[i][2]] for i in c['nodes']],
                            'edges':[[nodes[a][0],nodes[b][0]] for a,b in c['edges']], 'pathlike':c['pathlike'], 'degrees':{nodes[i][0]:d for i,d in c['degrees'].items()}})
    reveal={'protocol':PROTOCOL,'calibration':cal,'blind_candidate_count':len(con),'blind_stable_count':len(stable),'target_interpretation_licensed':blind_out['target_interpretation_licensed'],
            'edges':revealed,'stable_components':comp_reveal,
            'warning':'Components are undirected production-neighbour hypotheses, not original order. Path reversal is equivalent; no direction is licensed from text.'}
    (OUT/'reveal_validation.json').write_text(json.dumps(reveal,indent=2,sort_keys=True,default=list))

    promoted=[r for r in revealed if r['candidate_production_neighbour']]
    lines=['# Blind bifolium seriation v0.1 closeout','',f"Known-answer calibration qualified: **{cal['qualified']}** (overlap z={cal['overlap_null']['effect_over_null_sd']:.3f}, precision={cal['precision']:.3f}, recall={cal['recall']:.3f}).",
           f"Blind consensus candidates: **{len(con)}**; stable >=0.60: **{len(stable)}**.",
           f"Candidate-count null: mean {blind_out['candidate_count_null']['null_mean']:.3f} ± {blind_out['candidate_count_null']['null_sd']:.3f}; effect/SD **{blind_out['candidate_count_null']['effect_over_null_sd']:.3f}**; p={blind_out['candidate_count_null']['p_upper']:.6f}.",
           f"Target neighbourhood interpretation licensed: **{blind_out['target_interpretation_licensed']}**.",
           f"Edges surviving stability + revealed metadata matched-support z>=2: **{len(promoted)}**.",'']
    if promoted:
        lines.append('## Promoted candidate production-neighbour edges')
        for r in promoted:
            lines.append(f"- {r['unit_a']} ↔ {r['unit_b']}: support={r['support']}/12; bootstrap={r['bootstrap_consensus']:.3f}; matched-meta z={r['matched_meta_support_z']:.3f}; current-quire={r['same_current_quire']}; current-adj={r['present_folio_adjacency_exists']}")
    lines += ['', 'No text-derived edge is an original-order claim. Direction requires independent physical evidence.']
    (OUT/'CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'calibration':cal,'n_candidates':len(con),'n_stable':len(stable),'n_promoted':len(promoted),'candidate_count_z':blind_out['candidate_count_null']['effect_over_null_sd'],'licensed':blind_out['target_interpretation_licensed']},sort_keys=True),flush=True)

if __name__=='__main__': main()

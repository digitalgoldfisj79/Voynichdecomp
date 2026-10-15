#!/usr/bin/env python3
"""Read-only v138 architecture feasibility analysis. No fresh VMS model fit.
Usage: python preflight.py /mnt/data/Generator_v138_Pruefpaket_2026-10-14.zip
Each stage saves pickle atomically. No modification of the supplied archive.
"""
import collections as C, hashlib, importlib.util, json, os, pickle, shutil, sys, tempfile, zipfile
from pathlib import Path

BASE=Path(__file__).resolve().parent
P=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path('/mnt/data/Generator_v138_Pruefpaket_2026-10-14.zip')
CONTRACT=json.loads((BASE/'admission_contract.json').read_text())
def save(name,obj):
    dest=BASE/name; temp=dest.with_suffix(dest.suffix+'.tmp')
    with open(temp,'wb') as f: pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL);f.flush();os.fsync(f.fileno())
    os.replace(temp,dest)
def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Stage 0: supplied source provenance, preserve audit artefacts.
if not P.exists(): raise SystemExit(f'NO-GO: source package missing {P}')
root=BASE/'source'; root.mkdir(exist_ok=True)
with zipfile.ZipFile(P) as z:
    z.extractall(root)
source=root/'gen138'
state0={'package_sha256':hashfile(P),'contract_sha256':hashfile(BASE/'admission_contract.json'),
        'files':{str(x.relative_to(source)):hashfile(x) for x in sorted(source.rglob('*')) if x.is_file()},
        'complete_source_files':len([x for x in source.rglob('*') if x.is_file()])}
save('stage0_provenance.pkl',state0)

# Original program, in quarantined copy: imports fitted tables and reruns only original supplied output.
sys.path.insert(0,str(source));cwd=os.getcwd();os.chdir(source)
try:
    spec=importlib.util.spec_from_file_location('preflight_v138',source/'vbm_tafel_v138.py')
    gen=importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
finally:os.chdir(cwd)
if hashfile(source/'GENERAT_v138_kurzertext.txt')!=hashfile(source/'GENERAT_v138_kurzertext_REFERENZ.txt'):
    raise SystemExit('NO-GO: reference output has changed')

ev=gen.ev; RANK=gen.RANK; CELLS=gen.CELLS; pairs=gen.pair_v
assert len(CELLS)==len(set(CELLS)), 'Cell inventory duplicated!'
clusters=list(RANK)
# Support for key on the next cell, conditional on vowel between cells.
support={v:{p[1] for p,w in pairs.items() if w==v} for v in 'aeiou'}
# Use the actual chosen main cluster, accounting for long word joins.
def eff(e):
    v,K,v2,fl,aus,an=e
    return aus if (aus and an and len(aus)+len(an)>=3) else K

ctx=C.defaultdict(set);counts=C.Counter(); prior_events=0; empty=0
for i,e in enumerate(ev):
    k=eff(e)
    if not k: empty+=1; continue
    counts[k]+=1
    # A given cell is reached by the bridge from the preceding event.
    # The very first event carries a fixed initial key; there is no previous bridge.
    if i>0 and k in RANK: ctx[k].add(e[0])

# Stage 1a: true inventory and the actual rank collisions
inverse=C.defaultdict(list)
for c,cell in RANK.items():inverse[cell].append(c)
coll={cell:members for cell,members in inverse.items() if len(members)>1}
# Stage 1b: frozen mapping transition incompatibility, exact test
known=0; unsupported=0; bad=[]; counters=C.Counter()
for i,e in enumerate(ev[:-1]):
    current=eff(e); nextk=eff(ev[i+1]);v2=e[2]
    if not current or not nextk or nextk not in RANK:continue
    known+=1
    key=RANK[nextk][0]
    if key not in support[v2]:
        unsupported+=1
        counters[(v2,key)]+=1
        if len(bad)<16: bad.append({'event_index':i,'vowel':v2,'next_cluster':nextk,'next_key':key})
# Compare with previous audit definition. Note that this test's known includes the same predicate.
stage1={'events':len(ev),'empty_cluster_events':empty,'nonempty_cluster_events':sum(counts.values()),
       'unique_clusters':len(RANK),'distinct_cells':len(CELLS),'rank_cells_used':len(inverse),
       'duplicate_assigned_cells':len(coll), 'cluster_types_in_duplicate_cells':sum(map(len,coll.values())),
       'forced_collisions_lower_bound':max(0,len(RANK)-len(CELLS)),
       'known_next_events':known, 'unsupported_events':unsupported,
       'unsupported_by_vowel_and_key':{str(k):v for k,v in counters.items()},
       'supported_keys_by_vowel':{v:sorted(s) for v,s in support.items()},
       'first_failures':bad,'original_output_sha256':hashfile(source/'GENERAT_v138_kurzertext.txt')}
save('stage1_baseline.pkl',stage1)

# Stage 2: optimal injective matching, preserving every cluster's observed incoming vowel support.
# Since every cell with a given key has identical eligibility, optimal matching can be found
# by a min-cost-flow from clusters to 4 key-family capacities.
# Source->cluster cost=-occurrences, cluster->key capacity=1, key->sink capacity=#cells(key).
import networkx as nx
cellkeys=C.Counter(c[0] for c in CELLS)
G=nx.DiGraph();N=len(RANK)
G.add_node('S',demand=-N);G.add_node('T',demand=N)
for k in sorted(RANK):
    ck=('c',k);G.add_edge('S',ck,capacity=1,weight=0)
    # Unmapped clusters use direct discard link (explicit rejected cases, not an escape code).
    G.add_edge(ck,'T',capacity=1,weight=0)
    allowed=set.intersection(*(support[v] for v in ctx[k])) if ctx[k] else {'y'}
    for key in sorted(allowed):
        if cellkeys[key]:G.add_edge(ck,('key',key),capacity=1,weight=-counts[k])
for key,n in cellkeys.items():G.add_edge(('key',key),'T',capacity=n,weight=0)
fl=nx.min_cost_flow(G)
selected={}
for k in RANK:
    chosen=[key for key in sorted(cellkeys) if fl[('c',k)].get(('key',key),0)==1]
    if chosen:selected[k]=chosen[0]
# Allocate actual cells deterministically, once; inverse table is unique.
bykey=C.defaultdict(list)
for cell in CELLS:bykey[cell[0]].append(cell)
assign={}
for k in sorted(selected, key=lambda k:(-counts[k], k)):
    key=selected[k];assign[k]=bykey[key].pop(0)
assert len(set(assign.values()))==len(assign)
assert all(assign[k][0] in (set.intersection(*(support[v] for v in ctx[k])) if ctx[k] else {'y'}) for k in assign)
covered=sum(counts[k] for k in assign)
# Perfect forward+inverse uniqueness for selected symbolic clusters (not a Voynich cipher).
inv={tuple(c):k for k,c in assign.items()}
assert all(inv[tuple(assign[k])]==k for k in assign)

# Control: collision with same key family should trigger our invariant, without changing vowel compatibility.
byfamily=C.defaultdict(list)
for k,cell in assign.items():byfamily[cell[0]].append(k)
chosenfam=next((key for key,ks in sorted(byfamily.items()) if len(ks)>=2),None)
assert chosenfam
k1,k2=byfamily[chosenfam][:2]
badassign=dict(assign);badassign[k2]=badassign[k1]
assert len(set(badassign.values()))!=len(badassign)
assert all(badassign[k][0] in (set.intersection(*(support[v] for v in ctx[k])) if ctx[k] else {'y'}) for k in badassign)

stage2={'max_covered_distinct_cluster_types':len(assign),'unmapped_types':len(RANK)-len(assign),
       'max_covered_source_events':covered,'total_mapped_source_events':sum(counts.values()),
       'left_unmapped_source_events':sum(counts.values())-covered,
       'assigned_by_key':dict(C.Counter(c[0] for c in assign.values())),
       'cell_capacity_by_key':dict(cellkeys),'positive_control_unique_inverse_pass':True,
       'negative_control_same_key_collision_rejected':True,
       'negative_control_test_clusters':[k1,k2],
       'compatibility_test_scope':'incoming vowels for each source cluster, not full emitted-token seam grammar',
       'disambiguation':'No available keys or plaintext are inferred from the Voynich text',
       'assignment':{k:list(cell) for k,cell in assign.items()}}
save('stage2_optimized_bound.pkl',stage2)

# Independent invariants and frozen decision. The controls do not relax gates.
checks={'enough_distinct_cells':len(CELLS)>=len(RANK),
        'rank_map_injective':len(coll)==0,
        'all_known_bridge_transitions_supported':unsupported==0,
        'maximum_matching_covers_every_type':len(assign)==len(RANK)}
result={'status':'GO' if all(checks.values()) else 'NO-GO',
        'scope':'unmodified 443-cell, one-cell-per-cluster v138 mechanism',
        'checks':checks,'trace':['stage0_provenance.pkl','stage1_baseline.pkl','stage2_optimized_bound.pkl'],
        'reason':'Existing table inventory cannot satisfy its declared injectivity and key compatibility constraints. No encoder/decoder/held-out Voynich run admitted.' if not all(checks.values()) else 'Meets only necessary gates, decoder qualification still required.'}
save('stage3_admission.pkl',result)
(BASE/'results.json').write_text(json.dumps({'provenance':state0,'baseline':stage1,
    'optimized_bound':{k:v for k,v in stage2.items() if k!='assignment'},'admission':result},indent=2))
print(json.dumps({'baseline':stage1,'optimized_bound':{k:v for k,v in stage2.items() if k!='assignment'},'admission':result},indent=2))

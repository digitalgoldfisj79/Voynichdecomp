from pathlib import Path
from collections import Counter,defaultdict
import sys,os,runpy,pickle,hashlib,json,math
root=Path('/mnt/data/vbm138_review/gen138');os.chdir(root);sys.path.insert(0,str(root))
d=runpy.run_path('vbm_tafel_v138.py', run_name='audit_not_main')
A=d['_at2']; atom=d['_atoms']; ev=d['ev'];rank=d['RANK']; cm=d['_cu']; c=d['CLU']; cells=d['CELLS'];pair=d['pair_v']; R=d['R']; pool=d['POOL']; poolz=d['POOLZ']; an=d['ANr'];
# Pool forward collisions for Anlaut for each state-head and fallback
junct_events=[(aus,an) for v,K,v2,f,aus,an in ev if aus and an and len(aus)+len(an)>=3]
base_events=[(K if not (aus and an and len(aus)+len(an)>=3) else aus) for v,K,v2,f,aus,an in ev if K]
inv=defaultdict(list)
for K,cell in rank.items():inv[cell].append(K)
coll_keys={k for keys in inv.values() if len(keys)>1 for k in keys}
coll_mass=sum(cm[k] for k in coll_keys)
# unique clusters by VMS cell
stats={
 'n_input_words':len(d['words']),
 'n_charstream':len(d['s']),
 'n_events':len(ev),
 'vowel_counts':dict(Counter(v2 for v,K,v2,f,aus,an in ev)),
 'n_cluster_types':len(cm),
 'n_rank_types':len(rank),
 'n_output_cells':len(set(rank.values())),
 'n_collision_cells':sum(len(x)>1 for x in inv.values()),
 'n_collision_types':len(coll_keys),
 'cluster_base_events':len(base_events),
 'rank_collision_events':sum(1 for k in base_events if k in coll_keys),
 'n_zero_cluster_events':sum(not K for v,K,v2,f,aus,an in ev),
 'word_boundaries_encoded_as_junct':len(junct_events),
 'word_boundaries_total':len(d['bl'])-1,
 'naht_changes':d['NAHT'],
 'loss_counter':d['VERLUST'][0],
 'homophone_vowels':len(pair),
 'rare_cell_collisions_example':[(cell,keys[:7],sum(cm[k] for k in keys)) for cell,keys in sorted(inv.items(),key=lambda p:-sum(cm[k] for k in p[1])) if len(keys)>1][:8],
}
# Check generated trigram/bigram against VMS reference for every token
reference=root/'vms_B_REVA_atomstream.txt';generated=root/'GENERAT_v138_kurzertext.txt'
BI,TRI=set(),set()
for l in reference.read_text().splitlines():
 for tok in l.split('\t')[-1].split():
  a=A(tok)
  BI.update(zip(a,a[1:]));TRI.update(zip(a,a[1:],a[2:]))
allgen=[t for line in generated.read_text().splitlines() for t in line.split()]
viol2=[];viol3=[];missing2=Counter();missing3=Counter()
for tok in allgen:
 a=A(tok)
 b=[p for p in zip(a,a[1:]) if p not in BI]
 t=[p for p in zip(a,a[1:],a[2:]) if p not in TRI]
 if b:viol2.append(tok);missing2.update(b)
 if t:viol3.append(tok);missing3.update(t)
stats['generated_tokens']=len(allgen);stats['bigram_violating_tokens']=len(viol2);stats['trigram_violating_tokens']=len(viol3)
stats['bigram_violation_examples']=missing2.most_common(12);stats['trigram_violation_examples']=missing3.most_common(12)
stats['ngram_ex_bad_tokens']=Counter(viol2+viol3).most_common(10)
# key fit audit: output actual first atom (not including headings) at each expected event, measured by injecting event tracing separately
# naive check global prefix endings
stats['reference_sha256']=hashlib.sha256(reference.read_bytes()).hexdigest()
stats['generated_sha256']=hashlib.sha256(generated.read_bytes()).hexdigest()
# verify each map has injective source -> prefix for given (state,head), same as function
forward=defaultdict(lambda:defaultdict(list))
for head in pool:
 for st in {''}|{state for (state,h) in poolz if h==head}:
  lst=poolz.get((st,head)) or pool.get(head) or d['ALLP']
  for anlaut in an:
   j=an.index(anlaut)
   output=lst[j] if j<len(lst) else lst[(an.index(anlaut[0]) if anlaut[0] in an else 0)%len(lst)]
   forward[(st,head)][output].append(anlaut)
max_coll=max([len(ks) for entries in forward.values() for ks in entries.values()] or [0])
stats['prefix_contexts_with_collision']=sum(any(len(ks)>1 for ks in entries.values()) for entries in forward.values())
stats['prefix_contexts']=len(forward);stats['max_prefix_collapse']=max_coll
stats['prefix_collision_examples']=[(c,vs[:5]) for c,entries in forward.items() for form,vs in entries.items() if len(vs)>1][:6]
# for standalone pair no multiple vowels by construction; modified br can remove or be overridden
out=Path('/mnt/data/vbm138_review/stage1.pkl'); tmp=out.with_suffix('.pkl.tmp')
with tmp.open('wb') as fp:pickle.dump(stats,fp)
tmp.replace(out)
print(json.dumps(stats,ensure_ascii=False,indent=2,default=str))

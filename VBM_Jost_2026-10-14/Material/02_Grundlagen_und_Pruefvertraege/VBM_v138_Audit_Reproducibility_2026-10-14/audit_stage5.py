from pathlib import Path
import os,sys,runpy,pickle,json
root=Path('/mnt/data/vbm138_review/gen138'); os.chdir(root);sys.path.insert(0,str(root));os.environ['HOMK']='0.1'
a=runpy.run_path('vbm_tafel_v138.py',run_name='audit');b=runpy.run_path('vbm_tafel_v140.py',run_name='audit')
common=set(a['RANK'])&set(b['RANK']); identical=sum(a['RANK'][k]==b['RANK'][k] for k in common)
counts={'common_consonant_cluster_types':len(common),'same_cell_assignments':identical,'different_cell_assignments':len(common)-identical,'pct_changed':round(100*(len(common)-identical)/len(common),2) if common else None,
 'examples_changed':[(k,a['RANK'][k],b['RANK'][k]) for k in sorted(common) if a['RANK'][k]!=b['RANK'][k]][:10], 'description':'RANK is built afresh from the full source plaintext in each version. Comparing rank between bairisch_kurzertext and Hartlieb does not isolate text effects from other v140 model changes.'}
out=Path('/mnt/data/vbm138_review/stage5.pkl')
with out.with_suffix('.pkl.tmp').open('wb') as fp:pickle.dump(counts,fp)
out.with_suffix('.pkl.tmp').replace(out)
print(json.dumps(counts,indent=2))

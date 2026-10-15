from pathlib import Path
import os,sys,pickle,random,collections,time,json
import numpy as np
root=Path('/mnt/data/vbm138_review/gen138');os.chdir(root)
ns={}
source=Path('Paket_v100/mess_v118.py').read_text().split('ref=load(sys.argv[1])')[0]
exec(compile(source,'mess_functions','exec'),ns)
load=ns['load'];blocks=ns['blocks'];h2=ns['h2'];auc_h2=ns['auc_h2']
ref=load('vms_B_REVA_atomstream.txt')
synth=load('GENERAT_v138_kurzertext.txt')
A=np.array([h2(b) for b in blocks(ref,120)])
G=np.array([h2(b) for b in blocks(synth,120)])
rg=np.random.default_rng(42);sel=np.sort(rg.choice(len(G),len(A),replace=False));B=G[sel]
# matched null: every line retains EXACT VMS tokens, token frequencies, lengths, and length in tokens; order scrambled
metrics=[]
for seed in range(30):
 rng=random.Random(10000+seed)
 shuffled=[]
 for l in ref:
  l=l[:];rng.shuffle(l);shuffled.append(l)
 H=np.array([h2(b) for b in blocks(shuffled,120)])
 metrics.append((float(H.mean()),float(H.mean()-A.mean()),float(np.mean(np.abs(H-A)))))
H_mean=np.array([r[0] for r in metrics]);null_std=float(H_mean.std(ddof=1));target=float(G.mean()-A.mean())
# 1 shuffle control classifier; AUC vs synthetic benchmark
rng=random.Random(10000);ref_shuffled=[]
for l in ref:
 l=l[:];rng.shuffle(l);ref_shuffled.append(l)
H=np.array([h2(b) for b in blocks(ref_shuffled,120)])
# both real/negative same 180 blocks, want grouped fold - use same fake original group grouping
auc_fake=auc_h2(A,H,np.arange(len(A))//5,np.arange(len(H))//5)
auc_original=auc_h2(A,B,np.arange(len(A))//5,sel//5)
# pre-specified matched null: existing -5 within-line bootstrap offsets no multiple manipulations
stats={'n_blocks_ref':len(A),'n_blocks_gen':len(G),'ref_h2_mean':float(A.mean()),'gen_h2_mean':float(G.mean()),'difference_bits':target,
'null_shuffle_mean_h2':float(H_mean.mean()),'null_shuffle_sd_h2':null_std,
'null_shuffle_delta_mean':float(H_mean.mean()-A.mean()),
'null_shuffle_ratio_synthetic_gap_vs_null_std':float(abs(target)/null_std) if null_std else None,
'auc_original':float(auc_original),'auc_shuffled_ref':float(auc_fake),'h2_null_30_samples':metrics[:30],
'null_sample_note':'30 independently seeded within-line permutations; reference original fixed; h2 computed on 120-token blocks'}
out=Path('/mnt/data/vbm138_review/stage3.pkl')
with out.with_suffix('.pkl.tmp').open('wb') as f:pickle.dump(stats,f)
out.with_suffix('.pkl.tmp').replace(out)
print(json.dumps({k:v for k,v in stats.items() if k!='h2_null_30_samples'},indent=2))

from pathlib import Path
import os,sys,pickle,random,math,json
import numpy as np
root=Path('/mnt/data/vbm138_review/gen138');os.chdir(root)
ns={}
exec(compile(Path('Paket_v100/mess_v118.py').read_text().split('ref=load(sys.argv[1])')[0], 'h2_tools', 'exec'),ns)
load=ns['load']; blocks=ns['blocks'];h2=ns['h2'];auc=ns['auc_h2']; atom=ns['atom']
ref=load('vms_B_REVA_atomstream.txt'); gen=load('GENERAT_v138_kurzertext.txt')
A=np.array([h2(b) for b in blocks(ref,120)]); G=np.array([h2(b) for b in blocks(gen,120)])
sel=np.sort(np.random.default_rng(42).choice(len(G),len(A),replace=False)); B=G[sel]; ga=np.arange(len(A))//5;gb=sel//5
aucs=[]; hh=[]
for seed in range(16):
 rng=random.Random(400+seed)
 lines=[]
 for l in ref:
  w=l[1:-1].copy();rng.shuffle(w)
  lines.append([l[0]]+w+[l[-1]] if len(l)>1 else list(l))
 C=np.array([h2(b) for b in blocks(lines,120)])
 aucs.append(auc(A,C,ga,ga))
 hh.append(float(C.mean()-A.mean()))
# planted positive shuffle characters within tokens while preserving each token's atomic character inventory and lengths
rng=random.Random(633)
allsh=[]
for l in ref:
 ll=[]
 for t in l:
  a=atom.findall(t)
  rng.shuffle(a)
  ll.append(''.join(a))
 allsh.append(ll)
P=np.array([h2(b) for b in blocks(allsh,120)])
auc_positive=auc(A,P,ga,ga)
res={'observed_auc_generator':auc(A,B,ga,gb),'matched_negative_auc_mean':float(np.mean(aucs)),'matched_negative_auc_sd':float(np.std(aucs,ddof=1)),'matched_negative_auc_minmax':[min(aucs),max(aucs)],'matched_negative_aucs':aucs,'matched_negative_h2_delta_bits_mean':float(np.mean(hh)),'matched_negative_h2_delta_sd':float(np.std(hh,ddof=1)),'positive_control_auc_atomic_shuffle':float(auc_positive),'positive_control_h2_delta_bits':float(P.mean()-A.mean()),'note':'negative: 16 independently shuffled interior-token orders inside each VMS line; first and last token preserved. positive: atomic units shuffled within every token, preserving token atom multisets.'}
out=Path('/mnt/data/vbm138_review/stage4.pkl');tmp=out.with_suffix('.pkl.tmp')
with tmp.open('wb') as f:pickle.dump(res,f)
tmp.replace(out)
print(json.dumps(res,indent=2))

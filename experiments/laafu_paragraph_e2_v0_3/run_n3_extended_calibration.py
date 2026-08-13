#!/usr/bin/env python3
import importlib.util, json, os, sys, math
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
P3=os.path.join(HERE,'run_laafu_paragraph_e2.py')
spec=importlib.util.spec_from_file_location('v03',P3); v03=importlib.util.module_from_spec(spec); spec.loader.exec_module(v03)
v02=v03.v02
BASE_SEED=20261813
path=sys.argv[1] if len(sys.argv)>1 else v03.DATA_DEFAULT
out=sys.argv[2] if len(sys.argv)>2 else '/mnt/data/joint_lag/v03/N3_EXTENDED_CALIBRATION_20260813.json'
lines=v03.load_frame(path,'ZLZI','P0')
ratios=[]; zs=[]; reps=[]
for r in range(100):
    seed=BASE_SEED+1009*r
    pseudo=v02.permute_once(lines,'N1',seed)
    rr=v02.analyse(pseudo,'N1',500,seed+1,detail=True)['interior']
    ratios.append(rr['ratio']); zs.append(rr['z']); reps.append(rr)
    if (r+1)%10==0: print('rep',r+1,flush=True)
mean_ratio=float(np.mean(ratios)); mean_z=float(np.mean(zs)); sd_z=float(np.std(zs,ddof=1)); n2=sum(abs(z)>=2 for z in zs if math.isfinite(z))
criteria={'mean_ratio':0.97<=mean_ratio<=1.03,'mean_z':-0.25<=mean_z<=0.25,'sd_z':0.80<=sd_z<=1.20,'tail':n2<=8}
res={'n':100,'null_per_rep':500,'base_seed':BASE_SEED,'mean_ratio':mean_ratio,'mean_z':mean_z,'sd_z':sd_z,'n_abs_z_ge2':n2,'criteria':criteria,'pass':all(criteria.values()),'replicates':reps}
os.makedirs(os.path.dirname(out),exist_ok=True); json.dump(res,open(out,'w'),indent=2)
print(json.dumps({k:v for k,v in res.items() if k!='replicates'},indent=2),flush=True)

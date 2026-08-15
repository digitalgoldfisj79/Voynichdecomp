import ast, json, os, re
from pathlib import Path
import numpy as np
import requests

ROOT=Path('/manifest')
BRIDGE='https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/vtps_hf_bridge_20260814'
SECRET='frontier-u6-stageb-20260815'

out={'results_files':[],'npz':{},'json_samples':{},'src_files':{},'crop_dirs':{}}
for p in sorted((ROOT/'results').glob('*')):
    if not p.is_file(): continue
    rec={'name':p.name,'size':p.stat().st_size}
    out['results_files'].append(rec)
    if p.suffix=='.npz':
        try:
            z=np.load(p,allow_pickle=False)
            info={}
            for k in z.files:
                a=z[k]
                info[k]={'shape':list(a.shape),'dtype':str(a.dtype),'sample':a.reshape(-1)[:8].tolist() if a.size and a.dtype.kind in 'iufbSU' else None}
            out['npz'][p.name]=info
        except Exception as e: out['npz'][p.name]={'error':repr(e)}
    elif p.suffix in ('.json','.jsonl'):
        try:
            lines=[]
            with p.open('r',encoding='utf-8') as f:
                for _ in range(3):
                    line=f.readline()
                    if not line: break
                    lines.append(line[:3000])
            out['json_samples'][p.name]=lines
        except Exception as e: out['json_samples'][p.name]=[repr(e)]

for p in sorted((ROOT/'src').glob('*.py')):
    txt=p.read_text(encoding='utf-8',errors='replace')
    hits=[]
    for i,line in enumerate(txt.splitlines(),1):
        low=line.lower()
        if any(x in low for x in ['sha1','sha256','md5','hashlib','crop_id','word_index','_norm.png','crop_manifest','corpus_crop']):
            hits.append({'line':i,'text':line[:500]})
    out['src_files'][p.name]={'size':p.stat().st_size,'hits':hits[:120]}

# directory-level counts and a few filenames for every second-level child under crops
croot=ROOT/'crops'
for child in sorted(croot.iterdir()):
    if not child.is_dir(): continue
    count=0; samples=[]
    for dp,dn,fn in os.walk(child):
        count+=len(fn)
        for n in fn:
            if len(samples)<20: samples.append(str((Path(dp)/n).relative_to(ROOT)))
    out['crop_dirs'][child.name]={'count':count,'samples':samples}

r=requests.post(BRIDGE,json={'secret':SECRET,'id':'u6-stageb-20260815-pathfix-index-probe','payload':json.dumps(out,sort_keys=True,default=str),'meta':{'phase':'crop-index-probe'}},timeout=120)
r.raise_for_status()
print(json.dumps(out,indent=2,default=str)[:50000])

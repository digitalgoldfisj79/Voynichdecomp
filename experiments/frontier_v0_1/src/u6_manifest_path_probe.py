import json, os, re
from collections import Counter, defaultdict
from pathlib import Path
import requests

ROOT=Path('/manifest')
MAN=ROOT/'results/corpus_crop_manifest.jsonl'
BRIDGE='https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/vtps_hf_bridge_20260814'
SECRET='frontier-u6-stageb-20260815'

selected=[]
keysets=Counter(); path_present=0
examples=[]
with MAN.open('r',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line)
        if r.get('kind')=='word' and r.get('view')=='norm':
            keysets[tuple(sorted(r.keys()))]+=1
            if 'path' in r: path_present+=1
            if (r.get('folio'),r.get('word_index')) in {('f13r',68),('f13v',54),('f32r',14),('f39r',21),('f40r',1),('f10r',2)}:
                examples.append(r)
            selected.append(r)

files=[]; top=Counter(); ext=Counter(); norm=[]; folio_hits=defaultdict(list); id_hits=defaultdict(list)
target_ids={str(r.get('id')) for r in examples if r.get('id')}
for dp,dn,fn in os.walk(ROOT):
    rel_dp=Path(dp).relative_to(ROOT)
    topkey=rel_dp.parts[0] if rel_dp.parts else '.'
    for name in fn:
        p=Path(dp)/name; rel=str(p.relative_to(ROOT))
        files.append(rel); top[topkey]+=1; ext[p.suffix.lower()]+=1
        if name.endswith('_norm.png'):
            if len(norm)<100: norm.append(rel)
        low=rel.lower()
        for fol in ['f13r','f13v','f32r','f39r','f40r','f10r']:
            if fol in low and len(folio_hits[fol])<50: folio_hits[fol].append(rel)
        for tid in target_ids:
            if tid in low and len(id_hits[tid])<20: id_hits[tid].append(rel)

# summarize filename shapes for norm pngs across all files
shape=Counter()
for rel in files:
    if not rel.endswith('_norm.png'): continue
    name=Path(rel).name
    s=re.sub(r'[0-9a-f]{12}','<HEX12>',name)
    s=re.sub(r'\d+','<N>',s)
    shape[s]+=1

out={
 'manifest_norm_word_rows':len(selected),
 'manifest_norm_with_path':path_present,
 'manifest_keysets':[{'keys':list(k),'n':v} for k,v in keysets.most_common(10)],
 'examples':examples,
 'total_files':len(files),
 'top_level_counts':top,
 'extension_counts':ext,
 'norm_png_count':sum(1 for x in files if x.endswith('_norm.png')),
 'norm_png_samples':norm,
 'norm_filename_shapes':shape.most_common(20),
 'folio_path_hits':folio_hits,
 'id_path_hits':id_hits,
}
r=requests.post(BRIDGE,json={'secret':SECRET,'id':'u6-stageb-20260815-pathfix-manifest-probe','payload':json.dumps(out,sort_keys=True),'meta':{'phase':'asset-address-probe'}},timeout=120)
r.raise_for_status()
print(json.dumps(out,indent=2,default=str)[:30000])

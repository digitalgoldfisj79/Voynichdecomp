import hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import requests

ROOT=Path('/manifest')
CORPUS=ROOT/'results/corpus_crop_manifest.jsonl'
SHARD=ROOT/'crops/crop_shard_000/crop_manifest.jsonl'
BRIDGE='https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/vtps_hf_bridge_20260814'
SECRET='frontier-u6-stageb-20260815'
CALIB=set(['f10r','f10v','f11r','f11v','f13r','f13v','f14r','f14v','f15r','f15v','f16r','f16v','f17r','f17v','f18r','f18v','f19r','f19v','f1v','f20r','f20v','f21r','f21v','f22r','f22v','f23r','f23v','f24r','f24v','f25r','f25v','f26r','f26v','f27r','f27v','f28r','f28v','f29r','f29v','f2r','f2v','f30r','f30v','f31r','f31v','f32r','f32v','f33r','f33v','f34r','f34v','f35r','f35v','f36r','f36v','f37r','f37v','f38r','f38v','f39r','f39v','f3r','f3v','f40r','f40v','f41r','f42r','f42v','f43r','f43v','f44r','f44v','f45r','f45v','f46r','f46v','f47r','f47v','f48r','f48v','f49r','f49v','f4r','f4v','f50r','f50v','f51r','f51v','f52r','f52v','f53r','f53v','f54r','f54v','f55r','f55v','f56r','f56v','f5r','f6r','f6v','f7r','f7v','f8r','f8v','f9r','f9v'])
EXPECTED='c494eb695691e899d6e1dc648f9f7d7ec4afe49141a8890f9c1c40638b6a3f84'

required={}
with CORPUS.open('r',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line)
        if r.get('kind')=='word' and r.get('view')=='norm' and str(r.get('folio')) in CALIB:
            required[(str(r['folio']),int(r['word_index']))]=str(r['id'])
keys=sorted(required)
keyhash=hashlib.sha256(''.join(f'{a}|{b}\n' for a,b in keys).encode()).hexdigest()

hits=defaultdict(list); shard_rows=0; shard_folios=set(); wordnorm=0
with SHARD.open('r',encoding='utf-8') as f:
    for line in f:
        shard_rows+=1
        r=json.loads(line)
        if 'folio' in r: shard_folios.add(str(r['folio']))
        if r.get('kind')=='word' and r.get('view')=='norm':
            wordnorm+=1
            k=(str(r.get('folio')),int(r.get('word_index')))
            if k in required:
                hits[k].append({'id':str(r.get('id')),'path':r.get('path'),'word':r.get('word')})

missing=sorted(set(keys)-set(hits))
dups={f'{a}|{b}':v for (a,b),v in hits.items() if len(v)!=1}
resolved={k:v[0] for k,v in hits.items() if len(v)==1}
path_missing=[]; file_missing=[]; id_same=0
for k,r in resolved.items():
    p=r.get('path')
    if not p: path_missing.append(k); continue
    if not (ROOT/p).is_file(): file_missing.append((k,p))
    if r.get('id')==required[k]: id_same+=1

out={
 'required_n':len(required),'required_keyhash':keyhash,'expected_keyhash':EXPECTED,
 'shard_rows':shard_rows,'shard_unique_folios':len(shard_folios),'shard_folio_sample':sorted(shard_folios)[:30],
 'shard_word_norm_rows':wordnorm,'joined_keys':len(hits),'unique_resolved':len(resolved),
 'missing_n':len(missing),'missing_first':missing[:30],'duplicate_keys_n':len(dups),'duplicate_first':list(dups.items())[:10],
 'path_missing_n':len(path_missing),'file_missing_n':len(file_missing),'file_missing_first':file_missing[:10],
 'same_id_n':id_same,
 'example_joins':[{'key':list(k),'corpus_id':required[k],'crop':resolved[k]} for k in list(sorted(resolved))[:10]],
}
r=requests.post(BRIDGE,json={'secret':SECRET,'id':'u6-stageb-20260815-pathfix-join-probe','payload':json.dumps(out,sort_keys=True),'meta':{'phase':'manifest-join-probe'}},timeout=120)
r.raise_for_status()
print(json.dumps(out,indent=2))

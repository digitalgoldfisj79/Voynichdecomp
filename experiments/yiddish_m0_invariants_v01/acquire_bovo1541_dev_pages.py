from pathlib import Path
import hashlib, json, requests

LEAVES=[51,54,73,82,86,120]
OUT=Path('experiments/yiddish_m0_invariants_v01/bovo1541_dev_pages')
OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for leaf in LEAVES:
    urls=[
        f'https://iiif.archive.org/iiif/nybc207004${leaf}/full/full/0/default.jpg',
        f'https://iiif.archive.org/iiif/nybc207004%24{leaf}/full/full/0/default.jpg',
        f'https://archive.org/download/nybc207004/page/n{leaf}.jpg',
    ]
    ok=None
    errors=[]
    for url in urls:
        try:
            r=requests.get(url,timeout=60,allow_redirects=True)
            ct=r.headers.get('content-type','')
            if r.status_code==200 and ct.startswith('image/') and len(r.content)>10000:
                p=OUT/f'n{leaf}.jpg'
                p.write_bytes(r.content)
                ok={'leaf':leaf,'url':url,'final_url':r.url,'content_type':ct,'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest()}
                break
            errors.append({'url':url,'status':r.status_code,'ct':ct,'bytes':len(r.content),'final_url':r.url})
        except Exception as e:
            errors.append({'url':url,'error':repr(e)})
    if not ok:
        raise RuntimeError(f'No image acquired for leaf {leaf}: {errors}')
    rows.append(ok)
(OUT/'manifest.json').write_text(json.dumps({'source':'nybc207004','frozen_leaves':LEAVES,'pages':rows},indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))

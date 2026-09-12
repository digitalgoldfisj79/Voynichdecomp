#!/usr/bin/env python3
import json, os, pathlib, urllib.request, urllib.error

BASE='https://www.laudatio-repository.org'
OUT=pathlib.Path('experiments/yiddish_m0_invariants_v01/hsj_probe_output')
OUT.mkdir(parents=True, exist_ok=True)
headers={'Api-Version':'v1','Accept':'application/json','Content-Type':'application/json','User-Agent':'Voynichdecomp-HSJ-probe/1.0'}

def req(method,path,body=None,name=None):
    url=BASE+path
    data=None if body is None else json.dumps(body).encode()
    r=urllib.request.Request(url,data=data,headers=headers,method=method)
    rec={'url':url,'method':method}
    try:
        with urllib.request.urlopen(r,timeout=45) as resp:
            raw=resp.read()
            rec.update(status=resp.status,headers=dict(resp.headers),bytes=len(raw))
            if name:
                (OUT/name).write_bytes(raw)
            try: rec['json']=json.loads(raw)
            except Exception: rec['text_head']=raw[:5000].decode('utf-8','replace')
    except Exception as e:
        rec['error']=repr(e)
    return rec

results=[]
results.append(req('GET','/api/elasticapi/v1/corpora/latest/0/500',name='corpora_latest.json'))
for q in ['Historische Syntax des Jiddischen','Jiddisch','HSJ','8CRGCnMB7CArCQ9CUXL0']:
    results.append(req('POST','/api/elasticapi/v1/corpora/latest/searchMain',{'searchData':{'from':'0','size':'100','query':q}},name='corpus_search_'+q.replace(' ','_')+'.json'))
    results.append(req('POST','/api/elasticapi/v1/documents/latest/searchMain',{'searchData':{'from':'0','size':'200','query':q}},name='document_search_'+q.replace(' ','_')+'.json'))
    results.append(req('POST','/api/elasticapi/v1/annotations/latest/searchMain',{'searchData':{'from':'0','size':'200','query':q}},name='annotation_search_'+q.replace(' ','_')+'.json'))
# dynamic page and likely legacy/current routes for later manual inspection
for path,name in [
('/browse/corpus/8CRGCnMB7CArCQ9CUXL0/corpora','browse_hsj.html'),
('/api/elasticapi/v1/documents/latest/0/500','documents_latest.json'),
('/api/elasticapi/v1/annotations/latest/0/500','annotations_latest.json')]:
    results.append(req('GET',path,name=name))
(OUT/'probe_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps([{'url':r['url'],'status':r.get('status'),'bytes':r.get('bytes'),'error':r.get('error')} for r in results],ensure_ascii=False,indent=2))

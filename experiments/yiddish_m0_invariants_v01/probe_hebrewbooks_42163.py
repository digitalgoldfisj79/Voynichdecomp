#!/usr/bin/env python3
import re,json,pathlib,urllib.request,urllib.parse,hashlib
OUT=pathlib.Path('experiments/yiddish_m0_invariants_v01/hebrewbooks42163_probe'); OUT.mkdir(parents=True,exist_ok=True)
UA={'User-Agent':'Mozilla/5.0 Voynichdecomp-source-audit/1.0'}
def fetch(url,timeout=30):
 r=urllib.request.Request(url,headers=UA)
 with urllib.request.urlopen(r,timeout=timeout) as f:return f.geturl(),f.status,dict(f.headers),f.read()
rec=[]
for u in ['https://hebrewbooks.org/42163','https://www.hebrewbooks.org/42163']:
 try:
  final,status,h,b=fetch(u); (OUT/'detail.html').write_bytes(b)
  links=[]
  for x in re.findall(rb'href=["\']([^"\']+)',b,re.I):
   s=x.decode('utf-8','replace'); links.append(urllib.parse.urljoin(final,s))
  rec.append({'url':u,'final':final,'status':status,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'links':links})
  break
 except Exception as e:rec.append({'url':u,'error':repr(e)})
# probe only URLs discovered from the actual detail page whose text suggests PDF/download/reader/42163
cand=[]
for r in rec:
 for l in r.get('links',[]):
  ll=l.lower()
  if ('42163' in l or 'pdf' in ll or 'download' in ll) and l not in cand:cand.append(l)
probes=[]
for i,u in enumerate(cand[:40]):
 try:
  final,status,h,b=fetch(u,20)
  ct=h.get('Content-Type','')
  probes.append({'url':u,'final':final,'status':status,'content_type':ct,'content_length':h.get('Content-Length'),'bytes_read':len(b),'sha256':hashlib.sha256(b).hexdigest()})
  if b[:5]==b'%PDF-' or 'application/pdf' in ct.lower():
   (OUT/f'candidate_{i}.pdf').write_bytes(b)
 except Exception as e:probes.append({'url':u,'error':repr(e)})
(OUT/'summary.json').write_text(json.dumps({'detail':rec,'candidates':cand,'probes':probes},ensure_ascii=False,indent=2))
print(json.dumps({'detail':[ {k:v for k,v in r.items() if k!='links'} for r in rec], 'candidate_count':len(cand),'probes':probes},ensure_ascii=False,indent=2))

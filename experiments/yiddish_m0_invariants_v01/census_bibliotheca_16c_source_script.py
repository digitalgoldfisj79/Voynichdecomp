#!/usr/bin/env python3
import re,json,hashlib,pathlib,urllib.request,urllib.parse,concurrent.futures
from html import unescape
OUT=pathlib.Path('experiments/yiddish_m0_invariants_v01/bibliotheca16_census_output'); OUT.mkdir(parents=True,exist_ok=True)
INDEX='https://www.hs-augsburg.de/~harsch/iiddica/Khronologye/y_16yorh.html'
UA={'User-Agent':'Voynichdecomp-Yiddish16-source-census/1.0'}
HREF=re.compile(r'href=["\']([^"\']+)',re.I); TAG=re.compile(r'<[^>]+>'); HEB=re.compile(r'[\u0590-\u05FF\uFB1D-\uFB4F]+')
def get(u,timeout=10):
 r=urllib.request.Request(u,headers=UA)
 with urllib.request.urlopen(r,timeout=timeout) as f:return f.read()
def visible(raw):
 s=raw.decode('utf-8','replace');s=re.sub(r'(?is)<script.*?</script>|<style.*?</style>',' ',s);return unescape(TAG.sub(' ',s))
def canon(u):
 p=urllib.parse.urlparse(u);return urllib.parse.urlunparse((p.scheme,p.netloc,p.path,'','',''))
idx=get(INDEX); (OUT/'index.html').write_bytes(idx)
links=[]
for h in HREF.findall(idx.decode('utf-8','replace')):
 u=canon(urllib.parse.urljoin(INDEX,h))
 if '/iiddica/Khronologye/y_16yh/' in u and u.endswith(('.html','.htm')):links.append(u)
# Fetch work landing pages then follow their same-directory text/orig/manu/transcription links.
def fetch(u):
 try:return u,get(u)
 except Exception as e:return u,e
rows=[]; second=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
 for u,res in ex.map(fetch,sorted(set(links))):
  if isinstance(res,Exception):rows.append({'url':u,'error':repr(res)});continue
  txt=visible(res); rows.append({'url':u,'bytes':len(res),'sha256':hashlib.sha256(res).hexdigest(),'hebrew_chars':sum(len(x) for x in HEB.findall(txt)),'hebrew_runs':len(HEB.findall(txt))})
  base=u.rsplit('/',1)[0]+'/'
  for h in HREF.findall(res.decode('utf-8','replace')):
   v=canon(urllib.parse.urljoin(u,h))
   if v.startswith(base) and v.endswith(('.html','.htm')): second.append(v)
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
 for u,res in ex.map(fetch,sorted(set(second))):
  if isinstance(res,Exception):rows.append({'url':u,'error':repr(res)});continue
  txt=visible(res); hc=sum(len(x) for x in HEB.findall(txt)); hr=len(HEB.findall(txt))
  rows.append({'url':u,'bytes':len(res),'sha256':hashlib.sha256(res).hexdigest(),'hebrew_chars':hc,'hebrew_runs':hr})
  if hc: (OUT/(hashlib.sha256(u.encode()).hexdigest()[:12]+'.html')).write_bytes(res)
# dedupe URL
seen={};
for r in rows: seen[r['url']]=r
rows=list(seen.values()); ranked=sorted([r for r in rows if r.get('hebrew_chars',0)>0],key=lambda r:r['hebrew_chars'],reverse=True)
(OUT/'census.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(OUT/'hebrew_pages.json').write_text(json.dumps(ranked,ensure_ascii=False,indent=2))
print(json.dumps({'landing_links':len(set(links)),'pages_total':len(rows),'errors':sum('error'in r for r in rows),'hebrew_pages':len(ranked),'top':ranked[:30]},ensure_ascii=False,indent=2))

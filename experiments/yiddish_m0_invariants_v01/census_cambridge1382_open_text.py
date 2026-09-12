#!/usr/bin/env python3
import re,json,hashlib,pathlib,urllib.request,urllib.parse
from html import unescape

OUT=pathlib.Path('experiments/yiddish_m0_invariants_v01/cambridge1382_census_output'); OUT.mkdir(parents=True,exist_ok=True)
START=[
 'https://www.hs-augsburg.de/~harsch/iiddica/Khronologye/y_14yh/Dukus/duk_hor0.html',
 'https://www2.hs-augsburg.de/~harsch/iiddica/Khronologye/y_14yh/Dukus/duk_hor0.html',
 'https://www.hs-augsburg.de/~harsch/iiddica/Khronologye/y_14yh/Leib/lei_intr.html',
 'https://www2.hs-augsburg.de/~harsch/iiddica/Khronologye/y_14yh/Leib/lei_intr.html',
]
UA={'User-Agent':'Voynichdecomp-Cambridge1382-census/1.0'}
HEB=re.compile(r'[\u0590-\u05FF\uFB1D-\uFB4F]+')
TAG=re.compile(r'<[^>]+>')
HREF=re.compile(r'href=["\']([^"\']+)',re.I)

def get(u):
 r=urllib.request.Request(u,headers=UA); 
 with urllib.request.urlopen(r,timeout=20) as f: return f.read()

def visible(raw):
 s=raw.decode('utf-8','replace'); s=re.sub(r'(?is)<script.*?</script>|<style.*?</style>',' ',s); return unescape(TAG.sub(' ',s))

def canon(u):
 p=urllib.parse.urlparse(u); return urllib.parse.urlunparse((p.scheme,p.netloc,p.path,'','',''))

queue=list(START); seen=set(); rows=[]
while queue and len(seen)<120:
 u=queue.pop(0); u=canon(u)
 if u in seen: continue
 seen.add(u)
 try: raw=get(u)
 except Exception as e:
  rows.append({'url':u,'error':repr(e)}); continue
 txt=visible(raw)
 hs=HEB.findall(txt); heb_chars=sum(len(x) for x in hs); heb_tokens=len(re.findall(r'[\u0590-\u05FF\uFB1D-\uFB4F]+',txt))
 row={'url':u,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'hebrew_chars':heb_chars,'hebrew_runs':heb_tokens,'visible_chars':len(txt)}; rows.append(row)
 fn=OUT/(hashlib.sha256(u.encode()).hexdigest()[:12]+'.html'); fn.write_bytes(raw)
 # follow only y_14yh links, prioritising Cambridge-related directories
 s=raw.decode('utf-8','replace')
 for h in HREF.findall(s):
  v=canon(urllib.parse.urljoin(u,h))
  if '/iiddica/Khronologye/y_14yh/' in v and v.endswith(('.html','.htm')) and v not in seen:
   queue.append(v)
(OUT/'census.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
# Emit ranked Hebrew-bearing pages
ranked=sorted([r for r in rows if r.get('hebrew_chars',0)>0],key=lambda r:r['hebrew_chars'],reverse=True)
(OUT/'hebrew_pages.json').write_text(json.dumps(ranked,ensure_ascii=False,indent=2))
print(json.dumps({'pages_fetched':sum('bytes' in r for r in rows),'errors':sum('error' in r for r in rows),'hebrew_bearing_pages':len(ranked),'total_hebrew_chars_across_pages':sum(r['hebrew_chars'] for r in ranked),'top':ranked[:20]},ensure_ascii=False,indent=2))

#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

PPCHY_COMMIT='b5864bd02a315c1d436a82553667bbf81eab6537'
LEAF_RE=re.compile(r'\(([A-Z][A-Z0-9$=*-]*)\s+([^()\s]+)\)')
YEAR_RE=re.compile(r'^(\d{4})')
FID_RE=re.compile(r'(F\d{3})',re.I)
USED_YID={'1579e-shir','1589e-ester','1507w-bovo','1588e-letters-cracow','1590e-sam-hayyim','1620e-lev-tov-1','1648w-kine'}
USED_GER={'F014','F015','F016','F018','F034','F037','F148'}
YID_BUILD={'1579e-shir','1589e-ester'}
GER_BUILD={'F014','F015'}
MIN_WORDS=1056
MAX_YEAR=1750

def sha_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def norm_leaf(raw:str):
 if raw.startswith('*') or raw in {'0','-NONE-'}: return []
 raw=raw.replace('@','').split('^',1)[0]; out=[]
 for part in raw.split('_'):
  w=''.join(c for c in part.lower() if 'a'<=c<='z')
  if w: out.append(w)
 return out

def penn_words(p:Path):
 out=[]
 for tag,raw in LEAF_RE.findall(p.read_text(encoding='utf-8',errors='replace')):
  if tag.startswith(('ID','CODE','PUNC')): continue
  out.extend(norm_leaf(raw))
 return out

def penn_family_id(name:str)->str:
 s=name[:-4] if name.endswith('.psd') else name
 if s.endswith('-preface'): s=s[:-8]
 return s

def penn_year(fid:str):
 m=YEAR_RE.match(fid); return int(m.group(1)) if m else None

def norm_latin(s:str):
 w=''.join(c for c in s.lower() if 'a'<=c<='z')
 return w or None

def local(tag:str): return tag.rsplit('}',1)[-1]

def ref_words(p:Path):
 root=ET.parse(p).getroot(); out=[]
 for tok in root.iter():
  if local(tok.tag)!='token': continue
  frags=[]
  for el in tok.iter():
   if local(el.tag)=='tok_dipl': frags.append(el.attrib.get('utf') or (el.text or ''))
  w=norm_latin(''.join(frags))
  if w: out.append(w)
 return out

def ngrams(words,n): return {tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1))}

def overlap(a,b,n): return len(ngrams(a,n)&ngrams(b,n))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--penn-data',required=True); ap.add_argument('--ref-root',required=True); ap.add_argument('--out',required=True)
 a=ap.parse_args(); pdata=Path(a.penn_data); rroot=Path(a.ref_root); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

 # PPCHY family census.
 pf=defaultdict(list)
 for p in sorted(pdata.glob('*.psd')):
  fid=penn_family_id(p.name); y=penn_year(fid)
  if y is not None and y<=MAX_YEAR: pf[fid].append(p)
 yrows=[]; ywords={}
 for fid,files in sorted(pf.items()):
  words=[]; hashes={}
  for p in sorted(files): words.extend(penn_words(p)); hashes[p.name]=sha_file(p)
  ywords[fid]=words
  yrows.append({'family':fid,'year':penn_year(fid),'files':[p.name for p in sorted(files)],'sha256':hashes,'words':len(words),'used_in_l_v01b':fid in USED_YID,'eligible_transfer':fid not in USED_YID and len(words)>=MIN_WORDS})
 yeligible=[r for r in yrows if r['eligible_transfer']]

 # ReF census. Work ID is metadata/path-derived; if multiple XMLs resolve to one ID,
 # choose the nonempty candidate with maximum corrected word count, tie by path.
 rcand=defaultdict(list)
 for p in sorted(rroot.rglob('*.xml')):
  m=FID_RE.search(p.name) or FID_RE.search(str(p.parent))
  if not m: continue
  wid=m.group(1).upper()
  try: w=ref_words(p)
  except Exception: continue
  if w: rcand[wid].append((len(w),str(p),p,w))
 grows=[]; gwords={}
 for wid,arr in sorted(rcand.items()):
  arr.sort(key=lambda x:(-x[0],x[1])); n,path,p,w=arr[0]; gwords[wid]=w
  grows.append({'work':wid,'path':path,'sha256':sha_file(p),'words':n,'alternative_xml_candidates':[{'path':x[1],'words':x[0]} for x in arr[1:]],'used_in_l_v01b':wid in USED_GER,'eligible_transfer':wid not in USED_GER and n>=MIN_WORDS})
 gelig=[r for r in grows if r['eligible_transfer']]

 # Deterministic length-only German matching to all eligible Yiddish families.
 unused={r['work']:r for r in gelig}; matched=[]
 for yr in sorted(yeligible,key=lambda r:(r['words'],r['family'])):
  if not unused: raise RuntimeError('fewer eligible fresh German works than Yiddish families')
  gr=min(unused.values(),key=lambda r:(abs(r['words']-yr['words']),r['work']))
  matched.append({'yiddish_family':yr['family'],'yiddish_words':yr['words'],'german_work':gr['work'],'german_words':gr['words'],'abs_word_count_difference':abs(gr['words']-yr['words'])})
  del unused[gr['work']]
 selected_g={x['german_work'] for x in matched}

 # BUILD leakage audits.
 ybuild=[]
 for fid in sorted(YID_BUILD): ybuild.extend(ywords[fid])
 gbuild=[]
 for wid in sorted(GER_BUILD): gbuild.extend(gwords[wid])
 y_leak=[]
 for r in yeligible:
  w=ywords[r['family']]; y_leak.append({'family':r['family'],'shared_build_8word_types':overlap(ybuild,w,8),'shared_build_5word_types':overlap(ybuild,w,5)})
 g_leak=[]
 for wid in sorted(selected_g):
  w=gwords[wid]; g_leak.append({'work':wid,'shared_build_8word_types':overlap(gbuild,w,8),'shared_build_5word_types':overlap(gbuild,w,5)})

 # Cross-candidate duplicate audit inside each language.
 y_cross=[]
 for i,a1 in enumerate(sorted(r['family'] for r in yeligible)):
  for a2 in sorted(r['family'] for r in yeligible)[i+1:]:
   k=overlap(ywords[a1],ywords[a2],8)
   if k: y_cross.append({'a':a1,'b':a2,'shared_8word_types':k})
 gsel=sorted(selected_g); g_cross=[]
 for i,g1 in enumerate(gsel):
  for g2 in gsel[i+1:]:
   k=overlap(gwords[g1],gwords[g2],8)
   if k: g_cross.append({'a':g1,'b':g2,'shared_8word_types':k})

 blocked=any(x['shared_build_8word_types'] for x in y_leak+g_leak) or bool(y_cross) or bool(g_cross)
 result={
  'mode':'SOURCE_ONLY_CENSUS_AND_SELECTION__NO_SCORING','target_loaded':False,'voynich_loaded':False,
  'ppchy_commit_expected':PPCHY_COMMIT,'min_words':MIN_WORDS,'max_yiddish_year':MAX_YEAR,
  'yiddish_census':yrows,'german_census':grows,
  'eligible_yiddish_families':[r['family'] for r in yeligible],
  'eligible_german_work_count':len(gelig),'deterministic_length_matching':matched,
  'selected_german_works':sorted(selected_g),'yiddish_build_leakage':y_leak,'german_build_leakage':g_leak,
  'yiddish_cross_candidate_8word_overlap':y_cross,'german_cross_candidate_8word_overlap':g_cross,
  'selection_gate':'BLOCKED' if blocked else 'PASS'
 }
 (out/'census_selection.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'selection_gate':result['selection_gate'],'eligible_yiddish':len(yeligible),'eligible_german':len(gelig),'selected_german':len(selected_g),'yiddish_ids':result['eligible_yiddish_families'],'german_ids':result['selected_german_works']},indent=2))
if __name__=='__main__': main()

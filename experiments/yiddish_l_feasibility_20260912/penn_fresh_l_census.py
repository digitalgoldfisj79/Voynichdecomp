#!/usr/bin/env python3
"""Penn-only fresh Yiddish L-confirmation feasibility census.

No language scoring. No Voynich access. This asks only whether, after v01-v03,
there remains an independent 1501-1600 Penn source family long enough for the
registered 512 fit + 32 buffer + 512 audit L cell.
"""
from __future__ import annotations
import hashlib, json, re, subprocess
from pathlib import Path

URL='https://github.com/beatrice57/penn-parsed-corpus-of-historical-yiddish.git'
HERE=Path(__file__).resolve().parent
OUT=HERE/'penn_fresh_l_census.json'
NEED_SHORT=1056
NEED_LONG=4128
LEAF_RE=re.compile(r"\(([A-Z][A-Z0-9$=*-]*)\s+([^()\s]+)\)")
YEAR_RE=re.compile(r"^(\d{4})")

# Strongest obvious work-family grouping from Penn filenames plus prior-run lineage.
FAMILY={
 '1507w-bovo.psd':'bovo',
 '1579e-shir.psd':'shir','1579e-shir-preface.psd':'shir',
 '1589e-ester.psd':'ester','1589e-ester-preface.psd':'ester',
 '1620e-lev-tov-1.psd':'lev-tov-1','1620e-lev-tov-1-preface.psd':'lev-tov-1',
 '1815e-lev-tov-2.psd':'lev-tov-2',
}

def fam(name): return FAMILY.get(name,name[:-4])

# Any family exposed in development, solver/evaluator training, or confirmation is consumed.
CONSUMED_FILES={
 '1507w-bovo.psd','1579e-shir.psd','1579e-shir-preface.psd',
 '1589e-ester.psd','1589e-ester-preface.psd','1588e-letters-cracow.psd','1590e-sam-hayyim.psd',
 '1600e-magid-preface.psd','1600e-magid.psd','1600e-tsenerene.psd','1619w-letters-prague.psd',
 '1620e-lev-tov-1.psd','1620e-lev-tov-1-preface.psd','1624e-magen.psd','1648w-kine.psd',
 '1666w-messiah.psd','1671e-vaad.psd','1675e-ashkenaz-un-polak.psd','1677w-witzenhausen.psd',
 '1692e-vilna.psd','1697e-purim.psd','1704e-ellush.psd','1705w-glikl.psd','1712e-sarah.psd',
 '1716e-duties.psd','1717e-poznan.psd','1723w-simkhes.psd','1740w-drises.psd',
 '1743e-teshuat-preface.psd','1750w-moses.psd','1834e-ukraine-2.psd'
}
CONSUMED_FAMILIES={fam(x) for x in CONSUMED_FILES}

def normalize_leaf(raw):
 if raw.startswith('*') or raw in {'0','-NONE-'}: return []
 raw=raw.replace('@','').split('^',1)[0]
 out=[]
 for part in raw.split('_'):
  w=''.join(c for c in part.lower() if 'a'<=c<='z')
  if w: out.append(w)
 return out

def words(path):
 out=[]
 for tag,raw in LEAF_RE.findall(path.read_text(encoding='utf-8',errors='replace')):
  if tag.startswith(('ID','CODE','PUNC')): continue
  out.extend(normalize_leaf(raw))
 return out

def year(name):
 m=YEAR_RE.match(name); return int(m.group(1)) if m else None

def sha256(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()

def main():
 repo=HERE/'ppchy'
 if not repo.exists(): subprocess.run(['git','clone','--depth','1',URL,str(repo)],check=True)
 commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 data=repo/'data'
 rows=[]
 for p in sorted(data.glob('*.psd')):
  y=year(p.name)
  if y is None or y>1600: continue
  ws=words(p); f=fam(p.name)
  rows.append({
    'file':p.name,'family':f,'year_from_filename':y,'pipeline_words':len(ws),
    'short_eligible':len(ws)>=NEED_SHORT,'long_eligible':len(ws)>=NEED_LONG,
    'consumed_family':f in CONSUMED_FAMILIES,'sha256':sha256(p),
    'special_status':('ANTHOLOGY_RELATIONSHIP_RECONSTRUCTION_REQUIRED' if p.name=='1xxxx-court-testimony.psd' else None)
  })
 # Filename 1xxxx is an anthology and year() deliberately cannot assign it to 1501-1600.
 anthology=data/'1xxxx-court-testimony.psd'
 anthology_row=None
 if anthology.exists():
  anthology_row={'file':anthology.name,'pipeline_words':len(words(anthology)),'sha256':sha256(anthology),
    'status':'NOT_A_SINGLE_CONFIRMATION_WORK: multiple court-testimony items c1413-1686; relationship/date groups must be reconstructed before admission'}
 fresh_short=[r for r in rows if 1501<=r['year_from_filename']<=1600 and r['short_eligible'] and not r['consumed_family']]
 fresh_long=[r for r in rows if 1501<=r['year_from_filename']<=1600 and r['long_eligible'] and not r['consumed_family']]
 result={
  'question':'After v01-v03, does Penn retain a fresh independent 1501-1600 Yiddish work-family for L confirmation?',
  'ppchy_commit':commit,'need_short_words':NEED_SHORT,'need_long_words':NEED_LONG,
  'consumed_families':sorted(CONSUMED_FAMILIES),'pre1601_rows':rows,'court_testimony_anthology':anthology_row,
  'fresh_1501_1600_short_candidates':fresh_short,'fresh_1501_1600_long_candidates':fresh_long,
  'fresh_short_candidate_count':len(fresh_short),'fresh_long_candidate_count':len(fresh_long),
  'disposition':('PENN_FRESH_L_FEASIBLE' if fresh_short else 'PENN_FRESH_L_CONFIRMATION_CORPUS_LIMITED'),
  'interpretation':'No language score is computed. Zero fresh eligible works blocks a fresh Penn-only L confirmation; it is not evidence against Yiddish.'
 }
 OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({k:result[k] for k in ['ppchy_commit','fresh_short_candidate_count','fresh_long_candidate_count','fresh_1501_1600_short_candidates','fresh_1501_1600_long_candidates','court_testimony_anthology','disposition','interpretation']},indent=2))
if __name__=='__main__': main()

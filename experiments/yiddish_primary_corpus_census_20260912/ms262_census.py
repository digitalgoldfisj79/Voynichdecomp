#!/usr/bin/env python3
import json, re, subprocess, hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

URL='https://github.com/cu-mkp/ms-262-data.git'
HERE=Path(__file__).resolve().parent
OUT=HERE/'ms262_census.json'
HEB=re.compile(r'[\u0590-\u05FF]+')
PAGE=re.compile(r'tc_p(\d{3})([rv])\.xml$')

def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()

def key(p):
 m=PAGE.search(p.name)
 return (int(m.group(1)),0 if m.group(2)=='r' else 1) if m else (9999,9)

def main():
 repo=HERE/'ms262-data'
 if not repo.exists(): subprocess.run(['git','clone','--depth','1',URL,str(repo)],check=True)
 commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 files=sorted((repo/'xml'/'transcription').glob('tc_p*.xml'),key=key)
 pages=[]; total_words=0; total_atoms=0; language_abs={}
 present=[]
 for p in files:
  m=PAGE.search(p.name)
  if not m: continue
  label=f"{int(m.group(1))}{m.group(2)}"; present.append(label)
  root=ET.parse(p).getroot(); words=[]; langs={}
  for ab in root.iter('ab'):
   lang=ab.attrib.get('language','UNLABELED'); txt=' '.join(ab.itertext()); toks=HEB.findall(txt)
   langs[lang]=langs.get(lang,0)+len(toks); language_abs[lang]=language_abs.get(lang,0)+1
   if lang=='owy': words.extend(toks)
  atoms=sum(len(w) for w in words); total_words+=len(words); total_atoms+=atoms
  pages.append({'page':label,'file':p.name,'owy_word_tokens':len(words),'owy_hebrew_atoms':atoms,'language_token_counts':langs,'sha256':sha256(p)})
 expected=[f'{n}{s}' for n in range(1,30) for s in ('r','v')]
 missing=[x for x in expected if x not in present]
 # longest run of present physical sides, irrespective of whether each has OWY text
 idx={x:i for i,x in enumerate(expected)}; present_idx=sorted(idx[x] for x in present if x in idx)
 runs=[]
 if present_idx:
  st=pr=present_idx[0]
  for q in present_idx[1:]:
   if q==pr+1: pr=q
   else: runs.append((st,pr)); st=pr=q
  runs.append((st,pr))
 run_records=[]
 for a,b in runs:
  labels=expected[a:b+1]; wc=sum(x['owy_word_tokens'] for x in pages if x['page'] in labels)
  run_records.append({'start':expected[a],'end':expected[b],'sides':b-a+1,'owy_words':wc})
 result={
  'source':'Columbia Gen. MS 262 transcription repository cu-mkp/ms-262-data',
  'commit':commit,'transcription_files':len(files),'present_sides':present,'missing_sides_1r_to_29v':missing,
  'total_owy_hebrew_script_word_tokens':total_words,'total_owy_hebrew_atoms':total_atoms,
  'minimum_words_short_condition_including_buffer':1056,'minimum_words_long_condition_including_buffer':4128,
  'whole_transcription_short_length_possible':total_words>=1056,'whole_transcription_long_length_possible':total_words>=4128,
  'contiguous_present_side_runs':run_records,'pages':pages,'ab_element_counts_by_language':language_abs,
  'qualification_limit':'Counts establish transcription quantity only. They do not establish gold accuracy, completeness, relationship independence, or that missing sides may be crossed for a calibrated fit/audit span.'
 }
 OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps({k:result[k] for k in ['commit','transcription_files','total_owy_hebrew_script_word_tokens','total_owy_hebrew_atoms','whole_transcription_short_length_possible','whole_transcription_long_length_possible','missing_sides_1r_to_29v','contiguous_present_side_runs','qualification_limit']},indent=2,ensure_ascii=False))
if __name__=='__main__': main()

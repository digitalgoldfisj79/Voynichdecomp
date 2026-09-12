#!/usr/bin/env python3
"""Prepare blinded M0 Yiddish instrument-control cases.

Creates two deliberately separate bundles:
  public/  -> ciphertext cases + exact training words + pre-outcome manifest
  private/ -> plaintext/key truth + root reveal, withheld from solver jobs

No Voynich data is imported or referenced. All Yiddish sources used here are already
consumed development/control material. Fresh historical transfer is C6 and is not touched.
"""
from __future__ import annotations

import argparse, hashlib, hmac, json, math, os, random, secrets, sys, tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
V1B=REPO/'experiments'/'yiddish_qualification_v1b'
sys.path.insert(0,str(V1B))
import finite_panel_r_v02 as base  # noqa: E402
from encoder_v02 import encode_words  # noqa: E402
from independent_decoder_v02 import invert_permutation, decode_words  # noqa: E402

VERSION='yiddish_m0_full_control_v1_20260912'
A=base.A; ALPHABET=base.ALPHABET; BOUNDARY=base.BOUNDARY
BUFFER=32
LENGTHS=(128,256,512,1024,2048)
ERASURES=(0.0,0.01,0.03,0.05)
N_KEYS=32
NEG_N=94
MR_N=32

YID_BUILD=(
 '1600e-magid-preface.psd','1600e-magid.psd','1600e-tsenerene.psd',
 '1619w-letters-prague.psd','1624e-magen.psd','1671e-vaad.psd',
 '1677w-witzenhausen.psd','1692e-vilna.psd','1704e-ellush.psd',
 '1705w-glikl.psd','1712e-sarah.psd','1716e-duties.psd')
YID_DEV=('1507w-bovo.psd','1588e-letters-cracow.psd','1590e-sam-hayyim.psd',
         '1620e-lev-tov-1.psd','1648w-kine.psd','1666w-messiah.psd')
GERMAN_IDS=('F016','F018','F034','F037','F148')


def sha_file(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def sha_bytes(b:bytes): return hashlib.sha256(b).hexdigest()
def seed(root:bytes,*parts):
 return int.from_bytes(hmac.new(root,'|'.join(map(str,parts)).encode(),hashlib.sha256).digest()[:8],'big')
def pubseed(*parts):
 return int.from_bytes(hashlib.sha256('|'.join([VERSION,*map(str,parts)]).encode()).digest()[:8],'big')
def write_json(x,p:Path):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def append_jsonl(x,p:Path):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f: f.write(json.dumps(x,sort_keys=True)+'\n')

def make_key(root,*parts):
 r=random.Random(seed(root,'key',*parts)); p=list(range(A)); r.shuffle(p); return p

def enc(words,key,erasure,root,*parts):
 r=random.Random(seed(root,'erase',*parts,erasure)); return encode_words(words,key,erasure,r)[0]

def split(words,n):
 if len(words)<2*n+BUFFER: return None
 return words[:n],words[n+BUFFER:n+BUFFER+n]

def ngrams(words,n): return {tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1))}

def normalize_latin(s):
 w=''.join(c for c in s.lower() if 'a'<=c<='z'); return w or None

def ref_words(root:Path,wid:str):
 cand=[p for p in root.rglob('*.xml') if wid.lower() in p.name.lower() or wid.lower() in str(p.parent).lower()]
 if not cand: return [],None
 p=max(cand,key=lambda q:q.stat().st_size); out=[]
 tree=ET.parse(p).getroot()
 for el in tree.iter():
  if el.tag.rsplit('}',1)[-1]=='tok_dipl':
   w=normalize_latin(el.attrib.get('utf') or (el.text or ''))
   if w: out.append(w)
 return out,p

def relabel_words(words,perm):
 return [''.join(ALPHABET[perm[ord(c)-97]] for c in w) for w in words]

def per_block_encrypt(words,root,tag,block_size):
 out=[]
 for i,w in enumerate(words):
  k=make_key(root,tag,'block',i//block_size)
  out.append(''.join(ALPHABET[k[ord(c)-97]] for c in w))
 return out

def per_word_encrypt(words,root,tag): return per_block_encrypt(words,root,tag,1)

def kat(root):
 fixtures=[['a'],['aa'],['abba'],['abcdefghijklmnopqrstuvwxyz'],['abc','def','abc'],['mississippi','banana','letter']]
 keys=[('identity',list(range(A))),('cyclic5',list(range(5,A))+list(range(5))),('reverse',list(reversed(range(A)))),('random',make_key(root,'kat','random'))]
 rows=[]
 for fi,words in enumerate(fixtures):
  for kn,k in keys:
   for er in (0.0,0.01):
    c=enc(words,k,er,root,'kat',fi,kn)
    d=decode_words(c,invert_permutation(k)); ok=True
    for t,x in zip(words,d):
     for a,b in zip(t,x):
      if b!='~' and a!=b: ok=False
    rows.append({'fixture':fi,'key':kn,'erasure':er,'pass':ok})
 return rows

def sufficient_stat_mrs(sample_words,root):
 rows=[]
 C0,_=base.cipher_counts(sample_words); Cdup,_=base.cipher_counts(sample_words+sample_words)
 delta=max(abs(C0[i][j]-Cdup[i][j]) for i in range(A+1) for j in range(A+1))
 rows.append({'mr':'MR3_duplicate_normalized_counts','pass':delta<1e-15,'max_abs_delta':delta})
 recomb=sample_words[:171]+sample_words[171:341]+sample_words[341:]
 C1,_=base.cipher_counts(recomb); delta=max(abs(C0[i][j]-C1[i][j]) for i in range(A+1) for j in range(A+1))
 rows.append({'mr':'MR4_chunk_recombine','pass':delta<1e-15,'max_abs_delta':delta})
 # MR5 independent decoder, 32 planted examples.
 for i in range(MR_N):
  k=make_key(root,'mr5',i); c=enc(sample_words,k,0.01,root,'mr5',i)
  d=decode_words(c,invert_permutation(k)); ok=True
  for t,x in zip(sample_words,d):
   for a,b in zip(t,x):
    if b!='~' and a!=b: ok=False
  rows.append({'mr':'MR5_independent_decoder','rep':i,'pass':ok})
 # MR6 boundaries are a claimed evidence channel: deterministic merges must change counts.
 for i in range(MR_N):
  r=random.Random(pubseed('mr6',i)); w=list(sample_words)
  candidates=list(range(len(w)-1)); r.shuffle(candidates); take=sorted(candidates[:max(1,len(w)//20)],reverse=True)
  for j in take:
   w[j:j+2]=[w[j]+w[j+1]]
  Cb,_=base.cipher_counts(w); delta=max(abs(C0[a][b]-Cb[a][b]) for a in range(A+1) for b in range(A+1))
  rows.append({'mr':'MR6_boundary_destructive','rep':i,'pass':delta>0,'max_abs_delta':delta})
 return rows

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--ref-root',type=Path,required=True)
 args=ap.parse_args(); pub=args.out/'public'; priv=args.out/'private'; work=args.out/'work'; pub.mkdir(parents=True,exist_ok=True); priv.mkdir(parents=True,exist_ok=True); work.mkdir(parents=True,exist_ok=True)
 root=secrets.token_bytes(32); commitment=sha_bytes(root)
 repo,ppcommit=base.ensure_corpus(work/'sources'); data=repo/'data'

 # Deterministic representation check and exact source hashes.
 source_meta={}; dev_words={}; build=[]
 for name in YID_BUILD+YID_DEV:
  p=data/name; w1=base.extract_words(p); w2=base.extract_words(p)
  if w1!=w2: raise RuntimeError(f'nondeterministic extraction: {name}')
  source_meta[name]={'sha256':sha_file(p),'words':len(w1)}
  if name in YID_BUILD: build.extend(w1)
  else: dev_words[name]=w1
 if len(build)<10000: raise RuntimeError('BUILD too small')
 write_json(build,pub/'training_words.json')

 # Leakage screen BUILD vs development controls.
 b8=ngrams(build,8); b5=ngrams(build,5); overlaps=[]
 for name,w in dev_words.items(): overlaps.append({'work':name,'shared_8gram_types':len(b8&ngrams(w,8)),'shared_5gram_types':len(b5&ngrams(w,5))})

 kats=kat(root); mrs=sufficient_stat_mrs(dev_words['1507w-bovo.psd'][:512],root)
 if not all(x['pass'] for x in kats): raise RuntimeError('KAT failure during prepare')

 cases=pub/'cases.jsonl'; truth=priv/'truth.jsonl'
 if cases.exists(): cases.unlink()
 if truth.exists(): truth.unlink()
 case_count=Counter()

 def add_case(public,private):
  cid=public['case_id']; private={'case_id':cid,**private}; append_jsonl(public,cases); append_jsonl(private,truth); case_count[public['certificate']]+=1

 # C3: planted positives over full registered surface where quantity permits.
 for workname,w in dev_words.items():
  for n in LENGTHS:
   s=split(w,n)
   if not s: continue
   fit,audit=s
   for er in ERASURES:
    for ki in range(N_KEYS):
     cid=f'C3|{workname}|{n}|{er:.3f}|{ki:02d}'; k=make_key(root,cid)
     add_case({'case_id':cid,'certificate':'C3','family':'yiddish_m0','work':workname,'length':n,'erasure':er,
               'fit_cipher':enc(fit,k,er,root,cid,'fit'),'audit_cipher':enc(audit,k,er,root,cid,'audit'),'lm_relabel':None},
              {'fit_truth':fit,'audit_truth':audit,'oracle':invert_permutation(k),'expected_mechanism_positive':True,'expected_language':'yiddish'})

 # C4a: non-global substitution negatives. 94 per registered family supports Bonferroni bound if 0 false calls.
 basefit,baseaudit=split(dev_words['1507w-bovo.psd'],512)
 negspec={'per_word_key':1,'key_drift_4':4,'key_drift_16':16,'key_drift_64':64}
 for fam,block in negspec.items():
  for i in range(NEG_N):
   cid=f'C4a|{fam}|{i:03d}'
   fc=per_block_encrypt(basefit,root,cid+'|fit',block); ac=per_block_encrypt(baseaudit,root,cid+'|audit',block)
   add_case({'case_id':cid,'certificate':'C4a','family':fam,'work':'1507w-bovo.psd','length':512,'erasure':0.0,
             'fit_cipher':fc,'audit_cipher':ac,'lm_relabel':None},
            {'fit_truth':basefit,'audit_truth':baseaudit,'oracle':None,'expected_mechanism_positive':False,'expected_language':'yiddish'})

 # C4b: language/nuisance diagnostics under a genuine global M0 mechanism.
 # (a) shuffled-within-word Yiddish; (b) unigram-matched synthetic strings.
 for fam in ('within_word_shuffle','unigram_null'):
  for i in range(NEG_N):
   cid=f'C4b|{fam}|{i:03d}'
   if fam=='within_word_shuffle':
    def mutate(ws,side):
     out=[]
     for j,w in enumerate(ws):
      a=list(w); random.Random(pubseed(cid,side,j)).shuffle(a); out.append(''.join(a))
     return out
   else:
    c=Counter(''.join(basefit)); letters=list(c); weights=[c[x] for x in letters]
    def mutate(ws,side):
     r=random.Random(pubseed(cid,side)); return [''.join(r.choices(letters,weights=weights,k=len(w))) for w in ws]
   mf=mutate(basefit,'fit'); ma=mutate(baseaudit,'audit'); k=make_key(root,cid)
   add_case({'case_id':cid,'certificate':'C4b','family':fam,'work':'1507w-bovo.psd','length':512,'erasure':0.0,
             'fit_cipher':enc(mf,k,0.0,root,cid,'fit'),'audit_cipher':enc(ma,k,0.0,root,cid,'audit'),'lm_relabel':None},
            {'fit_truth':mf,'audit_truth':ma,'oracle':invert_permutation(k),'expected_mechanism_positive':True,'expected_language':'not_yiddish'})

 # Historical German diagnostic from official ReF diplomatic tokens.
 german_meta=[]
 for wid in GERMAN_IDS:
  gw,p=ref_words(args.ref_root,wid); german_meta.append({'work':wid,'words':len(gw),'source':str(p) if p else None,'sha256':sha_file(p) if p else None})
  s=split(gw,512)
  if not s: continue
  gf,ga=s
  for i in range(N_KEYS):
   cid=f'C4b|german|{wid}|{i:02d}'; k=make_key(root,cid)
   add_case({'case_id':cid,'certificate':'C4b','family':'historical_german','work':wid,'length':512,'erasure':0.0,
             'fit_cipher':enc(gf,k,0.0,root,cid,'fit'),'audit_cipher':enc(ga,k,0.0,root,cid,'audit'),'lm_relabel':None},
            {'fit_truth':gf,'audit_truth':ga,'oracle':invert_permutation(k),'expected_mechanism_positive':True,'expected_language':'german'})

 # C5 MR1: external ciphertext symbol labels must not affect the internal problem.
 for i in range(MR_N):
  cid0=f'C5|MR1|{i:02d}|base'; cid1=f'C5|MR1|{i:02d}|relabel'; k=make_key(root,'MR1',i)
  fc=enc(basefit,k,0.0,root,'MR1',i,'fit'); ac=enc(baseaudit,k,0.0,root,'MR1',i,'audit')
  rel=make_key(root,'MR1',i,'external_relabel')
  add_case({'case_id':cid0,'certificate':'C5','family':'MR1_global_cipher_relabel','pair_id':f'MR1|{i:02d}','variant':'base',
            'fit_cipher':fc,'audit_cipher':ac,'lm_relabel':None},
           {'fit_truth':basefit,'audit_truth':baseaudit,'oracle':invert_permutation(k),'expected_mechanism_positive':True,'expected_language':'yiddish'})
  # Relabel the ciphertext externally; truth plaintext is unchanged. Oracle is composed automatically by deriving from actual symbols.
  rfc=relabel_words(fc,rel); rac=relabel_words(ac,rel)
  # external symbol e=rel[c]; desired plaintext = oracle[c], so inv-rel composition:
  old_oracle=invert_permutation(k); new_oracle=[0]*A
  for oldc in range(A): new_oracle[rel[oldc]]=old_oracle[oldc]
  add_case({'case_id':cid1,'certificate':'C5','family':'MR1_global_cipher_relabel','pair_id':f'MR1|{i:02d}','variant':'relabel',
            'fit_cipher':rfc,'audit_cipher':rac,'lm_relabel':None},
           {'fit_truth':basefit,'audit_truth':baseaudit,'oracle':new_oracle,'expected_mechanism_positive':True,'expected_language':'yiddish'})

 # C5 MR2: rename plaintext alphabet AND language model consistently.
 for i in range(MR_N):
  pair=f'MR2|{i:02d}'; perm=make_key(root,'MR2',i,'plain_relabel')
  for variant in ('base','relabel'):
   if variant=='base': pf,pa,lmr=basefit,baseaudit,None
   else: pf,pa,lmr=relabel_words(basefit,perm),relabel_words(baseaudit,perm),perm
   cid=f'C5|MR2|{i:02d}|{variant}'; k=make_key(root,cid)
   add_case({'case_id':cid,'certificate':'C5','family':'MR2_plain_and_lm_relabel','pair_id':pair,'variant':variant,
             'fit_cipher':enc(pf,k,0.0,root,cid,'fit'),'audit_cipher':enc(pa,k,0.0,root,cid,'audit'),'lm_relabel':lmr},
            {'fit_truth':pf,'audit_truth':pa,'oracle':invert_permutation(k),'expected_mechanism_positive':True,'expected_language':'yiddish_relabelled' if variant=='relabel' else 'yiddish'})

 manifest={'version':VERSION,'created_before_solver_outcomes':True,'voynich_loaded':False,'target_access_allowed':False,
           'plant_root_commitment_sha256':commitment,'ppchy_commit':ppcommit,'representation':'Penn historical-Yiddish Romanisation -> literal a-z, secondary algorithm-control representation',
           'alphabet':''.join(ALPHABET),'buffer_words':BUFFER,'lengths':LENGTHS,'erasures':ERASURES,'keys_per_positive_cell':N_KEYS,
           'negative_trials_per_family':NEG_N,'metamorphic_reps':MR_N,'bonferroni_alpha':0.05/6,
           'source_meta':source_meta,'build_words':len(build),'build_development_overlap':overlaps,'german_meta':german_meta,
           'kat_rows':kats,'deterministic_metamorphic_rows':mrs,'case_counts':dict(case_count),
           'private_truth_withheld_from_solver':True,'c6_transfer':'NOT_RUN','c7_target':'SEALED'}
 write_json(manifest,pub/'pre_manifest.json'); (priv/'plant_root_reveal.hex').write_text(root.hex()+'\n')
 # Hash public evidence after generation; truth hash is private until scoring.
 public_hashes={p.name:sha_file(p) for p in pub.iterdir() if p.is_file()}; private_hashes={p.name:sha_file(p) for p in priv.iterdir() if p.is_file()}
 write_json({'public_hashes':public_hashes,'private_hashes':private_hashes},priv/'bundle_hashes.json')
 print(json.dumps({'status':'PREPARED_BLINDED_CASES','case_counts':dict(case_count),'public_hashes':public_hashes,'root_commitment':commitment},indent=2))

if __name__=='__main__': main()

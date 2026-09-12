#!/usr/bin/env python3
"""Prepare frozen C6 late-normalized transfer cases on fresh 1783 Ukraine-1.

Public/private truth separation matches the full-control architecture. No Voynich data.
"""
from __future__ import annotations
import argparse, hashlib, hmac, json, random, secrets, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
V1B=REPO/'experiments'/'yiddish_qualification_v1b'
sys.path.insert(0,str(V1B))
import finite_panel_r_v02 as base  # noqa:E402
from encoder_v02 import encode_words  # noqa:E402
from independent_decoder_v02 import invert_permutation  # noqa:E402

VERSION='yiddish_c6_late_normalized_v01_20260912'
SOURCE='1783e-ukraine-1.psd'
A=base.A
BUFFER=32
N=512
N_KEYS=32
ERASURES=(0.0,0.01)
YID_BUILD=(
 '1600e-magid-preface.psd','1600e-magid.psd','1600e-tsenerene.psd',
 '1619w-letters-prague.psd','1624e-magen.psd','1671e-vaad.psd',
 '1677w-witzenhausen.psd','1692e-vilna.psd','1704e-ellush.psd',
 '1705w-glikl.psd','1712e-sarah.psd','1716e-duties.psd')

def sha_file(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def sha_bytes(b:bytes): return hashlib.sha256(b).hexdigest()
def seed(root:bytes,*parts): return int.from_bytes(hmac.new(root,'|'.join(map(str,parts)).encode(),hashlib.sha256).digest()[:8],'big')
def write_json(x,p): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def append_jsonl(x,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:f.write(json.dumps(x,sort_keys=True)+'\n')
def ngrams(words,n): return {tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1))}
def make_key(root,*parts):
 r=random.Random(seed(root,'key',*parts)); p=list(range(A)); r.shuffle(p); return p
def enc(words,key,er,root,*parts):
 r=random.Random(seed(root,'erase',*parts,er)); return encode_words(words,key,er,r)[0]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
 pub=a.out/'public'; priv=a.out/'private'; work=a.out/'work'; pub.mkdir(parents=True,exist_ok=True); priv.mkdir(parents=True,exist_ok=True); work.mkdir(parents=True,exist_ok=True)
 repo,ppcommit=base.ensure_corpus(work/'sources'); data=repo/'data'
 root=secrets.token_bytes(32); commitment=sha_bytes(root)
 build=[]; build_hashes={}
 for name in YID_BUILD:
  p=data/name; ws=base.extract_words(p); build.extend(ws); build_hashes[name]=sha_file(p)
 if len(build)<10000: raise RuntimeError('BUILD too small')
 sourcep=data/SOURCE; sw1=base.extract_words(sourcep); sw2=base.extract_words(sourcep)
 if sw1!=sw2: raise RuntimeError('nondeterministic source extraction')
 quantity=len(sw1)
 b8=ngrams(build,8); b5=ngrams(build,5); s8=ngrams(sw1,8); s5=ngrams(sw1,5)
 overlap8=len(b8&s8); overlap5=len(b5&s5)
 pre_status='READY'
 if overlap8: pre_status='CONTAMINATION_FAIL'
 elif quantity < 2*N+BUFFER: pre_status='CORPUS_LIMITED'
 manifest={
  'version':VERSION,'source':SOURCE,'source_sha256':sha_file(sourcep),'source_words':quantity,
  'ppchy_commit':ppcommit,'build_words':len(build),'build_hashes':build_hashes,
  'shared_8gram_types_build_vs_source':overlap8,'shared_5gram_types_build_vs_source':overlap5,
  'plant_root_commitment_sha256':commitment,'private_truth_withheld_from_solver':True,
  'voynich_loaded':False,'target_access_allowed':False,'preflight_status':pre_status,
  'registered_cells':[{'length':N,'erasure':er,'keys':N_KEYS} for er in ERASURES],
  'thresholds':{'atom_recovery':0.90,'word_recovery':0.80,'cell_successes_required':29}
 }
 write_json(manifest,pub/'pre_manifest.json'); write_json(build,pub/'training_words.json')
 (priv/'plant_root_reveal.hex').write_text(root.hex()+'\n',encoding='utf-8')
 if pre_status!='READY':
  print(json.dumps({'status':pre_status,**manifest},indent=2)); return
 fit=sw1[:N]; audit=sw1[N+BUFFER:N+BUFFER+N]
 cases=pub/'cases.jsonl'; truth=priv/'truth.jsonl'
 for er in ERASURES:
  for i in range(N_KEYS):
   cid=f'C6LATE|{SOURCE}|{N}|{er:.3f}|{i:02d}'; key=make_key(root,cid)
   append_jsonl({'case_id':cid,'certificate':'C6','family':'late_normalized_yiddish_transfer','work':SOURCE,'length':N,'erasure':er,
                 'fit_cipher':enc(fit,key,er,root,cid,'fit'),'audit_cipher':enc(audit,key,er,root,cid,'audit'),'lm_relabel':None},cases)
   append_jsonl({'case_id':cid,'fit_truth':fit,'audit_truth':audit,'oracle':invert_permutation(key),'expected_mechanism_positive':True,'expected_language':'yiddish'},truth)
 print(json.dumps({'status':'C6_LATE_PREPARED','source':SOURCE,'source_words':quantity,'cases':2*N_KEYS,'overlap8':overlap8,'overlap5':overlap5,'commitment':commitment},indent=2))
if __name__=='__main__':main()

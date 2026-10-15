from pathlib import Path
import json, hashlib, collections
P=Path(__file__).resolve().parent
m=json.loads((P/'04_pair_bridge_manifest.json').read_text()); R=P/'VMS_B_REVA_reference.txt'
words=[w for line in R.read_text().splitlines() if '\t' in line for w in line.split('\t',1)[1].split()]
tri={w[i:i+3] for w in words for i in range(len(w)-2)}
bi={w[i:i+2] for w in words for i in range(len(w)-1)}
units=m['unit_list'];alpha='aeiou'
state={};badInternal={};code={};bridges=m['bridge_for_vowel_and_next_key']
for v in alpha:
 st=m['state'][v];cs=st['cells'];u=[]
 for i in range(len(units)):
  chunks=[cs[i]] if i<st['direct'] else [st['escape'],cs[i-st['direct']]]
  c=''.join(chunks)
  u.append(c)
 code[v]=u; badInternal[v]=sum(any(c[j:j+3] not in tri for j in range(len(c)-2)) for c in u)

def ok_concat(prev,v,next_):
 br=bridges[v][next_[0]]; c=prev+br+next_
 # Only ngrams straddling previously separate components can fail if interior valid
 n=len(prev)
 return all(c[i:i+3] in tri for i in range(max(0,n-2),min(len(c)-2,n+len(br)+1))) and all(c[i:i+2] in bi for i in range(max(0,n-1),min(len(c)-1,n+len(br))))
by={}
for pv in alpha:
 for nv in alpha:
  a=code[pv];b=code[nv];total=0;valid=0;out=0;ins=[False]*len(b)
  for w in a:
   local=0
   for j,x in enumerate(b):
    total+=1
    if ok_concat(w,nv,x):valid+=1;local+=1;ins[j]=True
   if local:out+=1
  by[pv+'>'+nv]={'valid_pairs':valid,'pairs':total,'percent':100*valid/total,'preceding_units_with_any_valid_continuation':out,'next_units_with_valid_predecessor':sum(ins),'total_units':len(b)}
res={'note':'Necessary local support screen for the frozen source-unit table, not a language or cipher test. Edge legality tested with original B REVA reference and bridge determined by following cell key. Some length-two escape codewords may already be internally illegal.','unattested_inside_source_unit_codeword':badInternal,'by_state_transition':by,'source':str(R)}
out=P/'VBM_v142b_Transition_Capacity_2026-10-14.json';out.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

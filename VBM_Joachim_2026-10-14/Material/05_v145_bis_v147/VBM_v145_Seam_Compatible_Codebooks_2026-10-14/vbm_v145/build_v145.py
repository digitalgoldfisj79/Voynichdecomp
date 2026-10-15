#!/usr/bin/env python3
"""VBM v145 prefix-code with observed local joins, independently decodable.
Engineering feasibility only: VMS support is used to BUILD codebooks, not held out.
"""
import sys,json,collections,re,hashlib,itertools,heapq,time
from pathlib import Path
P=Path(__file__).resolve().parent
B=(Path(__file__).resolve().parent/'inputs')
M=json.loads((B/'04_pair_bridge_manifest.json').read_text());U=M['unit_list']; assert len(U)==222
# Import only data and helper definitions, no side-effect import.
D=json.loads((P/'screen.json').read_text())
SLIM=json.loads((Path(__file__).resolve().parent/'inputs'/'voynich_transcriptions_slim.json').read_text())['pages']
B_PAGES=set('f26r f26v f31r f31v f33r f33v f34r f34v f39r f39v f40r f40v f41r f41v f43r f43v f46r f46v f48r f48v f50r f50v f55r f55v f57r f66r f66v f75r f75v f76r f76v f77r f77v f78r f78v f79r f79v f80r f80v f81r f81v f82r f82v f83r f83v f84r f84v f85r1 f85r2 fRos f86v4 f86v6 f86v5 f86v3 f94r f94v f95r1 f95r2 f95v2 f95v1 f103r f103v f104r f104v f105r f105v f106r f106v f107r f107v f108r f108v f111r f111v f112r f112v f113r f113v f114r f114v f115r f115v f116r'.split())
OTHER_PAGES=set('f57v f65r f65v f67r1 f67r2 f67v2 f67v1 f68r1 f68r2 f68r3 f68v3 f68v2 f68v1 f69r f69v f70r1 f70r2 f70v2 f70v1 f71r f71v f72r1 f72r2 f72r3 f72v3 f72v2 f72v1 f73r f73v f116v'.split())

def words_A():
 for fol,loci in SLIM.items():
  if fol in B_PAGES or fol in OTHER_PAGES:continue
  for p,record in loci.items():
   if not re.match(r'[@*+=~]P',record.get('u','')):continue
   raw=record.get('t',{}).get('ZLZB','')
   for word in raw.split():
    if re.fullmatch('[a-z]+',word):
     yield ''.join(reversed(re.findall('cth|ckh|cph|cfh|ch|sh|.',word)))

def words_B():
 for line in (Path(__file__).resolve().parent/'inputs'/'VMS_B_REVA_reference.txt').read_text().splitlines():
  if '\t' in line:yield from line.split('\t',1)[1].split()

def support(words):
 tri,big=set(),set()
 for w in words:
  for i in range(len(w)-2):tri.add(w[i:i+3])
  for i in range(len(w)-1):big.add(w[i:i+2])
 return tri,big

def newmanifest(classname,tri,big):
 def ok(s):return all(s[i:i+3] in tri for i in range(len(s)-2)) and all(s[i:i+2] in big for i in range(len(s)-1))
 allstate={};counts={}
 for vowel,vinfo in M['state'].items():
  cells=vinfo['cells'];br=M['bridge_for_vowel_and_next_key'][vowel]
  good=[c for c in cells if ok(br[c[0]]+c)]
  # one or two escapes are sufficient to exceed 222 candidates in this exact inventory.
  escoptions=[]
  for e in good:
   cont=[c for c in cells if ok(e+c)]
   escoptions.append((e,cont))
  one=max(escoptions,key=lambda x:len(good)-1+len(x[1])) if escoptions else (None,[])
  k=1 if len(good)-1+len(one[1])>=222 else 2
  # optimise capacity per escape; when two, choose disjoint prefixes from native leaf set
  choices=sorted(escoptions,key=lambda ec:(-len(ec[1]),len(ec[0]),ec[0]))[:k]
  assert len(choices)==k
  escapes=[e for e,cc in choices]
  codewords=[c for c in good if c not in escapes]
  for e,cont in choices: codewords.extend(e+c for c in cont)
  assert len(set(codewords))==len(codewords)
  assert len(codewords)>=len(U),(classname,vowel,k,len(codewords))
  # choose shortest permissible codewords first; exact 222 codes, other candidates unused
  codewords=sorted(codewords,key=lambda x:(len(x),x))[:len(U)]
  assert len(codewords)==222 and len(set(codewords))==222
  # Ensure selected codewords are all prefix-free.
  trie={}
  for c in codewords:
   node=trie
   for ch in c:
    assert '' not in node,(classname,vowel,'prefix',c)
    node=node.setdefault(ch,{})
   assert not node and '' not in node,(classname,vowel,'prefix collision',c)
   node['']=True
  for c in codewords:
   stem=next((e for e in escapes if c.startswith(e)),None)
   if stem is None:assert ok(br[c[0]]+c)
   else:assert ok(br[c[0]]+c)
  allstate[vowel]={'unitcode':codewords,'escapes':escapes}
  counts[vowel]={'direct_bridge_safe':len(good),'escapes_needed':k,'escape_firstcells':escapes,'admissible_codewords_before_truncate':len(good)-k+sum(len(cc) for _,cc in choices),'chosen_two_cell':sum(any(c.startswith(e) for e in escapes) for c in codewords),'mean_code_length':sum(map(len,codewords))/222}
 return {'format':'vbm145_seamlocal_prefixfree','source':'v142b fixed unit inventory and B-fitted cells','observed_reference':classname,'bridge_for_vowel_and_next_key':M['bridge_for_vowel_and_next_key'], 'vowel_by_pair':M['vowel_by_pair'],'initial_state':'e','unit_list':U,'state':allstate},counts

def runs(w):
 i=0
 while i<len(w) and w[i] not in 'aeiou':i+=1
 lead=w[:i];output=[]
 while i<len(w):
  v=w[i];j=i+1
  while j<len(w) and w[j] not in 'aeiou':j+=1
  output.append((v,w[i+1:j]));i=j
 return lead,output

def pieces(s):
 a=[]
 for i in range(0,len(s),2):
  p=s[i:i+2];a.extend([p] if p in U else list(p))
 assert ''.join(a)==s
 return a

def encode(w,m,lookup):
 lead,r=runs(w);o=''.join(lookup['e'][u] for u in pieces(lead))
 for v,run in r:
  parts=pieces(run) if run else [''];first=lookup[v][parts[0]]
  o+=m['bridge_for_vowel_and_next_key'][v][first[0]]+first
  for u in parts[1:]:o+=lookup[v][u]
 assert o;return o

def decoder(cwords,m):
 # Independent parsing from declared codeword list, not source unit indices or source vocabulary.
 tries={}
 for v,codes in cwords.items():
  t={}
  for i,c in enumerate(codes):
   n=t
   for ch in c:
    if 'unit' in n:raise ValueError('prefix problem')
    n=n.setdefault(ch,{})
   if n:raise ValueError('prefix problem')
   n['unit']=i
  tries[v]=t
 pairs={tuple(k.split('|')):v for k,v in m['vowel_by_pair'].items()}
 def decode_word(w):
  s='e';i=0;got='';need_first=False;after_epsilon=False
  while i<len(w):
   if w[i] not in 'ynr':
    if w.startswith('ch',i) or w.startswith('sh',i):br=w[i:i+2];i+=2
    else:br=w[i];i+=1
    if i>=len(w):raise ValueError('dangling bridge')
    state=pairs.get((br,w[i]));
    if not state:raise ValueError((br,w[i]))
    s=state;got+=s;need_first=True;after_epsilon=False
   else:
    if after_epsilon:raise ValueError('cell after epsilon')
    n=tries[s];old=i
    while 'unit' not in n:
     if i>=len(w) or w[i] not in n:raise ValueError(('invalid cell',w,old,i,s))
     n=n[w[i]];i+=1
    unit=m['unit_list'][n['unit']]
    if unit=='':
     if not need_first:raise ValueError('epsilon not after bridge')
     after_epsilon=True
    else:got+=unit
    need_first=False
  if not got or need_first:raise ValueError('bad word end')
  return got
 return decode_word

def main():
 result={};root=(Path(__file__).resolve().parent/'inputs')
 for cls,data in [('A',words_A()),('B',words_B())]:
  tri,big=support(data)
  mf,cnt=newmanifest(cls,tri,big)
  (P/f'v145_{cls}_manifest.json').write_text(json.dumps(mf,indent=2))
  lookup={v:dict(zip(U,mf['state'][v]['unitcode'])) for v in 'aeiou'}
  inverse=decoder({v:mf['state'][v]['unitcode'] for v in 'aeiou'},mf)
  emit={};
  for source in ['train','hartlieb']:
   words=(root/f'{source}.reference.txt').read_text().splitlines()
   ciphertext=[];firstbad=0;badtri=0;inside=0;length=0;badbi=0
   for idx,w in enumerate(words):
    c=encode(w,mf,lookup);ciphertext.append(c); length+=len(c)
    if inverse(c)!=w:raise ValueError((cls,source,idx,w,c,inverse(c)))
    if any(c[i:i+3] not in tri for i in range(len(c)-2)):firstbad+=1
    badtri+=sum(c[i:i+3] not in tri for i in range(len(c)-2))
    badbi+=sum(c[i:i+2] not in big for i in range(len(c)-1))
   (P/f'{cls}_{source}.vbm').write_text(' '.join(ciphertext)+'\n')
   emit[source]={'source_words':len(words),'decoded_exact':True,'word_invalid_trigrams':firstbad,'invalid_trigram_occurrences':badtri,'invalid_bigrams':badbi,'mean_cipher_length':length/len(words),'sha256':hashlib.sha256((' '.join(ciphertext)+'\n').encode()).hexdigest()}
  result[cls]={'training_support_trigrams':len(tri),'counts':cnt,'emissions':emit}
  print(cls,'built and roundtripped',emit,flush=True)
 (P/'v145_results.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()

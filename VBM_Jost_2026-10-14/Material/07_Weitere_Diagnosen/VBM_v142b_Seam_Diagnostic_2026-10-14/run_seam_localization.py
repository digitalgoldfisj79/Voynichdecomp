from pathlib import Path
import json, hashlib, collections, statistics
P=Path(__file__).resolve().parent
R=P/'VMS_B_REVA_reference.txt'
m=json.loads((P/'04_pair_bridge_manifest.json').read_text())
units=m['unit_list']; rank={u:i for i,u in enumerate(units)}
vows='aeiou'
wordsref=[]
for line in R.read_text().splitlines():
    if '\t' in line: wordsref.extend(line.split('\t',1)[1].split())
assert len(wordsref)==21679, len(wordsref)
support={n:set(g for w in wordsref for g in (w[i:i+n] for i in range(len(w)-n+1))) for n in [2,3]}
def runs(w):
 i=0
 while i<len(w) and w[i] not in vows:i+=1
 lead=w[:i];out=[]
 while i<len(w):
  v=w[i];j=i+1
  while j<len(w) and w[j] not in vows:j+=1
  out.append((v,w[i+1:j]));i=j
 return lead,out
def pieces(s):
 a=[]
 for i in range(0,len(s),2):
  p=s[i:i+2]
  a.extend([p] if p in rank else list(p))
 assert ''.join(a)==s
 return a
def cell(v,u):
 ix=rank[u];st=m['state'][v];cc=st['cells']
 return cc[ix] if ix<st['direct'] else st['escape']+cc[ix-st['direct']]
def segments(word):
 lead,rr=runs(word);o=[]
 for u in pieces(lead):o.append((cell('e',u),'leadcell'))
 for v,run in rr:
  ps=pieces(run) if run else ['']
  first=cell(v,ps[0])
  br=m['bridge_for_vowel_and_next_key'][v][first[0]]
  o.append((br,'bridge'))
  o.append((first,'vcell'))
  for u in ps[1:]:o.append((cell(v,u),'vcell'))
 return o
records={}
for src in ['train','hartlieb']:
 source=(P/(src+'.reference.txt')).read_text().splitlines()
 target=(P/(src+'.pair.vbm')).read_text().split()
 assert len(source)==len(target),(src,len(source),len(target))
 outcomes=collections.Counter();trigram_viol=collections.Counter();bigram_viol=collections.Counter();bad_by_segment=collections.Counter();sumcounts=collections.Counter()
 for i,(w,t) in enumerate(zip(source,target)):
  seg=segments(w)
  assert ''.join(x for x,_ in seg)==t,(src,i,w,t,seg)
  chars=''.join(x for x,_ in seg)
  char_to_segment=[]
  for segid,(strg,kind) in enumerate(seg):char_to_segment.extend([(segid,kind)]*len(strg))
  assert len(char_to_segment)==len(t)
  bad_here=False;bad_internal=False;bad_seam=False
  seen_labels=set()
  for n,track in [(2,bigram_viol),(3,trigram_viol)]:
   for off in range(len(t)-n+1):
    g=t[off:off+n]
    if g in support[n]:continue
    lo=char_to_segment[off:off+n]
    ids={z[0] for z in lo}
    typ='single_segment_'+lo[0][1] if len(ids)==1 else 'multi_segment_'+''.join(sorted(set(z[1] for z in lo)))
    track[typ]+=1
    track['TOTAL']+=1
    if n==3:
     bad_here=True;seen_labels.add(typ)
     if len(ids)==1: bad_internal=True
     else: bad_seam=True
  outcomes['tokens']+=1
  if bad_here:outcomes['bad_any']+=1
  if bad_internal:outcomes['bad_internal']+=1
  if bad_seam:outcomes['bad_seam']+=1
  if bad_internal and bad_seam:outcomes['bad_both']+=1
  for label in seen_labels: bad_by_segment[label]+=1
 records[src]={'counts':dict(outcomes),'trigram_invalid_instances_by_location':dict(trigram_viol),'bigram_invalid_instances_by_location':dict(bigram_viol),'violating_words_by_location':dict(bad_by_segment),'input_sha256':hashlib.sha256((P/(src+'.reference.txt')).read_bytes()).hexdigest(),'cipher_sha256':hashlib.sha256((P/(src+'.pair.vbm')).read_bytes()).hexdigest()}
res={'reference_words':len(wordsref),'reference_trigrams':len(support[3]),'reference_bigrams':len(support[2]),'frozen_code_sha256':hashlib.sha256((P/'04_pair_bridge_manifest.json').read_bytes()).hexdigest(),'measurement':'unseen character n-grams: entirely within emitted segment vs cross one-or-more segment boundaries, test all 81628 words exact reconstruction','source':'author supplied Currier B REVA, frozen v142b','results':records}
out=P/'VBM_v142b_Seam_Attribution_2026-10-14.json';out.write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))

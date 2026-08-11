# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import collections,hashlib,json,re,sys
sys.path.insert(0,'experiments/amadi_residuals_v1')
import amadi_residuals_v1 as ar
ar.HEADERS={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36','Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8','Accept-Language':'en-GB,en;q=0.9','Referer':'https://www.voynich.nu/transcr.html'}
NS='VBMV1'
BAB_C1=['f31v','f10r','f85r1','f53v','f23r','f28v','f81v','f33r','f34r','f111r','f5r','f88r']
H1=['f28v','f31v','f88r','f5r','f34r','f81v']; C1=['f85r1','f53v','f33r','f10r','f23r','f111r']

def raw_lines():
 b=ar.getb(ar.RF_URL,ar.HEADERS);assert hashlib.sha256(b).hexdigest()==ar.RF_SHA
 out=collections.defaultdict(list);meta={'raw_segments':0,'kept_segments':0,'words':0,'events':0,'singletons':0}
 for line in b.decode('utf-8','replace').splitlines():
  if not line.startswith('<') or '>' not in line:continue
  lab,rhs=line.split('>',1)
  if '.' not in lab or '<!' in rhs:continue
  pg=lab[1:].split('.',1)[0];rhs=re.sub(r'<(?:-|~)>','.',rhs);rhs=re.sub(r'<[^>]*>','.',rhs);rhs=rhs.replace(',','')
  seg=[]
  def flush():
   nonlocal seg
   if len(seg)>=2:out[pg].append(seg);meta['kept_segments']+=1;meta['words']+=len(seg)
   seg=[]
  for rw in rhs.split('.'):
   rw=rw.strip()
   if not rw:flush();continue
   meta['raw_segments']+=1
   if '[' in rw or ']' in rw or '?' in rw:flush();continue
   ch=[c for c in rw.replace('{','').replace('}','').lower() if 'a'<=c<='z']
   if not ch or any(c not in ar.S2I for c in ch) or len(ch)<2:
    if len(ch)==1:meta['singletons']+=1
    flush();continue
   seg.append(ch)
  flush()
 return dict(out),meta

def events(lines,folios,vocab=None):
 BC=collections.Counter();CC=collections.Counter();seqs=[];rawbr=keepbr=0
 for f in folios:
  for seg in lines.get(f,[]):
   ev=[];ok=True
   for wi,w in enumerate(seg):
    for c in w[1:-1]:ev.append(('C',c));CC[c]+=1
    if wi+1<len(seg):
     b=w[-1]+seg[wi+1][0];BC[b]+=1;rawbr+=1
     if vocab is not None and b not in vocab:ok=False;break
     ev.append(('B',b));keepbr+=1
   if ok and ev:seqs.append(ev)
 return seqs,BC,CC,{'segments':len(seqs),'bridge_events':rawbr,'kept_bridge_events':keepbr,'event_count':sum(len(x) for x in seqs)}

def main():
 pages,_=ar.parse_rf();T,H,priorC,H2,C2=ar.target_split(pages);FIT=T+H
 lines,lmeta=raw_lines();_,bc,cc,fmeta=events(lines,FIT)
 order=sorted(bc,key=lambda x:(-bc[x],x));tot=sum(bc.values());cum=0;kb=None
 for i,x in enumerate(order,1):
  cum+=bc[x]
  if cum/max(1,tot)>=.995:kb=i;break
 vocab=set(order[:kb]);fs,_,_,fm=events(lines,FIT,vocab);hs,_,_,hm=events(lines,H1,vocab)
 top=[(x,bc[x]) for x in order[:20]]
 out={'namespace':NS,'FIT_folios':len(FIT),'VBM_H1':H1,'VBM_C1':C1,'C1_opened':False,'bridge_K_995':kb,'bridge_vocab':order[:kb],'bridge_top20':top,'fit':fm,'H1_coverage_census':hm,'line_meta':lmeta,'core_surface_types':len(cc),'core_counts':cc}
 print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()

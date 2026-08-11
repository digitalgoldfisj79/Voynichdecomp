# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import json,sys,numpy as np
sys.path.insert(0,'experiments/amadi_residuals_v1');sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
import amadi_residuals_v1 as ar
import vbm_structure_v1 as v
ar.HEADERS={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36','Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8','Accept-Language':'en-GB,en;q=0.9','Referer':'https://www.voynich.nu/transcr.html'}

def train_ng(seqs,order):
 B=2;shape=(3,)*(order+1);C=np.full(shape,.25,float)
 for s in seqs:
  q=[B]*order+[1 if x=='V' else 0 for x in v.cv(s)]+[B]
  for i in range(order,len(q)):
   idx=tuple(q[i-order:i+1]);C[idx]+=1
 # normalize over final axis
 C/=C.sum(axis=-1,keepdims=True);return np.log(C)
def score(seq,lp,order):
 B=2;q=[B]*order+[1 if x=='V' else 0 for x in seq]+[B];z=0.;n=0
 for i in range(order,len(q)):
  z+=lp[tuple(q[i-order:i+1])];n+=1
 return z/max(1,n)
def vf(seqs):
 s=''.join(v.cv(x) for x in seqs);return s.count('V')/max(1,len(s))
def main():
 pages,_=ar.parse_rf();T,H,_,_,_=ar.target_split(pages);lines=v.raw_lines();hfolios=v.H1
 cs=v.corpora();out={'H1_folios':hfolios,'orders':{},'vowel_fraction':{'H1':None,'languages':{}},'per_folio_order4':{}}
 hseq,hm=v.vbm_types(lines,hfolios);H=''.join(hseq);out['vowel_fraction']['H1']=hm['vowel_event_fraction']
 for la,(tr,ct) in cs.items():out['vowel_fraction']['languages'][la]={'train':vf(tr),'control':vf(ct)}
 for order in [0,1,2,3]:
  rows=[]
  for la,(tr,ct) in cs.items():
   lp=train_ng(tr,order);rows.append((la,score(H,lp,order)))
  rows.sort(key=lambda x:-x[1]);out['orders'][str(order+1)]={'ranking':rows,'top_margin':rows[0][1]-rows[1][1]}
 lps={la:train_ng(tr,3) for la,(tr,ct) in cs.items()}
 for f in hfolios:
  sq,meta=v.vbm_types(lines,[f]);s=''.join(sq);rows=sorted([(la,score(s,lp,3)) for la,lp in lps.items()],key=lambda x:-x[1]);out['per_folio_order4'][f]={'events':meta['events'],'ranking':rows,'winner':rows[0][0],'margin':rows[0][1]-rows[1][1]}
 print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()

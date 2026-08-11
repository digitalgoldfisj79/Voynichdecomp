# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import collections,json
import sys
sys.path.insert(0,'experiments/amadi_residuals_v1')
import amadi_residuals_v1 as ar
ar.HEADERS={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36','Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8','Accept-Language':'en-GB,en;q=0.9','Referer':'https://www.voynich.nu/transcr.html'}

def main():
 pages,meta=ar.parse_rf();T,H,C1,H2,C2=ar.target_split(pages);fit=T+H
 C=collections.Counter(tuple(w) for f in fit for w in pages[f]);ordered=sorted(C,key=lambda x:(-C[x],x));tot=sum(C.values());k=1365;cov=sum(C[x] for x in ordered[:k])/tot
 out={'fit_folios':len(fit),'token_events':tot,'distinct_visible_word_types':len(C),'expanded_capacity':1365,'top1365_token_coverage':cov,'exact_capacity_pass':len(C)<=1365,'coverage_gate_0_995_pass':cov>=.995,'source':meta}
 print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()

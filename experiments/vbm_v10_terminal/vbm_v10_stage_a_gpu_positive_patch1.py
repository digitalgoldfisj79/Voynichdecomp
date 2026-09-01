# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
"""Execution-only patch for the frozen V10 Stage-A GPU positive runner.

The first launch failed at Triton compilation before any candidate score was
computed. The event surface id is scalar per line/event while the candidate
mask is vector-valued, so bpos/bmap/npos/nmap loads must use scalar event masks.
No candidate, objective, block, chain, gate, seed, or data rule changes.
"""
import urllib.request
URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v10-terminal-identifiability-20260901/experiments/vbm_v10_terminal/vbm_v10_stage_a_gpu_positive.py'
with urllib.request.urlopen(URL,timeout=120) as r: src=r.read().decode('utf-8')
repls={
"bp=tl.load(bpos+surf,mask=bm,other=-1).to(tl.int32)":"bp=tl.load(bpos+surf,mask=em & (typ==1),other=-1).to(tl.int32)",
"bv0=tl.load(bmap+surf,mask=bm,other=0).to(tl.int64)":"bv0=tl.load(bmap+surf,mask=em & (typ==1),other=0).to(tl.int64)",
"np=tl.load(npos+surf,mask=nm,other=-1).to(tl.int32)":"np=tl.load(npos+surf,mask=em & (typ==2),other=-1).to(tl.int32)",
"nv0=tl.load(nmap+surf,mask=nm,other=0).to(tl.int64)":"nv0=tl.load(nmap+surf,mask=em & (typ==2),other=0).to(tl.int64)",
}
for old,new in repls.items():
    if old not in src: raise RuntimeError('expected frozen source fragment missing: '+old)
    src=src.replace(old,new,1)
exec(compile(src,URL,'exec'),{'__name__':'__main__'})

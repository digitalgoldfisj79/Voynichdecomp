# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
"""Execution-only patch 2 for frozen V10 GPU runner.

Patch 1 correctly fixed scalar Triton load masks, but Triton requires @jit
functions to be backed by a physical Python source file and rejects dynamic
exec(). This wrapper applies the same four source substitutions, writes the
corrected runner to /tmp, and executes that file via runpy. Scientific logic
is unchanged and no candidate score was emitted by either failed launch.
"""
import runpy, urllib.request
from pathlib import Path
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
p=Path('/tmp/vbm_v10_stage_a_gpu_positive_fixed.py');p.write_text(src,encoding='utf-8')
runpy.run_path(str(p),run_name='__main__')

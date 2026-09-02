# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2", "triton>=3.0"]
# ///
"""Execution-only patch 1 for frozen V10 evolutionary runner.

The first smoke launch failed during Triton compilation before any scientific
candidate score was emitted because Triton disallowed the Python global KB in
kernel pointer arithmetic. Replace that compile-time global with its frozen
literal value 30. No data, key, objective, seed, population, mutation, chain,
gate, or scientific parameter changes.
"""
import runpy, urllib.request
from pathlib import Path
URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v10-terminal-identifiability-20260901/experiments/vbm_v10_terminal/vbm_v10_stage_a_gpu_evolution.py'
with urllib.request.urlopen(URL,timeout=120) as r: src=r.read().decode('utf-8')
old='maps+offs*STRIDE+KB+surf'
new='maps+offs*STRIDE+30+surf'
if old not in src: raise RuntimeError('expected frozen source fragment missing')
src=src.replace(old,new,1)
p=Path('/tmp/vbm_v10_stage_a_gpu_evolution_fixed.py'); p.write_text(src,encoding='utf-8')
runpy.run_path(str(p),run_name='__main__')

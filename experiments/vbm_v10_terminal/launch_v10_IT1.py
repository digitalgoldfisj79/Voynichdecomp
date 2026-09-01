# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
import runpy, sys, urllib.request
from pathlib import Path
URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v10-terminal-identifiability-20260901/experiments/vbm_v10_terminal/vbm_v10_stage_a_gpu_positive_patch3.py'
with urllib.request.urlopen(URL,timeout=120) as r: src=r.read().decode('utf-8')
p=Path('/tmp/v10_patch3.py'); p.write_text(src,encoding='utf-8')
sys.argv=['v10','--lang','IT','--rep','1','--sizes','100,250,500,1000,2000']
runpy.run_path(str(p),run_name='__main__')

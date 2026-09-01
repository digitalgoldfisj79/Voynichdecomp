# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
import subprocess, sys, urllib.request, time
URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v10-terminal-identifiability-20260901/experiments/vbm_v10_terminal/vbm_v10_stage_a_positive.py'
PATH='/tmp/vbm_v10_stage_a_positive.py'
with urllib.request.urlopen(URL,timeout=120) as r: open(PATH,'wb').write(r.read())
t0=time.time()
cmd=[sys.executable,PATH,'--lang','DE','--rep','0','--size','2000']
print('BENCH_START',flush=True)
subprocess.run(cmd,check=True)
print('BENCH_WALL_SECONDS',time.time()-t0,flush=True)

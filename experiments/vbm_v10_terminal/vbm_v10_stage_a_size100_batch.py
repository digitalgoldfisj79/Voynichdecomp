# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
import sys, urllib.request
URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v10-terminal-identifiability-20260901/experiments/vbm_v10_terminal/vbm_v10_stage_a_positive.py'
with urllib.request.urlopen(URL,timeout=120) as r: src=r.read().decode('utf-8')
ns={'__name__':'v10mod'}
exec(compile(src,URL,'exec'),ns)
for lang in ['DE','IT']:
    for rep in [0,1,2]:
        sys.argv=['v10','--lang',lang,'--rep',str(rep),'--size','100']
        ns['main']()

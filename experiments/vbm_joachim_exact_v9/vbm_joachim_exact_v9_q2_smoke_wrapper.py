# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
import sys, urllib.request
URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-joachim-exact-v9-20260901/experiments/vbm_joachim_exact_v9/vbm_joachim_exact_v9_q2_synth_ident.py'
with urllib.request.urlopen(URL,timeout=120) as r: src=r.read().decode('utf-8')
sys.argv=['q2','--mode','smoke']
exec(compile(src,URL,'exec'),{'__name__':'__main__'})

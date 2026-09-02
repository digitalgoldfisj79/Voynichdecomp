# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "scipy>=1.13,<2", "scikit-learn>=1.5,<2", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
import urllib.request
B='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-structural-constraints-v11-20260902/experiments/vbm_v11_structural/'
src=''
for f in ['vbm_v11_structural_part1.py','vbm_v11_structural_part2.py']:
    with urllib.request.urlopen(B+f,timeout=120) as r:
        src += r.read().decode('utf-8') + '\n'
g={'__name__':'__main__'}
exec(compile(src,'vbm_v11_structural_combined.py','exec'),g)

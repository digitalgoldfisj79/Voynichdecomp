# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
"""Execution-only patch for Q1 fresh-fit audit.

The frozen first runner computed all primary line scores but failed at the
structural-shuffle null because rows retained summary counts rather than the
parsed nucleus/bridge sequences. This wrapper changes only retained audit
metadata; parsing, candidate banks, sampling, fit tests, thresholds and all
statistics are unchanged.
"""
import urllib.request

URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-joachim-exact-v9-20260901/experiments/vbm_joachim_exact_v9/vbm_joachim_exact_v9_q1_freshfit.py'
with urllib.request.urlopen(URL,timeout=120) as r:
    src=r.read().decode('utf-8')
old="rows.append({'folio':r['folio'],'line':r['line'],'key':r['key'],'B':r['B'],'tokens':len(r['tokens']),'unique_nuclei':len(set(x for x in r['nuclei'] if x)),'unique_bridges':len(set(r['bridges'])),'scores':scores})"
new="rows.append({'folio':r['folio'],'line':r['line'],'key':r['key'],'B':r['B'],'tokens':len(r['tokens']),'nuclei':list(r['nuclei']),'bridges':list(r['bridges']),'unique_nuclei':len(set(x for x in r['nuclei'] if x)),'unique_bridges':len(set(r['bridges'])),'scores':scores})"
if old not in src:
    raise RuntimeError('expected frozen runner line not found; refusing broad patch')
src=src.replace(old,new,1)
exec(compile(src,URL,'exec'),{'__name__':'__main__'})

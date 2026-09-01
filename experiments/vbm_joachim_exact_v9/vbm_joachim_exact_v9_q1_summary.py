# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
"""Summary-only execution wrapper for frozen Q1.

Applies the already-recorded audit metadata retention patch, then suppresses the
720-row payload at final print. Scientific computation is unchanged.
"""
import urllib.request
URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-joachim-exact-v9-20260901/experiments/vbm_joachim_exact_v9/vbm_joachim_exact_v9_q1_freshfit.py'
with urllib.request.urlopen(URL,timeout=120) as r: src=r.read().decode('utf-8')
old="rows.append({'folio':r['folio'],'line':r['line'],'key':r['key'],'B':r['B'],'tokens':len(r['tokens']),'unique_nuclei':len(set(x for x in r['nuclei'] if x)),'unique_bridges':len(set(r['bridges'])),'scores':scores})"
new="rows.append({'folio':r['folio'],'line':r['line'],'key':r['key'],'B':r['B'],'tokens':len(r['tokens']),'nuclei':list(r['nuclei']),'bridges':list(r['bridges']),'unique_nuclei':len(set(x for x in r['nuclei'] if x)),'unique_bridges':len(set(r['bridges'])),'scores':scores})"
if old not in src: raise RuntimeError('retention patch target missing')
src=src.replace(old,new,1)
oldprint="print('VBM_V9_Q1_RESULT='+json.dumps(out,sort_keys=True,separators=(',',':')))"
newprint="summary={k:out[k] for k in ['protocol','namespace','fixture','eligible_by_B','selected_by_B','sample_lines','banks','structural_shuffle_null','interpretation_band','decision','target_firewall']}; print('VBM_V9_Q1_SUMMARY='+json.dumps(summary,sort_keys=True,separators=(',',':')))"
if oldprint not in src: raise RuntimeError('summary print target missing')
src=src.replace(oldprint,newprint,1)
exec(compile(src,URL,'exec'),{'__name__':'__main__'})

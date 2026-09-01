# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
"""Execution patch applied before any Q2 output.

Frozen protocol defines STAB as split-half dictionary agreement weighted over
surface types as they occur in HOLDOUT. The initial runner accidentally used
the FIT+SELECT events as the weighting reference. This wrapper changes only
that reference set and its call signature; all solver/search/scoring settings
remain frozen.
"""
import urllib.request
URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-joachim-exact-v9-20260901/experiments/vbm_joachim_exact_v9/vbm_joachim_exact_v9_q2_synth_ident.py'
with urllib.request.urlopen(URL,timeout=120) as r: src=r.read().decode('utf-8')
old="""def stability(train,asset,tag,restarts,passes):
    a=train[::2];b=train[1::2];ma=apply_defaults(fit_map(a,asset,tag+':SA',restarts,passes),a,asset);mb=apply_defaults(fit_map(b,asset,tag+':SB',restarts,passes),b,asset)
    ab=weighted_agree(ma,mb,train,'b');an=weighted_agree(ma,mb,train,'n');return .5*(ab+an),ab,an
"""
new="""def stability(train,hold,asset,tag,restarts,passes):
    a=train[::2];b=train[1::2];ma=apply_defaults(fit_map(a,asset,tag+':SA',restarts,passes),a,asset);mb=apply_defaults(fit_map(b,asset,tag+':SB',restarts,passes),b,asset)
    ab=weighted_agree(ma,mb,hold,'b');an=weighted_agree(ma,mb,hold,'n');return .5*(ab+an),ab,an
"""
if old not in src: raise RuntimeError('stability definition patch target missing')
src=src.replace(old,new,1)
oldcall="stab,sb,sn=stability(train,A[chosen],f'{phase}:{fam}:R{rep}:STAB:{chosen}',1 if smoke else 2,2 if smoke else 4)"
newcall="stab,sb,sn=stability(train,hold,A[chosen],f'{phase}:{fam}:R{rep}:STAB:{chosen}',1 if smoke else 2,2 if smoke else 4)"
if oldcall not in src: raise RuntimeError('stability call patch target missing')
src=src.replace(oldcall,newcall,1)
exec(compile(src,URL,'exec'),{'__name__':'__main__'})

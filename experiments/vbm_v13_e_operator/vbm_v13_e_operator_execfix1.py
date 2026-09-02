#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "scipy>=1.13,<2", "scikit-learn>=1.5,<2"]
# ///
import urllib.request

URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v13-e-operator-geometry-20260902/experiments/vbm_v13_e_operator/vbm_v13_e_operator.py'
req=urllib.request.Request(URL,headers={'User-Agent':'VBMV13ExecFix/2026-09-02'})
src=urllib.request.urlopen(req,timeout=120).read().decode('utf-8')
src=src.replace("import collections, hashlib, json, re, urllib.request", "import collections, hashlib, json, re, urllib.request, sys, types", 1)
old="def load_remote(url,name):\n    ns={'__name__':name}; exec(compile(get_text(url),url,'exec'),ns); return ns"
new="def load_remote(url,name):\n    mod=types.ModuleType(name); mod.__file__=url; sys.modules[name]=mod; exec(compile(get_text(url),url,'exec'),mod.__dict__); return mod.__dict__"
if old not in src:
    raise RuntimeError('expected load_remote source fragment missing')
src=src.replace(old,new,1)
ns={'__name__':'v13patched'}
exec(compile(src,URL,'exec'),ns)
ns['main']()

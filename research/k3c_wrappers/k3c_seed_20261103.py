#!/usr/bin/env python3
import sys,urllib.request
sys.argv=["k3c","20261103"]
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/dff9dcc0ebc3ff016b62f1e1490546c8ca8478e8/research/select_form_asym_f1_replicate_phaseK3c_20261004.py"
exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),globals())

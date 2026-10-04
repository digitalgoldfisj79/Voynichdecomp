#!/usr/bin/env python3
import sys,urllib.request
sys.argv=["k4c","20263222"]
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/4b089fa1cdf7a917ae8bde8299737b584442bcbf/research/select_form_normalized_blind_phaseK4c_20261004.py"
exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),globals())

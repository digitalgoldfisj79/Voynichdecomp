#!/usr/bin/env python3
import sys, urllib.request
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/004a50b91f91183a548b2b2effa3f0ac172e6310/research/inverse_renderer_recoverability_phaseB_20261004.py"
src=urllib.request.urlopen(URL,timeout=60).read().decode()
sys.argv=["inverse_renderer_recoverability_phaseB_20261004.py","--strength","2.5"]
exec(compile(src,URL,"exec"),{"__name__":"__main__"})

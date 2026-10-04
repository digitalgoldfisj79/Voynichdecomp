#!/usr/bin/env python3
import json,urllib.request
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/ea8d96668fed07b6eb3f33ff6ff405a3b1ef2972/research/structured_source_calibration_phaseL0_20261004.py"
ns={"__name__":"phaseL0lib"}
exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),ns)
r=ns["one_dataset"]("LANG",20262003)
print("STRUCTURED_SOURCE_REP_JSON="+json.dumps(r,separators=(",",":")),flush=True)

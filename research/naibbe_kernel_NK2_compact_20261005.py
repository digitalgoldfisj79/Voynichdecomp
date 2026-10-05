#!/usr/bin/env python3
import urllib.request, json, builtins
U="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/b17d0160a4e9b791ac60342a3b9a1867bb2d5b73/research/naibbe_kernel_NK2_20261005.py"
SRC=urllib.request.urlopen(U,timeout=60).read().decode()
def cap(*args,**kwargs):
    s=" ".join(str(x) for x in args)
    if s.startswith("NAIBBE_NK2_JSON="):
        o=json.loads(s.split("=",1)[1])
        out={
            "phase":o["phase"],"status":o["status"],"nblocks":o["nblocks"],"block":o["block"],
            "test_per_block":o["test_per_block"],"source_states":o["source_states"],
            "channel":o["channel"],"arms":o["arms"],"summary":o["summary"],
            "A_conditional_MI_block_family_given_source_bits":o["A_conditional_MI_block_family_given_source_bits"],
            "prior_L3c_reference":o["prior_L3c_reference"],"closure_note":o["closure_note"]
        }
        builtins.print("NK2_COMPACT_JSON="+json.dumps(out,separators=(",",":")),flush=True)
exec(compile(SRC,U,"exec"),{"__name__":"nk2_compact","print":cap})

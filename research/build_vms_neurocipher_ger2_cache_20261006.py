#!/usr/bin/env python3
import json, urllib.request
import vms_neurocipher_ger1_20261005 as v

SCRAMBLE_SEEDS=[9001,7001,11003,17011]
out={
  "schema":"VMS_NEUROCIPHER_GER2_CACHE_V1",
  "screen_commit":"a6516bb5733fc82bf77da383800dcacbf28cac5e",
  "solver_commit":"e36031a9a6b8b67fcebb4d6f4af1c3753fad4287",
  "vms":{
    "discovery":v.DISC,
    "validation":v.VAL,
    "final_unseen_discovery":v.FIN_UNSEEN,
    "final_strict_novel":v.FIN_STRICT,
    "chars":v.VCHARS,
    "audit":v.V_AUDIT
  },
  "dialects":{}
}
for d in ("BAV","ALEM"):
    ranked,freq,docs=v.load_ref_ranked(d)
    base=ranked[:v.FULL_K]
    scr={}
    audits={}
    for ss in SCRAMBLE_SEEDS:
        z,a=v.scrambled_unique_vocab(base,ss)
        scr[str(ss)]=z
        audits[str(ss)]=a
    out["dialects"][d]={
      "docs":docs,
      "ref_types":len(ranked),
      "real":base,
      "scramble":scr,
      "scramble_audit":audits
    }
raw=json.dumps(out,ensure_ascii=False,separators=(",",":"))
print("GER2_CACHE_SHA="+__import__("hashlib").sha256(raw.encode()).hexdigest(),flush=True)
print("GER2_CACHE_JSON="+raw,flush=True)

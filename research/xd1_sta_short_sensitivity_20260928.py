#!/usr/bin/env python3
"""Post-outcome STA short-word sensitivity; downgrade-only."""
import hashlib,json,re,urllib.request,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import xd1_adapters_20260920 as adapters
import xd1_recipe_repetition_closeout_20260928 as xd
URL="https://voynich.nu/data/sta/RF1b.txt"
CODE_RE=re.compile(r"[A-Z][0-9a-z]")
LOCUS_RE=re.compile(r"^<(?P<loc>f[^,>]+),(?P<mode>[^>]+)>\s*(?P<body>.*)$")
FOLIO_RE=re.compile(r"^(f\d+[rv]\d*)")
SEED=2026092807
req=urllib.request.Request(URL,headers={"User-Agent":"XD1-STA-SHORT-SENS-20260928"})
data=urllib.request.urlopen(req,timeout=120).read(); text=data.decode("utf-8")
rows=[]; loci=codes=short_words=uncertain=0
for raw in text.splitlines():
    m=LOCUS_RE.match(raw.strip())
    if not m: continue
    loci+=1; body=m.group("body"); uncertain += body.count("<->")
    allchunks=re.split(r"\.|<->",body)
    short_words += sum(bool(CODE_RE.findall(c)) for c in allchunks)
    codes += sum(len(CODE_RE.findall(c)) for c in allchunks)
    if "P" not in m.group("mode"): continue
    fm=FOLIO_RE.match(m.group("loc"))
    if not fm: continue
    fol=fm.group(1); nm=re.match(r"f(\d+)",fol)
    if not nm or int(nm.group(1)) not in adapters.CANON_FOLIO_NUMS: continue
    ws=["".join(CODE_RE.findall(c)) for c in allchunks if CODE_RE.findall(c)]
    if ws: rows.append({"folio":fol,"tokens":ws})
gate=(loci,codes,short_words,uncertain)
expected=(5385,157254,37848,761)
if gate!=expected: raise SystemExit(f"source gate {gate} != {expected}")
res={"protocol_id":"XD1-STA-SHORT-SENS-20260928","post_outcome":True,
     "source_sha256":hashlib.sha256(data).hexdigest(),"source_gate":gate,
     "n_lines":len(rows),"n_tokens":sum(len(r["tokens"]) for r in rows),"lags":{}}
for lag in (1,2):
    res["lags"][str(lag)]={"sensitivity_2000_ALL":xd.metric(rows,lag,2000,SEED+lag*1000),
                           "bootstrap":xd.bootstrap(rows,lag,SEED+lag*1000+77)}
l1=res["lags"]["1"]["sensitivity_2000_ALL"]; l2=res["lags"]["2"]["sensitivity_2000_ALL"]
z1=l1["effect"]/l1["null_sd"]; z2=l2["effect"]/l2["null_sd"]
res["signed_effect_over_null_sd"]={"lag1":z1,"lag2":z2}
res["downgrade_triggered"]=not (z2>=2 and z1>-2)
payload=json.dumps(res,sort_keys=True,separators=(",",":"))
res["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
out=Path("research/xd1_sta_short_sens_result_20260928.json"); out.write_text(json.dumps(res,indent=2,sort_keys=True))
print("STA_SHORT_SENS_RESULT="+json.dumps(res,sort_keys=True))

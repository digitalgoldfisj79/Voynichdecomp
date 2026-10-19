#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""ANCHOR-1 empirical corpus QA, label ecology firewalls and comparable negatives.

This is an audit, not a decipherment. No glyphs are interpreted semantically.
ZLZI and TTLI are audited independently; never silently coalesce layers.
"""
import collections,hashlib,json,re,urllib.request
from pathlib import Path

URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/19fc6f2262dc2d184b370fb7c6960f11278f4778/voynich_transcriptions_slim.json"
EXPECTED_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
FOCUS=("f2r","f47r")
def emit(tag,obj):print(tag+"="+json.dumps(obj,sort_keys=True,separators=(",",":")),flush=True)

def get():
    raw=urllib.request.urlopen(URL,timeout=120).read()
    sh=hashlib.sha256(raw).hexdigest()
    if sh!=EXPECTED_SHA:raise RuntimeError(f"SHA_CHANGED {sh}")
    return json.loads(raw),sh
def section(f):
    v=re.match(r"f(\d+)",f)
    if not v:return "UNKNOWN"
    n=int(v.group(1))
    if n<=66:return "HERBAL_PRELIMINARY"
    if 75<=n<=84:return "BALNEOLOGICAL_PRELIMINARY"
    if 88<=n<=102:return "PHARMA_PRELIMINARY"
    return "OTHER"
def rows(pop,layer):
    out=[]
    for f,recs in pop["pages"].items():
        for line,rec in recs.items():
            u=rec.get("u","")
            if not isinstance(u,str):continue
            unit="LABEL" if "L" in u and "P" not in u else ("RUNNING" if "P" in u else "OTHER")
            txt=rec.get("t",{}).get(layer)
            if txt is None:continue
            tokens=[x.lower() for x in txt.split() if re.fullmatch("[a-z]+",x.lower())]
            mline=re.match(r"^[0-9]+",str(line))
            if not mline:continue
            out.append({"folio":f,"line":int(mline.group(1)),"original_line_key":str(line),"unit":u,"ecology":unit,
                        "section_prelim":section(f),"raw":txt,"tokens":tokens,
                        "source_layer":layer})
    return out
def target(a,f):
    return [x for x in a if x["folio"]==f]
def main():
    pop,sha=get()
    layers={}
    for lay in ("TTLI","ZLZI"):
        allrows=rows(pop,lay)
        herbal=[x for x in allrows if x["section_prelim"]=="HERBAL_PRELIMINARY"]
        labels=[x for x in herbal if x["ecology"]=="LABEL"]
        active=[x for x in labels if x["tokens"]]
        running=[x for x in herbal if x["ecology"]=="RUNNING"]
        lab_types=collections.Counter(t for x in active for t in x["tokens"])
        run_types=collections.Counter(t for x in running for t in x["tokens"])
        focus={f:[{"line":x["line"],"unit":x["unit"],"ecology":x["ecology"],
                    "raw":x["raw"],"tokens":x["tokens"]}
                  for x in target(allrows,f)
                  if x["ecology"]=="LABEL" or x["line"]==1]
              for f in FOCUS}
        layers[lay]={
            "herbal_loci":len(herbal),"herbal_running_loci":len(running),
            "herbal_label_loci":len(labels),"nonempty_herbal_label_loci":len(active),
            "label_folios":len(set(x["folio"] for x in active)),
            "all_herbal_label_types":len(lab_types),
            "label_token_top20":lab_types.most_common(20),
            "focus":focus,
            "ytoail_label_count":lab_types["ytoail"],
            "ytoail_running_count":run_types["ytoail"],
            "eligible_label_loci":sum(len(x["tokens"])>=1 for x in active),
            "label_control_examples":[{"folio":x["folio"],"line":x["line"],
                                       "raw":x["raw"],"unit":x["unit"]}
                                      for x in active if x["folio"] not in FOCUS][:25]}
    def find(layer,f,li):
        return next((x for x in layers[layer]["focus"][f] if x["line"]==li),None)
    checks={
        "f2r_ZLZI_label_14_ytoail":bool((t:=find("ZLZI","f2r",14)) and t["ecology"]=="LABEL" and t["tokens"]==["ytoail"]),
        "f2r_ZLZI_label_15_three_tokens":bool((t:=find("ZLZI","f2r",15)) and t["tokens"]==["ios","an","on"]),
        "f47r_source_first_word_conflict":bool((t:=find("TTLI","f47r",1)) and t["tokens"][:1]==["pchair"] and (z:=find("ZLZI","f47r",1)) and z["tokens"][:1]!=["pchair"])
    }
    if not all(checks.values()):raise RuntimeError("SOURCE_AUDIT_ASSERTIONS "+json.dumps(checks))
    out={"schema":"ANCHOR1-REAL-CORPUS-AUDIT","sha256":sha,
         "note":"Herbal folio range used only for preliminary source counts; controls require canonical section/hand/physical bifolium metadata before hypothesis testing.",
         "unit_rule":"Unit indicator P is running; L is label; no pooled training or scoring.",
         "layers":layers,"checks":checks,
         "promotion":"NO: f2r plant identity is disputed; Egerton labelled peony and centaury are admissible historical vocabulary but not aligned proof."}
    emit("ANCHOR1_AUDIT",out)
if __name__=="__main__":main()

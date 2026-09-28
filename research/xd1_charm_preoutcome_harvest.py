#!/usr/bin/env python3
"""Pre-outcome harvest of CUL Add.9308 diplomatic physical lines. NO recurrence."""
import json,re,html,hashlib,urllib.request
from pathlib import Path

BASE="https://services.cudl.lib.cam.ac.uk"
ITEM="MS-ADD-09308"
UA={"User-Agent":"Voynichdecomp-XD1-CHARM-harvest/1.0"}

def get(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r:return r.read()

def strip_tags(s):
    s=re.sub(r"<style\b.*?</style>"," ",s,flags=re.I|re.S)
    s=re.sub(r"<script\b.*?</script>"," ",s,flags=re.I|re.S)
    s=re.sub(r"<[^>]+>"," ",s)
    return re.sub(r"\s+"," ",html.unescape(s)).strip()

def parse_lines(s):
    # Every physical transcription line is emitted as a <br> carrying geometry and an id.
    ms=list(re.finditer(r"<br\b([^>]*)>",s,re.I))
    out=[]
    for i,m in enumerate(ms):
        attrs=m.group(1)
        end=ms[i+1].start() if i+1<len(ms) else len(s)
        frag=s[m.end():end]
        xid=re.search(r"(?:xml:id|id)=['\"]([^'\"]+)['\"]",attrs,re.I)
        pts=re.search(r"data-points=['\"]([^'\"]+)['\"]",attrs,re.I)
        out.append({
          "line_index":i+1,
          "line_id":xid.group(1) if xid else None,
          "data_points":pts.group(1) if pts else None,
          "text":strip_tags(frag),
          "raw_fragment":frag
        })
    return out

def main():
    meta=json.loads(get(f"{BASE}/v1/metadata/json/{ITEM}"))
    pages=[]
    for p in meta["pages"]:
        u=p.get("transcriptionDiplomaticURL")
        if not u:continue
        b=get(BASE+u)
        s=b.decode("utf-8","replace")
        lines=parse_lines(s)
        pages.append({
          "label":p.get("label"),"physID":p.get("physID"),"sequence":p.get("sequence"),
          "url":BASE+u,"sha256":hashlib.sha256(b).hexdigest(),
          "physical_line_count":sum(1 for x in lines if x["data_points"]),
          "lines":lines
        })
    out={
      "protocol":"XD1-CHARM-20260928","stage":"PRE_OUTCOME_SOURCE_HARVEST",
      "outcomes_computed":False,"item":ITEM,
      "metadata_sha256":hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest(),
      "pages":pages
    }
    Path("xd1_charm_harvest").mkdir(exist_ok=True)
    Path("xd1_charm_harvest/add9308_diplomatic_lines.json").write_text(
      json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    # Plain TSV for manual source classification. No token/repetition columns.
    with open("xd1_charm_harvest/add9308_lines.tsv","w",encoding="utf-8") as f:
        f.write("folio\tline_index\tline_id\ttext\n")
        for p in pages:
            for x in p["lines"]:
                if x["data_points"]:
                    f.write(f"{p['label']}\t{x['line_index']}\t{x['line_id'] or ''}\t{x['text'].replace(chr(9),' ')}\n")
    print(json.dumps({"pages":len(pages),"physical_lines":sum(p["physical_line_count"] for p in pages),
                      "outcomes_computed":False},indent=2))
if __name__=="__main__":main()

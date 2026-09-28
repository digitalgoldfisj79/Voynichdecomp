#!/usr/bin/env python3
"""Focused pre-outcome source extraction for independently attested Add.9308 charm loci."""
import json,re,html,urllib.request,hashlib
from pathlib import Path
BASE="https://services.cudl.lib.cam.ac.uk"; ITEM="MS-ADD-09308"
FOLIOS=["14v","22v","23r","25r","25v","26r","32v","33r","35r","35v","36v",
        "49r","49v","50r","51v","52r","52v","53r","53v","54r",
        "61r","61v","62r","62v","68r","68v","78r","78v","86v","87r"]
UA={"User-Agent":"Voynichdecomp-XD1-CHARM-focused-source/1.0"}
def get(u):
 req=urllib.request.Request(u,headers=UA)
 with urllib.request.urlopen(req,timeout=60) as r:return r.read()
def textify(s):
 s=re.sub(r"<[^>]+>"," ",s);return re.sub(r"\s+"," ",html.unescape(s)).strip()
def lines(s):
 ms=list(re.finditer(r"<br\b([^>]*)>",s,re.I));out=[]
 for i,m in enumerate(ms):
  end=ms[i+1].start() if i+1<len(ms) else len(s)
  attrs=m.group(1);frag=s[m.end():end]
  xid=re.search(r"(?:xml:id|id)=['\"]([^'\"]+)['\"]",attrs,re.I)
  pts=re.search(r"data-points=['\"]([^'\"]+)['\"]",attrs,re.I)
  if not pts: continue
  out.append({"line":len(out)+1,"id":xid.group(1) if xid else None,
              "text":textify(frag),"raw_fragment":frag})
 return out
def main():
 meta=json.loads(get(f"{BASE}/v1/metadata/json/{ITEM}"))
 by={p.get("label"):p for p in meta["pages"]}
 out={"protocol":"XD1-CHARM-20260928","stage":"FOCUSED_PRE_OUTCOME_SOURCE","outcomes_computed":False,"folios":{}}
 for f in FOLIOS:
  p=by.get(f)
  if not p or not p.get("transcriptionDiplomaticURL"):
   out["folios"][f]={"error":"no diplomatic page"};continue
  b=get(BASE+p["transcriptionDiplomaticURL"]);s=b.decode("utf-8","replace")
  out["folios"][f]={"url":BASE+p["transcriptionDiplomaticURL"],"sha256":hashlib.sha256(b).hexdigest(),"lines":lines(s)}
 Path("xd1_charm_focused_source").mkdir(exist_ok=True)
 Path("xd1_charm_focused_source/focused_lines.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
 with open("xd1_charm_focused_source/focused_lines.txt","w",encoding="utf-8") as w:
  for f,d in out["folios"].items():
   w.write(f"\n### {f}\n")
   for x in d.get("lines",[]):w.write(f"{x['line']:02d}\t{x['id']}\t{x['text']}\n")
 print("folios",len(out["folios"]),"physical_lines",sum(len(x.get("lines",[])) for x in out["folios"].values()),"outcomes",False)
if __name__=="__main__":main()

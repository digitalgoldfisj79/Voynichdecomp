#!/usr/bin/env python3
"""XD1-CHARM source-only preflight. NO recurrence statistics."""
import json, re, sys, urllib.request, urllib.error, xml.etree.ElementTree as ET
from pathlib import Path

BASE="https://services.cudl.lib.cam.ac.uk"
ITEMS={
  "MS-ADD-09308":{"folios":["49r","49v","50r","68r","68v"]},
  "MS-DD-00005-00053":{"folios":["107r","107v","114v","117v","122v","144r"]},
  "MS-DD-00003-00052":{"folios":[]},
}
UA={"User-Agent":"Voynichdecomp-XD1-CHARM-source-preflight/1.0"}

def get(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r:
        return r.status, r.headers.get("content-type",""), r.read()

def safe_get(url):
    try:
        return get(url)
    except Exception as e:
        return None,None,(type(e).__name__+": "+str(e)).encode()

def walk_strings(x,path=""):
    if isinstance(x,dict):
        for k,v in x.items():
            yield from walk_strings(v,path+"/"+str(k))
    elif isinstance(x,list):
        for i,v in enumerate(x):
            yield from walk_strings(v,path+"/"+str(i))
    elif isinstance(x,str):
        yield path,x

def page_summary(obj):
    out=[]
    pages=obj.get("pages") if isinstance(obj,dict) else None
    if not isinstance(pages,list):
        # recursively locate first plausible pages list
        def rec(x,p=""):
            if isinstance(x,dict):
                for k,v in x.items():
                    if k=="pages" and isinstance(v,list):
                        return v,p+"/"+k
                    z=rec(v,p+"/"+str(k))
                    if z:return z
            elif isinstance(x,list):
                for i,v in enumerate(x):
                    z=rec(v,p+"/"+str(i))
                    if z:return z
            return None
        z=rec(obj)
        if z: pages,where=z
        else: return [],None
    else: where="/pages"
    for i,p in enumerate(pages):
        if not isinstance(p,dict): continue
        trans=p.get("transcriptionDiplomaticURL") or p.get("transcriptionNormalisedURL")
        label=p.get("label") or p.get("title") or p.get("pageLabel") or p.get("physID")
        out.append({"index":i+1,"label":label,"physID":p.get("physID"),"sequence":p.get("sequence"),
                    "transcriptionDiplomaticURL":p.get("transcriptionDiplomaticURL"),
                    "transcriptionNormalisedURL":p.get("transcriptionNormalisedURL"),
                    "surfaceID":p.get("surfaceID"),"keys":sorted(p.keys())})
    return out,where

def lineation_probe(url):
    if url.startswith("/"): url=BASE+url
    st,ct,b=safe_get(url)
    if st is None: return {"url":url,"error":b.decode(errors="replace")}
    txt=b.decode("utf-8","replace")
    # Source-only structural facts. Do NOT tokenize or compare word equality.
    return {
      "url":url,"status":st,"content_type":ct,"bytes":len(b),
      "lb_tag_count":len(re.findall(r"<(?:\w+:)?lb\b",txt,re.I)),
      "line_element_count":len(re.findall(r"<(?:\w+:)?(?:line|l)\b",txt,re.I)),
      "page_break_count":len(re.findall(r"<(?:\w+:)?pb\b",txt,re.I)),
      "contains_tei":bool(re.search(r"<(?:\w+:)?TEI\b",txt,re.I)),
      "contains_html":bool(re.search(r"<html\b",txt,re.I)),
      "head":txt[:1200]
    }

def main():
    Path("xd1_charm_preflight").mkdir(exist_ok=True)
    result={"protocol":"XD1-CHARM-20260928","outcomes_computed":False,"items":{}}
    for item,cfg in ITEMS.items():
        rec={"requested_folios":cfg["folios"]}
        # JSON is the viewer-oriented transformed metadata and should expose page URLs.
        jurl=f"{BASE}/v1/metadata/json/{item}"
        st,ct,b=safe_get(jurl)
        rec["json_fetch"]={"url":jurl,"status":st,"content_type":ct,"bytes":len(b)}
        obj=None
        if st:
            try:
                obj=json.loads(b)
                rec["json_top_keys"]=sorted(obj.keys()) if isinstance(obj,dict) else [type(obj).__name__]
                pages,where=page_summary(obj)
                rec["pages_path"]=where
                rec["page_count"]=len(pages)
                rec["pages"]=pages
            except Exception as e:
                rec["json_error"]=repr(e); rec["json_head"]=b[:2000].decode(errors="replace")
        # TEI metadata raw attrs can provide additional media URLs / folio labels.
        turl=f"{BASE}/v1/metadata/tei/{item}"
        st2,ct2,b2=safe_get(turl)
        rec["tei_fetch"]={"url":turl,"status":st2,"content_type":ct2,"bytes":len(b2)}
        if st2:
            text=b2.decode("utf-8","replace")
            rec["tei_lb_count"]=len(re.findall(r"<(?:\w+:)?lb\b",text,re.I))
            rec["tei_media_diplomatic_urls"]=sorted(set(re.findall(r"""url=["']([^"']*transcription[^"']*)["']""",text,re.I)))[:500]
            rec["tei_internal_transcription_paths"]=sorted(set(re.findall(r"(/v1/transcription/[^\"'<> ]+)",text)))[:500]
            rec["tei_folio_mentions"]={f: len(re.findall(re.escape(f),text,re.I)) for f in cfg["folios"]}
        # Find page entries by exact/contained folio labels, before touching transcript contents.
        selected=[]
        for p in rec.get("pages",[]):
            blob=json.dumps(p,ensure_ascii=False).lower()
            if any(f.lower() in blob for f in cfg["folios"]):
                selected.append(p)
        rec["selected_pages"]=selected
        # If exact labels are absent, record every page with diplomatic URLs for later mapping, but do not score.
        trans_urls=[]
        for p in (selected if selected else rec.get("pages",[])):
            u=p.get("transcriptionDiplomaticURL")
            if u: trans_urls.append(u)
        # Limit fallback to structural probing of first 8 transcripts; selected pages are all probed.
        if not selected: trans_urls=trans_urls[:8]
        rec["transcription_structures"]=[lineation_probe(u) for u in dict.fromkeys(trans_urls)]
        result["items"][item]=rec

    # Qualification is source-structural only.
    for item,rec in result["items"].items():
        probes=rec.get("transcription_structures",[])
        rec["has_explicit_physical_line_marker"]=any(
            (p.get("lb_tag_count",0)>0 or p.get("line_element_count",0)>0) for p in probes
        )
    out=Path("xd1_charm_preflight/source_preflight.json")
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
      k:{
        "json_status":v.get("json_fetch",{}).get("status"),
        "page_count":v.get("page_count"),
        "selected_pages":len(v.get("selected_pages",[])),
        "transcripts_probed":len(v.get("transcription_structures",[])),
        "has_explicit_physical_line_marker":v.get("has_explicit_physical_line_marker")
      } for k,v in result["items"].items()
    },indent=2))

if __name__=="__main__":
    main()

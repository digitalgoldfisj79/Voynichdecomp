#!/usr/bin/env python3
"""XD1-CHARM source preflight v2. Structural/source metadata only; NO recurrence."""
import json,re,hashlib,urllib.request
from pathlib import Path

BASE="https://services.cudl.lib.cam.ac.uk"
ITEMS=[
 "MS-ADD-09308",      # Taylor Appendix 1
 "MS-DD-00006-00029", # Taylor Appendix 1
 "MS-DD-00005-00076", # Taylor Appendix 1
 "MS-DD-00004-00044", # Taylor Appendix 1
 "MS-DD-00005-00053", # independently catalogued medical charms; non-Taylor
 "MS-DD-00003-00052", # independently catalogued ritual charms; non-Taylor
]
UA={"User-Agent":"Voynichdecomp-XD1-CHARM-source-preflight/2.0"}

def fetch(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r:
        b=r.read()
        return r.status,r.headers.get("content-type",""),b

def safe(url):
    try:return fetch(url)
    except Exception as e:return None,None,(type(e).__name__+": "+str(e)).encode()

def nodes_with_terms(x, terms, path="$"):
    hits=[]
    if isinstance(x,dict):
        blob=json.dumps(x,ensure_ascii=False).lower()
        if any(t in blob for t in terms):
            # capture leaf-ish logical nodes, not giant roots
            scalar={k:v for k,v in x.items() if isinstance(v,(str,int,float,bool)) or v is None}
            if scalar:
                hits.append({"path":path,"scalar":scalar})
        for k,v in x.items():
            hits.extend(nodes_with_terms(v,terms,path+"/"+str(k)))
    elif isinstance(x,list):
        for i,v in enumerate(x): hits.extend(nodes_with_terms(v,terms,path+f"/{i}"))
    return hits

def probe_transcript(url):
    if url.startswith("/"): url=BASE+url
    st,ct,b=safe(url)
    if not st:return {"url":url,"error":b.decode(errors="replace")}
    s=b.decode("utf-8","replace")
    brs=re.findall(r"<br\b[^>]*>",s,re.I)
    physical=[x for x in brs if re.search(r"(?:xml:id|id)=['\"][^'\"]*(?:tl|line)[^'\"]*['\"]",x,re.I) or "data-points=" in x.lower()]
    return {
      "url":url,"status":st,"content_type":ct,"bytes":len(b),
      "sha256":hashlib.sha256(b).hexdigest(),
      "br_count":len(brs),
      "br_with_line_id_or_geometry":len(physical),
      "has_polygon_geometry":any("data-points=" in x.lower() for x in brs),
      "title":(re.search(r"<title>(.*?)</title>",s,re.I|re.S).group(1) if re.search(r"<title>(.*?)</title>",s,re.I|re.S) else None),
      "first_br_tags":brs[:3]
    }

def main():
    out={"protocol":"XD1-CHARM-20260928","preflight_version":2,"outcomes_computed":False,
         "selection_note":"First four CUL items are independently listed in Taylor 2025 Appendix 1; no recurrence was inspected.",
         "items":{}}
    for item in ITEMS:
        rec={}
        url=f"{BASE}/v1/metadata/json/{item}"
        st,ct,b=safe(url)
        rec["json"]={"status":st,"content_type":ct,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest() if st else None}
        if not st:
            rec["error"]=b.decode(errors="replace");out["items"][item]=rec;continue
        obj=json.loads(b)
        rec["title"]=obj.get("descriptiveMetadata",{}).get("title") if isinstance(obj.get("descriptiveMetadata"),dict) else None
        rec["numberOfPages"]=obj.get("numberOfPages")
        rec["useDiplomaticTranscriptions"]=obj.get("useDiplomaticTranscriptions")
        pages=obj.get("pages",[])
        tx=[p for p in pages if isinstance(p,dict) and p.get("transcriptionDiplomaticURL")]
        rec["diplomatic_pages"]=len(tx)
        rec["total_pages"]=len(pages)
        rec["first_diplomatic_pages"]=[{
          "index":i+1,"label":p.get("label"),"physID":p.get("physID"),
          "url":p.get("transcriptionDiplomaticURL")
        } for i,p in enumerate(pages) if isinstance(p,dict) and p.get("transcriptionDiplomaticURL")][:8]
        # Independently supplied metadata strings identifying charm/experiment loci.
        terms=["charm","incant","spell","experiment","amulet","ritual"]
        hits=nodes_with_terms(obj.get("logicalStructures",[]),terms)
        # de-duplicate compact hit representations
        seen=set(); compact=[]
        for h in hits:
            key=json.dumps(h,sort_keys=True,ensure_ascii=False)
            if key not in seen:
                seen.add(key);compact.append(h)
        rec["logical_term_hits"]=compact[:300]
        # Probe first, middle and last available diplomatic page plus every page whose label
        # occurs in a logical hit string. Structural only.
        wanted=[]
        if tx:
            wanted.extend([tx[0],tx[len(tx)//2],tx[-1]])
        hitblob=json.dumps(compact,ensure_ascii=False).lower()
        for p in tx:
            lab=str(p.get("label") or "").lower()
            if lab and lab in hitblob:wanted.append(p)
        uniq=[];u=set()
        for p in wanted:
            x=p.get("transcriptionDiplomaticURL")
            if x and x not in u:u.add(x);uniq.append(p)
        rec["structural_transcript_probes"]=[{
          "page_label":p.get("label"),"physID":p.get("physID"),**probe_transcript(p.get("transcriptionDiplomaticURL"))
        } for p in uniq[:30]]
        rec["passes_physical_lineation"]=any(
          q.get("br_with_line_id_or_geometry",0)>0 for q in rec["structural_transcript_probes"]
        )
        out["items"][item]=rec
    Path("xd1_charm_preflight_v2").mkdir(exist_ok=True)
    Path("xd1_charm_preflight_v2/source_preflight_v2.json").write_text(
      json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:{
      "json_status":v.get("json",{}).get("status"),
      "pages":v.get("total_pages"),
      "diplomatic_pages":v.get("diplomatic_pages"),
      "term_hits":len(v.get("logical_term_hits",[])),
      "passes_physical_lineation":v.get("passes_physical_lineation")
    } for k,v in out["items"].items()},indent=2))
if __name__=="__main__":main()

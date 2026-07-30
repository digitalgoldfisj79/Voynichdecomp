#!/usr/bin/env python3
"""Historical Wikisource acquisition without the Wikidata Query Service.

Enumerates main-namespace root pages deterministically, resolves their Wikidata
claims in batches, and admits only author-attributed works dated no later than
the registered cutoff. Exact Wikisource revisions are frozen. BLOCKED is a valid
result; no replacement text is generated.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, time, unicodedata
import urllib.error, urllib.parse, urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

UA="VoynichCompressionTransfer/0.1 (corpus acquisition; github.com/digitalgoldfisj79/Voynichdecomp)"
LIC="Wikisource CC BY-SA 4.0; underlying work public-domain candidate by registered date cutoff"
LICURL="https://creativecommons.org/licenses/by-sa/4.0/"
LANGS=["en","de","fi","tr","el","he","ar","la"]

def get(url:str, tries:int=6)->dict[str,Any]:
    last=None
    for i in range(tries):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":UA,"Accept":"application/json"})
            with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)
        except (urllib.error.URLError,urllib.error.HTTPError,TimeoutError,json.JSONDecodeError) as e:
            last=e;time.sleep(.35*(2**i))
    raise RuntimeError(f"request failed: {url}: {last}")

def api(host:str,**p:Any)->dict[str,Any]:
    p.update(format="json",formatversion=2,maxlag=5)
    return get(f"https://{host}/w/api.php?{urllib.parse.urlencode(p,doseq=True)}")

def norm(s:str)->str:return re.sub(r"\s+"," ",unicodedata.normalize("NFC",s).replace("\u00a0"," ")).strip()
def h(b:bytes)->str:return hashlib.sha256(b).hexdigest()

def year(claim:dict[str,Any])->int|None:
    try:t=claim["mainsnak"]["datavalue"]["value"]["time"]
    except (KeyError,TypeError):return None
    m=re.match(r"([+-])(\d+)-",t)
    if not m:return None
    y=int(m.group(2));return -y if m.group(1)=="-" else y

def enumerate_roots(lang:str,max_pages:int=25000)->list[dict[str,Any]]:
    host=f"{lang}.wikisource.org";cont=None;scanned=0;rows=[]
    while scanned<max_pages:
        p=dict(action="query",generator="allpages",gapnamespace=0,gapfilterredir="nonredirects",gaplimit=500,prop="pageprops",ppprop="wikibase_item")
        if cont:p["gapcontinue"]=cont
        x=api(host,**p);pages=x.get("query",{}).get("pages",[]);scanned+=len(pages)
        for q in pages:
            title=str(q.get("title",""));qid=q.get("pageprops",{}).get("wikibase_item")
            if qid and "/" not in title:rows.append(dict(qid=qid,title=title,pageid=int(q["pageid"])))
        cont=x.get("continue",{}).get("gapcontinue")
        if not cont or not pages:break
    uniq={}
    for r in rows:uniq.setdefault(r["qid"],r)
    return sorted(uniq.values(),key=lambda r:(int(r["qid"][1:]) if r["qid"][1:].isdigit() else 10**30,r["title"]))

def entities(qids:list[str],lang:str)->dict[str,Any]:
    if not qids:return {}
    return api("www.wikidata.org",action="wbgetentities",ids="|".join(qids),props="claims|labels",languages=f"{lang}|en",languagefallback=1).get("entities",{})

def candidates(lang:str,cutoff:int,limit:int=600)->list[dict[str,Any]]:
    roots=enumerate_roots(lang);out=[]
    for off in range(0,len(roots),50):
        batch=roots[off:off+50];es=entities([r["qid"] for r in batch],lang)
        for r in batch:
            e=es.get(r["qid"],{});cs=e.get("claims",{});authors=[]
            for c in cs.get("P50",[]):
                try:authors.append(c["mainsnak"]["datavalue"]["value"]["id"])
                except (KeyError,TypeError):pass
            ys=[y for prop in ("P577","P571") for c in cs.get(prop,[]) if (y:=year(c)) is not None and y<=cutoff]
            if not authors or not ys:continue
            label=e.get("labels",{}).get(lang,e.get("labels",{}).get("en",{})).get("value",r["qid"])
            out.append(dict(work_id=r["qid"],page_title=r["title"],pageid=r["pageid"],author_id=sorted(authors)[0],author_label=sorted(authors)[0],work_label=label,date=str(min(ys))))
            if len(out)>=limit:return out
    return out

def extract(host:str,title:str)->dict[str,Any]:
    x=api(host,action="query",redirects=1,prop="extracts|revisions|info",explaintext=1,exsectionformat="plain",rvprop="ids|timestamp",inprop="url",titles=title)
    ps=x.get("query",{}).get("pages",[])
    if len(ps)!=1 or ps[0].get("missing"):raise ValueError(f"missing {host}:{title}")
    p=ps[0];rv=p.get("revisions") or []
    if not rv:raise ValueError(f"no revision {host}:{title}")
    return dict(pageid=int(p["pageid"]),title=str(p["title"]),url=str(p.get("canonicalurl") or ""),revid=int(rv[0]["revid"]),timestamp=str(rv[0]["timestamp"]),text=str(p.get("extract","")))

def subpages(host:str,root:str,limit:int=50)->list[str]:
    out=[];cont=None
    while len(out)<limit:
        p=dict(action="query",list="allpages",apnamespace=0,apprefix=root.rstrip("/")+"/",aplimit=min(50,limit-len(out)))
        if cont:p["apcontinue"]=cont
        x=api(host,**p);out += [r["title"] for r in x.get("query",{}).get("allpages",[])]
        cont=x.get("continue",{}).get("apcontinue")
        if not cont:break
    return out

def work(lang:str,title:str,min_units:int)->dict[str,Any]:
    host=f"{lang}.wikisource.org";parts=[extract(host,title)];text=norm(parts[0]["text"])
    if len(text)<min_units:
        for t in subpages(host,parts[0]["title"]):
            try:parts.append(extract(host,t))
            except Exception:continue
            text=norm("\n".join(p["text"] for p in parts))
            if len(text)>=min_units:break
    return dict(parts=parts,text=text,units=len(text))

def split(i:int)->str:return "train" if i<8 else ("dev" if i<10 else "test")

def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--output",default="data/stage1_historical_wikisource");ap.add_argument("--target-docs",type=int,default=12);ap.add_argument("--min-units",type=int,default=4096);ap.add_argument("--cutoff-year",type=int,default=1800);a=ap.parse_args()
    out=Path(a.output);raw=out/"source_raw";normalized=out/"normalized";raw.mkdir(parents=True,exist_ok=True);normalized.mkdir(parents=True,exist_ok=True)
    rows=[];states={};logs={}
    for lang in LANGS:
        accepted=[];rejected=[]
        try:pool=candidates(lang,a.cutoff_year)
        except Exception as e:states[lang]=dict(status="BLOCKED_DISCOVERY_ERROR",accepted=0,error=repr(e));logs[lang]=dict(accepted=[],rejected=[]);continue
        for c in pool:
            if len(accepted)>=a.target_docs:break
            try:w=work(lang,c["page_title"],a.min_units)
            except Exception as e:rejected.append({**c,"reason":"fetch_error","error":repr(e)});continue
            if w["units"]<a.min_units:rejected.append({**c,"reason":"short","units":w["units"]});continue
            auth={x["author_id"] for x in accepted}
            if a.target_docs-len(accepted)==1 and len(auth)<2 and c["author_id"] in auth:rejected.append({**c,"reason":"reserved_for_second_author"});continue
            accepted.append({**c,**w})
        authors=sorted({x["author_id"] for x in accepted});state="ELIGIBLE" if len(accepted)>=a.target_docs and len(authors)>=2 else "BLOCKED_INSUFFICIENT_ELIGIBLE_WORKS"
        states[lang]=dict(status=state,accepted=len(accepted),authors=authors,candidate_count=len(pool),rejected_count=len(rejected));logs[lang]=dict(accepted=[],rejected=rejected)
        for i,x in enumerate(accepted):
            rb=("\n\n".join(p["text"] for p in x["parts"])).encode();nb=x["text"].encode();sig=h(json.dumps([{k:p[k] for k in ("pageid","title","revid","timestamp")} for p in x["parts"]],ensure_ascii=False,sort_keys=True,separators=(",",":")).encode())
            stem=f"{i:02d}_{lang}_{x['work_id']}_{sig[:12]}";rp=raw/f"{stem}.txt";np=normalized/f"{stem}.txt";rp.write_bytes(rb);np.write_bytes(nb);root=x["parts"][0];url=root["url"]+("&" if "?" in root["url"] else "?")+f"oldid={root['revid']}"
            row=dict(corpus_id="wikisource_historical_domain_20260730",document_id=f"wikisource-{lang}-{x['work_id']}-{sig[:12]}",split=split(i),class_label=lang,language=lang,family="historical_plaintext",path=np.as_posix(),sha256=h(nb),encoding="utf-8",license=LIC,author_id=x["author_id"],work_id=x["work_id"],date_band=f"dated_not_later_than_{a.cutoff_year};wikidata_date={x['date']}",notes=f"root_page={root['title']}; source={url}; components={len(x['parts'])}; component_signature={sig}; license_url={LICURL}; raw_sha256={h(rb)}; normalized_units={x['units']}",_text=x["text"])
            rows.append(row);logs[lang]["accepted"].append({k:v for k,v in row.items() if k!="_text"})
    dup=[];groups=defaultdict(list)
    for r in rows:groups[r["language"]].append(r)
    for lang,ds in groups.items():
        for i,x in enumerate(ds):
            for y in ds[i+1:]:
                ratio=SequenceMatcher(None,x["_text"][:20000],y["_text"][:20000],autojunk=False).ratio()
                if x["sha256"]==y["sha256"] or ratio>=.85:dup.append(dict(language=lang,document_a=x["document_id"],document_b=y["document_id"],exact=x["sha256"]==y["sha256"],ratio=ratio))
    eligible=[l for l,s in states.items() if s["status"]=="ELIGIBLE"];status="ACQUIRED_NOT_SCIENTIFICALLY_EVALUATED" if len(eligible)==len(LANGS) and not dup else ("BLOCKED_DUPLICATES" if dup else "BLOCKED")
    fields=["corpus_id","document_id","split","class_label","language","family","path","sha256","encoding","license","author_id","work_id","date_band","notes"]
    with (out/"manifest.csv").open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();[w.writerow({k:r[k] for k in fields}) for r in rows]
    (out/"discovery_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2,sort_keys=True)+"\n");(out/"duplicate_screen.json").write_text(json.dumps(dup,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    s=dict(programme="compression-transfer-v0.1",panel="stage1_historical_domain_wikisource",status=status,voynich_accessed=False,registered_cutoff_year=a.cutoff_year,minimum_units=a.min_units,target_documents_per_language=a.target_docs,language_status=states,eligible_languages=eligible,duplicate_findings_count=len(dup),selection_rule="deterministic Wikisource main-page enumeration plus batched Wikidata P50 and P577/P571 claims; first 12 eligible works with at least two authors",scientific_boundary="Blocked classes are not replaced synthetically.")
    s["scientific_payload_sha256"]=h(json.dumps(s,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode());(out/"summary.json").write_text(json.dumps(s,ensure_ascii=False,indent=2,sort_keys=True)+"\n");print(json.dumps(s,ensure_ascii=False,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())

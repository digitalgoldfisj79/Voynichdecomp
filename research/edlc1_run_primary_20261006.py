#!/usr/bin/env python3
"""
EDLC1 primary runner.
Requires: numpy regex rapidfuzz
Modes:
  pilot        200 matched draws, 100 shuffle draws, 100 cluster bootstraps
  publication  1000 matched draws, 1000 shuffle draws, 2000 cluster bootstraps
"""
from __future__ import annotations
import argparse, collections, hashlib, io, json, os, pathlib, re, subprocess, sys, tarfile, tempfile, urllib.request, time
import xml.etree.ElementTree as ET

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1f6cda14ca4ba8277812a1391ba16af616bc18ef/research/edlc1_core_20261006.py"
m={"__name__":"edlc1_core"}
exec(compile(urllib.request.urlopen(CORE_URL,timeout=120).read().decode(),CORE_URL,"exec"),m)

V_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
V_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
REF_URL="https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content"
DEEDS_REPO="https://github.com/DEEDS-Project/htr-dataset.git"
DEEDS_COMMIT="6a148076247c6c666c1051141f6e80286d4a3b6d"
SEED=20261006

PAIRS=[(1,8),(2,7),(3,6),(4,5),(9,16),(10,15),(11,14),(17,24),(18,23),(19,22),(20,21),
(25,32),(26,31),(27,30),(28,29),(33,40),(34,39),(35,38),(36,37),(41,48),(42,47),
(43,46),(44,45),(49,56),(50,55),(51,54),(52,53),(57,66),(58,65),(67,68),(69,70),
(71,72),(75,84),(76,83),(77,82),(78,81),(79,80),(85,86),(87,90),(88,89),(93,96),
(94,95),(99,102),(100,101),(103,116),(104,115),(105,114),(106,113),(107,112),(108,111)]
BIF={n:f"B{a:03d}_{b:03d}" for a,b in PAIRS for n in (a,b)}

def urlread_retry(url,timeout=300,attempts=6):
    last=None
    for i in range(attempts):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"EDLC1/1.0 reproducibility runner"})
            return urllib.request.urlopen(req,timeout=timeout).read()
        except Exception as e:
            last=e
            if i+1<attempts:
                time.sleep(min(30,2**i))
    raise last

def fnum(f):
    z=re.match(r"f(\d+)",str(f));return int(z.group(1)) if z else None

def local(tag):
    return tag.split("}")[-1]

def child_tag(el,name):
    for x in el:
        if local(x.tag)==name:return x.attrib.get("tag","")
    return ""

def load_voynich():
    raw=urlread_retry(V_URL,120,4)
    got=hashlib.sha256(raw).hexdigest()
    if got!=V_SHA:raise RuntimeError(("Voynich SHA mismatch",got))
    obj=json.loads(raw)
    out={}
    for tid in ("ZLZI","ZLZB","TTLI"):
        rr=[];excluded=0
        for fol,ld in obj["pages"].items():
            n=fnum(fol)
            if n not in BIF:continue
            for lid,rec in ld.items():
                if str(rec.get("u",""))!="+P0":continue
                txt=rec.get("t",{}).get(tid,"")
                for rawtok in txt.split():
                    t=m["clean_surface"](rawtok)
                    if t is None or not re.fullmatch(r"[a-z]+",t):
                        excluded+=1;continue
                    rr.append({"form":t,"block":BIF[n],"source":fol,"lemma":None,
                               "lemma_verified":False,"norm":None})
        out[tid]={"records":rr,"excluded":excluded}
    return out,got

def header_meta(root):
    h=next((x for x in root.iter() if local(x.tag)=="header"),None)
    out={}
    for line in ((h.text or "") if h is not None else "").splitlines():
        if ":" in line:
            k,v=line.split(":",1);out[k.strip().lower()]=v.strip()
    return out

def region_flags(md):
    region=md.get("language-region","").lower()
    area=md.get("language-area","").lower()
    return {
      "east":region=="ostoberdeutsch" or "ostoberdeutsch" in region,
      "west":region=="westoberdeutsch" or "westoberdeutsch" in region,
      "north":region=="nordoberdeutsch" or "nordoberdeutsch" in region,
      "bav":(("bair" in area or "bayr" in area or "österreich" in area or "oesterreich" in area)
             and "alemann" not in area),
      "alem":(("alemann" in area or "schwäb" in area or "elsäss" in area)
              and "bair" not in area and "bayr" not in area)
    }

def load_ref():
    local_archive=os.environ.get("EDLC1_REF_ARCHIVE")
    if local_archive:
        raw=pathlib.Path(local_archive).read_bytes()
    else:
        raw=urlread_retry(os.environ.get("EDLC1_REF_URL",REF_URL),300,6)
    sha=hashlib.sha256(raw).hexdigest()
    tf=tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz")
    allr=[];normr=[];docs=[];excluded=0;ambig=0;verified_n=0;lemma_n=0
    panels=collections.defaultdict(list)
    for name in tf.getnames():
        if not name.endswith(".xml"):continue
        try:root=ET.fromstring(tf.extractfile(name).read())
        except Exception:
            excluded+=1;continue
        md=header_meta(root)
        if "handschrift" not in md.get("medium","").lower():continue
        if not md.get("time","").lower().startswith("15,"):continue
        block=md.get("corpus-sigle") or md.get("text") or name
        flags=region_flags(md);n_before=len(allr)
        for tok in root.iter():
            if local(tok.tag)!="token":continue
            ds=[x for x in tok if local(x.tag) in ("tok_dipl","dipl")]
            ms=[x for x in tok if local(x.tag) in ("tok_anno","mod")]
            # Exclude annotation-designated punctuation/foreign material.
            poss=[child_tag(x,"pos") for x in ms]
            usable_anno=[x for x,p in zip(ms,poss) if p and not p.startswith("$") and p!="FM"]
            if ms and not usable_anno:
                excluded+=1;continue
            lemma=None;verified=False;norm=None
            if len(ds)==1 and len(usable_anno)==1:
                a=usable_anno[0]
                le=child_tag(a,"lemma")
                if le and le not in ("--","[!]"):lemma=le.casefold();lemma_n+=1
                verified=any(local(x.tag)=="cora-flag" and x.attrib.get("name")=="lemma verified" for x in a)
                if verified and lemma:verified_n+=1
                norm=m["clean_surface"](a.attrib.get("ascii") or a.attrib.get("utf") or a.attrib.get("trans"))
            elif len(ds)!=1 or len(usable_anno)!=1:
                ambig+=1
            dforms=[]
            for d in ds:
                rawform=d.attrib.get("utf") or d.attrib.get("trans") or ""
                parts=rawform.split()
                for p in parts:
                    f=m["clean_surface"](p)
                    if f:dforms.append(f)
            if not dforms:
                excluded+=1;continue
            for j,f in enumerate(dforms):
                rr={"form":f,"block":block,"source":name,
                    "lemma":lemma if len(dforms)==1 else None,
                    "lemma_verified":bool(verified and len(dforms)==1),
                    "norm":norm if len(dforms)==1 else None}
                allr.append(rr)
                ff=m["folded_surface"](f,"german")
                if ff:panels["folded"].append({**rr,"form":ff})
            if norm:
                normr.append({"form":norm,"block":block,"source":name,
                              "lemma":lemma,"lemma_verified":verified,"norm":norm})
        if len(allr)>n_before:
            docs.append({"block":block,"name":name,"date":md.get("date"),"time":md.get("time"),
                         "region":md.get("language-region"),"area":md.get("language-area"),
                         "text":md.get("text"),"n":len(allr)-n_before,**flags})
            for key,val in flags.items():
                if val:
                    panels[key].extend([r for r in allr[n_before:]])
    return {"diplomatic":allr,"folded":panels["folded"],"normalized":normr,
            "regional":{k:v for k,v in panels.items() if k!="folded"},
            "docs":docs,"excluded":excluded,"ambiguous_alignment":ambig,
            "lemma_records":lemma_n,"verified_lemma_records":verified_n,
            "archive_sha256":sha,"archive_bytes":len(raw)}

def clone_deeds(base):
    p=pathlib.Path(base)/"deeds"
    subprocess.run(["git","clone","-q","--filter=blob:none","--no-checkout",DEEDS_REPO,str(p)],check=True)
    subprocess.run(["git","-C",str(p),"sparse-checkout","init","--cone"],check=True)
    subprocess.run(["git","-C",str(p),"sparse-checkout","set","data/bl-cotton-nero-e-vi"],check=True)
    subprocess.run(["git","-C",str(p),"checkout","-q",DEEDS_COMMIT],check=True)
    got=subprocess.check_output(["git","-C",str(p),"rev-parse","HEAD"],text=True).strip()
    if got!=DEEDS_COMMIT:raise RuntimeError(("DEEDS commit mismatch",got))
    return p

def alto_main_lines(path):
    root=ET.parse(path).getroot()
    labels={}
    for x in root.iter():
        if local(x.tag)=="OtherTag":labels[x.attrib.get("ID")]=x.attrib.get("LABEL")
    lines=[]
    for b in root.iter():
        if local(b.tag)!="TextBlock":continue
        refs=b.attrib.get("TAGREFS","").split()
        if "MainZone" not in {labels.get(x) for x in refs}:continue
        for ln in b:
            if local(ln.tag)!="TextLine":continue
            chunks=[]
            for x in ln.iter():
                if local(x.tag)=="String" and x.attrib.get("CONTENT"):
                    chunks.append(x.attrib["CONTENT"])
            if chunks:lines.append(" ".join(chunks))
    return lines

def load_latin(base):
    repo=clone_deeds(base)
    d=repo/"data"/"bl-cotton-nero-e-vi"
    files=sorted(p for p in d.glob("*.xml") if not p.name.endswith(".chocomufin.xml"))
    rr=[];fold=[];excluded=0;joins=0;unresolved=0
    for p in files:
        lines=alto_main_lines(p);pending=None
        for line in lines:
            parts=line.split()
            if pending is not None:
                if parts:
                    parts[0]=pending+parts[0];joins+=1
                else:
                    unresolved+=1
                pending=None
            if parts and (parts[-1].endswith("-") or parts[-1].endswith("‐")):
                pending=parts.pop()[:-1]
            for rawtok in parts:
                f=m["clean_surface"](rawtok)
                if not f:
                    excluded+=1;continue
                rec={"form":f,"block":p.name,"source":p.name,
                     "lemma":None,"lemma_verified":False,"norm":None}
                rr.append(rec)
                ff=m["folded_surface"](f,"latin")
                if ff:fold.append({**rec,"form":ff})
        if pending:
            unresolved+=1;pending=None
    return {"diplomatic":rr,"folded":fold,"files":[p.name for p in files],
            "excluded":excluded,"hyphen_joins":joins,"unresolved_hyphens":unresolved,
            "commit":DEEDS_COMMIT}

def brief_metrics(records,do_graph=True):
    z={"summary":m["corpus_summary"](records)}
    if do_graph:z["ed"]=m["ed_panels"](records)
    return z

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=("pilot","publication"),default="pilot")
    ap.add_argument("--skip-shuffle",action="store_true")
    ap.add_argument("--skip-bootstrap",action="store_true")
    args=ap.parse_args()
    if args.mode=="publication":
        nmatch,nshuffle,nboot=1000,1000,2000
    else:
        nmatch,nshuffle,nboot=200,100,100
    print("EDLC1_CONFIG="+json.dumps({"mode":args.mode,"nmatch":nmatch,
      "nshuffle":0 if args.skip_shuffle else nshuffle,
      "nboot":0 if args.skip_bootstrap else nboot,"seed":SEED},separators=(",",":")),flush=True)

    V,vsha=load_voynich()
    print("EDLC1_SOURCE_V="+json.dumps({"sha256":vsha,**{k:{"tokens":len(v["records"]),"excluded":v["excluded"]} for k,v in V.items()}},separators=(",",":")),flush=True)

    R=load_ref()
    print("EDLC1_SOURCE_REF="+json.dumps({"sha256":R["archive_sha256"],"bytes":R["archive_bytes"],
      "documents":len(R["docs"]),"diplomatic_tokens":len(R["diplomatic"]),
      "normalized_tokens":len(R["normalized"]),"excluded":R["excluded"],
      "ambiguous_alignment":R["ambiguous_alignment"],
      "lemma_records":R["lemma_records"],"verified_lemma_records":R["verified_lemma_records"],
      "doc_meta":R["docs"]},ensure_ascii=False,separators=(",",":")),flush=True)

    with tempfile.TemporaryDirectory() as td:
        L=load_latin(td)
    print("EDLC1_SOURCE_LATIN="+json.dumps({"commit":L["commit"],"pages":len(L["files"]),
      "tokens":len(L["diplomatic"]),"excluded":L["excluded"],"hyphen_joins":L["hyphen_joins"],
      "unresolved_hyphens":L["unresolved_hyphens"],"files":L["files"]},ensure_ascii=False,separators=(",",":")),flush=True)

    corp={
      "VOYNICH_ZLZI":V["ZLZI"]["records"],
      "VOYNICH_ZLZB":V["ZLZB"]["records"],
      "VOYNICH_TTLI":V["TTLI"]["records"],
      "GERMAN_REF15_DIPL":R["diplomatic"],
      "GERMAN_REF15_FOLDED":R["folded"],
      "GERMAN_REF15_NORM":R["normalized"],
      "LATIN_COTTON1442_47_DIPL":L["diplomatic"],
      "LATIN_COTTON1442_47_FOLDED":L["folded"]
    }
    raw={}
    for i,(name,records) in enumerate(corp.items()):
        print("EDLC1_RAW_BEGIN",name,flush=True)
        raw[name]=brief_metrics(records,True)
        print("EDLC1_RAW_"+name+"="+json.dumps(raw[name],ensure_ascii=False,separators=(",",":")),flush=True)

    morph=m["morphology_decomposition"](R["diplomatic"],3)
    print("EDLC1_MORPH_GERMAN="+json.dumps(morph,ensure_ascii=False,separators=(",",":")),flush=True)

    oov={}
    for i,name in enumerate(("VOYNICH_ZLZI","VOYNICH_ZLZB","VOYNICH_TTLI","GERMAN_REF15_DIPL","LATIN_COTTON1442_47_DIPL")):
        oov[name]=m["oov_repair"](corp[name],5)
        print("EDLC1_OOV_"+name+"="+json.dumps(oov[name],separators=(",",":")),flush=True)

    matched={}
    for ti,tname in enumerate(("VOYNICH_ZLZI","VOYNICH_TTLI")):
        matched[tname]={}
        for ci,cname in enumerate(("GERMAN_REF15_DIPL","LATIN_COTTON1442_47_DIPL")):
            z=m["matched_resample"](corp[tname],corp[cname],3,nmatch,SEED+1000*ti+100*ci)
            matched[tname][cname]=z
            print("EDLC1_MATCH_"+tname+"__"+cname+"="+json.dumps(z,separators=(",",":")),flush=True)

    shuffle={}
    if not args.skip_shuffle:
        for i,name in enumerate(("VOYNICH_ZLZI","VOYNICH_TTLI","GERMAN_REF15_DIPL","LATIN_COTTON1442_47_DIPL")):
            z=m["positional_shuffle_null"](corp[name],3,nshuffle,SEED+5000+i)
            shuffle[name]=z
            print("EDLC1_SHUFFLE_"+name+"="+json.dumps(z,separators=(",",":")),flush=True)

    boot={}
    if not args.skip_bootstrap:
        for i,name in enumerate(("VOYNICH_ZLZI","VOYNICH_TTLI","GERMAN_REF15_DIPL","LATIN_COTTON1442_47_DIPL")):
            z=m["cluster_bootstrap_primary"](corp[name],3,nboot,SEED+8000+i)
            boot[name]=z
            print("EDLC1_BOOT_"+name+"="+json.dumps(z,separators=(",",":")),flush=True)

    regional={}
    for key,records in R["regional"].items():
        if len(records)<500:continue
        regional[key]={"summary":m["corpus_summary"](records),"ed_mf3":m["ed_graph_from_counter"](
          collections.Counter(r["form"] for r in records),3)}
        regional[key]["ed_mf3"].pop("_edges",None)
    print("EDLC1_GERMAN_REGIONAL="+json.dumps(regional,ensure_ascii=False,separators=(",",":")),flush=True)

    final={"study":"EDLC1","status":"pilot_complete" if args.mode=="pilot" else "publication_run_complete",
      "mode":args.mode,
      "sources":{"voynich_sha256":vsha,"ref_sha256":R["archive_sha256"],"deeds_commit":L["commit"]},
      "raw":raw,"morphology_german":morph,"oov":oov,"matched":matched,
      "shuffle":shuffle,"bootstrap":boot,"german_regional":regional,
      "warnings":[
        "ZLZI and ZLZB are transcription checks, not independent replications.",
        "Cotton Nero pages are blocks within one 1442-1447 manuscript, not independent manuscripts.",
        "RIDGES 1487, FnhdC 1450-1500 and John-of-Burgundy medical Latin remain predeclared Tier-B replications."
      ]}
    print("EDLC1_FINAL="+json.dumps(final,ensure_ascii=False,separators=(",",":")),flush=True)

if __name__=="__main__":main()

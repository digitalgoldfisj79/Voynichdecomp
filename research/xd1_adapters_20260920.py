#!/usr/bin/env python3
import argparse, collections, hashlib, json, os, re, unicodedata, urllib.request, urllib.parse, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

VMS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
VMS_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
NUR_RECORD="https://zenodo.org/api/records/13881575"
CREMMA_COMMIT="292525969ad98380b398e6606a9c2a36d51913ae"
CREMMA_ARCHIVE=f"https://github.com/HTR-United/CREMMA-Medieval-LAT/archive/{CREMMA_COMMIT}.zip"
PAIRS=[(1,8),(2,7),(3,6),(4,5),(9,16),(10,15),(11,14),(17,24),(18,23),(19,22),(20,21),(25,32),(26,31),(27,30),(28,29),(33,40),(34,39),(35,38),(36,37),(41,48),(42,47),(43,46),(44,45),(49,56),(50,55),(51,54),(52,53),(57,66),(58,65),(67,68),(69,70),(71,72),(75,84),(76,83),(77,82),(78,81),(79,80),(85,86),(87,90),(88,89),(93,96),(94,95),(99,102),(100,101),(103,116),(104,115),(105,114),(106,113),(107,112),(108,111)]
CANON_FOLIO_NUMS={n for a,b in PAIRS for n in (a,b)}

def local(tag): return tag.split("}")[-1]
def nfc(s): return unicodedata.normalize("NFC",s)

def tokenize(text):
    text=nfc(text).replace("\n"," ")
    out=[];buf=[]
    for ch in text:
        cat=unicodedata.category(ch)
        if cat[0] in ("L","M") or cat=="Nd":
            buf.append(ch.lower())
        else:
            if buf: out.append("".join(buf));buf=[]
    if buf:out.append("".join(buf))
    return [t for t in out if any(unicodedata.category(ch).startswith("L") for ch in t)]

def vms():
    p=Path("/tmp/xd1_vms.json");urllib.request.urlretrieve(VMS_URL,p)
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    if got!=VMS_SHA:raise RuntimeError(f"VMS source SHA mismatch {got}")
    obj=json.load(open(p));lines=[]
    for fol,ld in obj["pages"].items():
        mfol=re.match(r"f(\d+)",str(fol))
        if not mfol or int(mfol.group(1)) not in CANON_FOLIO_NUMS:continue
        for ls,rec in ld.items():
            if "P" not in str(rec.get("u","")):continue
            txt=rec.get("t",{}).get("ZLZI","")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower())]
            if not toks:continue
            m=re.match(r"(\d+)",str(ls));order=int(m.group(1)) if m else len(lines)
            lines.append(dict(block=fol,unit=fol,line_order=order,tokens=toks,writer=None))
    return dict(label="VMS_ZLZI_XD1_GENERIC",source_sha256=got,representation="ZLZI_RUNNING_TEXT",lines=lines)

def render_unicode(el,drop_ex=False):
    def rec(x):
        if drop_ex and local(x.tag)=="ex":
            return x.tail or ""
        s=x.text or ""
        for ch in x:
            s+=rec(ch)
        return s+(x.tail or "")
    # avoid including Unicode's own tail outside the element
    s=el.text or ""
    for ch in el:s+=rec(ch)
    return s

def baseline_y(line):
    bl=next((x for x in line if local(x.tag)=="Baseline"),None)
    if bl is None:return None
    pts=bl.attrib.get("points","")
    ys=[]
    for p in pts.split():
        try:ys.append(float(p.split(",")[1]))
        except:pass
    return sum(ys)/len(ys) if ys else None

def nuremberg():
    meta=json.load(urllib.request.urlopen(NUR_RECORD))
    f=next(x for x in meta["files"] if x["key"]=="labels.zip")
    url=f["links"]["self"];p=Path("/tmp/nuremberg_labels.zip");urllib.request.urlretrieve(url,p)
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    bykey={}
    with zipfile.ZipFile(p) as z:
        names=[n for n in z.namelist() if "/diplomatic-regularised/" in n and n.endswith(".xml")]
        for fn in names:
            root=ET.fromstring(z.read(fn))
            page=next((x for x in root.iter() if local(x.tag)=="Page"),None)
            if page is None:continue
            image=page.attrib.get("imageFilename") or fn
            band_match=re.search(r"/Band(\d+)/",fn);band=band_match.group(1) if band_match else "UNK"
            for tl in (x for x in root.iter() if local(x.tag)=="TextLine"):
                ue=next((x for x in tl.iter() if local(x.tag)=="Unicode"),None)
                if ue is None:continue
                exp=render_unicode(ue,False);unexp=render_unicode(ue,True)
                te=tokenize(exp);tu=tokenize(unexp)
                if not te or not tu:continue
                y=baseline_y(tl)
                if y is None:
                    custom=tl.attrib.get("custom","");m=re.search(r"index:(\d+)",custom);y=float(m.group(1)) if m else 0.0
                writer=tl.attrib.get("writerID")
                # Physical image + baseline ordinate + normalized expanded text gives stable de-duplication
                k=(image,round(y,2)," ".join(te))
                row=dict(block=image,unit=f"Band{band}",line_order=float(y),writer=writer,
                         tokens_expanded=te,tokens_unexpanded=tu,source_file=fn)
                if k not in bykey:bykey[k]=row
    rows=sorted(bykey.values(),key=lambda r:(r["block"],r["line_order"],r["source_file"]))
    common=[r for r in rows if r["tokens_expanded"] and r["tokens_unexpanded"]]
    def pack(rep,key):
        return dict(label=f"NUREMBERG_2_5_{rep}",source_sha256=sha,representation=rep,
                    lines=[dict(block=r["block"],unit=r["unit"],line_order=r["line_order"],
                                tokens=r[key],writer=r["writer"]) for r in common])
    return pack("DIPLOMATIC_UNEXPANDED_DROP_EX","tokens_unexpanded"),pack("DIPLOMATIC_EXPANDED","tokens_expanded")

def cremmas():
    import csv, io
    api=f"https://api.github.com/repos/HTR-United/CREMMA-Medieval-LAT/git/trees/{CREMMA_COMMIT}?recursive=1"
    req=urllib.request.Request(api,headers={"User-Agent":"XD1-control-audit"})
    tree=json.load(urllib.request.urlopen(req))["tree"]
    paths=[x["path"] for x in tree if x.get("type")=="blob"]
    rawbase=f"https://raw.githubusercontent.com/HTR-United/CREMMA-Medieval-LAT/{CREMMA_COMMIT}/"
    regtxt=urllib.request.urlopen(rawbase+"data-registry.csv").read().decode("utf-8-sig")
    metas=list(csv.DictReader(io.StringIO(regtxt)))
    source_manifest={"commit":CREMMA_COMMIT,"registry_sha256":hashlib.sha256(regtxt.encode()).hexdigest(),
                     "txt_paths":sorted(p for p in paths if p.startswith("data/") and p.endswith(".txt"))}
    source_sha=hashlib.sha256(json.dumps(source_manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    out=[]
    for meta in metas:
        folder=meta["Folder"].strip().rstrip("/")+"/"
        txts=sorted(p for p in source_manifest["txt_paths"] if p.startswith(folder))
        lines=[]
        for path in txts:
            page=path.rsplit("/",1)[-1].rsplit(".",1)[0]
            raw=urllib.request.urlopen(rawbase+urllib.parse.quote(path,safe="/")).read().decode("utf-8","replace")
            for i,line in enumerate(raw.splitlines()):
                toks=tokenize(line)
                if toks:
                    lines.append(dict(block=page,unit=meta["Shelfmark ID"],line_order=i,tokens=toks,
                                      writer=(meta.get("Scribe") or None)))
        if lines:
            out.append(dict(label="CREMMA_"+meta["Shelfmark ID"],source_sha256=source_sha,
                            source_commit=CREMMA_COMMIT,representation="GRAPHEMATIC_TXT",
                            metadata=dict(meta),lines=lines))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("source",choices=["vms","nuremberg","cremma"]);ap.add_argument("--out-prefix",required=True)
    a=ap.parse_args()
    if a.source=="vms":
        x=vms();open(a.out_prefix+".json","w",encoding="utf-8").write(json.dumps(x,ensure_ascii=False))
        print("ADAPTER",x["label"],len(x["lines"]),sum(len(r["tokens"]) for r in x["lines"]),x["source_sha256"])
    elif a.source=="nuremberg":
        u,e=nuremberg()
        open(a.out_prefix+"_unexpanded.json","w",encoding="utf-8").write(json.dumps(u,ensure_ascii=False))
        open(a.out_prefix+"_expanded.json","w",encoding="utf-8").write(json.dumps(e,ensure_ascii=False))
        print("ADAPTER",u["label"],len(u["lines"]),sum(len(r["tokens"]) for r in u["lines"]),u["source_sha256"])
        print("ADAPTER",e["label"],len(e["lines"]),sum(len(r["tokens"]) for r in e["lines"]),e["source_sha256"])
    else:
        xs=cremmas()
        manifest=[]
        for i,x in enumerate(xs):
            slug=re.sub(r"[^A-Za-z0-9._-]+","_",x["label"])
            path=a.out_prefix+"_"+slug+".json"
            open(path,"w",encoding="utf-8").write(json.dumps(x,ensure_ascii=False))
            row=dict(path=path,label=x["label"],lines=len(x["lines"]),tokens=sum(len(r["tokens"]) for r in x["lines"]),metadata=x["metadata"])
            manifest.append(row);print("ADAPTER",json.dumps(row,ensure_ascii=False))
        open(a.out_prefix+"_manifest.json","w",encoding="utf-8").write(json.dumps(manifest,ensure_ascii=False))

if __name__=="__main__":main()

#!/usr/bin/env python3
import collections,hashlib,io,itertools,json,random,tarfile,urllib.request,xml.etree.ElementTree as ET
import vms_neurocipher_ger1_20261005 as v

REF_URL=v.REF_URL
SCRAMBLE_SEEDS=[9001,7001,11003,17011]

def child(el,name):
    for x in el:
        if x.tag.split("}")[-1]==name:return x.attrib.get("tag","")
    return ""
def header(root):
    h=next((x for x in root.iter() if x.tag.split("}")[-1]=="header"),None);d={}
    for line in ((h.text or "") if h is not None else "").splitlines():
        if ":" in line:
            k,val=line.split(":",1);d[k.strip().lower()]=val.strip()
    return d

print("GER2_PRECOMPUTE=download_ref_once",flush=True)
rb=urllib.request.urlopen(REF_URL,timeout=300).read()
print("GER2_PRECOMPUTE=downloaded bytes="+str(len(rb)),flush=True)
tar=tarfile.open(fileobj=io.BytesIO(rb),mode="r:gz")
freq={"BAV":collections.Counter(),"ALEM":collections.Counter()}
docs=collections.Counter()
for name in [n for n in tar.getnames() if n.endswith(".xml")]:
    try:root=ET.fromstring(tar.extractfile(name).read())
    except:continue
    md=header(root);med=md.get("medium","").lower();tm=md.get("time","").lower();area=md.get("language-area","").lower()
    if "handschrift" not in med or not tm.startswith("15,"):continue
    bav=(("bair" in area or "bayr" in area or "österreich" in area or "oesterreich" in area) and "alemann" not in area)
    alem=(("alemann" in area or "schwäb" in area or "elsäss" in area) and "bair" not in area and "bayr" not in area)
    side="BAV" if bav else ("ALEM" if alem else None)
    if not side:continue
    docs[side]+=1
    for tok in root.iter():
        if tok.tag.split("}")[-1]!="token":continue
        for x in [z for z in tok if z.tag.split("}")[-1] in ("tok_anno","mod")]:
            form=(x.attrib.get("ascii") or x.attrib.get("utf") or x.attrib.get("trans") or "").strip().lower()
            pos=child(x,"pos")
            if form and pos and not pos.startswith("$") and all(ch.isalpha() for ch in form):
                freq[side][form]+=1
print("GER2_PRECOMPUTE=parsed",flush=True)

out={
 "schema":"VMS_NEUROCIPHER_GER2_CACHE_V1",
 "screen_commit":"a6516bb5733fc82bf77da383800dcacbf28cac5e",
 "solver_commit":"e36031a9a6b8b67fcebb4d6f4af1c3753fad4287",
 "vms":{"discovery":v.DISC,"validation":v.VAL,"final_unseen_discovery":v.FIN_UNSEEN,
        "final_strict_novel":v.FIN_STRICT,"chars":v.VCHARS,"audit":v.V_AUDIT},
 "dialects":{}
}
for d in ("BAV","ALEM"):
    ranked=[w for w,n in freq[d].most_common()]
    base=ranked[:v.FULL_K]
    if len(base)!=v.FULL_K:raise RuntimeError((d,len(base)))
    scr={};audits={}
    for ss in SCRAMBLE_SEEDS:
        z,a=v.scrambled_unique_vocab(base,ss);scr[str(ss)]=z;audits[str(ss)]=a
    out["dialects"][d]={"docs":docs[d],"ref_types":len(ranked),"real":base,
                        "scramble":scr,"scramble_audit":audits}
raw=json.dumps(out,ensure_ascii=False,separators=(",",":"))
print("GER2_CACHE_SHA="+hashlib.sha256(raw.encode()).hexdigest(),flush=True)
print("GER2_CACHE_JSON="+raw,flush=True)

#!/usr/bin/env python3
# VMS R2 topology-only alignment checkpoint.
import hashlib,json,re,urllib.request

ZL_URL="https://raw.githubusercontent.com/noah-chelednik/voynich-data/472ef7366606a799fc8f1044c037e06b413f6ddd/data_sources/cache/ZL3b-n.txt"
ZL_SHA="bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc"
CORPUS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json"

zl=urllib.request.urlopen(ZL_URL,timeout=120).read()
got=hashlib.sha256(zl).hexdigest()
if got!=ZL_SHA: raise RuntimeError(("ZL_SHA_MISMATCH",got))
txt=zl.decode("utf-8")
obj=json.loads(urllib.request.urlopen(CORPUS_URL,timeout=120).read())

TARGET=set(f"f{n}{s}" for n in list(range(103,109))+list(range(111,117)) for s in ("r","v"))
TARGET.discard("f116v")

entries=[];current={}
pat=re.compile(r"^<(f\d+[rv]\d*)\.(\d+),[^>]+>\s+(.*)$")
for ln in txt.splitlines():
    m=pat.match(ln)
    if not m: continue
    fol,lno,body=m.group(1),int(m.group(2)),m.group(3)
    if fol not in TARGET: continue
    if "<%>" in body:
        if fol in current: raise RuntimeError(("nested_entry",fol,lno))
        current[fol]={"folio":fol,"eno":1+sum(e["folio"]==fol for e in entries),"lines":[]}
    if fol in current: current[fol]["lines"].append(lno)
    if "<$>" in body and fol in current: entries.append(current.pop(fol))
if current: raise RuntimeError(("open_entries",sorted(current)))
if len(entries)!=285: raise RuntimeError(("entry_count",len(entries)))

line_to_entry={}
for eid,e in enumerate(entries):
    for li,lno in enumerate(e["lines"]):
        key=(e["folio"],int(lno))
        if key in line_to_entry: raise RuntimeError(("duplicate_line",key))
        line_to_entry[key]=(eid,li,len(e["lines"]))

# Strict Recipes +P0 line universe; count scored events as all but opener.
n_lines=0;n_events=0;unmapped=[]
for fol,lines in obj["pages"].items():
    m=re.match(r"f(\d+)",fol)
    if not m: continue
    n=int(m.group(1))
    if not (103<=n<=116): continue
    if fol=="f116v": continue
    for ls,rec in lines.items():
        if str(rec.get("u",""))!="+P0": continue
        toks=[t.lower() for t in str(rec.get("t",{}).get("ZLZI","")).split() if re.fullmatch(r"[a-z]+",t.lower())]
        if not toks: continue
        n_lines+=1;n_events+=max(0,len(toks)-1)
        if (fol,int(ls)) not in line_to_entry: unmapped.append((fol,int(ls),len(toks)))

if n_lines!=1083 or n_events!=9616 or unmapped:
    raise RuntimeError(("TOPOLOGY_ALIGN_FAIL",n_lines,n_events,unmapped[:20],len(unmapped)))

multi=sum(1 for e in entries if len(e["lines"])>1)
out={
 "programme":"VMS-R2-TOPOLOGY",
 "status":"complete",
 "zl_sha":got,
 "entries":len(entries),
 "multi_line_entries":multi,
 "strict_plusp0_lines":n_lines,
 "scored_events":n_events,
 "unmapped_lines":0
}
print("R2_TOPOLOGY="+json.dumps(out,separators=(",",":")),flush=True)

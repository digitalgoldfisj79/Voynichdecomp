#!/usr/bin/env python3
import glob,os,re,sys,io,zipfile,urllib.request,json,lzma,hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
URL="https://zenodo.org/record/13982324/files/ReM-v2.1_tei.zip?download=1"
NS="{http://www.tei-c.org/ns/1.0}";XMLID="{http://www.w3.org/XML/1998/namespace}id"
CLEAN=re.compile(r"[\[\]<>|\\/*()=+#%$\"'{}0-9\-.,;:!?]")
EXPECTED_CONTENT_SHA="79f298a9f40b27da413bd07cf445de834038fa0ebc1749f470685dd263fc463f"
def build(tei_dir="ReM-v2.1_tei/tei"):
    docs={}
    for fp in sorted(glob.glob(os.path.join(tei_dir,"*.xml"))):
        did=os.path.basename(fp)[:-4];root=ET.parse(fp).getroot();groups,order=defaultdict(str),[]
        for w in root.iter(NS+"w"):
            wid=w.get(XMLID) or "";base=wid.split("_m")[0] if "_m" in wid else wid
            txt=re.sub(r"\s+","","".join(w.itertext()))
            if base not in groups:order.append(base)
            groups[base]+=txt
        words=[CLEAN.sub("",groups[b].lower()) for b in order if groups[b]];words=[w for w in words if w]
        if words:docs[did]=words
    return docs
if __name__=="__main__":
    out=sys.argv[1] if len(sys.argv)>1 else "rem_docs.json.xz"
    if not os.path.isdir("ReM-v2.1_tei"):
        print("downloading ReM",flush=True);data=urllib.request.urlopen(URL,timeout=600).read();zipfile.ZipFile(io.BytesIO(data)).extractall(".")
    docs=build();n=sum(map(len,docs.values()));chars=sum(len(w) for v in docs.values() for w in v);alpha=len({c for v in docs.values() for w in v for c in w})
    assert (len(docs),n,chars,alpha)==(406,2236137,9967570,118),(len(docs),n,chars,alpha)
    canonical=json.dumps(docs,ensure_ascii=False,separators=(",",":")).encode("utf-8")
    content_sha=hashlib.sha256(canonical).hexdigest();assert content_sha==EXPECTED_CONTENT_SHA,(content_sha,EXPECTED_CONTENT_SHA)
    raw=lzma.compress(canonical);open(out,"wb").write(raw)
    print(json.dumps({"docs":len(docs),"tokens":n,"chars":chars,"alpha":alpha,"content_sha256":content_sha,"file_sha256":hashlib.sha256(raw).hexdigest()}),flush=True)

#!/usr/bin/env python3
import glob,os,re,sys,io,zipfile,urllib.request,json,lzma,hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
URL="https://zenodo.org/record/13982324/files/ReM-v2.1_tei.zip?download=1"
NS="{http://www.tei-c.org/ns/1.0}";XMLID="{http://www.w3.org/XML/1998/namespace}id"
CLEAN=re.compile(r"[\[\]<>|\\/*()=+#%$\"'{}0-9\-.,;:!?]")
EXPECTED_SHA="ac7e5c24743c7e8faac819a5f331ec99baecc6e5aef37025294c4252d4d4487c"
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
    raw=lzma.compress(json.dumps(docs,ensure_ascii=False,separators=(",",":")).encode("utf-8"));open(out,"wb").write(raw)
    sha=hashlib.sha256(raw).hexdigest();print(json.dumps({"docs":len(docs),"tokens":n,"chars":chars,"alpha":alpha,"sha256":sha}),flush=True)
    # JSON serialization is required to match the frozen bundled corpus byte-for-byte.
    assert sha==EXPECTED_SHA,(sha,EXPECTED_SHA)

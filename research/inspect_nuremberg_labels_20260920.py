#!/usr/bin/env python3
import json, urllib.request, zipfile, tempfile, os, re
REC="https://zenodo.org/api/records/13881575"
meta=json.load(urllib.request.urlopen(REC))
files=meta.get("files",[])
print("FILES",[(f.get("key"),f.get("size"),f.get("links",{}).get("self")) for f in files])
lab=next(f for f in files if f.get("key")=="labels.zip")
url=lab["links"]["self"]
path="/tmp/labels.zip"
print("DOWNLOADING",url)
urllib.request.urlretrieve(url,path)
print("ZIP_SIZE",os.path.getsize(path))
with zipfile.ZipFile(path) as z:
    names=z.namelist()
    print("NFILES",len(names))
    for n in names[:120]: print("NAME",n)
    xmls=[n for n in names if n.lower().endswith(".xml")]
    print("NXML",len(xmls))
    for n in xmls[:8]:
        b=z.read(n)
        print("XML",n,"BYTES",len(b))
        print(b[:5000].decode("utf-8","replace"))
        print("----")

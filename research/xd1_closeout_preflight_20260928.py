#!/usr/bin/env python3
import hashlib, urllib.request

WITNESSES = ("a1","bs1","so1","w1")
TEMPLATES = (
    "https://gams.uni-graz.at/archive/objects/o:corema.{w}/datastreams/TEI_SOURCE/content",
    "https://gams.uni-graz.at/o:corema.{w}/TEI_SOURCE",
    "https://gams.uni-graz.at/o:corema.{w}",
)

for w in WITNESSES:
    ok = False
    errs = []
    for template in TEMPLATES:
        url = template.format(w=w)
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"XD1-closeout-20260928"})
            data = urllib.request.urlopen(req, timeout=60).read()
            head = data[:100].decode("utf-8","replace").replace("\n"," ")
            print(f"SOURCE_OK witness={w.upper()} url={url} bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()} head={head!r}")
            ok = True
            break
        except Exception as e:
            errs.append(f"{url} -> {type(e).__name__}: {e}")
    if not ok:
        print(f"SOURCE_FAIL witness={w.upper()} :: " + " || ".join(errs))
        raise SystemExit(2)

#!/usr/bin/env python3
"""Source-only STA/RF count diagnostic. No repetition statistic is computed."""
import hashlib,re,urllib.request
URL="https://voynich.nu/data/sta/RF1b.txt"
CODE_RE=re.compile(r"[A-Z][0-9a-z]")
LOCUS_RE=re.compile(r"^<(?P<loc>f[^,>]+),(?P<mode>[^>]+)>\s*(?P<body>.*)$")
EXPECTED_LOCI=5385
EXPECTED_CODES=157254
req=urllib.request.Request(URL,headers={"User-Agent":"XD1-STA-RF-source-diagnostic-20260928"})
data=urllib.request.urlopen(req,timeout=120).read()
text=data.decode("utf-8")
loci=codes=long_words=short_words=uncertain_markers=0
for raw in text.splitlines():
    m=LOCUS_RE.match(raw.strip())
    if not m: continue
    loci+=1
    body=m.group("body")
    uncertain_markers += body.count("<->")
    # Long-word convention: remove uncertain separators before dot splitting.
    for chunk in body.replace("<->","").split("."):
        cc=CODE_RE.findall(chunk)
        if cc:
            long_words+=1
            codes+=len(cc)
    # Short-word diagnostic: uncertain separators act as boundaries.
    for chunk in re.split(r"\.|<->",body):
        if CODE_RE.findall(chunk):
            short_words+=1
print(f"STA_SOURCE_DIAGNOSTIC sha256={hashlib.sha256(data).hexdigest()} loci={loci} codes={codes} long_words={long_words} short_words={short_words} uncertain_markers={uncertain_markers}")
assert loci==EXPECTED_LOCI,(loci,EXPECTED_LOCI)
assert codes==EXPECTED_CODES,(codes,EXPECTED_CODES)

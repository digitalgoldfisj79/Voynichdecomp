#!/usr/bin/env python3
from __future__ import annotations
import collections, hashlib, json, re, urllib.request

RF_URL='https://www.voynich.nu/data/sta/RF1b.txt'
RF_SHA='81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17'
HEADERS={
 'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36',
 'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
 'Accept-Language':'en-GB,en;q=0.9',
 'Referer':'https://www.voynich.nu/extra/sta.html'
}

def main():
    req=urllib.request.Request(RF_URL,headers=HEADERS)
    b=urllib.request.urlopen(req,timeout=60).read()
    sha=hashlib.sha256(b).hexdigest()
    assert sha==RF_SHA,(sha,RF_SHA)
    text=b.decode('utf-8')
    rx=re.compile(r'[A-Z][1-9a-z]')
    counts=collections.Counter(); total=0
    for line in text.splitlines():
        if not line.startswith('<') or '>' not in line: continue
        lab,rhs=line.split('>',1)
        if '.' not in lab: continue
        rhs=re.sub(r'\[[^\]]*\]','<BREAK>',rhs)
        for p in re.split(r'<(?:-|~)>|<BREAK>',rhs):
            toks=rx.findall(p)
            if toks:
                counts.update(toks); total+=len(toks)
    order=sorted(counts,key=lambda x:(-counts[x],x))
    k=92; kept=sum(counts[x] for x in order[:k])
    out={
      'source_sha256':sha,
      'parsed_full_sta_chars':total,
      'observed_member_types':len(counts),
      'K':k,
      'coverage':kept/total,
      'retained_events':kept,
      'tail_events':total-kept,
      'vocab':order[:k],
      'gate': bool(total==157254 and len(counts)>=92 and kept/total>=.995)
    }
    print('B0_RF_STA92='+json.dumps(out,separators=(',',':')),flush=True)
    raise SystemExit(0 if out['gate'] else 3)

if __name__=='__main__': main()

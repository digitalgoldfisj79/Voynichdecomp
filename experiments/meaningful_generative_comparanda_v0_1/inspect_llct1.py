#!/usr/bin/env python3
import sys, xml.etree.ElementTree as ET
from collections import Counter, defaultdict
p=sys.argv[1]
tagc=Counter(); attrc=Counter(); interesting=[]; vals=defaultdict(Counter)
for ev,e in ET.iterparse(p,events=('start','end')):
    tag=e.tag.split('}')[-1]
    if ev=='start':
        tagc[tag]+=1
        for k,v in e.attrib.items():
            kk=k.split('}')[-1]; attrc[(tag,kk)]+=1
            low=(kk+' '+v).lower()
            if any(x in low for x in ['formula','free','sentence','sent','form']): vals[(tag,kk)][v]+=1
        if any(x in tag.lower() for x in ['formula','free','sentence','sent','form','token','word']):
            if len(interesting)<200: interesting.append(('start',tag,dict(e.attrib), (e.text or '')[:100]))
    else:
        if any(x in tag.lower() for x in ['formula','free','sentence','sent','form','token','word']):
            if len(interesting)<200: interesting.append(('end',tag,dict(e.attrib), (e.text or '')[:100]))
        e.clear()
print('TOP TAGS')
for x,n in tagc.most_common(40): print(x,n)
print('\nATTRS')
for (t,k),n in attrc.most_common(80): print(t,k,n)
print('\nINTERESTING VALUES')
for key,c in vals.items():
    print(key,c.most_common(30))
print('\nSAMPLE ELEMENTS')
for row in interesting[:200]: print(repr(row))

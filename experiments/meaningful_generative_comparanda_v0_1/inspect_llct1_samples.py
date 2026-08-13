#!/usr/bin/env python3
import sys, xml.etree.ElementTree as ET
p=sys.argv[1]
count=roots=0
for ev,e in ET.iterparse(p,events=('start',)):
    if e.tag.split('}')[-1] != 'LM':
        continue
    a=dict(e.attrib)
    if count < 60:
        print('LM',count,repr(a))
        count += 1
    if 'document_id' in a and roots < 30:
        print('ROOT',roots,repr(a))
        roots += 1
    if count>=60 and roots>=30:
        break

#!/usr/bin/env python3
import sys, xml.etree.ElementTree as ET
from collections import Counter
p=sys.argv[1]
tree=ET.parse(p)
segc=Counter(); sent_classes=Counter(); mixed=[]; roots=0
for e in tree.getroot().iter():
    if e.tag.split('}')[-1]=='LM' and 'document_id' in e.attrib:
        roots+=1
        toks=[]
        for d in e.iter():
            if d is e: continue
            if d.tag.split('}')[-1]=='LM' and 'form' in d.attrib:
                toks.append(d.attrib)
        ss=tuple(sorted(set(a.get('seg','') for a in toks)))
        sent_classes[ss]+=1
        for a in toks: segc[a.get('seg','')]+=1
        if len(ss)>1 and len(mixed)<20:
            mixed.append((e.attrib, [(a['id'],a.get('form'),a.get('seg')) for a in sorted(toks,key=lambda a:int(a['id']))]))
print('roots',roots)
print('seg_counts',segc)
print('sentence_class_sets',sent_classes)
print('mixed_examples')
for x in mixed: print(repr(x))

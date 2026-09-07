import urllib.request,re,collections
u='https://raw.githubusercontent.com/CS-433/ml-project-2-scikit-learn2/e370e7f14f49c30026b572fa7bcd06b5fd1ec1c7/data/GC_ivtff_0c.txt'
s=urllib.request.urlopen(u).read().decode('utf-8','ignore')
print('raw8am',len(re.findall(r'8am',s)))
for split_commas in (False,True):
    toks=[]
    lines=0
    for ln in s.splitlines():
        m=re.match(r'^<([^>]+)>\s+(.*)$',ln)
        if not m:
            continue
        locus=m.group(1)
        if '.' not in locus:
            continue
        lines+=1
        t=m.group(2)
        t=re.sub(r'<[^>]*>','',t)
        t=re.sub(r'@[0-9]+;','X',t)
        parts=re.split(r'[.,]' if split_commas else r'[.]',t)
        toks.extend([x.strip() for x in parts if x.strip()])
    c=collections.Counter(toks)
    print('mode','dotcomma' if split_commas else 'dot','lines',lines,'tokens',len(toks),'types',len(c),'8am',c['8am'],'top10',c.most_common(10))

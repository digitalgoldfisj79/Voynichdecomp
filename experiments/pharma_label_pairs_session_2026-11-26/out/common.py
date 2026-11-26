import csv, re, pickle
C=list(csv.DictReader(open('data/concordance_upload.csv')))
def norm(s):
    s=re.sub(r'\[([^:\]]*):[^\]]*\]',r'\1',s); s=re.sub(r'<[^>]*>','',s); s=re.sub(r'[{}?!<>0-9 ]','',s); return s
def vords(s): return [v for v in re.split(r'[.,]',norm(s)) if v]
def joined(s): return ''.join(vords(s))

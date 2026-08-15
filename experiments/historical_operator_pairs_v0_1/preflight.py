from __future__ import annotations
import json, re, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path('experiments/historical_operator_pairs_v0_1/preflight_out')
OUT.mkdir(parents=True, exist_ok=True)

def classify_names(names):
    toks = ['dipl','diplom','abbrev','abbrevi','expand','expan','regular','basic','alto','text','tei','page','xml','norm']
    c = Counter()
    examples = defaultdict(list)
    for n in names:
        low=n.lower()
        for t in toks:
            if t in low:
                c[t]+=1
                if len(examples[t])<12: examples[t].append(n)
    return c, examples

def xml_probe(zf, names, label):
    probes=[]
    xmls=[n for n in names if n.lower().endswith(('.xml','.alto'))]
    ranked=sorted(xmls, key=lambda n:(0 if any(x in n.lower() for x in ['dipl','alto','text','tei']) else 1, len(n)))
    for n in ranked[:80]:
        try:
            b=zf.read(n)
        except Exception:
            continue
        s=b[:250000].decode('utf-8','replace')
        score=sum(s.lower().count(x) for x in ['<expan','<abbr','<ex>','&lt;expan','&lt;abbr','<unicode>','content='])
        if score or len(probes)<12:
            probes.append({
                'name':n,'bytes':len(b),'score':score,
                'counts':{x:s.lower().count(x) for x in ['<expan','<abbr','<ex>','&lt;expan','&lt;abbr','<unicode>','content=','<string']},
                'snippet':re.sub(r'\s+',' ',s[:4000])[:4000]
            })
        if len(probes)>=30: break
    Path(OUT/f'{label}_probes.json').write_text(json.dumps(probes,ensure_ascii=False,indent=2))
    return probes

def inspect_zip(path,label):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        c,ex=classify_names(names)
        top=Counter('/'.join(n.split('/')[:min(4,len(n.split('/')))]) for n in names)
        info={
            'label':label,'file_count':len(names),'xml_count':sum(n.lower().endswith('.xml') for n in names),
            'first_300_names':names[:300],
            'token_counts':dict(c),'token_examples':dict(ex),
            'extensions':dict(Counter(Path(n).suffix.lower() for n in names if Path(n).suffix).most_common(30)),
            'top_prefixes':top.most_common(100)
        }
        probes=xml_probe(z,names,label)
        info['probe_count']=len(probes)
    Path(OUT/f'{label}_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    return info

def main():
    if len(sys.argv)!=3:
        raise SystemExit('usage: preflight.py NUREMBERG.zip ORIFLAMMS.zip')
    a=inspect_zip(sys.argv[1],'nuremberg')
    b=inspect_zip(sys.argv[2],'oriflamms')
    summary={'nuremberg':a,'oriflamms':b}
    Path(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('NUREMBERG', a['file_count'], 'files', a['xml_count'], 'xml', a['token_counts'])
    print('ORIFLAMMS', b['file_count'], 'files', b['xml_count'], 'xml', b['token_counts'])

if __name__=='__main__': main()

#!/usr/bin/env python3
import io, os, re, json, tarfile, urllib.request, hashlib
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
import numpy as np
from bs4 import BeautifulSoup

# Import frozen v1.0 machinery without executing its main.
PARENT='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/c7c50f74e1f1f88004a0f08ea379324a3d42c16d/experiments/bnf_m19_german_confirm_v1_0/run_confirm.py'
src=urllib.request.urlopen(PARENT,timeout=90).read().decode('utf-8')
src=src.rsplit("if __name__=='__main__':main()",1)[0]
lib={'__name__':'parent'}; exec(compile(src,'run_confirm.py','exec'),lib)
b=lib['b']; inner=lib['inner']; M=lib['M']; SYMS=lib['SYMS']

REF_TABLE='https://www.linguistics.rub.de/ref/corpus/texts.html'
REF_ARCH='https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content'
WINDOW={'14,2','15,1','15,2'}


def seed(*x):
    return int.from_bytes(hashlib.sha256(('20260809|WHYREF|'+'|'.join(map(str,x))).encode()).digest()[:8],'big') & 0xffffffff


def metadata():
    h=urllib.request.urlopen(REF_TABLE,timeout=90).read()
    s=BeautifulSoup(h,'html.parser'); t=s.find('table')
    heads=[x.get_text(' ',strip=True) for x in t.find_all('th')]
    out={}
    for r in t.find_all('tr')[1:]:
        cells=[x.get_text(' ',strip=True) for x in r.find_all(['td','th'])]
        if len(cells)<9: continue
        row=dict(zip(heads,cells)); fid=row.get('ID','').strip()
        if fid: out[fid]=row
    return out


def norm_token(x):
    # CorA diplomatic utf can contain grapheme-separating spaces; parent norm_words
    # supplies Unidecode and the frozen j/v/w normalization.
    ws=b['norm_words'](x)
    return ''.join(ws) if ws else ''


def parse_xml(blob):
    try: root=ET.fromstring(blob)
    except Exception: return []
    toks=[]
    # Primary: diplomatic token layer.
    for e in root.iter():
        tag=e.tag.split('}')[-1]
        if tag=='dipl':
            x=e.attrib.get('utf') or e.attrib.get('trans') or (e.text or '')
            z=norm_token(x)
            if z: toks.append(z)
    if toks: return toks
    # Fallback for UP-style material: anno ascii/utf.
    for e in root.iter():
        tag=e.tag.split('}')[-1]
        if tag=='anno':
            x=e.attrib.get('ascii') or e.attrib.get('utf') or (e.text or '')
            z=norm_token(x)
            if z: toks.append(z)
    return toks


def load_ref(meta):
    p='/tmp/ReF-v1.0.2.tar.gz'
    if not os.path.exists(p): urllib.request.urlretrieve(REF_ARCH,p)
    tf=tarfile.open(p,'r:gz'); byid={}
    for member in tf.getmembers():
        if not member.isfile() or not member.name.endswith('.xml'): continue
        fid=os.path.basename(member.name)[:-4]
        if fid not in meta: continue
        f=tf.extractfile(member)
        if f is None: continue
        toks=parse_xml(f.read())
        # Prefer whichever representation supplies more diplomatic material.
        if len(toks)>len(byid.get(fid,[])): byid[fid]=toks
    return byid


def lm_from_docs(docs):
    # build_lm iterates words in each string; document boundaries do not induce transitions.
    return b['build_lm']([' '.join(ws) for ws in docs if ws])


def split_docs(items,label):
    # items = [(fid, words)] deterministic doc-disjoint split; accumulate ~20% hold.
    a=sorted(items,key=lambda q:seed(label,q[0])); total=sum(sum(map(len,w)) for _,w in a); target=max(10000,int(.20*total)); hold=[]; hn=0
    for x in a:
        if len(a)-len(hold)<=1: break
        hold.append(x); hn+=sum(map(len,x[1]))
        if hn>=target: break
    hs={x[0] for x in hold}; train=[x for x in a if x[0] not in hs]
    return train,hold


def invert_map():
    d=defaultdict(list)
    for i,s in enumerate(SYMS): d[int(M[i])].append(s)
    return d


def encode_words(words,tag,nletters=25000):
    forms=invert_map(); rng=np.random.default_rng(seed('enc',tag)); out=[]; n=0
    for w in words:
        z=[]
        for c in w:
            if c not in b['A2I']: continue
            vals=b['LETTER_VALS'][b['A2I'][c]]
            raw=int(rng.choice(vals)); vi=b['V2I'][raw]
            if vi not in forms: raise RuntimeError(('missing frozen value',raw,vi))
            z.append(str(rng.choice(forms[vi]))); n+=1
            if n>=nletters: break
        if z: out.append(''.join(z))
        if n>=nletters: break
    return out


def rank(words,lms):
    rows=[]
    for la,lm in lms.items():
        sc,n,sk,cov=inner['forward_words'](words,M,SYMS,lm); rows.append((la,float(sc),int(n),float(cov)))
    rows.sort(key=lambda x:x[1],reverse=True); return rows


def main():
    meta=metadata(); print('META',len(meta),flush=True)
    docs=load_ref(meta); print('XML_DOCS',len(docs),flush=True)
    chosen={fid:ws for fid,ws in docs.items() if meta[fid].get('Datierung') in WINDOW and meta[fid].get('Dialekt')}
    groups=defaultdict(list)
    for fid,ws in chosen.items(): groups[meta[fid]['Dialekt']].append((fid,ws))
    census=[]
    for g,xs in sorted(groups.items()): census.append({'dialect':g,'docs':len(xs),'letters':sum(sum(map(len,w)) for _,w in xs),'ids':[x[0] for x in xs]})
    print('REF_CENSUS='+json.dumps(census,ensure_ascii=False,separators=(',',':')),flush=True)

    # Exact dialect models: threshold from frozen protocol.
    models={}; holds={}; model_meta={}
    for g,xs in sorted(groups.items()):
        total=sum(sum(map(len,w)) for _,w in xs)
        if len(xs)<2 or total<40000: continue
        tr,ho=split_docs(xs,g); tn=sum(sum(map(len,w)) for _,w in tr); hn=sum(sum(map(len,w)) for _,w in ho)
        if tn<30000 or hn<10000: continue
        models[g]=lm_from_docs([w for _,w in tr]); holds[g]=[w for _,ws in ho for w in ws]
        model_meta[g]={'train_docs':[f for f,_ in tr],'hold_docs':[f for f,_ in ho],'train_letters':tn,'hold_letters':hn}
    print('QUALIFIABLE='+json.dumps(model_meta,ensure_ascii=False,separators=(',',':')),flush=True)

    # Add aggregates from full window; these are ranking comparators, not dialect qualification targets.
    bav_labels=[g for g in groups if 'bairisch' in g.lower()]
    bav_docs=[ws for g in bav_labels for _,ws in groups[g]]
    all_docs=[ws for xs in groups.values() for _,ws in xs]
    aggregates={'ReF_1350_1500':lm_from_docs(all_docs)}
    if bav_docs: aggregates['ReF_Bavarian_1350_1500']=lm_from_docs(bav_docs)
    nonb=[ws for g,xs in groups.items() if 'bairisch' not in g.lower() for _,ws in xs]
    if nonb: aggregates['ReF_nonBavarian_1350_1500']=lm_from_docs(nonb)

    # Dialect positive controls through the actual M19 channel and unchanged frozen surface map.
    q=[]
    for g,plain in holds.items():
        cipher=encode_words(plain,('qual',g),25000); rr=rank(cipher,models); pos=1+next(i for i,x in enumerate(rr) if x[0]==g); margin=(rr[0][1]-rr[1][1]) if rr and rr[0][0]==g and len(rr)>1 else None
        row={'dialect':g,'rank':pos,'margin':margin,'ranking':[(x[0],x[1]) for x in rr],'cipher_letters':sum(map(len,cipher))};q.append(row); print('DIALECT_QUAL',json.dumps(row,ensure_ascii=False,separators=(',',':')),flush=True)

    # Frozen C10 panel.
    data=json.loads(b['fetch'](b['SLIM'])); sample,hold,pages,required=inner['split_vms'](data); T={f for f,_,_ in sample};H={f for f,_,_ in hold};A={f for f,_,_ in pages};C=sorted(A-T-H); words=lib['words_for'](data,C,'ZLZI')
    historical={**models,**aggregates}; hr=rank(words,historical)
    print('VMS_HIST_RANK='+json.dumps([(x[0],x[1],x[2],x[3]) for x in hr],ensure_ascii=False,separators=(',',':')),flush=True)

    # Viterbi vocabulary-hit diagnostics for qualified dialects and aggregates; <=2 separated.
    lex={}
    for la,lm in historical.items():
        cnt=Counter(words); hit=tot=hit3=tot3=0
        for w,n in cnt.items():
            dec=b['viterbi'](w,M,SYMS,lm)
            if dec is None: continue
            tot+=n; hit+=n*int(dec in lm['vocab'])
            if len(dec)>=3: tot3+=n; hit3+=n*int(dec in lm['vocab'])
        lex[la]={'all_fraction':hit/max(1,tot),'hits':hit,'tokens':tot,'len3plus_fraction':hit3/max(1,tot3),'hits3plus':hit3,'tokens3plus':tot3}
    print('HIST_LEX='+json.dumps(lex,ensure_ascii=False,separators=(',',':')),flush=True)

    out={'meta_n':len(meta),'xml_docs':len(docs),'census':census,'qualifiable':model_meta,'dialect_controls':q,'vms_historical_ranking':[(x[0],x[1]) for x in hr],'lexical':lex}
    print('RESULT_JSON='+json.dumps(out,ensure_ascii=False,separators=(',',':')),flush=True)

if __name__=='__main__': main()

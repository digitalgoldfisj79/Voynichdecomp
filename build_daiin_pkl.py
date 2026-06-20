#!/usr/bin/env python3
"""
Build a clean, self-documented pickle of the daiin.net (K. Hamidullin) .vml transliteration.
Output: daiin_vms.pkl  -- dict with 'meta', 'folios', 'corpus_freq'.
Re-derivable: point VML_DIR at the unzipped daiin .vml folder and run.
"""
import re, os, glob, pickle, datetime
from collections import Counter
VML_DIR="vml_data"; OUT="daiin_vms.pkl"
HDR=re.compile(r'folio\s*\{\s*\{\s*(\d+)\s*,\s*(\d+)\s*\}\s*,\s*([0-9]+|null)\s*,\s*"([^"]*)"')
WORD_BB=re.compile(r'word\s*\{\s*"((?:[^"\\]|\\.)*)"(?:\s*,\s*rect\s*\{\s*\{\s*(\d+)\s*,\s*(\d+)\s*\}\s*,\s*\{\s*(\d+)\s*,\s*(\d+)\s*\}\s*\})?')
MASTER=re.compile(r'(?P<word>word\s*\{\s*"(?:[^"\\]|\\.)*")|(?P<brk>\b(?:line|column|panel)\s*\{)')
def parse(fn):
    raw=open(fn,'rb').read().decode('utf-8','replace')
    m=HDR.search(raw)
    w=h=quire=lang=None
    if m: w,h=int(m.group(1)),int(m.group(2)); quire=None if m.group(3)=='null' else int(m.group(3)); lang=m.group(4)
    bb={mm.start():(int(mm.group(2)),int(mm.group(3)),int(mm.group(4)),int(mm.group(5))) if mm.group(2) else None
        for mm in WORD_BB.finditer(raw)}
    lines=[]; cur=[]; boxes=[]
    for tok in MASTER.finditer(raw):
        if tok.group('word'):
            v=re.search(r'"((?:[^"\\]|\\.)*)"',tok.group('word')).group(1)
            cur.append(v); boxes.append((v, bb.get(tok.start())))
        elif cur: lines.append(cur); cur=[]
    if cur: lines.append(cur)
    return dict(folio=os.path.basename(fn)[:-4], lang=lang, quire=quire, width=w, height=h,
                lines=lines, tokens=[v for ln in lines for v in ln], boxes=boxes)
def main():
    files=[f for f in sorted(glob.glob(f"{VML_DIR}/*.vml")) if os.path.basename(f) not in ("def.vml","vms.vml")]
    folios={}
    for f in files:
        d=parse(f)
        if d['tokens']: folios[d['folio']]=d
    corpus=Counter(t for d in folios.values() for t in d['tokens'])
    meta=dict(
        source="daiin.net (K. Hamidullin) .vml transliteration; browser based on the Voynichese project",
        built=str(datetime.date.today()),
        n_folios=len(folios), n_tokens=sum(len(d['tokens']) for d in folios.values()), n_types=len(corpus),
        transliteration="daiin.net BASE stored values (raw); the site's 'voynichedy' weighted variant-expansion layer is NOT applied",
        caveats=[
            "Word DIVISION differs from EVA/ZL/enriched_records (e.g. f105r first word 'paiindar' vs 'paiin'); token counts are NOT comparable across transliterations.",
            "'lang' is Currier A/B at folio level, NOT content-section (Herbal/Pharmaceutical/Recipes...). e.g. f102r2 = lang 'A'.",
            "Alphabet is a daiin variant of EVA (uses 'z', capital C-ligatures per regex C?[a-z][A-BD-Z0-9]*).",
            "Excludes def.vml (grammar) and vms.vml (index). Includes 'ros' (rosettes foldout) keyed as 'ros'.",
            "bbox coords are in daiin image space (per-folio width/height in this record), ~half Yale IIIF resolution.",
        ],
        schema="{'meta':{...}, 'folios': {name: {folio,lang,quire,width,height, lines:[[tok,...],...], tokens:[tok,...], boxes:[(tok,(x0,y0,x1,y1)|None),...]}}, 'corpus_freq': {tok:count}}",
        usage="import pickle; D=pickle.load(open('daiin_vms.pkl','rb')); D['folios']['f105r']['lines']; D['corpus_freq']['daiin']",
    )
    pickle.dump(dict(meta=meta, folios=folios, corpus_freq=dict(corpus)), open(OUT,"wb"))
    return meta, folios, corpus
if __name__=="__main__":
    meta,folios,corpus=main()
    # verification + manifest
    print("BUILD OK:", meta['n_folios'],"folios,",meta['n_tokens'],"tokens,",meta['n_types'],"types")
    from collections import Counter as C
    langs=C(d['lang'] for d in folios.values()); print("lang counts:",dict(langs))
    qs=[d['quire'] for d in folios.values() if d['quire'] is not None]; print("quire range:",min(qs),"-",max(qs))
    # spot checks vs what we saw
    for f in ("f105r","f102r2","f1r"):
        d=folios.get(f); print(f"  {f}: lang={d['lang']} quire={d['quire']} {len(d['tokens'])}tok {len(d['lines'])}lines first='{d['tokens'][0]}'")
    print("top tokens:", corpus.most_common(6))
    # manifest CSV
    import csv
    with open("daiin_manifest.csv","w",newline="") as fh:
        wr=csv.writer(fh); wr.writerow(["folio","lang","quire","n_tokens","n_lines","first_token"])
        for name,d in sorted(folios.items()):
            wr.writerow([name,d['lang'],d['quire'],len(d['tokens']),len(d['lines']),d['tokens'][0] if d['tokens'] else ""])
    print("wrote daiin_manifest.csv")

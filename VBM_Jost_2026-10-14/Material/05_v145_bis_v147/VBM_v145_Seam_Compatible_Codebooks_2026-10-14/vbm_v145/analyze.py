from pathlib import Path
import re,json,collections,statistics,hashlib
P=Path(__file__).resolve().parent; B=(Path(__file__).resolve().parent/'inputs'); ROOT=(Path(__file__).resolve().parent/'inputs')
M=json.loads((B/'04_pair_bridge_manifest.json').read_text()); data=json.loads((Path(__file__).resolve().parent/'inputs'/'voynich_transcriptions_slim.json').read_text())['pages'];sections=json.loads((P/'voynich_section_map.json').read_text())['mapping'];
areva=[];breva=[];bysection=collections.defaultdict(list)
# Exact Currier B metadata from ZL3b header ($L=B), recovered independently from the primary GitHub mirror.
B_PAGES=set('f26r f26v f31r f31v f33r f33v f34r f34v f39r f39v f40r f40v f41r f41v f43r f43v f46r f46v f48r f48v f50r f50v f55r f55v f57r f66r f66v f75r f75v f76r f76v f77r f77v f78r f78v f79r f79v f80r f80v f81r f81v f82r f82v f83r f83v f84r f84v f85r1 f85r2 fRos f86v4 f86v6 f86v5 f86v3 f94r f94v f95r1 f95r2 f95v2 f95v1 f103r f103v f104r f104v f105r f105v f106r f106v f107r f107v f108r f108v f111r f111v f112r f112v f113r f113v f114r f114v f115r f115v f116r'.split())
OTHER_PAGES=set('f57v f65r f65v f67r1 f67r2 f67v2 f67v1 f68r1 f68r2 f68r3 f68v3 f68v2 f68v1 f69r f69v f70r1 f70r2 f70v2 f70v1 f71r f71v f72r1 f72r2 f72r3 f72v3 f72v2 f72v1 f73r f73v f116v'.split())
bref=(ROOT/'VMS_B_REVA_reference.txt').read_text().splitlines(); b_pages={ln.split('.',1)[0] for ln in bref if '\t' in ln}
for fol, lines in data.items():
    sec=sections.get(fol,'unknown');
    if fol in OTHER_PAGES:continue
    isB=fol in B_PAGES
    # ZLZB normalized line split; only @ or + etc P line, skip uncertain units
    for i,loc in lines.items():
        if not re.match(r'[@*+=~]P',loc.get('u','')):continue
        text=loc.get('t',{}).get('ZLZB')
        if not text:continue
        ws=[w for w in text.split() if re.fullmatch(r'[a-z]+',w)]
        # original REVA reverses word order and compound glyph ordering
        vals=[''.join(reversed(re.findall('cth|ckh|cph|cfh|ch|sh|.',w))) for w in reversed(ws)]
        bysection[sec+'|'+('B' if isB else 'A')].extend(vals)
        (breva if isB else areva).extend(vals)
# Original B support from pinned  B REVA reference.
def support(words):return {w[i:i+3] for w in words for i in range(len(w)-2)}, {w[i:i+2] for w in words for i in range(len(w)-1)}
Bwords=[w for ln in bref if '\t' in ln for w in ln.split('\t',1)[1].split()]; trisB,bisB=support(Bwords);trisA,bisA=support(areva)
bridge=M['bridge_for_vowel_and_next_key'];U=len(M['unit_list']);
def wordbad(w,tris):return any(w[i:i+3] not in tris for i in range(len(w)-2))
def testsupport(ref):
 tris,bis=ref
 def ok(s):return all(s[i:i+3] in tris for i in range(len(s)-2)) and all(s[i:i+2] in bis for i in range(len(s)-1))
 res={}
 for v,state in M['state'].items():
    cells=state['cells'];good=[c for c in cells if ok(bridge[v][c[0]]+c)]
    best=[]
    for esc in good:
       cont=[c for c in cells if ok(esc+c)]
       best.append((len(good)-1+len(cont),esc,len(cont)))
    best.sort(reverse=True)
    res[v]={'native_cells':len(cells),'compatible_direct':len(good),'optimistic_1escape':best[0][0] if best else 0, 'escape':best[0][1] if best else None,'one_cell_shortfall':max(0,U-len(good))}
 return res
output={'corpus':{'A':len(areva),'B_from_slim':len(breva),'B_original':len(Bwords),'by_section':dict((k,len(v))for k,v in bysection.items()),'b_pages':len(b_pages)},'per_inventory':{'B':testsupport((trisB,bisB)),'A':testsupport((trisA,bisA))},'support_size':{'A':len(trisA),'B':len(trisB)}}
for src in ('train','hartlieb'):
 gen=(B/f'{src}.pair.vbm').read_text().split()
 output.setdefault('generated',{})[src]={'n':len(gen),'unattested_by_A':sum(wordbad(w,trisA) for w in gen),'unattested_by_B':sum(wordbad(w,trisB) for w in gen)}
(P/'screen.json').write_text(json.dumps(output,indent=2)); print(json.dumps(output,indent=2))

#!/usr/bin/env python3
import argparse,collections,json,pickle,re,sys,xml.etree.ElementTree as ET
from pathlib import Path
from trigram import train
from diagnose import c,v,read_pickle

def ref_extract(p):
    root=ET.parse(p).getroot();head=root.find('header').text or ''
    area=re.search(r'^language-area:\s*(.+)$',head,re.M)
    words=[]
    for tok in root.iter('token'):
        w=c.normalize_latin(''.join(ch.attrib.get('utf') or (ch.text or '') for ch in tok if ch.tag=='tok_dipl'))
        if w:words.append(w)
    return words,area.group(1).strip() if area else 'MISSING'

def run(data,penn,out):
    banned_y=set(sum(v.YID_DEV.values(),[]))|{'1507w-bovo.psd','1588e-letters-cracow.psd','1590e-sam-hayyim.psd','1620e-lev-tov-1.psd','1620e-lev-tov-1-preface.psd','1648w-kine.psd'}
    all_y={}
    for p in sorted(penn.glob('*.psd')):
        if p.name[:4].isdigit() and int(p.name[:4])<=1750:
            fam=p.stem.removesuffix('-preface');all_y.setdefault(fam,[]).append(p)
    eval_y=[c.penn_words(penn/f) for f in sorted(banned_y)]
    ygrams=set().union(*(c.ngrams(ws,8) for ws in eval_y));ypool={};ysources={};excluded=[]
    for fam,paths in all_y.items():
        if any(p.name in banned_y for p in paths):continue
        ws=sum((c.penn_words(p) for p in paths),[])
        ov=len(c.ngrams(ws,8)&ygrams)
        if ov:excluded.append({'family':fam,'overlap8':ov});continue
        ypool[fam]=ws;ysources[fam]={p.name:c.sha_file(p) for p in paths}
    ytrain,yalloc=c.work_balanced(ypool,10000);assert len(ypool)>2
    banned_g=set(v.GER_BUILD+v.GER_DEV)|{'F016','F018','F034','F037','F148'}
    eval_g=[v.ref_words_v02(data/'ref_extract',f)[0] for f in v.GER_DEV];ggrams=set().union(*(c.ngrams(ws,8) for ws in eval_g))
    byarea=collections.defaultdict(list);gexcluded=[];seen=set()
    for p in sorted((data/'ref_extract').rglob('*.xml')):
        wid=p.stem
        if wid in banned_g or not re.fullmatch(r'F\d+',wid):continue
        assert wid not in seen;seen.add(wid)
        ws,area=ref_extract(p)
        if len(ws)<5000 or area=='MISSING':continue
        ov=len(c.ngrams(ws,8)&ggrams)
        if ov:gexcluded.append({'family':wid,'overlap8':ov});continue
        byarea[area].append((wid,p,ws))
    for a in byarea:byarea[a].sort(key=lambda x:x[0])
    selected=[];depth=0
    while len(selected)<len(ypool):
        changed=False
        for area in sorted(byarea):
            if depth<len(byarea[area]):
                wid,p,ws=byarea[area][depth];selected.append((area,wid,p,ws));changed=True
                if len(selected)==len(ypool):break
        assert changed;depth+=1
    gsources={};galloc={};gtrain=[]
    for (yf,amount),(area,wid,p,ws) in zip(sorted(yalloc.items()),selected):
        gtrain.extend(ws[:amount]);galloc[wid]=amount;gsources[wid]={'sha256':c.sha_file(p),'words':len(ws),'area':area,'matched_yiddish_work':yf}
    assert sum(galloc.values())==10000 and sorted(galloc.values())==sorted(yalloc.values())
    manifest={'status':'SOURCE_SELECTION_FROZEN_BEFORE_MODEL_FITTING','yiddish_sources':ysources,'german_sources':gsources,'yiddish_alloc':yalloc,'german_alloc':galloc,'yiddish_overlap_exclusions':excluded,'german_overlap_exclusions':gexcluded,'target_loaded':False}
    c.atomic_json(manifest,out/'source_manifest.json')
    truth=read_pickle(data/'result/checkpoint.pkl')['truth'];trains={'yiddish':ytrain,'german':gtrain};models={mid:train(trains[lang]) for mid,lang in truth['model_truth'].items()}
    c.atomic_pickle(models,out/'models.pkl');c.atomic_json({'status':'COVERAGE_MODELS_FROZEN','source_manifest_sha256':c.sha_file(out/'source_manifest.json'),'model_sha256':c.sha_file(out/'models.pkl')},out/'model_manifest.json')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--penn',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run(a.data,a.penn,a.out)

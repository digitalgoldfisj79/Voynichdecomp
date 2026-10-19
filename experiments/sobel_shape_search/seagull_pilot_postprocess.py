#!/usr/bin/env python3
import argparse, glob, io, json, os, re
from pathlib import Path
import requests
from PIL import Image, ImageDraw


def load_rows(root):
    groups={'seagull_target':[],'seagull_control':[]}; summaries={k:[] for k in groups}
    for fn in glob.glob(str(Path(root)/'**/*.json'),recursive=True):
        base=os.path.basename(fn)
        q='seagull_target' if 'seagull_target' in base else ('seagull_control' if 'seagull_control' in base else None)
        if not q: continue
        try:
            d=json.load(open(fn)); summaries[q].append(d.get('summary',{})); groups[q].extend(d.get('results',[]))
        except Exception as e: print('skip',fn,e)
    for q in groups: groups[q].sort(key=lambda r:r['score'])
    return groups,summaries


def contact_sheet(rows,out,title,n=40):
    sess=requests.Session(); tiles=[]
    for rank,r in enumerate(rows[:n],1):
        u=r.get('region_url') or r.get('search_url')
        if not u: continue
        try:
            rr=sess.get(u,timeout=(8,25),headers={'User-Agent':'ManuComp-SeagullPilot/0.1 (+research)'}); rr.raise_for_status()
            im=Image.open(io.BytesIO(rr.content)).convert('RGB'); im.thumbnail((360,240))
        except Exception: continue
        tile=Image.new('RGB',(390,300),'white'); tile.paste(im,((390-im.width)//2,32))
        d=ImageDraw.Draw(tile); d.text((8,6),f"#{rank} score={r['score']:.4f} rot={r.get('rotation_deg',0):g} w={r.get('base_width')}",fill='black')
        ms=str(r.get('manuscript_id',''))[:48]; fol=str(r.get('folio_label') or r.get('canvas_index'))
        d.text((8,278),f"{ms} | {fol}",fill='black'); tiles.append(tile)
    cols=4; rows_n=max(1,(len(tiles)+cols-1)//cols)
    sheet=Image.new('RGB',(cols*390,40+rows_n*300),'white'); ImageDraw.Draw(sheet).text((10,8),title,fill='black')
    for i,t in enumerate(tiles): sheet.paste(t,((i%cols)*390,40+(i//cols)*300))
    sheet.save(out,quality=92)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='pilot_parts'); ap.add_argument('--outdir',default='seagull_pilot'); args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    groups,sums=load_rows(args.root)
    payload={'method':'paired Sobel/chamfer pilot','queries':{}}
    for q,rows in groups.items():
        ss=sums[q]; payload['queries'][q]={
            'pages_seen':sum(s.get('seen',0) for s in ss),'pages_ok':sum(s.get('ok',0) for s in ss),'errors':sum(s.get('errors',0) for s in ss),
            'best_score':rows[0]['score'] if rows else None,'results':rows[:250]}
        contact_sheet(rows,out/f'{q}_top40.jpg',q,n=40)
    (out/'seagull_pilot_combined.json').write_text(json.dumps(payload,indent=2))
    with open(out/'summary.md','w') as f:
        f.write('# Voynich f1r seagull paired pilot\n\n')
        f.write('Target = EVA &253 / STA Yy (vertical squiggle); control = adjacent EVA &252 / STA Yx.\n\n')
        for q in ('seagull_target','seagull_control'):
            p=payload['queries'][q]; f.write(f"## {q}\nScanned {p['pages_seen']:,}; ok {p['pages_ok']:,}; errors {p['errors']:,}; best score {p['best_score']}.\n\n")
            f.write('|rank|score|manuscript|folio|region|\n|---:|---:|---|---|---|\n')
            for i,r in enumerate(p['results'][:25],1):
                f.write(f"|{i}|{r['score']:.5f}|`{r['manuscript_id']}`|{r.get('folio_label') or r.get('canvas_index')}|{r.get('region_url') or ''}|\n")
            f.write('\n')
    print((out/'summary.md').read_text())
if __name__=='__main__': main()

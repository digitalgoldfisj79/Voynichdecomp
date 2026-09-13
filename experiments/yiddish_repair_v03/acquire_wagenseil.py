#!/usr/bin/env python3
import argparse,concurrent.futures,hashlib,json,pickle,urllib.request,time
from pathlib import Path
from html.parser import HTMLParser
from diagnose import c

PAGES=list(range(261,394,2))+list(range(395,405))
AUDIT=[273,289,337,381,389,391]
class HocrParser(HTMLParser):
    def __init__(self):
        super().__init__();self.lines=[];self.stack=[];self.line=None;self.word=None
    def handle_starttag(self,tag,attrs):
        if tag!='span':return
        d=dict(attrs);cl=d.get('class','').split();kind=''
        if 'ocr_line' in cl:
            self.line={'bbox':d.get('title'),'words':[]};self.lines.append(self.line);kind='line'
        if 'ocrx_word' in cl or 'ocr_word' in cl:self.word=[];kind='word'
        self.stack.append(kind)
    def handle_endtag(self,tag):
        if tag!='span' or not self.stack:return
        kind=self.stack.pop()
        if kind=='word':
            if self.line is not None:self.line['words'].append(''.join(self.word or []))
            self.word=None
        if kind=='line':self.line=None
    def handle_data(self,data):
        if self.word is not None:self.word.append(data)
def get(url,path):
    if path.exists():return
    for i in range(3):
        try:
            with urllib.request.urlopen(url,timeout=25) as r:b=r.read()
            tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(b);tmp.replace(path);return
        except Exception:
            if i==2:raise
def run(manifest,out):
    out.mkdir(parents=True,exist_ok=True);m=json.loads(manifest.read_text());canvases=m['sequences'][0]['canvases'];byid={int(x['@id'].rsplit('/',1)[-1]):x for x in canvases}
    c.atomic_json({'pages':PAGES,'audit_pages':AUDIT,'manifest_sha256':c.sha_file(manifest),'status':'OFFICIAL_OCR_PIPELINE_FROZEN_BEFORE_DOWNLOAD'},out/'acquisition_freeze.json')
    def page(p):
        cv=byid[p];url=cv['seeAlso']['@id'];dest=out/f'p{p:05d}.hocr';get(url,dest)
        if p in AUDIT:get(cv['images'][0]['resource']['@id'],out/f'p{p:05d}.jpg')
        parser=HocrParser();parser.feed(dest.read_text(encoding='utf-8'));lines=parser.lines
        # Fail closed rather than silently changing extraction if source schema differs.
        if not lines:raise RuntimeError(f'No hOCR lines on page {p}')
        row={'page':p,'hocr_url':url,'hocr_sha256':c.sha_file(dest),'image_url':cv['images'][0]['resource']['@id'],'lines':lines}
        c.atomic_json(row,out/f'p{p:05d}.json');return row
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(page,PAGES):
            rows.append(r);c.atomic_pickle({'status':'ACQUIRING_NO_LANGUAGE_SCORE','pages':rows,'target_loaded':False},out/'checkpoint.pkl')
    c.atomic_json({'status':'OFFICIAL_OCR_ACQUIRED_NOT_QUALIFIED','n_pages':len(rows),'pages':rows,'target_loaded':False},out/'ocr_all.json')
    print(json.dumps({'pages':len(rows),'status':'ACQUIRED_NOT_QUALIFIED'}))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--manifest',type=Path,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args();run(x.manifest,x.out)

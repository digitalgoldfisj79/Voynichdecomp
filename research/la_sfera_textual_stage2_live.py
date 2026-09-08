#!/usr/bin/env python3
from __future__ import annotations
import csv, io, json, math, random, re, statistics, sys, urllib.request
from collections import defaultdict
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

DATA_PAGE='https://sferaproject.org/resources/data/'
PAIR=('Laur2','Yale4')
RIGHT=('Fn12','He1','He2','NYPL2','Spe','Par4','Barb4','Cap1','Urb2','Vat3')
KNOWN=('Ba','Barb1','Barb2','Barb3','Barb4','Borg','Bos','Cam','Cap1','Chig1','Chig2','Chig3','Fn12','Fn2','Fn3','Fn8','He1','He2','He3','Laur1','Laur2','Laur3','Laur4','Laur5','Laur6','Laur7','LW01','LW02','Pal5','Par1','Par2','Par3','Par4','Spe','Urb1','Urb2','Vat1','Vat2','Vat3','Vat4','Yale1','Yale2','Yale3','Yale4','NYPL1','NYPL2','NYPL3')
SEED=20260908
NNULL=100000

class AParser(HTMLParser):
    def __init__(self): super().__init__(); self.a=[]; self.cur=None; self.txt=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a': self.cur=dict(attrs).get('href'); self.txt=[]
    def handle_data(self,d):
        if self.cur is not None: self.txt.append(d)
    def handle_endtag(self,tag):
        if tag=='a' and self.cur is not None:
            self.a.append((' '.join(''.join(self.txt).split()),self.cur)); self.cur=None; self.txt=[]

def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 LaSfera-research/1.0'})
    with urllib.request.urlopen(req,timeout=60) as r: return r.read()

def norm(s):
    s=unescape(str(s or '')).lower().replace('’',"'").replace('‘',"'")
    s=re.sub(r'<[^>]+>',' ',s); s=re.sub(r"[^\wà-ÿ']+",' ',s,flags=re.UNICODE)
    return re.sub(r'\s+',' ',s).strip()

def hnorm(s): return re.sub(r'[^a-z0-9]+','',str(s or '').lower())
def mean(x): return sum(x)/len(x) if x else float('nan')
def sd(x): return statistics.stdev(x) if len(x)>1 else float('nan')
def zclose(obs,null):
    s=sd(null); return (mean(null)-obs)/s if null and s and math.isfinite(s) else float('nan')
def plower(obs,null): return (sum(v<=obs for v in null)+1)/(len(null)+1) if null else float('nan')

def read_csv(raw):
    txt=raw.decode('utf-8-sig',errors='replace')
    try: dialect=csv.Sniffer().sniff(txt[:10000],delimiters=',;\t')
    except Exception: dialect=csv.excel
    return list(csv.DictReader(io.StringIO(txt),dialect=dialect))

def findcol(fields,terms):
    nf={hnorm(x):x for x in fields}
    for t in terms:
        if hnorm(t) in nf: return nf[hnorm(t)]
    for f in fields:
        hf=hnorm(f)
        if any(hnorm(t) in hf for t in terms): return f
    return None

def matrix_from_csv(rows, kind):
    if not rows: return {},{}, {'mode':'empty'}
    fields=list(rows[0].keys()); known_cols={s:next((f for f in fields if hnorm(f)==hnorm(s)),None) for s in KNOWN}
    wide={s:c for s,c in known_cols.items() if c}
    readings=defaultdict(lambda:defaultdict(list)); coverage=defaultdict(set)
    diag={'fields':fields,'mode':None,'n_rows':len(rows),'wide_witness_columns':sorted(wide)}
    locus_terms=['line code','line_code','linecode','code','variant line','stanza','place_id','place id','placename id','toponym id','id']
    locus=findcol(fields,locus_terms)
    if len(wide)>=3:
        diag['mode']='wide'; diag['locus_col']=locus
        for i,r in enumerate(rows):
            loc=norm(r.get(locus)) if locus else str(i)
            if not loc: loc=str(i)
            for s,c in wide.items():
                val=norm(r.get(c))
                if val:
                    readings[s][loc].append(val); coverage[s].add(loc)
    else:
        ms=findcol(fields,['siglum','manuscript','ms','manuscript siglum'])
        if kind=='text': val=findcol(fields,['variant','reading','text','stanza variation','variation'])
        else: val=findcol(fields,['label','variant','toponym','placename','place name','reading'])
        diag.update({'mode':'long','ms_col':ms,'locus_col':locus,'value_col':val})
        if ms and val:
            for i,r in enumerate(rows):
                s=str(r.get(ms) or '').strip()
                # common patterns may include shelfmark plus siglum; recover known siglum token
                if s not in KNOWN:
                    hit=next((k for k in KNOWN if re.search(r'(?<![A-Za-z0-9])'+re.escape(k)+r'(?![A-Za-z0-9])',s)),None)
                    if hit: s=hit
                if not s: continue
                loc=norm(r.get(locus)) if locus else str(i)
                valn=norm(r.get(val))
                if loc and valn:
                    readings[s][loc].append(valn); coverage[s].add(loc)
    R={s:{l:tuple(sorted(set(v))) for l,v in d.items()} for s,d in readings.items()}
    C={s:set(v) for s,v in coverage.items()}
    diag['matrix_mss']=len(C); diag['matrix_counts']={s:len(C[s]) for s in sorted(C)}
    return R,C,diag

def pdist(R,C,a,b,min_common=1):
    loci=C.get(a,set())&C.get(b,set())
    if len(loci)<min_common:return None
    return sum(R[a][l]!=R[b][l] for l in loci)/len(loci),len(loci)

def pairtest(R,C,pair):
    d=pdist(R,C,*pair)
    if not d:return None
    obs,cov=d; vals=[]
    mss=sorted(C)
    for i,a in enumerate(mss):
        for b in mss[i+1:]:
            q=pdist(R,C,a,b)
            if q: vals.append((q[0],q[1],a,b))
    for tol in (.15,.30,.50,1.0):
        null=[x for x,c,a,b in vals if cov*(1-tol)<=c<=cov*(1+tol) and {a,b}!=set(pair)]
        if len(null)>=30:break
    return {'observed':obs,'common_loci':cov,'null_mean':mean(null),'null_sd':sd(null),'z_closer':zclose(obs,null),'empirical_p_lower':plower(obs,null),'n_null':len(null)}

def gstat(R,C,g):
    vals=[]; cov=[]
    for i,a in enumerate(g):
        for b in g[i+1:]:
            q=pdist(R,C,a,b)
            if q: vals.append(q[0]);cov.append(q[1])
    return (mean(vals),mean(cov),len(vals)) if vals else None

def grouptest(R,C,g):
    g=[s for s in g if s in C]
    if len(g)<3:return None,g
    obs=gstat(R,C,g)
    if not obs:return None,g
    od,oc,np=obs; rng=random.Random(SEED); mss=sorted(C); cand=[]
    if len(mss)<len(g):return None,g
    for _ in range(NNULL):
        qg=rng.sample(mss,len(g)); q=gstat(R,C,qg)
        if q and q[2]>=max(1,int(.9*np)):cand.append(q)
    for tol in (.10,.20,.35,.60,1.0):
        null=[d for d,c,n in cand if oc*(1-tol)<=c<=oc*(1+tol)]
        if len(null)>=100:break
    return {'observed':od,'mean_pair_common_loci':oc,'valid_pairs':np,'n_group':len(g),'members':g,'null_mean':mean(null),'null_sd':sd(null),'z_closer':zclose(od,null),'empirical_p_lower':plower(od,null),'n_null':len(null)},g

def decision(x):
    if not x:return 'NOT TESTABLE'
    z=x.get('z_closer')
    if not isinstance(z,(int,float)) or not math.isfinite(z):return 'NOT RESOLVED'
    if z>=2:return f'RESOLVES AS UNUSUALLY TIGHT ({z:.2f} null SD)'
    if z<=-2:return f'RESOLVES OPPOSITE: UNUSUALLY DISPERSED ({z:.2f} null SD)'
    return f'THE METRIC DOES NOT RESOLVE ({z:.2f} null SD)'

def nearest(R,C,target,n=10):
    out=[]
    if target not in C:return out
    for s in C:
        if s==target:continue
        q=pdist(R,C,target,s)
        if q:out.append((q[0],q[1],s))
    return sorted(out)[:n]

def main():
    od=Path(sys.argv[1] if len(sys.argv)>1 else 'research_out_live');od.mkdir(parents=True,exist_ok=True)
    html=fetch(DATA_PAGE).decode('utf-8',errors='replace'); p=AParser();p.feed(html)
    links=[]
    for text,href in p.a:
        u=urljoin(DATA_PAGE,href)
        if '.csv' in u.lower() or 'sfera-production.s3.amazonaws.com' in u.lower(): links.append((text,u))
    # de-dupe
    seen=set(); links=[x for x in links if not (x[1] in seen or seen.add(x[1]))]
    print('DISCOVERED LINKS',json.dumps(links,indent=2),flush=True)
    exports={}; meta={}
    for text,u in links:
        try:
            raw=fetch(u); rows=read_csv(raw); key=hnorm(text) or Path(u).name
            exports[key]=rows
            meta[key]={'label':text,'url':u,'bytes':len(raw),'rows':len(rows),'fields':list(rows[0]) if rows else [],'sample':rows[:2]}
            (od/(re.sub(r'[^A-Za-z0-9_.-]+','_',key)+'.csv')).write_bytes(raw)
        except Exception as e: meta[hnorm(text) or u]={'label':text,'url':u,'error':repr(e)}
    (od/'exports_meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('EXPORT META',json.dumps(meta,indent=2,ensure_ascii=False)[:40000],flush=True)
    def pick(tokens):
        for k,rows in exports.items():
            label=hnorm(meta[k].get('label',''))
            if all(t in label for t in tokens):return k,rows
        for k,rows in exports.items():
            if all(t in k for t in tokens):return k,rows
        return None,[]
    tk,trows=pick(['textual','variant'])
    pk,prows=pick(['toponym','variant'])
    sk,srows=pick(['stanza'])
    # avoid translated/English for robustness if possible
    if sk and ('english' in hnorm(meta[sk]['label']) or 'translation' in hnorm(meta[sk]['label'])):
        candidates=[(k,r) for k,r in exports.items() if 'stanza' in hnorm(meta[k].get('label','')) and 'english' not in hnorm(meta[k].get('label','')) and 'translation' not in hnorm(meta[k].get('label',''))]
        if candidates:sk,srows=candidates[0]
    Rt,Ct,dt=matrix_from_csv(trows,'text')
    Rp,Cp,dp=matrix_from_csv(prows,'toponym')
    Rs,Cs,ds=matrix_from_csv(srows,'text') if srows else ({},{},{'mode':'missing'})
    text_pair=pairtest(Rt,Ct,PAIR); text_group,_=grouptest(Rt,Ct,RIGHT)
    topo_pair=pairtest(Rp,Cp,PAIR); topo_group,_=grouptest(Rp,Cp,RIGHT)
    stanza_pair=pairtest(Rs,Cs,PAIR); stanza_group,_=grouptest(Rs,Cs,RIGHT)
    res={'data_page':DATA_PAGE,'exports':meta,'selected_exports':{'textual':tk,'toponym':pk,'stanzas':sk},
         'diagnostics':{'textual':dt,'toponym':dp,'stanzas':ds},
         'frozen':{'pair':PAIR,'right':RIGHT,'right_text_available':[s for s in RIGHT if s in Ct],'right_toponym_available':[s for s in RIGHT if s in Cp],'right_stanza_available':[s for s in RIGHT if s in Cs]},
         'textual':{'pair':text_pair,'group':text_group,'Laur2_nearest':nearest(Rt,Ct,'Laur2'),'Yale4_nearest':nearest(Rt,Ct,'Yale4')},
         'toponym':{'pair':topo_pair,'group':topo_group},'stanza_robustness':{'pair':stanza_pair,'group':stanza_group}}
    (od/'result_live.json').write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')
    md=f'''# La Sfera Stage-2 LIVE export test\n\nFrozen visual selections were not changed after seeing text.\n\n## Curated textual variants\n- Laur2 ↔ Yale4: **{decision(text_pair)}**\n- Right frozen set (available intersection): **{decision(text_group)}**\n\n## Independent toponym variants\n- Laur2 ↔ Yale4: **{decision(topo_pair)}**\n- Right frozen set (available intersection): **{decision(topo_group)}**\n\n## Stanza-text robustness\n- Laur2 ↔ Yale4: **{decision(stanza_pair)}**\n- Right set: **{decision(stanza_group)}**\n\n## Frozen-set availability\n```json\n{json.dumps(res['frozen'],indent=2)}\n```\n\n## Core statistics\n```json\n{json.dumps({'text_pair':text_pair,'text_group':text_group,'toponym_pair':topo_pair,'toponym_group':topo_group,'stanza_pair':stanza_pair,'stanza_group':stanza_group},indent=2)}\n```\n\n## Parsing diagnostics\n```json\n{json.dumps(res['diagnostics'],indent=2,ensure_ascii=False)}\n```\n'''
    (od/'summary_live.md').write_text(md,encoding='utf-8');print(md,flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
import collections, functools, glob, json, math, os, random, re, statistics, urllib.request, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

SEED=20260813
NREP=50
URL='https://zenodo.org/record/13982324/files/ReM-v2.1_tei.zip?download=1'
NS='{http://www.tei-c.org/ns/1.0}'
XMLID='{http://www.w3.org/XML/1998/namespace}id'
CLEAN=re.compile(r'[\[\]<>|\\/*()=+#%$"\'{}0-9\-.,;:!?]')
CLASSES=('short','mid','long')
HIST_OBS={'short':598,'mid':501,'long':146}
HIST_NULL={'short':483.46,'mid':471.405,'long':109.88}

def build_rem(cache=Path('/tmp/rem_docs.json')):
    if cache.exists(): return json.loads(cache.read_text(encoding='utf-8'))
    zp=Path('/tmp/rem.zip'); out=Path('/tmp/remtei')
    urllib.request.urlretrieve(URL,zp)
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(zp) as z: z.extractall(out)
    xmls=list(out.rglob('*.xml'))
    docs={}
    for fp in sorted(xmls):
        try: root=ET.parse(fp).getroot()
        except Exception: continue
        groups=collections.defaultdict(str); order=[]
        for w in root.iter(NS+'w'):
            wid=w.get(XMLID) or ''
            base=wid.split('_m')[0] if '_m' in wid else wid
            txt=re.sub(r'\s+','',''.join(w.itertext()))
            if base not in groups: order.append(base)
            groups[base]+=txt
        words=[]
        for base in order:
            s=CLEAN.sub('',groups[base]).strip().lower()
            if s: words.append(s)
        if words: docs[fp.stem]=words
    cache.write_text(json.dumps(docs,ensure_ascii=False),encoding='utf-8')
    return docs

@functools.lru_cache(maxsize=500000)
def ed1(a,b):
    if a==b: return False
    la,lb=len(a),len(b)
    if abs(la-lb)>1: return False
    if la==lb: return sum(x!=y for x,y in zip(a,b))==1
    if la>lb: a,b=b,a; la,lb=lb,la
    i=j=d=0
    while i<la and j<lb:
        if a[i]==b[j]: i+=1; j+=1
        else:
            d+=1; j+=1
            if d>1:return False
    return True

def cls(a,b):
    m=(len(a)+len(b))/2
    return 'short' if m<=4 else ('mid' if m<=6 else 'long')

def score(lines):
    obs=collections.Counter(); null=collections.Counter()
    for line in lines:
        n=len(line)
        if n<2: continue
        for a,b in zip(line,line[1:]):
            if ed1(a,b): obs[cls(a,b)]+=1
        p=2.0/n
        for i in range(n):
            a=line[i]
            for j in range(i+1,n):
                b=line[j]
                if ed1(a,b): null[cls(a,b)]+=p
    return {k:{'obs':float(obs[k]),'null':float(null[k]),'ratio':float(obs[k]/null[k]) if null[k] else None} for k in CLASSES}

def load_vms(path='enriched_records.json'):
    obj=json.load(open(path,encoding='utf-8')); recs=obj['records']
    by=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in recs: by[r['folio']][int(r['line_no'])].append(r)
    folios=[]; all_lines=[]
    for f,d in by.items():
        lines=[]
        for ln,rr in sorted(d.items()):
            rr.sort(key=lambda r:int(r['pos']))
            line=[r['token'] for r in rr]
            lines.append(line); all_lines.append(line)
        folios.append((f,[len(x) for x in lines]))
    return recs,folios,all_lines

def pseudo(folios,docs,rng):
    vals=list(docs.values()); by_need={}
    lines=[]
    for _,lens in folios:
        need=sum(lens)
        elig=by_need.get(need)
        if elig is None:
            elig=[d for d in vals if len(d)>=need]; by_need[need]=elig
        if not elig: raise RuntimeError(f'No ReM doc long enough for pseudo-folio need={need}')
        d=rng.choice(elig); start=rng.randrange(len(d)-need+1); seg=d[start:start+need]
        p=0
        for n in lens: lines.append(seg[p:p+n]); p+=n
    return lines

def summ(xs):
    return {'mean':statistics.mean(xs),'sd':statistics.stdev(xs) if len(xs)>1 else 0.0,'median':statistics.median(xs),'p10':sorted(xs)[max(0,math.ceil(.10*len(xs))-1)],'p90':sorted(xs)[max(0,math.ceil(.90*len(xs))-1)]}

def main():
    recs,folios,vms_lines=load_vms()
    v=score(vms_lines)
    valid_obs={k:int(v[k]['obs'])==HIST_OBS[k] for k in CLASSES}
    delta={k:v[k]['null']-HIST_NULL[k] for k in CLASSES}
    # historical MC means should lie well within ~2 counts of exact expectations; threshold only cancels gross mismatch
    validation=all(valid_obs.values()) and all(abs(delta[k])<3.0 for k in CLASSES)
    out={'metadata':{'seed':SEED,'n_replicates':NREP,'vms_tokens':len(recs),'vms_lines':len(vms_lines),'vms_folios':len(folios)},'vms_exact_null_validation':{'score':v,'historical_obs':HIST_OBS,'historical_200perm_null_mean':HIST_NULL,'obs_exact_match':valid_obs,'exact_minus_historical_null':delta,'pass':validation}}
    if not validation:
        out['status']='CANCELLED_VMS_NULL_VALIDATION_FAILED'
    else:
        docs=build_rem(); nt=sum(map(len,docs.values()))
        if len(docs)!=406 or nt!=2236137: raise RuntimeError((len(docs),nt))
        rows=[]
        for r in range(NREP):
            rng=random.Random(SEED+r)
            s=score(pseudo(folios,docs,rng)); s['replicate']=r; rows.append(s)
            ed1.cache_clear()
        agg={}
        for k in CLASSES:
            agg[k]={'observed':summ([r[k]['obs'] for r in rows]),'null':summ([r[k]['null'] for r in rows]),'ratio':summ([r[k]['ratio'] for r in rows]),'fraction_ratio_gt_1':sum(r[k]['ratio']>1 for r in rows)/NREP}
        mono=[r['short']['ratio']>r['mid']['ratio']>r['long']['ratio'] for r in rows]
        hist=[m and r['long']['ratio']<=1.05 for m,r in zip(mono,rows)]
        out.update({'status':'COMPLETE','rem':{'documents':len(docs),'tokens':nt},'replicates':rows,'aggregate':agg,'gradient':{'fraction_strict_short_gt_mid_gt_long':sum(mono)/NREP,'fraction_historical_crowding_pattern':sum(hist)/NREP}})
    p=Path('results/abc_fw_recovery_p70_bridge_v0_1'); p.mkdir(parents=True,exist_ok=True)
    (p/'REM_LENGTH_CONTROL_20260815.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    md=['# Reconstructed ABC-B ReM length control — 2026-08-15','',f"Status: **{out['status']}**.",'','## VMS exact-null validation','|class|obs|exact null|historical 200-perm null|delta|ratio|','|---|---:|---:|---:|---:|---:|']
    for k in CLASSES:
        q=v[k]; md.append(f"|{k}|{q['obs']:.0f}|{q['null']:.3f}|{HIST_NULL[k]:.3f}|{delta[k]:+.3f}|{q['ratio']:.3f}|")
    if out['status']=='COMPLETE':
        md += ['','## ReM — 50 structure-matched pseudo-corpora','|class|mean obs|mean null|mean ratio|median ratio|p10–p90|fraction ratio>1|','|---|---:|---:|---:|---:|---:|---:|']
        for k in CLASSES:
            a=out['aggregate'][k]; md.append(f"|{k}|{a['observed']['mean']:.2f}|{a['null']['mean']:.2f}|{a['ratio']['mean']:.3f}|{a['ratio']['median']:.3f}|{a['ratio']['p10']:.3f}–{a['ratio']['p90']:.3f}|{a['fraction_ratio_gt_1']:.2f}|")
        g=out['gradient']; md += ['',f"Strict short > mid > long ratio gradient: **{g['fraction_strict_short_gt_mid_gt_long']:.2f}** of replicates.",f"Full historical crowding pattern (strict gradient and long <=1.05): **{g['fraction_historical_crowding_pattern']:.2f}** of replicates.",'','## Guardrail','This is a frozen reconstruction of the missing historical control, not recovery of the original `rem_matched.py`. The VMS ABC-B criterion remains the primary preregistered decision; this arm tests whether the proposed generic ReM crowding-gradient rationale calibrates.']
    (p/'REM_LENGTH_CONTROL_20260815.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__':main()

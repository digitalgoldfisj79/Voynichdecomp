#!/usr/bin/env python3
"""NSA historical Voynich replication v0.1.

Stdlib-only Stage 0/1 runner. It deliberately fails closed on unknown corpus
schemas rather than silently inventing a folio mapping. Stage 2/3 are recorded
as pending unless line/token structure can be recovered without ambiguity.
"""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, math, os, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEED = 1978


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def norm_folio(x: str) -> str:
    x = str(x).strip().lower().replace(' ', '')
    x = x.replace('folio', 'f')
    return x if x.startswith('f') else 'f' + x


def flatten_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for v in obj: yield from flatten_strings(v)
    elif isinstance(obj, dict):
        for k in ('text','tokens','lines','content','transcription','zlzi','ZLZI'):
            if k in obj: yield from flatten_strings(obj[k])


def detect_folios(data):
    """Return folio->raw object for common slim-corpus schemas."""
    candidates = {}
    if isinstance(data, dict):
        for wrapper in ('folios','pages','documents','ZLZI','zlzi'):
            if wrapper in data and isinstance(data[wrapper], (dict,list)):
                nested = detect_folios(data[wrapper])
                if len(nested) > len(candidates): candidates = nested
        for k,v in data.items():
            nk = norm_folio(k)
            if re.fullmatch(r'f\d+[rv](?:\d+)?', nk): candidates[nk] = v
        if candidates: return candidates
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict): continue
            key = next((item.get(k) for k in ('folio','folio_id','page','id','name') if item.get(k)), None)
            if key:
                nk = norm_folio(key)
                if re.fullmatch(r'f\d+[rv](?:\d+)?', nk): candidates[nk] = item
    return candidates


def raw_text(obj):
    parts = list(flatten_strings(obj))
    return '\n'.join(parts)


def glyphs(text):
    # Preserve alphabetic transcription symbols only; remove markup/digits.
    return ''.join(ch.lower() for ch in text if ch.isalpha())


def js_distance(a, b, alphabet):
    sa, sb = sum(a.values()), sum(b.values())
    pa = [(a.get(c,0)+1e-12)/(sa+1e-12*len(alphabet)) for c in alphabet]
    pb = [(b.get(c,0)+1e-12)/(sb+1e-12*len(alphabet)) for c in alphabet]
    m = [(x+y)/2 for x,y in zip(pa,pb)]
    def kl(p,q): return sum(x*math.log(x/y) for x,y in zip(p,q) if x)
    return math.sqrt(max(0.0,(kl(pa,m)+kl(pb,m))/2))


def centroid(rows, omit=None):
    c=Counter()
    for r in rows:
        if r is omit: continue
        total=sum(r['counts'].values()) or 1
        for k,v in r['counts'].items(): c[k]+=v/total
    n=max(1,len(rows)-(1 if omit in rows else 0))
    return Counter({k:v/n for k,v in c.items()})


def stage1(rows):
    alphabet=sorted(set().union(*(r['counts'] for r in rows)))
    by=defaultdict(list)
    for r in rows: by[r['class']].append(r)
    correct=0
    for r in rows:
        ds={cl:js_distance(r['counts'],centroid(rs, r if r in rs else None),alphabet) for cl,rs in by.items()}
        r['predicted']=min(ds,key=ds.get); r['centroid_distance']=ds[r['class']]
        correct += r['predicted']==r['class']
    accuracy=correct/len(rows); majority=max(map(len,by.values()))/len(rows)
    within=[]; between=[]
    for i,j in itertools.combinations(range(len(rows)),2):
        d=js_distance(rows[i]['counts'],rows[j]['counts'],alphabet)
        (within if rows[i]['class']==rows[j]['class'] else between).append(d)
    contrast=sum(between)/len(between)-sum(within)/len(within)
    labels=[r['class'] for r in rows]; rng=random.Random(SEED); ge=0; nperm=10000
    for _ in range(nperm):
        p=labels[:]; rng.shuffle(p); w=[]; b=[]
        for i,j in itertools.combinations(range(len(rows)),2):
            d=js_distance(rows[i]['counts'],rows[j]['counts'],alphabet)
            (w if p[i]==p[j] else b).append(d)
        stat=sum(b)/len(b)-sum(w)/len(w)
        ge += stat>=contrast-1e-15
    pval=(ge+1)/(nperm+1)
    hb=sorted(by['Herbal_B'],key=lambda r:r['centroid_distance'],reverse=True)
    for rank,r in enumerate(hb,1): r['anomaly_rank_HB']=rank
    frozen={'f36r','f39v','f41r','f41v','f48v'}
    frozen_top_half=sum(r['folio'] in frozen for r in hb[:math.ceil(len(hb)/2)])
    metrics={'n':len(rows),'loo_accuracy':accuracy,'majority_baseline':majority,
             'accuracy_margin':accuracy-majority,'distance_contrast':contrast,
             'permutation_p':pval,'frozen_anomalies_top_half':frozen_top_half,
             'gate':bool(pval<=.01 and accuracy-majority>=.20 and frozen_top_half>=3)}
    return metrics


def synthetic_smoke():
    rows=[]
    for cl,ch in [('Biological_B','a'),('Herbal_A','b'),('Herbal_B','c')]:
        for i in range(5): rows.append({'folio':f'f{i+1}r','class':cl,'counts':Counter((ch*390+'xyz'*3)[:400])})
    m=stage1(rows)
    return m['loo_accuracy']==1.0 and m['permutation_p']<=.01


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','full'],default='full'); args=ap.parse_args()
    out=HERE/'results'; out.mkdir(exist_ok=True)
    if not synthetic_smoke(): raise SystemExit('synthetic smoke failed')
    if args.mode=='smoke':
        (out/'SMOKE_RESULT.md').write_text('# Smoke result\n\nPASS\n',encoding='utf-8'); return
    corpus=ROOT/'voynich_transcriptions_slim.json'; panel=HERE/'DATA_PANEL.csv'
    integrity={'smoke':'PASS','corpus_exists':corpus.exists(),'files':{}}
    for p in (corpus,panel,HERE/'PROTOCOL.md',HERE/'run.py'):
        if p.exists(): integrity['files'][str(p.relative_to(ROOT))]=sha256(p)
    if not corpus.exists():
        integrity['status']='BLOCKED_CORPUS_MISSING'; (out/'INTEGRITY.json').write_text(json.dumps(integrity,indent=2)); raise SystemExit(2)
    data=json.loads(corpus.read_text(encoding='utf-8')); fmap=detect_folios(data)
    rows=[]; missing=[]; undersized=[]
    with panel.open(newline='',encoding='utf-8') as f:
        for rec in csv.DictReader(f):
            fol=norm_folio(rec['folio']); obj=fmap.get(fol)
            if obj is None: missing.append(fol); continue
            g=glyphs(raw_text(obj))
            if len(g)<350: undersized.append([fol,len(g)])
            rows.append({'sample_id':rec['sample_id'],'folio':fol,'class':rec['historical_class'],'n_glyphs':len(g),'counts':Counter(g[:400])})
    integrity.update({'resolved_folios':len(rows),'missing':missing,'undersized':undersized,'detected_folios':len(fmap)})
    if missing or undersized or len(rows)!=40:
        integrity['status']='BLOCKED_SCHEMA_OR_PANEL'; (out/'INTEGRITY.json').write_text(json.dumps(integrity,indent=2)); raise SystemExit(3)
    integrity['status']='PASS'; (out/'INTEGRITY.json').write_text(json.dumps(integrity,indent=2))
    m=stage1(rows); (out/'stage1_metrics.json').write_text(json.dumps(m,indent=2))
    with (out/'stage1_samples.csv').open('w',newline='',encoding='utf-8') as f:
        cols=['sample_id','folio','class','n_glyphs','predicted','centroid_distance','anomaly_rank_HB']; w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,'') for k in cols})
    result=f"""# NSA Historical Replication v0.1 — Result\n\n## Integrity\n\n**PASS** — 40/40 historical folios resolved; all supplied at least 350 normalized glyph characters.\n\n## Stage 1\n\n- LOO accuracy: {m['loo_accuracy']:.4f}\n- Majority baseline: {m['majority_baseline']:.4f}\n- Accuracy margin: {m['accuracy_margin']:.4f}\n- Distance contrast: {m['distance_contrast']:.6f}\n- Permutation p: {m['permutation_p']:.6g}\n- Frozen anomalies in Herbal-B top half: {m['frozen_anomalies_top_half']}/5\n- Gate S1: **{'PASS' if m['gate'] else 'FAIL'}**\n\nEvidence label: **EMPIRICAL**, conditional on transcription/parser audit.\n\n## Stage 2 and Stage 3\n\n**OPEN** in this stdlib runner. They require an audited extraction of line and token boundaries and, for Stage 3, a held-out categorical HMM implementation. No inference is made from their non-execution.\n"""
    (out/'RESULT.md').write_text(result,encoding='utf-8')

if __name__=='__main__': main()

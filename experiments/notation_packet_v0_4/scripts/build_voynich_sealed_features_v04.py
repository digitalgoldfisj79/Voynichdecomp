#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, pickle, sys
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from surface_features_v04 import extract_surface_features, canonicalize_events

IN=Path('/mnt/data/notation_voynich_inputs/enriched_records.pkl')
OUT=HERE/'voynich_surface_features_sealed_v04.csv'
MAN=HERE/'voynich_surface_features_sealed_v04_manifest.json'

records=pickle.load(open(IN,'rb'))
by=defaultdict(list)
for r in records:
    by[str(r['folio'])].append(r)
rows=[]
for folio, recs in sorted(by.items()):
    recs=sorted(recs,key=lambda r:(int(r.get('line_no',0)),int(r.get('pos',0))))
    events=[str(r['token']) for r in recs]
    section=max(set(str(r['section']) for r in recs),key=lambda s:sum(str(x['section'])==s for x in recs))
    starts = []
    if len(events) >= 48:
        starts.append(0)
    if len(events) >= 96:
        starts.append(len(events) - 48)
    for wi, a in enumerate(starts):
        b = a + 48
        w = events[a:b]
        f=extract_surface_features(canonicalize_events(w))
        rows.append({'corpus':'Voynich','family':'sealed_target','group':folio,'section':section,'window_id':f'{folio}:{wi}','start':a,'end':b,**f})
fields=list(rows[0])
with open(OUT,'w',newline='',encoding='utf-8') as fh:
    wr=csv.DictWriter(fh,fieldnames=fields);wr.writeheader();wr.writerows(rows)
raw=OUT.read_bytes()
manifest={
    'schema':'voynich-canonical-surface-features-sealed-v0.4',
    'source':'enriched_records.pkl','source_records':len(records),
    'source_sha256':hashlib.sha256(IN.read_bytes()).hexdigest(),
    'window':48,'selection':'first and last non-overlapping window per folio when available','minimum':48,
    'character_policy':'within-window frequency-rank canonicalisation','rows':len(rows),'folios':len(by),
    'feature_columns':[x for x in fields if x not in {'corpus','family','group','section','window_id','start','end'}],
    'csv_sha256':hashlib.sha256(raw).hexdigest(),
}
MAN.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))

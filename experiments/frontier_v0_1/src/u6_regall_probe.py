import json
from pathlib import Path
import requests

ROOT=Path('/manifest')
REG=ROOT/'register/reg_all.jsonl'
BRIDGE='https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/vtps_hf_bridge_20260814'
SECRET='frontier-u6-stageb-20260815'
TARGET={'f32r','f39r','f40r'}
rows=[]; total=0; passed=0; folios=set(); keysets={}
with REG.open('r',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line); total+=1
        fol=str(r.get('folio')); folios.add(fol)
        if r.get('passed'): passed+=1
        keysets[tuple(sorted(r.keys()))]=keysets.get(tuple(sorted(r.keys())),0)+1
        if fol in TARGET: rows.append(r)
out={'exists':REG.is_file(),'size':REG.stat().st_size,'total_rows':total,'unique_folios':len(folios),'passed_rows':passed,'targets':rows,'keysets':[{'keys':list(k),'n':v} for k,v in keysets.items()]}
r=requests.post(BRIDGE,json={'secret':SECRET,'id':'u6-stageb-20260815-pathfix-regall-probe','payload':json.dumps(out,sort_keys=True),'meta':{'phase':'frozen-registration-probe'}},timeout=120)
r.raise_for_status(); print(json.dumps(out,indent=2)[:40000])

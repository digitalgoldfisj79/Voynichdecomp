#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,lzma,sys,os
import scorer
from mechanisms import make_plan,encode_with_plan,self_tests as mechanism_self_tests
from phase2_operators import NEW_OPS,transform_lines,self_tests as p2tests
ROOT=Path(__file__).resolve().parent
shard=int(sys.argv[1]);nshards=int(sys.argv[2]);REPS=('ATOMIC','LITERAL');SCHEDULE='SWITCH_LINE'
mechanism_self_tests();p2tests();scorer.self_test_ed1()
rem=json.loads(lzma.decompress((ROOT/'inputs'/'rem_docs.json.xz').read_bytes()).decode('utf-8'))
docs=sorted(d for d,w in rem.items() if len(w)>=2000);assert len(docs)==190
chosen=docs[shard::nshards]
def w10(words):
    f=words[:2000];return [f[i:i+10] for i in range(0,len(f),10) if len(f[i:i+10])>=2]
def score_cell(did,plain,op,representation):
    e1=scorer.adj_repeats(plain)>0;rows=[]
    for rep in range(20):
        plan=make_plan(plain,SCHEDULE,scorer.seed_of('PLADDER-VAL-plan',did,rep))
        out=encode_with_plan(plain,plan,representation)
        out=transform_lines(out,op,scorer.seed_of('PLADDER-P2-op',did,op,rep))
        rows.append(scorer.one_eval(scorer.prep(out),scorer.seed_of('PLADDER-VAL-stat',did,representation,rep),e1,50))
    return {'corpus':did,'operator':op,'representation':representation,**scorer.aggregate(rows),'rows':rows}
def one(did):
    plain=w10(rem[did]);return did,[score_cell(did,plain,op,r) for op in NEW_OPS for r in REPS]
cells=[]
with ProcessPoolExecutor(max_workers=min(12,os.cpu_count() or 1)) as ex:
    fs=[ex.submit(one,d) for d in chosen]
    for f in as_completed(fs):
        n,x=f.result();print('DONE',n,flush=True);cells.extend(x)
cells.sort(key=lambda c:(c['corpus'],c['operator'],c['representation']))
p=ROOT/'results'/f'P2_SHARD_{shard:02d}_OF_{nshards:02d}.json';p.write_text(json.dumps({'shard':shard,'nshards':nshards,'docs':chosen,'cells':cells},ensure_ascii=False),encoding='utf-8')
print(p,len(chosen),len(cells))

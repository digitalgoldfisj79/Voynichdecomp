# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4"]
# ///
import json, os, time, torch, torch.multiprocessing as mp

SECONDS=30.0
BATCH=262144
K=5
V=32
C=64

def worker(rank, q):
    torch.cuda.set_device(rank)
    dev=torch.device(f'cuda:{rank}')
    props=torch.cuda.get_device_properties(rank)
    contrib=torch.randn(K,V,C,device=dev,dtype=torch.float32)
    weights=torch.randn(C,device=dev,dtype=torch.float32)
    ids=torch.arange(BATCH,device=dev,dtype=torch.int64)
    for _ in range(4):
        x=ids.clone(); score=torch.zeros((BATCH,C),device=dev)
        for k in range(K):
            d=x.remainder(V); x=torch.div(x,V,rounding_mode='floor'); score += contrib[k,d]
        _=(score*weights).sum(1).max()
    torch.cuda.synchronize(rank)
    start=time.time(); total=0; base=rank*BATCH; maxid=V**K
    while time.time()-start<SECONDS:
        ids=(torch.arange(BATCH,device=dev,dtype=torch.int64)+base).remainder(maxid)
        x=ids.clone(); score=torch.zeros((BATCH,C),device=dev)
        for k in range(K):
            d=x.remainder(V); x=torch.div(x,V,rounding_mode='floor'); score += contrib[k,d]
        _=(score*weights).sum(1).max(); total += BATCH; base=(base+BATCH)%maxid
    torch.cuda.synchronize(rank); elapsed=time.time()-start
    q.put({'rank':rank,'name':props.name,'mem_gb':props.total_memory/2**30,'cand':total,'elapsed':elapsed,'rate':total/elapsed})

if __name__=='__main__':
    n=torch.cuda.device_count()
    if n<1: raise SystemExit('CUDA unavailable')
    ctx=mp.get_context('spawn'); q=ctx.Queue(); ps=[]
    for r in range(n):
        p=ctx.Process(target=worker,args=(r,q)); p.start(); ps.append(p)
    rows=[q.get() for _ in range(n)]
    for p in ps: p.join()
    rate=sum(r['rate'] for r in rows)
    out={'gpus':n,'devices':rows,'aggregate_cand_per_s':rate,'aggregate_cand_10min':rate*600,'nucleus_32pow7':32**7,'nucleus_32pow8':32**8,'bridge_5pow15':5**15,'bridge_5pow16':5**16,'seconds_32pow7':(32**7)/rate,'seconds_32pow8':(32**8)/rate,'seconds_5pow15':(5**15)/rate,'seconds_5pow16':(5**16)/rate}
    print('BENCH_X8='+json.dumps(out,sort_keys=True),flush=True)

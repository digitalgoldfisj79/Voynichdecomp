# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4"]
# ///
import argparse, json, math, time, torch

ap=argparse.ArgumentParser()
ap.add_argument('--seconds',type=float,default=45.0)
ap.add_argument('--batch',type=int,default=262144)
ap.add_argument('--vars',type=int,default=5)
ap.add_argument('--values',type=int,default=32)
ap.add_argument('--contexts',type=int,default=64)
a=ap.parse_args()

if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable')
dev=torch.device('cuda')
props=torch.cuda.get_device_properties(0)
print('DEVICE='+json.dumps({'name':props.name,'mem_gb':props.total_memory/2**30,'cc':[props.major,props.minor]}),flush=True)

# Representative exact block-key scoring kernel:
# each candidate is a tuple of assignments for `vars` surface types; score is
# the sum of dense precomputed per-type/per-value context contributions.
# This isolates combinatorial enumeration throughput from Python decoder cost.
V=a.values; K=a.vars; C=a.contexts; B=a.batch
contrib=torch.randn(K,V,C,device=dev,dtype=torch.float32)
weights=torch.randn(C,device=dev,dtype=torch.float32)

# warm-up
ids=torch.arange(B,device=dev,dtype=torch.int64)
for _ in range(5):
    x=ids.clone(); score=torch.zeros((B,C),device=dev)
    for k in range(K):
        d=x.remainder(V); x=torch.div(x,V,rounding_mode='floor'); score += contrib[k,d]
    z=(score*weights).sum(1); _=z.max()
torch.cuda.synchronize()

start=time.time(); total=0; batches=0; maxid=V**K
base=0
while time.time()-start<a.seconds:
    ids=(torch.arange(B,device=dev,dtype=torch.int64)+base).remainder(maxid)
    x=ids.clone(); score=torch.zeros((B,C),device=dev)
    for k in range(K):
        d=x.remainder(V); x=torch.div(x,V,rounding_mode='floor'); score += contrib[k,d]
    z=(score*weights).sum(1); _=z.max()
    total += B; batches += 1; base=(base+B)%maxid
torch.cuda.synchronize(); elapsed=time.time()-start
rate=total/elapsed
out={'vars':K,'values':V,'contexts':C,'batch':B,'elapsed_s':elapsed,'candidates':total,'cand_per_s':rate,'cand_10min':rate*600,'search_space':maxid,'full_space_seconds':maxid/rate}
print('BENCH='+json.dumps(out,sort_keys=True),flush=True)

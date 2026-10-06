#!/usr/bin/env python3
import argparse,base64,collections,gzip,importlib.util,json,math,pathlib,re,urllib.request
import numpy as np

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/df57f396ca185d1d07a5313e455be545143f0b97/research/"
CORPUS="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/19fc6f2262dc2d184b370fb7c6960f11278f4778/voynich_transcriptions_slim.json"
NK4="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/df57f396ca185d1d07a5313e455be545143f0b97/research/naibbe_kernel_NK4_20261005.py"
R=10
MODES={
 "standard":(1.0,1.0),
 "stop_short":(1.50,1.0),
 "stop_long":(0.67,1.0),
 "reuse_low":(1.0,0.50),
 "reuse_high":(1.0,2.00),
}
METRICS=["stolfi_novel","mauro_novel","joint_novel","type_rate","hapax_type_fraction",
         "first_novel_slope","length_mean","length_sd","space_edge_MI","lag1_repeat","lag2_repeat",
         "page_repeat_fraction","opener_MI","relative_position_head_MI"]

def dl(u,p): pathlib.Path(p).write_bytes(urllib.request.urlopen(u,timeout=120).read())
def loadmod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def load_acceptors():
    src=urllib.request.urlopen(NK4,timeout=120).read().decode()
    a=src.index("# --- classical whole-token diagnostics")
    b=src.index("# --- frozen FORM generator ---",a)
    snippet=src[a:b]
    ns={"re":re}
    exec(snippet,ns)
    return ns["stolfi_ok"],ns["mauro_ok"]

def page_repeat_fraction(rows):
    h=0;n=0
    for fol in sorted({r["folio"] for r in rows}):
        seen=set()
        for r in [x for x in rows if x["folio"]==fol]:
            for t in r["tokens"]:
                h+=int(t in seen);n+=1;seen.add(t)
    return h/max(n,1)

def diag(h,rows,train_types,stolfi_ok,mauro_ok):
    ts=[t for r in rows for t in r["tokens"]];N=len(ts);freq=collections.Counter(ts)
    novel=sorted(set(ts)-train_types)
    base=h.diagnostics(rows)
    first=[];seen=set(train_types);idx=0
    for r in rows:
        for t in r["tokens"]:
            if t not in seen:first.append(idx);seen.add(t)
            idx+=1
    q=np.zeros(4);den=np.zeros(4)
    firstset=set(first)
    for i in range(N):
        z=min(3,int(4*i/max(N,1)));den[z]+=1;q[z]+=int(i in firstset)
    rates=np.divide(q,den,out=np.zeros(4),where=den>0)
    slope=float(np.polyfit(np.linspace(0,1,4),rates,1)[0]) if N else 0.0
    lens=np.array([len(t) for t in ts],float) if ts else np.array([0.])
    return {
      "stolfi_novel":float(np.mean([stolfi_ok(t) for t in novel])) if novel else 0.0,
      "mauro_novel":float(np.mean([mauro_ok(t) for t in novel])) if novel else 0.0,
      "joint_novel":float(np.mean([stolfi_ok(t) and mauro_ok(t) for t in novel])) if novel else 0.0,
      "type_rate":len(freq)/max(N,1),
      "hapax_type_fraction":sum(v==1 for v in freq.values())/max(len(freq),1),
      "first_novel_slope":slope,
      "length_mean":float(lens.mean()),"length_sd":float(lens.std(ddof=1)) if len(lens)>1 else 0.0,
      "space_edge_MI":base["space_edge_MI"],"lag1_repeat":base["line_repeat_lag1"],
      "lag2_repeat":base["line_repeat_lag2"],"page_repeat_fraction":page_repeat_fraction(rows),
      "opener_MI":base["opener_MI"],"relative_position_head_MI":base["relative_position_head_MI"],
      "n_tokens":N,"n_types":len(freq),"n_novel_distinct":len(novel)
    }

def odds_scale(pi,s):
    if pi<=0:return 0.0
    if pi>=1:return 1.0
    o=pi/(1-pi);o*=s;return o/(1+o)

def draw_mod(h,m,e,state,rng,stop_scale=1.0,reuse_scale=1.0):
    pi=odds_scale(m.gate_pi(e),reuse_scale);hist=e["history"][-64:]
    if hist and rng.random()<pi:
        ws=collections.Counter()
        for lag,t in enumerate(reversed(hist),1):ws[t]+=1/math.sqrt(lag)
        kk=list(ws);pp=np.array([ws[x] for x in kk],float);pp/=pp.sum()
        return kk[int(rng.choice(len(kk),p=pp))],False
    pref=""
    for _ in range(2048):
        p=m.nextp(e,pref,state).copy()
        if pref:
            j=h.SI[h.EOS];p[j]*=stop_scale;p/=p.sum()
        j=int(rng.choice(len(h.SYMS),p=p));ch=h.SYMS[j]
        if ch==h.EOS:return pref,False
        pref+=ch
    return pref,True

def gen_mod(h,m,template_rows,seed,stop_scale,reuse_scale):
    rng=np.random.default_rng(seed);out=[];hist=[];last_bif=None;last_para=None;cache={}
    for r in template_rows:
        if r["fold"] not in {0,1}:continue
        if r["bifolium"]!=last_bif or r["para"]!=last_para:
            hist=[];last_bif=r["bifolium"];last_para=r["para"]
        toks=[]
        for i in range(len(r["tokens"])):
            prev=hist[-1] if hist else None;hv=np.zeros(64,np.float32)
            H=m.horizon
            for lag,x in enumerate(reversed(hist[-H:]),1):
                if x not in cache:cache[x]=h.strhash(x,64)
                hv+=cache[x]/math.sqrt(lag)
            n=np.linalg.norm(hv)
            if n:hv/=n
            e=dict(token="",section=r["section"],currier=r["currier"],
                   posbin=min(4,int(5*i/max(1,len(r["tokens"])))),
                   line_start=int(i==0),prev_final=(prev[-1] if prev else "^"),
                   prev_line_opener=(out[-1]["tokens"][0][0] if i==0 and out and out[-1]["folio"]==r["folio"]
                     and out[-1]["para"]==r["para"] and out[-1]["line"]+1==r["line"] and out[-1]["tokens"] else "^"),
                   history=tuple(hist[-64:]),hvec=hv)
            st=h.simhash_state(m,e)
            t,overflow=draw_mod(h,m,e,st,rng,stop_scale,reuse_scale)
            if overflow:raise RuntimeError("generation overflow")
            toks.append(t);hist.append(t)
        nr=dict(r);nr["tokens"]=toks;out.append(nr)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,required=True);a=ap.parse_args();seed=a.seed
    dl(BASE+"complete_form_harness_v4_20261006.py.gz.b64","/tmp/h.b64")
    pathlib.Path("/tmp/h.py").write_bytes(gzip.decompress(base64.b64decode(pathlib.Path("/tmp/h.b64").read_text().strip())))
    dl(BASE+"complete_form_hf_qualification_20261006.py","/tmp/q.py")
    dl(BASE+"data/complete_form_hf_meta_20261006.json.gz.b64","/tmp/m.b64");dl(CORPUS,"/tmp/c.json")
    h=loadmod("h","/tmp/h.py");q=loadmod("q","/tmp/q.py");meta=q.load_json_b64("/tmp/m.b64")
    rows,audit=q.recover_from_public("/tmp/c.json",meta)
    stolfi_ok,mauro_ok=load_acceptors()
    # Synthetic truth generated by C using discovery-only source fit, matching predictive qualification contract.
    tr=h.flatten_events(rows,{2,3},16)
    src=h.fit_model("C",tr,alpha=20,beta=5,horizon=16,k=8,seed=seed)
    syn=h.generate_rows(src,rows,seed=seed*1009+307,broken_context=False)
    selected=h.select_models(syn,seed=seed,fast=False)
    mods=h.refit_selected(syn,selected,seed=seed);m=mods["C"]
    train_types={e["token"] for e in h.flatten_events(syn,{2,3,4},64)}
    truth=[r for r in syn if r["fold"] in {0,1}]
    td=diag(h,truth,train_types,stolfi_ok,mauro_ok)
    modes={}
    for mi,(name,(ss,rs)) in enumerate(MODES.items()):
        ds=[]
        for j in range(R):
            g=gen_mod(h,m,syn,seed*100000+mi*1000+j,ss,rs)
            ds.append(diag(h,g,train_types,stolfi_ok,mauro_ok))
        mean={k:float(np.mean([d[k] for d in ds])) for k in METRICS}
        delta={k:mean[k]-td[k] for k in METRICS}
        modes[name]={"mean":mean,"delta":delta}
    out={"seed":seed,"audit":audit,"selected_C":selected["C"][1],"truth_diag":{k:td[k] for k in METRICS},
         "modes":modes,"metrics":METRICS,"replicates_per_mode":R}
    print("GENQUAL="+json.dumps(out,separators=(",",":"),sort_keys=True),flush=True)
if __name__=="__main__":main()

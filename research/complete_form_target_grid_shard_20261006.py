#!/usr/bin/env python3
import argparse,base64,gzip,importlib.util,json,pathlib,urllib.request

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5f4e0ef14ef78e0fde3039c298f69d1fdc0694f0/research/"
CORPUS="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/19fc6f2262dc2d184b370fb7c6960f11278f4778/voynich_transcriptions_slim.json"
SEED=7001
ALPHAS=[5.,20.,80.]
BETAS=[5.,20.,80.]

def dl(u,p): pathlib.Path(p).write_bytes(urllib.request.urlopen(u,timeout=120).read())
def loadmod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def prep():
    dl(BASE+"complete_form_harness_v4_20261006.py.gz.b64","/tmp/h.b64")
    pathlib.Path("/tmp/h.py").write_bytes(gzip.decompress(base64.b64decode(pathlib.Path("/tmp/h.b64").read_text().strip())))
    dl(BASE+"complete_form_hf_qualification_20261006.py","/tmp/q.py")
    dl(BASE+"data/complete_form_hf_meta_20261006.json.gz.b64","/tmp/m.b64")
    dl(CORPUS,"/tmp/c.json")
    h=loadmod("h","/tmp/h.py");q=loadmod("q","/tmp/q.py");meta=q.load_json_b64("/tmp/m.b64")
    rows,aud=q.recover_from_public("/tmp/c.json",meta)
    assert aud["n_lines"]==4117 and aud["n_tokens"]==34229 and aud["n_within"]==30112
    return h,rows,aud

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--h",type=int,required=True);ap.add_argument("--k",type=int,required=True);a=ap.parse_args()
    h,rows,aud=prep()
    tr=h.flatten_events(rows,{2,3},a.h);va=h.flatten_events(rows,{4},a.h)
    mc=h.fit_model("C",tr,alpha=20.,beta=20.,horizon=a.h,k=a.k,seed=SEED)
    ms=h.fit_shuffled_model(tr,alpha=20.,beta=20.,horizon=a.h,k=a.k,seed=SEED,
        shuffle_seed=h._shuf_seed(SEED,a.h,a.k,"train"))
    bestC=None;bestS=None
    for al in ALPHAS:
      for be in BETAS:
        mc.alpha=al;mc.beta=be;loss=float(h.score(mc,va).mean())
        cand=(loss,{"alpha":al,"beta":be,"horizon":a.h,"k":a.k})
        if bestC is None or loss<bestC[0]:bestC=cand
        ms.alpha=al;ms.beta=be
        sloss=float(h.score_shuffled(ms,va,h._shuf_seed(SEED,a.h,a.k,"val"))[0].mean())
        scand=(sloss,{"alpha":al,"beta":be,"horizon":a.h,"k":a.k})
        if bestS is None or sloss<bestS[0]:bestS=scand
    out={"h":a.h,"k":a.k,"C":{"loss":bestC[0],"params":bestC[1]},"S":{"loss":bestS[0],"params":bestS[1]}}
    if a.h==2 and a.k==4:
        ev0=h.flatten_events(rows,{2,3},64);v0=h.flatten_events(rows,{4},64)
        mb=h.fit_model("B",ev0,alpha=20.,seed=SEED);bestB=None
        for al in ALPHAS:
            mb.alpha=al;loss=float(h.score(mb,v0).mean());cand=(loss,{"alpha":al})
            if bestB is None or loss<bestB[0]:bestB=cand
        A=h.fit_model("A",ev0,seed=SEED);la=float(h.score(A,v0).mean())
        out["A"]={"loss":la,"params":{}}
        out["B"]={"loss":bestB[0],"params":bestB[1]}
    print("GRID_SHARD="+json.dumps(out,separators=(",",":"),sort_keys=True),flush=True)

if __name__=="__main__":main()

#!/usr/bin/env python3
import base64,gzip,importlib.util,json,pathlib,urllib.request,numpy as np

GQ_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/6072424401a5da2157c03e43a14b162128e079f6/research/complete_form_generation_qualification_seed_20261006.py"
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/df57f396ca185d1d07a5313e455be545143f0b97/research/"
CORPUS="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/19fc6f2262dc2d184b370fb7c6960f11278f4778/voynich_transcriptions_slim.json"
MU={"first_novel_slope":0.007473207776427702,"hapax_type_fraction":-0.02278618660152514,"joint_novel":-0.06646213850918886,"lag1_repeat":0.014874708775563034,"lag2_repeat":0.011683160415003991,"length_mean":-0.11016023693803163,"length_sd":0.16434866020119884,"mauro_novel":-0.020445432479622508,"opener_MI":0.002577286955856154,"page_repeat_fraction":0.0406962332928311,"relative_position_head_MI":-0.003213046428580171,"space_edge_MI":-7.233050790752579e-05,"stolfi_novel":-0.06661131796558023,"type_rate":-0.0038841889428918586}
SD={"first_novel_slope":0.008443369039105614,"hapax_type_fraction":0.008707981110557152,"joint_novel":0.013349331893179162,"lag1_repeat":0.002130164350982004,"lag2_repeat":0.00192462646003738,"length_mean":0.0286747117018329,"length_sd":0.036826093929490315,"mauro_novel":0.00827376631395167,"opener_MI":0.012419368236655244,"page_repeat_fraction":0.004928586194461099,"relative_position_head_MI":0.001915514499691016,"space_edge_MI":0.002615302126960894,"stolfi_novel":0.01331352267714193,"type_rate":0.004865997492055621}
THR=3.0044094891036517
METRICS=list(MU)
SEED=7001
NREP=10

def dl(u,p): pathlib.Path(p).write_bytes(urllib.request.urlopen(u,timeout=120).read())
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 dl(GQ_URL,"/tmp/gq.py");gq=loadmod("gq","/tmp/gq.py")
 dl(BASE+"complete_form_harness_v4_20261006.py.gz.b64","/tmp/h.b64")
 pathlib.Path("/tmp/h.py").write_bytes(gzip.decompress(base64.b64decode(pathlib.Path("/tmp/h.b64").read_text().strip())))
 dl(BASE+"complete_form_hf_qualification_20261006.py","/tmp/q.py")
 dl(BASE+"data/complete_form_hf_meta_20261006.json.gz.b64","/tmp/m.b64")
 dl(CORPUS,"/tmp/c.json")
 h=loadmod("h","/tmp/h.py");q=loadmod("q","/tmp/q.py");meta=q.load_json_b64("/tmp/m.b64")
 rows,audit=q.recover_from_public("/tmp/c.json",meta)
 assert audit["n_lines"]==4117 and audit["n_tokens"]==34229 and audit["n_within"]==30112
 stolfi_ok,mauro_ok=gq.load_acceptors()
 ev=h.flatten_events(rows,{2,3,4},16)
 m=h.fit_model("C",ev,alpha=80.0,beta=80.0,horizon=16,k=16,seed=SEED)
 train_types={e["token"] for e in h.flatten_events(rows,{2,3,4},64)}
 truth=[r for r in rows if r["fold"] in {0,1}]
 td=gq.diag(h,truth,train_types,stolfi_ok,mauro_ok)
 ds=[]
 for j in range(NREP):
   gen=gq.gen_mod(h,m,rows,2026100600+j,1.0,1.0)
   ds.append(gq.diag(h,gen,train_types,stolfi_ok,mauro_ok))
 mean={k:float(np.mean([d[k] for d in ds])) for k in METRICS}
 delta={k:mean[k]-td[k] for k in METRICS}
 z={k:(delta[k]-MU[k])/SD[k] for k in METRICS}
 az={k:abs(z[k]) for k in METRICS}
 win=max(az,key=az.get); stat=az[win]
 out={"stage":"voynich_free_generation_target_zlzi","audit":audit,
      "model":{"kind":"C","alpha":80.0,"beta":80.0,"horizon":16,"k":16,"fit_folds":[2,3,4],"test_folds":[0,1]},
      "n_replicates":NREP,"truth_diag":{k:td[k] for k in METRICS},"generated_mean":mean,"delta":delta,
      "calibration":{"mu":MU,"sd":SD,"maxabs_threshold":THR},
      "standardized":z,"maxabs":{"stat":stat,"winning_metric":win,"threshold":THR,"passes_adequacy":stat<=THR},
      "guard":"Joint generation adequacy only for this frozen battery/model/representation; no semantics/language/cipher/historical mechanism claim."}
 print("VOYNICH_GEN="+json.dumps(out,separators=(",",":"),sort_keys=True),flush=True)

if __name__=="__main__": main()

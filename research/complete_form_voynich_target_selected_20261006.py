#!/usr/bin/env python3
import base64,gzip,importlib.util,json,pathlib,urllib.request

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5f4e0ef14ef78e0fde3039c298f69d1fdc0694f0/research/"
CORPUS="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/19fc6f2262dc2d184b370fb7c6960f11278f4778/voynich_transcriptions_slim.json"
SEED=7001
SEL={
 "B":{"alpha":80.0},
 "C":{"alpha":80.0,"beta":80.0,"horizon":16,"k":16},
 "S":{"alpha":80.0,"beta":80.0,"horizon":64,"k":8}
}
VAL={"A":10.828944083066963,"B":11.065253343313927,"C":10.96063155444689,"S":11.007111802235219}
MU={"CB_all":-0.03411204935776476,"CB_first":-0.09564742376311867,"CS_all":-0.0027640390443200236,"CS_first":-0.0051135594307158165}
SD={"CB_all":0.009664896668776842,"CB_first":0.09612664454670879,"CS_all":0.006074534937792764,"CS_first":0.03957571653374771}
RAW_THR={"CB_all":-0.014962293745300033,"CB_first":0.07735081637696763,"CS_all":0.009183378331300273,"CS_first":0.039507590665440254}
FW_THR=2.4213793808791584

def dl(u,p): pathlib.Path(p).write_bytes(urllib.request.urlopen(u,timeout=120).read())
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 dl(BASE+"complete_form_harness_v4_20261006.py.gz.b64","/tmp/h.b64")
 pathlib.Path("/tmp/h.py").write_bytes(gzip.decompress(base64.b64decode(pathlib.Path("/tmp/h.b64").read_text().strip())))
 dl(BASE+"complete_form_hf_qualification_20261006.py","/tmp/q.py")
 dl(BASE+"data/complete_form_hf_meta_20261006.json.gz.b64","/tmp/m.b64")
 dl(CORPUS,"/tmp/c.json")
 h=loadmod("h","/tmp/h.py");q=loadmod("q","/tmp/q.py");meta=q.load_json_b64("/tmp/m.b64")
 rows,aud=q.recover_from_public("/tmp/c.json",meta)
 assert aud["n_lines"]==4117 and aud["n_tokens"]==34229 and aud["n_within"]==30112
 ev0=h.flatten_events(rows,{2,3,4},64)
 A=h.fit_model("A",ev0,seed=SEED)
 B=h.fit_model("B",ev0,seed=SEED,**SEL["B"])
 pc=SEL["C"]; ec=h.flatten_events(rows,{2,3,4},pc["horizon"])
 C=h.fit_model("C",ec,seed=SEED,**pc)
 ps=SEL["S"]; es=h.flatten_events(rows,{2,3,4},ps["horizon"])
 S=h.fit_shuffled_model(es,seed=SEED,shuffle_seed=h._shuf_seed(SEED,ps["horizon"],ps["k"],"refit"),**ps)
 cmp=h.compare_BCS(rows,{"A":A,"B":B,"C":C,"S":S},seed=SEED)
 heads={
  "CB_all":cmp["C_minus_B"]["all"]["mean"],
  "CB_first":cmp["C_minus_B"]["first_unseen_distinct"]["mean"],
  "CS_all":cmp["C_minus_S"]["all"]["mean"],
  "CS_first":cmp["C_minus_S"]["first_unseen_distinct"]["mean"]
 }
 z={k:(heads[k]-MU[k])/SD[k] for k in heads}
 out={
  "stage":"voynich_predictive_target_zlzi",
  "seed":SEED,"audit":aud,"selected":SEL,"validation_loss":VAL,
  "headline":{k:{"value":heads[k],"cal_null_mean":MU[k],"cal_null_sd":SD[k],"centered_effect":heads[k]-MU[k],
                   "effect_over_cal_null_sd":z[k],"raw_threshold":RAW_THR[k],"passes_raw_threshold":heads[k]>RAW_THR[k]} for k in heads},
  "familywise":{"stat":max(z.values()),"threshold":FW_THR,"passes":max(z.values())>FW_THR,"winning_metric":max(z,key=z.get)},
  "panels":{"C_minus_B":{k:cmp["C_minus_B"][k] for k in ["all","seen","unseen","first_unseen_distinct","line_start","within_line"]},
            "C_minus_S":{k:cmp["C_minus_S"][k] for k in ["all","seen","unseen","first_unseen_distinct","line_start","within_line"]}},
  "shuffle_audit":cmp["shuffle_audit"],
  "guard":"Predictive result only; free-generation adequacy not yet tested."
 }
 print("VOYNICH_TARGET="+json.dumps(out,separators=(",",":"),sort_keys=True),flush=True)

if __name__=="__main__":main()

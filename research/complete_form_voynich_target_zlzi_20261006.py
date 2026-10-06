#!/usr/bin/env python3
import argparse, base64, gzip, importlib.util, json, pathlib, urllib.request

HARNESS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5f4e0ef14ef78e0fde3039c298f69d1fdc0694f0/research/complete_form_harness_v4_20261006.py.gz.b64"
QUAL_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5f4e0ef14ef78e0fde3039c298f69d1fdc0694f0/research/complete_form_hf_qualification_20261006.py"
META_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5f4e0ef14ef78e0fde3039c298f69d1fdc0694f0/research/data/complete_form_hf_meta_20261006.json.gz.b64"
CORPUS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/19fc6f2262dc2d184b370fb7c6960f11278f4778/voynich_transcriptions_slim.json"

MU={"CB_all":-0.03411204935776476,"CB_first":-0.09564742376311867,"CS_all":-0.0027640390443200236,"CS_first":-0.0051135594307158165}
SD={"CB_all":0.009664896668776842,"CB_first":0.09612664454670879,"CS_all":0.006074534937792764,"CS_first":0.03957571653374771}
RAW_THR={"CB_all":-0.014962293745300033,"CB_first":0.07735081637696763,"CS_all":0.009183378331300273,"CS_first":0.039507590665440254}
FW_THR=2.4213793808791584
SEED=7001

def dl(u,p):
    pathlib.Path(p).write_bytes(urllib.request.urlopen(u,timeout=120).read())

def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    dl(HARNESS_URL,"/tmp/harness.b64")
    raw=pathlib.Path("/tmp/harness.b64").read_text().strip()
    pathlib.Path("/tmp/harness.py").write_bytes(gzip.decompress(base64.b64decode(raw)))
    dl(QUAL_URL,"/tmp/qual.py");dl(META_URL,"/tmp/meta.b64");dl(CORPUS_URL,"/tmp/corpus.json")
    h=loadmod("h","/tmp/harness.py");q=loadmod("q","/tmp/qual.py")
    meta=q.load_json_b64("/tmp/meta.b64")
    rows,audit=q.recover_from_public("/tmp/corpus.json",meta)
    if audit["n_lines"]!=4117 or audit["n_tokens"]!=34229 or audit["n_within"]!=30112:
        raise RuntimeError("population mismatch "+json.dumps(audit))
    selected=h.select_models(rows,seed=SEED,fast=False)
    models=h.refit_selected(rows,selected,seed=SEED)
    cmp=h.compare_BCS(rows,models,seed=SEED)
    heads={
      "CB_all":cmp["C_minus_B"]["all"]["mean"],
      "CB_first":cmp["C_minus_B"]["first_unseen_distinct"]["mean"],
      "CS_all":cmp["C_minus_S"]["all"]["mean"],
      "CS_first":cmp["C_minus_S"]["first_unseen_distinct"]["mean"],
    }
    z={k:(heads[k]-MU[k])/SD[k] for k in heads}
    fw=max(z.values())
    out={
      "stage":"voynich_predictive_target_zlzi",
      "seed":SEED,
      "audit":audit,
      "selected":{k:{"validation_loss":selected[k][0],"params":selected[k][1]} for k in "ABCS"},
      "headline":{k:{"value":heads[k],"cal_null_mean":MU[k],"cal_null_sd":SD[k],"z_vs_cal_null":z[k],"raw_threshold":RAW_THR[k],"passes_raw_threshold":heads[k]>RAW_THR[k]} for k in heads},
      "familywise":{"stat":fw,"threshold":FW_THR,"passes":fw>FW_THR,"winning_metric":max(z,key=z.get)},
      "panels":{
        "C_minus_B":{k:cmp["C_minus_B"][k] for k in ["all","seen","unseen","first_unseen_distinct","line_start","within_line"]},
        "C_minus_S":{k:cmp["C_minus_S"][k] for k in ["all","seen","unseen","first_unseen_distinct","line_start","within_line"]},
      },
      "shuffle_audit":cmp["shuffle_audit"],
      "interpretation_guard":"Predictive result only. No generative sufficiency, semantics, plaintext, language, or causal historical mechanism claim."
    }
    print("VOYNICH_TARGET="+json.dumps(out,separators=(",",":"),sort_keys=True),flush=True)

if __name__=="__main__":main()

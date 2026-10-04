#!/usr/bin/env python3
# Phase K4b: fresh confirmation of the frozen normalized calibration class.
# Acceptance uses ONLY parameter geometry + observable token length.
# Oracle truth is revealed only after 20 accepted instances are frozen.
# Synthetic-only. NO P70. No real Voynich inversion.
import json,urllib.request
import numpy as np

K4A_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/2c43acf5ff8b619b4e8eb50ae98dc27beb2f43d2/research/select_form_identifiability_screen_phaseK4a_20261004.py"
k4={"__name__":"k4a"}
exec(compile(urllib.request.urlopen(K4A_URL,timeout=60).read().decode(),K4A_URL,"exec"),k4)

Q10_MIN=.30
STAT_MIN=.015
LEN_LO=.80
LEN_HI=1.20
TARGET=20
START=20263001
MAX_SCAN=5000

def parameter_geometry(seed):
    rng=np.random.default_rng(seed)
    A,pi=k4["k0"]["source_graph"](k4["K"],k4["D"],rng)
    U,Ve0,Vr0=k4["k0"]["controls"](k4["K"],k4["R"],1.,rng,"F1")
    Ve=Ve0*k4["ENTRY"];Vr=Vr0*k4["ROUTE"]
    g=k4["geometry"](A,U,Ve,Vr)
    return g

if __name__=="__main__":
    accepted=[]
    scanned=0
    for seed in range(START,START+MAX_SCAN):
        scanned+=1
        g=parameter_geometry(seed)
        if g["q10_pair_js"] < Q10_MIN: continue
        if g["stationary_min"] < STAT_MIN: continue

        # Generate observation and oracle diagnostic. Acceptance decision below
        # references only observable mean_len_ratio, NOT oracle metrics.
        rec=k4["generate"](seed)
        if not (LEN_LO <= rec["mean_len_ratio"] <= LEN_HI): continue

        # At this point the instance is accepted and frozen.
        out={
          "accept_index":len(accepted),
          "seed":seed,
          "q10_pair_js":g["q10_pair_js"],
          "stationary_min":g["stationary_min"],
          "mean_len_ratio":rec["mean_len_ratio"],
          "mean_len":rec["mean_len"],
          "q95_len":rec["q95_len"],
          "oracle_nmi":rec["oracle_nmi"],
          "oracle_ari":rec["oracle_ari"],
          "oracle_ll":rec["oracle_ll"]
        }
        accepted.append(out)
        print("K4B_ACCEPTED_JSON="+json.dumps(out,separators=(",",":")),flush=True)
        if len(accepted)>=TARGET:break

    if len(accepted)<TARGET:
        raise RuntimeError(f"only {len(accepted)} accepted after {scanned} scans")

    y=np.array([x["oracle_nmi"] for x in accepted])
    def q(a,p):return float(np.quantile(a,p))
    summary={
      "phase":"K4b_normalized_oracle_confirmation",
      "acceptance_rule":{"q10_pair_js_min":Q10_MIN,"stationary_min":STAT_MIN,
                         "mean_len_ratio":[LEN_LO,LEN_HI]},
      "scanned":scanned,"accepted_n":len(accepted),
      "accepted_seeds":[x["seed"] for x in accepted],
      "oracle":{"median_nmi":float(np.median(y)),"q10_nmi":q(y,.10),
                "min_nmi":float(y.min()),"ge70_n":int(np.sum(y>=.70)),
                "ge70_fraction":float(np.mean(y>=.70))},
      "gate":{"median_ge75":bool(np.median(y)>=.75),
              "q10_ge70":bool(q(y,.10)>=.70),
              "ge90pct_ge70":bool(np.mean(y>=.70)>=.90)}
    }
    summary["gate"]["pass_all"]=all(summary["gate"].values())
    print("SELECT_FORM_PHASEK4B_JSON="+json.dumps(summary,separators=(",",":")),flush=True)

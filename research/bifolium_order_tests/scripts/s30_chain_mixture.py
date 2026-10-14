#!/usr/bin/env python3
"""PREREG 16: intermediate-peak forms vs a page-mixture null on eo777's Echo chain. -> results_chain_mixture.json"""
import json, os, pickle, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import binom
ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261023)
meta = pickle.load(open(os.path.join(D, "s1_words.pkl"), "rb"))["meta"]
meta = meta.assign(folio=meta.folio.astype(str), word=meta.word.astype(str))
pagewords = {f: Counter(g.word) for f, g in meta.groupby("folio")}
P = pd.read_csv(os.path.join(D, "eo777_chain_pages.csv")); P = P[P.page.isin(pagewords)]
chain = ["1a", "1b", "1.5a", "2a", "3a", "3b", "3c", "4a", "4c"]
rule_outputs = set(json.load(open(os.path.join(D, "echo_rule_outputs.json")))["rule_outputs"])
def N_peaks(pages_k, pages_E, pages_L, forms=None, return_forms=False):
    ck, cE, cL = Counter(), Counter(), Counter()
    for p in pages_k: ck.update(pagewords[p])
    for p in pages_E: cE.update(pagewords[p])
    for p in pages_L: cL.update(pagewords[p])
    nk, nE, nL = sum(ck.values()), sum(cE.values()), sum(cL.values())
    if forms is None: forms = [w for w in set(ck) | set(cE) | set(cL) if ck[w] + cE[w] + cL[w] >= 10]
    w = np.array(forms); k = np.array([ck[x] for x in w]); e = np.array([cE[x] for x in w]); l = np.array([cL[x] for x in w])
    rE = (e + .5) / (nE + 1); rL = (l + .5) / (nL + 1)
    pE = binom.sf(k - 1, nk, rE); pL = binom.sf(k - 1, nk, rL)
    peak = (k / nk > rE) & (k / nk > rL) & (pE < .01) & (pL < .01)
    if return_forms:
        order = np.argsort(pE + pL)
        return int(peak.sum()), [(w[i], int(k[i]), round(float(k[i] / nk * 1e3), 2), round(float(rE[i] * 1e3), 2), round(float(rL[i] * 1e3), 2), w[i] in rule_outputs)
                                 for i in order if peak[i]][:12], forms
    return int(peak.sum())
def test(pages_by_stage, stages, tag, nperm=2000):
    out = {}
    for i in range(1, len(stages) - 1):
        k = stages[i]; Pk = pages_by_stage[k]; PE = [p for s in stages[:i] for p in pages_by_stage[s]]
        PL = [p for s in stages[i + 1:] for p in pages_by_stage[s]]
        obs, top, forms = N_peaks(Pk, PE, PL, return_forms=True)
        allp = np.array(Pk + PE + PL); nl = []
        for _ in range(nperm):
            q = rng.permutation(allp); a, b = len(Pk), len(Pk) + len(PE)
            nl.append(N_peaks(list(q[:a]), list(q[a:b]), list(q[b:]), forms))
        nl = np.array(nl)
        out[k] = dict(pages=len(Pk), N_peaks=obs, null_mean=float(nl.mean()), null_sd=float(nl.std()),
                      z=float((obs - nl.mean()) / nl.std()) if nl.std() > 0 else None, p=float((np.sum(nl >= obs) + 1) / (nperm + 1)),
                      top_peaking=top)
        print(tag, k, {x: (round(v, 2) if isinstance(v, float) else v) for x, v in out[k].items() if x != "top_peaking"}, flush=True)
    return out
pbs = {s: list(P[P.group == s].page) for s in chain}
res = json.load(open(os.path.join(ROOT, "results_chain_mixture.json"))) if os.path.exists(os.path.join(ROOT, "results_chain_mixture.json")) else {}
if "full_chain" not in res: res["full_chain"] = test(pbs, chain, "full")
P1 = P[P.scribe.astype(str) == "1.0"]; pbs1 = {s: list(P1[P1.group == s].page) for s in ["1a", "1b", "1.5a", "2a"]}
res["scribe1_chain"] = test(pbs1, ["1a", "1b", "1.5a", "2a"], "scribe1")
json.dump(res, open(os.path.join(ROOT, "results_chain_mixture.json"), "w"), indent=1, default=str)

# ---- exploratory (not pre-registered): are the peaking forms enriched for Echo rule outputs?
from scipy.stats import fisher_exact
enr = {}
for i in range(1, len(chain) - 1):
    k = chain[i]; Pk = pbs[k]; PE = [p for s in chain[:i] for p in pbs[s]]; PL = [p for s in chain[i + 1:] for p in pbs[s]]
    ck, cE, cL = Counter(), Counter(), Counter()
    for p in Pk: ck.update(pagewords[p])
    for p in PE: cE.update(pagewords[p])
    for p in PL: cL.update(pagewords[p])
    nk, nE, nL = sum(ck.values()), sum(cE.values()), sum(cL.values())
    forms = [w for w in set(ck) | set(cE) | set(cL) if ck[w] + cE[w] + cL[w] >= 10]
    pk = [w for w in forms if ck[w] / nk > (cE[w] + .5) / (nE + 1) and ck[w] / nk > (cL[w] + .5) / (nL + 1)
          and binom.sf(ck[w] - 1, nk, (cE[w] + .5) / (nE + 1)) < .01 and binom.sf(ck[w] - 1, nk, (cL[w] + .5) / (nL + 1)) < .01]
    a = sum(w in rule_outputs for w in pk); b = len(pk) - a; c = sum(w in rule_outputs for w in forms) - a; d = len(forms) - len(pk) - c
    OR, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    enr[k] = dict(peaks=len(pk), peaks_rule_outputs=a, tested=len(forms), tested_rule_outputs=a + c, odds_ratio=float(OR), p=float(p))
    print("enrichment", k, enr[k])
json.dump(enr, open(os.path.join(ROOT, "results_chain_rule_enrichment.json"), "w"), indent=1)

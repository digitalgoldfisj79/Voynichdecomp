#!/usr/bin/env python3
"""Unblind only after measurement artifacts are frozen; run manuscript-scale inference."""
from __future__ import annotations
import argparse, glob, hashlib, itertools, json, math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260820
AUTHORS_SIX = {"f39v", "f23r", "f56v", "f42v", "f7v", "f14v"}


def load_many(pattern):
    fs = sorted(glob.glob(pattern))
    if not fs:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True), fs


def gap_summary(d, value="gap_px"):
    out = {}
    for g in ("certain", "uncertain"):
        x = d.loc[d.group == g, value].dropna().to_numpy(float)
        out[g] = {
            "n": int(len(x)),
            "mean": float(x.mean()) if len(x) else None,
            "median": float(np.median(x)) if len(x) else None,
            "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
        }
    c = d.loc[d.group == "certain", value].dropna().to_numpy(float)
    u = d.loc[d.group == "uncertain", value].dropna().to_numpy(float)
    diff = float(c.mean() - u.mean()) if len(c) and len(u) else float("nan")
    if len(c) > 1 and len(u) > 1:
        sp = math.sqrt(((len(c)-1)*c.var(ddof=1) + (len(u)-1)*u.var(ddof=1)) / (len(c)+len(u)-2))
        cohen = diff / sp if sp else float("nan")
    else:
        cohen = float("nan")
    out["difference_certain_minus_uncertain"] = diff
    out["cohen_d"] = float(cohen)
    return out


def folio_bootstrap(d, value="gap_px", nrep=10000, seed=SEED):
    rows = []
    for fo, g in d.groupby("folio"):
        c = g.loc[g.group == "certain", value].dropna().to_numpy(float)
        u = g.loc[g.group == "uncertain", value].dropna().to_numpy(float)
        rows.append((fo, c.sum(), len(c), u.sum(), len(u)))
    if not rows:
        return {}
    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows], float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    sims = []
    for _ in range(nrep):
        s = arr[rng.integers(0, n, size=n)].sum(axis=0)
        if s[1] > 0 and s[3] > 0:
            sims.append(s[0]/s[1] - s[2]/s[3])
    sims = np.asarray(sims)
    return {
        "n_folios": n, "nrep": int(len(sims)), "mean": float(sims.mean()),
        "sd": float(sims.std(ddof=1)),
        "ci95": [float(np.quantile(sims, .025)), float(np.quantile(sims, .975))],
    }


def line_permutation(d, value="gap_px", nrep=50000, seed=SEED):
    opts, observed = [], []
    for key, g in d.groupby(["folio", "line"], sort=False):
        if g.group.nunique() != 2:
            continue
        vals = g[value].to_numpy(float)
        is_u = g.group.to_numpy() == "uncertain"
        nu = int(is_u.sum())
        nc = len(vals) - nu
        if not nu or not nc:
            continue
        observed.append(vals[~is_u].mean() - vals[is_u].mean())
        ncomb = math.comb(len(vals), nu)
        if ncomb <= 20000:
            sums = np.fromiter((sum(vals[list(ix)]) for ix in itertools.combinations(range(len(vals)), nu)), float, count=ncomb)
            o = (vals.sum() - sums) / nc - sums / nu
        else:
            stable = int.from_bytes(hashlib.sha256((str(key) + "|" + str(SEED)).encode()).digest()[:8], "big")
            rr = np.random.default_rng(stable)
            vv = []
            for _ in range(20000):
                ix = rr.choice(len(vals), size=nu, replace=False)
                su = vals[ix].sum()
                vv.append((vals.sum() - su) / nc - su / nu)
            o = np.asarray(vv, float)
        opts.append(o)
    if not opts:
        return {}
    obs = float(np.mean(observed))
    rng = np.random.default_rng(seed)
    sims = np.empty(nrep, float)
    batch = 1000
    for a in range(0, nrep, batch):
        b = min(batch, nrep-a)
        z = np.zeros(b, float)
        for o in opts:
            z += rng.choice(o, size=b)
        sims[a:a+b] = z / len(opts)
    null_mean = float(sims.mean())
    null_sd = float(sims.std(ddof=1))
    p1 = float((1 + (sims >= obs).sum()) / (nrep + 1))
    p2 = float((1 + (np.abs(sims-null_mean) >= abs(obs-null_mean)).sum()) / (nrep + 1))
    return {
        "observed_mean_line_diff_px": obs,
        "informative_lines": len(opts),
        "positive_lines": int(sum(x > 0 for x in observed)),
        "nrep": nrep,
        "null_mean_px": null_mean,
        "null_sd_px": null_sd,
        "effect_over_null_sd": float((obs-null_mean)/null_sd) if null_sd else None,
        "p_one_sided": p1,
        "p_two_sided": p2,
    }


def per_folio_sign(d, value="gap_px"):
    diffs = []
    for fo, g in d.groupby("folio"):
        if g.group.nunique() != 2:
            continue
        a = g.groupby("group")[value].mean()
        diffs.append((fo, float(a["certain"] - a["uncertain"])))
    n = len(diffs)
    pos = sum(x[1] > 0 for x in diffs)
    p = sum(math.comb(n, k) for k in range(pos, n+1)) / 2**n if n else float("nan")
    return {"informative_folios": n, "positive_folios": pos, "one_sided_exact_sign_p": float(p), "diffs": diffs}


def inclusion_by_group(d):
    z = []
    for g, q in d.groupby("group"):
        z.append({
            "group": g, "n": int(len(q)), "qc_ok": int((q.qc == "ok").sum()),
            "finite": int(q.gap_px.notna().sum()), "qc_ok_rate": float((q.qc == "ok").mean()),
        })
    return z


def strata_table(d, value="gap_px", col="hand", min_u=10, min_c=50):
    rows = []
    for k, g in d.groupby(col):
        c = g[g.group == "certain"][value].dropna()
        u = g[g.group == "uncertain"][value].dropna()
        if len(c) >= min_c and len(u) >= min_u:
            rows.append({
                col: str(k), "n_certain": len(c), "n_uncertain": len(u),
                "certain_mean": float(c.mean()), "uncertain_mean": float(u.mean()),
                "diff": float(c.mean()-u.mean()),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--section-map", default="voynich_section_map.json")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    d700, _ = load_many(str(root / "measure_700_s*.csv"))
    d2500, _ = load_many(str(root / "measure_2500_s*.csv"))
    key, _ = load_many(str(root / "sealed_key_s*.csv"))

    d700 = d700.merge(key, on=["blind_id", "folio", "line"], how="inner", validate="many_to_one")
    d2500 = d2500.merge(key, on=["blind_id", "folio", "line"], how="inner", validate="one_to_one")
    for d in (d700, d2500):
        d["group"] = d.label.map({".": "certain", ",": "uncertain"})

    secmap = {}
    p = Path(args.section_map)
    if p.exists():
        secmap = json.loads(p.read_text()).get("mapping", {})
    for d in (d700, d2500):
        d["section"] = d.folio.map(secmap).fillna("Unassigned")

    z0 = d700[d700.threshold_offset == 0].copy()
    primary = z0[(z0.qc == "ok") & z0.gap_px.notna()].copy()
    finite = z0[z0.gap_px.notna()].copy()
    h0 = d2500[(d2500.threshold_offset == 0) & (d2500.qc == "ok") & d2500.gap_px.notna()].copy()

    res = {
        "schema": "direct-pixel-full-analysis-v0.1",
        "blinding": "measurement jobs received blind manifests only; labels merged in analysis job",
        "primary_definition": "700px, offset 0, automatic qc == ok, certain-minus-uncertain",
        "primary_700": gap_summary(primary),
        "primary_inclusion": inclusion_by_group(z0),
        "primary_folio_bootstrap": folio_bootstrap(primary),
        "primary_line_permutation": line_permutation(primary),
        "primary_folio_sign": per_folio_sign(primary),
        "all_finite_700_sensitivity": gap_summary(finite),
        "highres_2500_normalized": gap_summary(h0, "gap_700eq"),
        "highres_2500_raw_px": gap_summary(h0, "gap_px"),
        "authors_six_folios_full_eligible": gap_summary(primary[primary.folio.isin(AUTHORS_SIX)]),
        "strata": {
            "hand": strata_table(primary, col="hand"),
            "quire": strata_table(primary, col="quire"),
            "section": strata_table(primary, col="section"),
        },
    }

    a = primary[["blind_id", "gap_px", "group"]].rename(columns={"gap_px": "gap700"})
    b = h0[["blind_id", "gap_700eq"]].rename(columns={"gap_700eq": "gap2500eq"})
    pair = a.merge(b, on="blind_id")
    if len(pair) > 2:
        res["resolution_pairing"] = {
            "n": int(len(pair)),
            "pearson": float(pair[["gap700", "gap2500eq"]].corr().iloc[0, 1]),
            "mean_abs_difference_700eq": float(np.mean(np.abs(pair.gap700-pair.gap2500eq))),
        }

    ts = []
    for off, g in d700.groupby("threshold_offset"):
        q = g[(g.qc == "ok") & g.gap_px.notna()]
        s = gap_summary(q)
        ts.append({
            "offset": int(off), "n_certain": s["certain"]["n"], "n_uncertain": s["uncertain"]["n"],
            "certain_mean": s["certain"]["mean"], "uncertain_mean": s["uncertain"]["mean"],
            "diff": s["difference_certain_minus_uncertain"],
        })
    res["threshold_sensitivity_700"] = ts

    freezes = []
    for f in sorted(root.glob("freeze_s*.json")):
        try:
            freezes.append(json.loads(f.read_text()))
        except Exception:
            pass
    if freezes:
        nums = ["raw_boundaries", "raw_uncertain", "aligned_boundaries", "aligned_uncertain"]
        agg = {k: int(sum(float(x.get("counts", {}).get(k, 0)) for x in freezes)) for k in nums}
        agg["alignment_rate"] = agg["aligned_boundaries"] / agg["raw_boundaries"] if agg["raw_boundaries"] else None
        agg["uncertain_alignment_rate"] = agg["aligned_uncertain"] / agg["raw_uncertain"] if agg["raw_uncertain"] else None
        res["alignment_counts"] = agg

    lp = res["primary_line_permutation"]
    bs = res["primary_folio_bootstrap"]
    D = res["primary_700"]["difference_certain_minus_uncertain"]
    replicated = bool(D > 0 and lp and lp["p_two_sided"] < .05 and bs and bs["ci95"][0] > 0)
    core_offsets = [x for x in ts if -15 <= x["offset"] <= 15]
    threshold_stable = bool(core_offsets and all(x["diff"] > 0 for x in core_offsets))
    Dh = res["highres_2500_normalized"]["difference_certain_minus_uncertain"]
    highres_stable = bool(np.isfinite(Dh) and Dh > 0)
    res["frozen_adjudication"] = {
        "replicates_primary": replicated,
        "threshold_core_all_positive": threshold_stable,
        "highres_normalized_positive": highres_stable,
        "status": (
            "REPLICATES_AND_SCALE_STABLE" if replicated and threshold_stable and highres_stable else
            "PRIMARY_REPLICATES_BUT_SENSITIVITY_FAILS" if replicated else
            "PRIMARY_DOES_NOT_REPLICATE"
        ),
    }

    (out / "DIRECT_PIXEL_FULL_RESULTS.json").write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Full-manuscript blind direct-pixel boundary replication", "",
        f"**Frozen adjudication:** `{res['frozen_adjudication']['status']}`", "",
        f"Primary 700px objective-QC certain-minus-uncertain mean gap: **{D:.4f} px**.",
    ]
    if lp:
        ratio = abs(lp["effect_over_null_sd"] or 0)
        prefix = "**The metric does not resolve this:** " if ratio < 2 else ""
        lines.append(prefix + f"within-line effect **{lp['observed_mean_line_diff_px']:.4f} px** with permutation null SD **{lp['null_sd_px']:.4f} px** ({lp['effect_over_null_sd']:.2f} SD), two-sided p={lp['p_two_sided']:.6g}.")
    if bs:
        lines.append(f"Folio-block bootstrap 95% CI: **[{bs['ci95'][0]:.4f}, {bs['ci95'][1]:.4f}] px**.")
    lines += [f"High-resolution arm, rescaled to 700px-equivalent units: **{Dh:.4f} px** certain-minus-uncertain.", ""]
    if "alignment_counts" in res:
        a0 = res["alignment_counts"]
        lines += [
            f"Aligned eligible boundaries: **{a0['aligned_boundaries']:,}/{a0['raw_boundaries']:,}** ({a0['alignment_rate']:.1%}); uncertain **{a0['aligned_uncertain']:,}/{a0['raw_uncertain']:,}** ({a0['uncertain_alignment_rate']:.1%}).",
            "",
        ]
    lines += ["## Threshold sensitivity", "", "| offset | n certain | n uncertain | difference px |", "|---:|---:|---:|---:|"]
    for x in ts:
        lines.append(f"| {x['offset']} | {x['n_certain']} | {x['n_uncertain']} | {x['diff']:.4f} |")
    (out / "DIRECT_PIXEL_FULL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()

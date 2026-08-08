#!/usr/bin/env python3
"""Control matcher v2 for Alpine–Venetian Corridor Programme v0.1.

This executable supersedes only the matching stage of corridor_programme.py for
run01. It was frozen after pre-outcome QA showed that the scalar v1 matcher did
not respect the preregistered priority ordering. No VMS similarity score had
been computed or inspected when this file was introduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import psycopg

SEED = 20260808
CONTROL_GEOS = {"control_lombardy", "control_bavaria_swabia", "control_tuscany", "control_east_alpine"}


def db():
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise SystemExit("SUPABASE_DB_URL is required")
    return psycopg.connect(url)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def jaccard_parts(a, b):
    a, b = set(a or []), set(b or [])
    inter = len(a & b)
    union = len(a | b)
    j = 0.0 if union == 0 else 1.0 - inter / union
    return inter, union, j


def coverage_rank(x):
    return {"none": 0, "unknown": 1, "partial": 2, "complete": 3}.get(x, 1)


def priority(c, d):
    if c["time_bin"] != d["time_bin"]:
        return None
    inter, union, j = jaccard_parts(c["content_tags"], d["content_tags"])
    substrate_penalty = 0 if (not c["substrate"] or not d["substrate"] or c["substrate"] == d["substrate"]) else 1
    coverage_penalty = abs(coverage_rank(c["image_coverage"]) - coverage_rank(d["image_coverage"]))
    holder_penalty = 1 if c["holding_institution"] and c["holding_institution"] == d["holding_institution"] else 0
    key = (
        0 if inter > 0 else 1,  # content overlap before all lower priorities
        j,
        substrate_penalty,
        coverage_penalty,
        holder_penalty,
        sha(d["candidate_key"] + str(SEED)),
    )
    scalar_diagnostic = j + 0.5 * substrate_penalty + 0.25 * coverage_penalty + 0.25 * holder_penalty
    factors = {
        "time_bin": c["time_bin"],
        "shared_tags": inter,
        "union_tags": union,
        "jaccard": j,
        "substrate_penalty": substrate_penalty,
        "coverage_penalty": coverage_penalty,
        "holder_penalty": holder_penalty,
        "control_ecology": d["geography_class"],
        "matcher": "lexicographic_preregister_priority_v2",
    }
    return key, scalar_diagnostic, factors


def main(label: str):
    with db() as con, con.cursor() as cur:
        cur.execute("select run_id from public.corridor_runs where run_label=%s", (label,))
        row = cur.fetchone()
        if not row:
            raise SystemExit(f"unknown run: {label}")
        run_id = row[0]
        cur.execute("""
            select candidate_key,geography_class,time_bin,content_tags,substrate,image_coverage,holding_institution
            from public.corridor_candidates
            where inclusion_status='included' and time_bin in ('A_primary','B_antecedent')
        """)
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        corridor = [r for r in rows if r["geography_class"] == "corridor_core"]
        controls = [r for r in rows if r["geography_class"] in CONTROL_GEOS]
        cur.execute("delete from public.corridor_control_matches where run_id=%s", (run_id,))
        n = 0
        for c in corridor:
            ranked = []
            for d in controls:
                x = priority(c, d)
                if x is not None:
                    key, dist, factors = x
                    ranked.append((key, dist, factors, d))
            ranked.sort(key=lambda z: z[0])
            for rank, (_, dist, factors, d) in enumerate(ranked[:3], 1):
                cur.execute("""
                    insert into public.corridor_control_matches
                    (run_id,corridor_candidate_key,control_candidate_key,match_rank,match_distance,match_factors)
                    values(%s,%s,%s,%s,%s,%s::jsonb)
                """, (run_id, c["candidate_key"], d["candidate_key"], rank, dist, json.dumps(factors)))
                n += 1
        con.commit()
        cur.execute("""
            select count(*), count(distinct corridor_candidate_key),
                   count(*) filter(where (match_factors->>'shared_tags')::int=0),
                   count(distinct corridor_candidate_key) filter(where (match_factors->>'shared_tags')::int>0)
            from public.corridor_control_matches where run_id=%s
        """, (run_id,))
        total, manuscripts, zero, with_overlap = cur.fetchone()
    print(json.dumps({"match_rows": total, "corridor_manuscripts": manuscripts,
                      "zero_overlap_fallback_rows": zero,
                      "corridor_with_at_least_one_content_overlap": with_overlap}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-label", required=True)
    args = ap.parse_args()
    main(args.run_label)

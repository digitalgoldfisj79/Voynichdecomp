#!/usr/bin/env python3
"""Alpine–Venetian Corridor Programme v0.1.

This runner is deliberately conservative: discovery/seeding uses metadata only;
no VMS similarity is inspected until the cohort can be frozen.

Requires:
    pip install psycopg[binary] requests numpy

Environment:
    SUPABASE_DB_URL=postgresql://...

Examples:
    python corridor_programme.py audit
    python corridor_programme.py seed-existing
    python corridor_programme.py coverage
    python corridor_programme.py freeze-run --label corridor_v01_20260808
    python corridor_programme.py resolve-iiif
    python corridor_programme.py match-controls --run-label corridor_v01_20260808
    python corridor_programme.py analyse --run-label corridor_v01_20260808
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psycopg
import requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())
SEED = int(CONFIG["rng_seed"])
TIME_BINS = CONFIG["time_bins"]
CONTROL_GEOS = {
    "control_lombardy", "control_tuscany", "control_bavaria_swabia", "control_east_alpine"
}


def db() -> psycopg.Connection:
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise SystemExit("SUPABASE_DB_URL is required")
    return psycopg.connect(url)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def intersect_years(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def assign_time_bin(start: int | None, end: int | None) -> tuple[str | None, list[str]]:
    if start is None or end is None:
        return None, []
    overlaps = []
    for label, (b0, b1) in TIME_BINS.items():
        n = intersect_years(start, end, int(b0), int(b1))
        if n:
            overlaps.append((n, int(b0), label))
    if not overlaps:
        return None, []
    # Largest overlap; tie -> earlier bin. This is an analysis label, never a precise date claim.
    overlaps.sort(key=lambda x: (-x[0], x[1]))
    return overlaps[0][2], [x[2] for x in sorted(overlaps, key=lambda x: x[1])]


def classify_geography(place: str | None) -> tuple[str, float]:
    p = (place or "").lower()
    core = [
        "bressanone", "brixen", "bolzano", "bozen", "trento", "trient", "rovereto",
        "verona", "padua", "padova", "venice", "venezia"
    ]
    if any(x in p for x in core):
        return "corridor_core", 0.95
    if any(x in p for x in ["tyrol", "tirol", "trentino", "south tyrol", "südtirol", "veneto"]):
        return "corridor_buffer", 0.60
    if any(x in p for x in ["milan", "milano", "pavia", "lombard"]):
        return "control_lombardy", 0.90
    if any(x in p for x in ["florence", "firenze", "siena", "tuscany", "toscana"]):
        return "control_tuscany", 0.90
    if any(x in p for x in ["munich", "münchen", "augsburg", "regensburg", "bavaria", "swabia", "schwaben"]):
        return "control_bavaria_swabia", 0.85
    if any(x in p for x in ["salzburg", "vienna", "wien", "graz", "styria", "steiermark"]):
        return "control_east_alpine", 0.85
    return "unresolved", 0.0


def audit() -> None:
    q = """
    select
      count(*) filter (where date_start is not null and date_end is not null) as dated,
      count(*) as total,
      count(*) filter (where iiif_manifest_url is not null) as iiif,
      count(*) filter (where dating_authority is not null) as dating_authority,
      count(*) filter (where place_of_origin is not null) as place_known
    from public.manuscripts;
    """
    with db() as con, con.cursor() as cur:
        cur.execute(q)
        row = cur.fetchone()
        print(json.dumps(dict(zip([d.name for d in cur.description], row)), indent=2, default=str))
        cur.execute("select count(*) from public.corridor_candidates")
        print(json.dumps({"corridor_candidates": cur.fetchone()[0]}))


def seed_existing() -> None:
    """Seed neutral date/geography candidates from the existing manuscript registry.

    This does not mark them included; it only stages them for authority/image verification.
    Existing registry records are marked legacy_voynich_context because they come from a
    Voynich comparanda database and therefore cannot be treated as neutrally discovered.
    """
    q = """
    select id,name,date_start,date_end,date_display,dating_authority,place_of_origin,
           library_held,substrate,language,genre_tags,iiif_manifest_url,source_url,research_note
    from public.manuscripts
    where date_start is not null and date_end is not null
      and date_start <= 1500 and date_end >= 1350
    order by id
    """
    ins = """
    insert into public.corridor_candidates(
      candidate_key,registry_id,title,holding_institution,production_place_verbatim,
      geography_class,geography_confidence,dating_verbatim,date_start,date_end,dating_authority,
      time_bin,antecedent_eligible,substrate,language,content_tags,facsimile_url,iiif_manifest_url,
      discovery_source,discovery_bias,inclusion_status,metadata,updated_at
    ) values (
      %(candidate_key)s,%(registry_id)s,%(title)s,%(holding)s,%(place)s,
      %(geo)s,%(geo_conf)s,%(date_text)s,%(date_start)s,%(date_end)s,%(dating_authority)s,
      %(time_bin)s,%(antecedent)s,%(substrate)s,%(language)s,%(tags)s,%(source_url)s,%(iiif)s,
      'public.manuscripts seed','legacy_voynich_context','needs_review',%(metadata)s::jsonb,now()
    ) on conflict(candidate_key) do update set
      registry_id=excluded.registry_id,title=excluded.title,holding_institution=excluded.holding_institution,
      production_place_verbatim=excluded.production_place_verbatim,geography_class=excluded.geography_class,
      geography_confidence=excluded.geography_confidence,dating_verbatim=excluded.dating_verbatim,
      date_start=excluded.date_start,date_end=excluded.date_end,dating_authority=excluded.dating_authority,
      time_bin=excluded.time_bin,antecedent_eligible=excluded.antecedent_eligible,substrate=excluded.substrate,
      language=excluded.language,content_tags=excluded.content_tags,facsimile_url=excluded.facsimile_url,
      iiif_manifest_url=excluded.iiif_manifest_url,metadata=excluded.metadata,updated_at=now()
    """
    staged = 0
    with db() as con, con.cursor() as cur:
        cur.execute(q)
        cols = [d.name for d in cur.description]
        for tup in cur.fetchall():
            r = dict(zip(cols, tup))
            geo, conf = classify_geography(r["place_of_origin"])
            if geo == "unresolved":
                continue
            time_bin, intersections = assign_time_bin(r["date_start"], r["date_end"])
            payload = {
                "candidate_key": f"registry:{r['id']}", "registry_id": r["id"], "title": r["name"],
                "holding": r["library_held"], "place": r["place_of_origin"], "geo": geo,
                "geo_conf": conf, "date_text": r["date_display"], "date_start": r["date_start"],
                "date_end": r["date_end"], "dating_authority": r["dating_authority"],
                "time_bin": time_bin, "antecedent": bool(r["date_end"] <= CONFIG["antecedent_latest_end_year"]),
                "substrate": r["substrate"], "language": r["language"], "tags": r["genre_tags"] or [],
                "source_url": r["source_url"], "iiif": r["iiif_manifest_url"],
                "metadata": json.dumps({"intersecting_time_bins": intersections, "research_note": r["research_note"]})
            }
            cur.execute(ins, payload)
            staged += 1
        con.commit()
    print(json.dumps({"staged": staged}, indent=2))


def parse_iiif(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # IIIF Presentation 2
    if manifest.get("sequences"):
        canvases = manifest["sequences"][0].get("canvases", [])
        for i, c in enumerate(canvases, 1):
            img = ((c.get("images") or [{}])[0].get("resource") or {})
            service = img.get("service") or {}
            if isinstance(service, list):
                service = service[0] if service else {}
            base = service.get("@id") or service.get("id")
            url = f"{base}/full/1200,/0/default.jpg" if base else img.get("@id") or img.get("id")
            out.append({"seq": i, "folio": c.get("label"), "canvas_id": c.get("@id"),
                        "iiif_image_base": base, "image_url": url,
                        "width": c.get("width"), "height": c.get("height")})
        return out
    # IIIF Presentation 3
    for i, c in enumerate(manifest.get("items", []), 1):
        body = {}
        try:
            body = c["items"][0]["items"][0]["body"]
        except (KeyError, IndexError, TypeError):
            pass
        service = body.get("service") or []
        if isinstance(service, dict):
            service = [service]
        base = (service[0].get("id") if service else None)
        url = f"{base}/full/1200,/0/default.jpg" if base else body.get("id")
        label = c.get("label")
        if isinstance(label, dict):
            vals = next(iter(label.values()), [])
            label = vals[0] if vals else None
        out.append({"seq": i, "folio": label, "canvas_id": c.get("id"),
                    "iiif_image_base": base, "image_url": url,
                    "width": c.get("width"), "height": c.get("height")})
    return out


def resolve_iiif(limit: int | None = None) -> None:
    sel = """
    select candidate_key,iiif_manifest_url from public.corridor_candidates
    where iiif_manifest_url is not null and inclusion_status <> 'excluded'
    order by candidate_key
    """
    ins = """
    insert into public.corridor_pages(candidate_key,folio,seq,canvas_id,iiif_image_base,image_url,
      image_width,image_height,opaque_blind_id)
    values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    on conflict(candidate_key,seq) do update set folio=excluded.folio,canvas_id=excluded.canvas_id,
      iiif_image_base=excluded.iiif_image_base,image_url=excluded.image_url,
      image_width=excluded.image_width,image_height=excluded.image_height,updated_at=now()
    """
    done, failed, pages = 0, 0, 0
    with db() as con, con.cursor() as cur:
        cur.execute(sel)
        rows = cur.fetchall()
        if limit:
            rows = rows[:limit]
        for key, url in rows:
            try:
                r = requests.get(url, timeout=30, headers={"User-Agent": "VoynichCorridorResearch/0.1"})
                r.raise_for_status()
                parsed = parse_iiif(r.json())
                for p in parsed:
                    blind = "C" + sha256_text(f"{key}|{p['seq']}|{SEED}")[:15]
                    cur.execute(ins, (key,p["folio"],p["seq"],p["canvas_id"],p["iiif_image_base"],
                                      p["image_url"],p["width"],p["height"],blind))
                cur.execute("update public.corridor_candidates set image_coverage=%s,updated_at=now() where candidate_key=%s",
                            ("complete" if parsed else "none", key))
                done += 1; pages += len(parsed)
            except Exception as e:
                failed += 1
                cur.execute("update public.corridor_candidates set metadata=metadata||%s::jsonb,updated_at=now() where candidate_key=%s",
                            (json.dumps({"iiif_error": str(e)[:500]}), key))
            con.commit()
    print(json.dumps({"manifests_done": done, "failed": failed, "pages": pages}, indent=2))


def coverage() -> None:
    q = """
    select geography_class,time_bin,count(*) n,
      count(*) filter(where inclusion_status='included') included,
      count(*) filter(where iiif_manifest_url is not null) iiif,
      count(*) filter(where image_coverage in ('partial','complete')) imaged,
      count(*) filter(where illustration_status in ('catalogue_attested','image_verified')) illustrated
    from public.corridor_candidates
    group by geography_class,time_bin order by geography_class,time_bin
    """
    with db() as con, con.cursor() as cur:
        cur.execute(q)
        cols=[d.name for d in cur.description]
        print(json.dumps([dict(zip(cols,r)) for r in cur.fetchall()], indent=2, default=str))


def freeze_run(label: str) -> None:
    cfg_sha = hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest()
    prot_sha = hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest()
    with db() as con, con.cursor() as cur:
        cur.execute("""
        insert into public.corridor_runs(run_label,git_branch,config_sha256,protocol_sha256,status,stage,started_at)
        values(%s,%s,%s,%s,'built','stage0',now()) on conflict(run_label) do nothing returning run_id
        """, (label, "experiment/alpine-venetian-corridor-v0.1-20260808", cfg_sha, prot_sha))
        row=cur.fetchone(); con.commit()
    print(json.dumps({"run_label": label, "run_id": str(row[0]) if row else "already_exists",
                      "config_sha256": cfg_sha, "protocol_sha256": prot_sha}, indent=2))


def get_run_id(cur, label: str):
    cur.execute("select run_id from public.corridor_runs where run_label=%s", (label,))
    row=cur.fetchone()
    if not row: raise SystemExit(f"Unknown run label: {label}")
    return row[0]


def match_distance(c: dict[str, Any], d: dict[str, Any]) -> float:
    if c["time_bin"] != d["time_bin"]:
        return 1e6
    ct, dt = set(c["content_tags"] or []), set(d["content_tags"] or [])
    j = 1.0 - (len(ct & dt) / len(ct | dt) if (ct | dt) else 1.0)
    substrate = 0.0 if (not c["substrate"] or not d["substrate"] or c["substrate"] == d["substrate"]) else 0.5
    cov_rank = {"none":0,"unknown":1,"partial":2,"complete":3}
    cov = 0.25 * abs(cov_rank.get(c["image_coverage"],1)-cov_rank.get(d["image_coverage"],1))
    same_holder = 0.25 if c["holding_institution"] and c["holding_institution"] == d["holding_institution"] else 0.0
    return j + substrate + cov + same_holder


def match_controls(label: str) -> None:
    with db() as con, con.cursor() as cur:
        run_id=get_run_id(cur,label)
        cur.execute("""select candidate_key,geography_class,time_bin,content_tags,substrate,image_coverage,holding_institution
                       from public.corridor_candidates where inclusion_status='included'""")
        cols=[d.name for d in cur.description]
        rows=[dict(zip(cols,r)) for r in cur.fetchall()]
        corridor=[r for r in rows if r["geography_class"]=="corridor_core" and r["time_bin"] in ("A_primary","B_antecedent")]
        controls=[r for r in rows if r["geography_class"] in CONTROL_GEOS]
        cur.execute("delete from public.corridor_control_matches where run_id=%s",(run_id,))
        n=0
        for c in corridor:
            ranked=[]
            for d in controls:
                dist=match_distance(c,d)
                if dist>=1e6: continue
                tie=sha256_text(d["candidate_key"]+str(SEED))
                ranked.append((dist,tie,d))
            ranked.sort(key=lambda x:(x[0],x[1]))
            for rank,(dist,_,d) in enumerate(ranked[:CONFIG["matching"]["target_controls_per_corridor"]],1):
                cur.execute("""insert into public.corridor_control_matches(run_id,corridor_candidate_key,control_candidate_key,match_rank,match_distance,match_factors)
                               values(%s,%s,%s,%s,%s,%s::jsonb)""",
                            (run_id,c["candidate_key"],d["candidate_key"],rank,dist,json.dumps({"time_bin":c["time_bin"]})))
                n+=1
        con.commit()
    print(json.dumps({"matched_rows":n,"corridor_manuscripts":len(corridor)},indent=2))


def signflip_p(diffs: np.ndarray, n_perm: int, seed: int) -> float:
    if len(diffs)==0: return float("nan")
    obs=float(np.mean(diffs)); rng=np.random.default_rng(seed); exceed=0
    chunk=10000
    for start in range(0,n_perm,chunk):
        m=min(chunk,n_perm-start)
        signs=rng.choice(np.array([-1.0,1.0]),size=(m,len(diffs)))
        vals=(signs*diffs).mean(axis=1)
        exceed += int(np.count_nonzero(np.abs(vals)>=abs(obs)))
    return (exceed+1)/(n_perm+1)


def bh(pairs: list[tuple[str,float]]) -> dict[str,float]:
    valid=sorted([(k,p) for k,p in pairs if np.isfinite(p)],key=lambda x:x[1])
    m=len(valid); out={}; prev=1.0
    for i in range(m-1,-1,-1):
        k,p=valid[i]; q=min(prev,p*m/(i+1)); out[k]=q; prev=q
    return out


def analyse(label: str) -> None:
    n_perm=int(CONFIG["statistics"]["permutations"])
    with db() as con, con.cursor() as cur:
        run_id=get_run_id(cur,label)
        cur.execute("""select candidate_key,object_class,arm,calibrated_score from public.corridor_scores
                       where run_id=%s and calibrated_score is not null""",(run_id,))
        rows=cur.fetchall()
        if not rows: raise SystemExit("No calibrated corridor_scores for this run")
        bycand={}; byfamily={}
        for cand,cls,arm,score in rows:
            bycand.setdefault(cand,{}).setdefault(arm,[]).append(float(score))
            byfamily.setdefault(cls,{}).setdefault(cand,[]).append(float(score))
        composite={c:statistics.fmean(statistics.fmean(v) for v in arms.values()) for c,arms in bycand.items()}
        famscore={f:{c:statistics.fmean(v) for c,v in d.items()} for f,d in byfamily.items()}
        cur.execute("select corridor_candidate_key,control_candidate_key from public.corridor_control_matches where run_id=%s",(run_id,))
        match={}
        for c,d in cur.fetchall(): match.setdefault(c,[]).append(d)
        def diffs_for(scores):
            ds=[]; keys=[]
            for c,ctrls in match.items():
                vals=[scores[x] for x in ctrls if x in scores]
                if c in scores and vals:
                    ds.append(scores[c]-statistics.fmean(vals)); keys.append(c)
            return np.asarray(ds,dtype=float),keys
        diffs,keys=diffs_for(composite)
        p=signflip_p(diffs,n_perm,SEED)
        estimate=float(np.mean(diffs)) if len(diffs) else float("nan")
        fam_results=[]
        for fam,scores in sorted(famscore.items()):
            d,_=diffs_for(scores); fp=signflip_p(d,n_perm,SEED+int(sha256_text(fam)[:6],16)%100000)
            fam_results.append((fam,float(np.mean(d)) if len(d) else float("nan"),fp,len(d)))
        qmap=bh([(f,pv) for f,_,pv,_ in fam_results])
        contrib=np.abs(diffs); max_frac=float(contrib.max()/contrib.sum()) if contrib.sum()>0 else 0.0
        loo=[]
        if len(diffs)>1:
            for i in range(len(diffs)): loo.append(float(np.mean(np.delete(diffs,i))))
        loo_positive=float(np.mean(np.array(loo)>0)) if loo else float("nan")
        cur.execute("delete from public.corridor_results where run_id=%s",(run_id,))
        cur.execute("""insert into public.corridor_results(run_id,result_key,analysis_family,estimate,p_value,n_corridor,n_control,verdict,detail)
                       values(%s,'primary_composite','composite',%s,%s,%s,%s,%s,%s::jsonb)""",
                    (run_id,estimate,p,len(diffs),sum(len(v) for v in match.values()),
                     "positive" if estimate>0 and p<CONFIG["statistics"]["alpha"] else "not_established",
                     json.dumps({"max_single_manuscript_effect_fraction":max_frac,"loo_positive_fraction":loo_positive})))
        for fam,est,fp,n in fam_results:
            cur.execute("""insert into public.corridor_results(run_id,result_key,analysis_family,estimate,p_value,q_value,n_corridor,verdict)
                           values(%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (run_id,"family:"+fam,fam,est,fp,qmap.get(fam),n,
                         "positive" if est>0 else "nonpositive"))
        con.commit()
    print(json.dumps({"primary_estimate":estimate,"primary_p":p,"n_matched_sets":len(diffs),
                      "max_single_manuscript_effect_fraction":max_frac,"loo_positive_fraction":loo_positive,
                      "families":[{"family":f,"estimate":e,"p":p0,"q":qmap.get(f),"n":n} for f,e,p0,n in fam_results]},indent=2))


def main() -> None:
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)
    sub.add_parser("audit")
    sub.add_parser("seed-existing")
    r=sub.add_parser("resolve-iiif"); r.add_argument("--limit",type=int)
    sub.add_parser("coverage")
    f=sub.add_parser("freeze-run"); f.add_argument("--label",required=True)
    m=sub.add_parser("match-controls"); m.add_argument("--run-label",required=True)
    a=sub.add_parser("analyse"); a.add_argument("--run-label",required=True)
    args=ap.parse_args()
    if args.cmd=="audit": audit()
    elif args.cmd=="seed-existing": seed_existing()
    elif args.cmd=="resolve-iiif": resolve_iiif(args.limit)
    elif args.cmd=="coverage": coverage()
    elif args.cmd=="freeze-run": freeze_run(args.label)
    elif args.cmd=="match-controls": match_controls(args.run_label)
    elif args.cmd=="analyse": analyse(args.run_label)

if __name__ == "__main__":
    main()

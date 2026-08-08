#!/usr/bin/env python3
"""Mandatory internal Voynich-archive prior-art scanner.

Run before external interpretation and before any novelty claim.

Environment:
    SUPABASE_DB_URL=postgresql://...

Examples:
    python archive_prior_art.py global
    python archive_prior_art.py candidates
    python archive_prior_art.py candidate --candidate-key registry:cr_british_library_egerton_ms_2020_erbario_carrarese
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable

import psycopg


GLOBAL_TOPICS = {
    "broad_corridor_hypothesis": [
        "Bolzano", "Bozen", "Brixen", "Bressanone", "Trento", "Trient", "Trentino",
        "Rovereto", "Verona", "Padua", "Padova", "Venice", "Venezia", "Tyrol", "Tirol",
        "South Tyrol", "Brenner", "Adige", "Etsch", "Eischtal", "Alps", "Alpine"
    ],
    "route_transmission": [
        "Brenner Pass", "route between Italy and Northern Europe", "Val d'Adige", "Eischtal",
        "crossed the Alps", "Alpine passes", "Padua Venice", "Verona Padua", "Trento Padua"
    ],
    "german_italian_interface": [
        "German Italian", "Italian German", "German scribe", "North Italian origin", "Tyrol",
        "Cimbrian", "Ladin", "Mòcheno", "South Tyrol", "German-speaking"
    ],
    "corridor_visual_culture": [
        "Runkelstein", "Buonconsiglio", "Aquila tower", "swallowtail merlons", "Ghibelline",
        "baggy sleeves", "steep roof", "Alpine castle", "Trento fresco"
    ],
    "corridor_scientific_manuscripts": [
        "Carrara Herbal", "Egerton 2020", "Roccabonella", "Rinio", "Liber de Simplicibus",
        "Trento Herbal", "1591", "Tacuinum", "2644", "Fontana", "Hartlieb", "Ulrich Putsch"
    ],
}


def db():
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise SystemExit("SUPABASE_DB_URL is required")
    return psycopg.connect(url)


def aliases_from_candidate(title: str, shelfmark: str | None, place: str | None) -> list[str]:
    vals = [title, shelfmark or "", place or ""]
    out: list[str] = []
    for val in vals:
        val = re.sub(r"\s+", " ", val).strip()
        if len(val) >= 4:
            out.append(val)
    # Extract useful shelfmark-like fragments and parenthetical names.
    for token in re.findall(r"(?:MS\.?\s*)?[A-Z]?[A-Za-z]{0,8}\.?\s*[A-Z]?\.?\s*\d+[A-Za-z./=-]*", title):
        token = token.strip(" ,()")
        if len(token) >= 4:
            out.append(token)
    for token in re.findall(r"\(([^)]{4,80})\)", title):
        out.append(token.strip())
    # Historic/modern place aliases relevant to the frozen corridor.
    alias_map = {
        "bolzano": ["Bolzano", "Bozen"], "bozen": ["Bolzano", "Bozen"],
        "bressanone": ["Bressanone", "Brixen"], "brixen": ["Bressanone", "Brixen"],
        "trento": ["Trento", "Trient"], "trient": ["Trento", "Trient"],
        "padua": ["Padua", "Padova"], "padova": ["Padua", "Padova"],
        "venice": ["Venice", "Venezia"], "venezia": ["Venice", "Venezia"],
        "tyrol": ["Tyrol", "Tirol"], "tirol": ["Tyrol", "Tirol"],
    }
    low = " ".join(vals).lower()
    for key, aa in alias_map.items():
        if key in low:
            out.extend(aa)
    # Stable unique order.
    seen = set(); ans = []
    for x in out:
        k = x.casefold()
        if k not in seen:
            seen.add(k); ans.append(x)
    return ans[:20]


def search_terms(cur, topic_key: str, terms: Iterable[str], candidate_key: str | None = None) -> int:
    terms = [t for t in terms if len(t.strip()) >= 3]
    if not terms:
        return 0
    clauses = " or ".join(["sp.text ilike %s"] * len(terms))
    params = [f"%{t}%" for t in terms]
    cur.execute(f"""
      select sp.source_id,sp.paragraph_index,sp.passage_year,sp.passage_author,sp.passage_url,
             s.title,s.source_type,sp.text
      from public.source_passages sp
      join public.sources s on s.source_id=sp.source_id
      where ({clauses})
      order by sp.passage_year nulls last,sp.source_id,sp.paragraph_index
      limit 500
    """, params)
    rows = cur.fetchall()
    qtext = " | ".join(terms)
    for r in rows:
        cur.execute("""
          insert into public.corridor_archive_prior_art(
            topic_key,candidate_key,query_text,source_id,paragraph_index,passage_year,passage_author,
            passage_url,source_title,source_type,hit_text,relevance,reviewed)
          values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'unreviewed',false)
          on conflict(topic_key,source_id,paragraph_index) do update set
            candidate_key=coalesce(public.corridor_archive_prior_art.candidate_key,excluded.candidate_key)
        """, (topic_key,candidate_key,qtext,*r))
    return len(rows)


def scan_global() -> None:
    with db() as con, con.cursor() as cur:
        counts = {}
        for topic, terms in GLOBAL_TOPICS.items():
            counts[topic] = search_terms(cur, topic, terms)
        con.commit()
    print(json.dumps(counts, indent=2))


def scan_candidate(candidate_key: str) -> dict:
    with db() as con, con.cursor() as cur:
        cur.execute("""select candidate_key,title,shelfmark,production_place_verbatim
                       from public.corridor_candidates where candidate_key=%s""", (candidate_key,))
        row = cur.fetchone()
        if not row:
            raise SystemExit(f"Unknown candidate: {candidate_key}")
        aliases = aliases_from_candidate(row[1], row[2], row[3])
        topic = "candidate:" + candidate_key
        n = search_terms(cur, topic, aliases, candidate_key)
        status = "none_found" if n == 0 else "discussed" if n < 10 else "heavily_discussed"
        cur.execute("""update public.corridor_candidates
                       set archive_hit_count=%s,archive_prior_art_status=%s,updated_at=now()
                       where candidate_key=%s""", (n,status,candidate_key))
        con.commit()
    return {"candidate_key": candidate_key, "aliases": aliases, "archive_hits": n, "status": status}


def scan_candidates() -> None:
    with db() as con, con.cursor() as cur:
        cur.execute("select candidate_key from public.corridor_candidates order by candidate_key")
        keys = [r[0] for r in cur.fetchall()]
    out = []
    for key in keys:
        out.append(scan_candidate(key))
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("global")
    sub.add_parser("candidates")
    c = sub.add_parser("candidate")
    c.add_argument("--candidate-key", required=True)
    args = ap.parse_args()
    if args.cmd == "global":
        scan_global()
    elif args.cmd == "candidates":
        scan_candidates()
    elif args.cmd == "candidate":
        print(json.dumps(scan_candidate(args.candidate_key), indent=2))


if __name__ == "__main__":
    main()

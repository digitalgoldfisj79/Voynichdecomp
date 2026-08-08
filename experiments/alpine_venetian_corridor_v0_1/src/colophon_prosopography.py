#!/usr/bin/env python3
"""Signed-colophon / prosopography arm for Alpine–Venetian Corridor v0.1.

The raw-record layer is intentionally conservative. A named person is NOT
assumed to be a scribe unless the colophon or an authoritative catalogue
supports that role.

Environment:
    SUPABASE_DB_URL=postgresql://...

Examples:
    python colophon_prosopography.py audit
    python colophon_prosopography.py ingest-jsonl data/bouveret_t2.jsonl
    python colophon_prosopography.py show-entry --source bouveret --volume 2 --entry 5912
    python colophon_prosopography.py unresolved-leads

JSONL record example:
    {
      "source_short": "Bouveret, Colophons de manuscrits occidentaux",
      "source_volume": 2,
      "entry_no": 5912,
      "page_no": 0,
      "person_name_raw": "...",
      "person_role": "scribe|illuminator|binder|recipient|owner|unknown_person_role",
      "colophon_text": "...",
      "manuscript_ref_raw": "...",
      "date_text": "...", "date_start": 1420, "date_end": 1420,
      "place_raw": "...", "place_norm": "...",
      "geography_class": "corridor_core",
      "source_url": "...",
      "extraction_method": "manual_transcription",
      "verification_status": "transcribed"
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import psycopg

DEFAULT_SOURCE = "Bouveret, Colophons de manuscrits occidentaux"
VALID_ROLES = {
    "scribe", "copyist", "illuminator", "rubricator", "binder", "recipient",
    "commissioner", "owner", "corrector", "book_worker", "unknown_person_role"
}
VALID_GEOS = {
    "corridor_core", "corridor_buffer", "control_lombardy", "control_tuscany",
    "control_bavaria_swabia", "control_east_alpine", "unresolved", None
}


def db() -> psycopg.Connection:
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise SystemExit("SUPABASE_DB_URL is required")
    return psycopg.connect(url)


def person_key(name: str) -> str:
    norm = " ".join(name.lower().split())
    return "person:" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:20]


def validate_record(r: dict[str, Any]) -> None:
    for k in ("source_short", "entry_no"):
        if not r.get(k):
            raise ValueError(f"missing required field {k}")
    role = r.get("person_role")
    if role and role not in VALID_ROLES:
        raise ValueError(f"invalid person_role={role!r}")
    geo = r.get("geography_class")
    if geo not in VALID_GEOS:
        raise ValueError(f"invalid geography_class={geo!r}")
    if role == "scribe" and not r.get("person_name_raw"):
        raise ValueError("scribe role requires person_name_raw")
    if r.get("verification_status") == "verified" and not r.get("colophon_text"):
        raise ValueError("verified record requires colophon_text")


def audit() -> None:
    q = """
    select
      count(*) as raw_records,
      count(*) filter(where verification_status in ('transcribed','verified')) as transcribed_or_verified,
      count(*) filter(where person_name_raw is not null) as named_people_records,
      count(*) filter(where person_role in ('scribe','copyist')) as scribal_records,
      count(*) filter(where geography_class='corridor_core') as corridor_core,
      count(*) filter(where candidate_key is not null) as linked_candidates
    from public.corridor_colophon_records
    """
    with db() as con, con.cursor() as cur:
        cur.execute(q)
        row = cur.fetchone()
        cols = [d.name for d in cur.description]
        print(json.dumps(dict(zip(cols, row)), indent=2, default=str))
        cur.execute("select count(*) from public.corridor_people")
        people = cur.fetchone()[0]
        cur.execute("select count(*) from public.corridor_person_manuscript_links")
        links = cur.fetchone()[0]
        print(json.dumps({"normalized_people": people, "person_manuscript_links": links}, indent=2))


def ingest_jsonl(path: Path) -> None:
    sql = """
    insert into public.corridor_colophon_records(
      source_short,source_volume,entry_no,page_no,person_name_raw,person_role,colophon_text,
      manuscript_ref_raw,candidate_key,date_text,date_start,date_end,place_raw,place_norm,
      geography_class,source_url,extraction_method,verification_status,notes,updated_at
    ) values (
      %(source_short)s,%(source_volume)s,%(entry_no)s,%(page_no)s,%(person_name_raw)s,%(person_role)s,%(colophon_text)s,
      %(manuscript_ref_raw)s,%(candidate_key)s,%(date_text)s,%(date_start)s,%(date_end)s,%(place_raw)s,%(place_norm)s,
      %(geography_class)s,%(source_url)s,%(extraction_method)s,%(verification_status)s,%(notes)s,now()
    ) on conflict(source_short,source_volume,entry_no) do update set
      page_no=excluded.page_no,
      person_name_raw=excluded.person_name_raw,
      person_role=excluded.person_role,
      colophon_text=excluded.colophon_text,
      manuscript_ref_raw=excluded.manuscript_ref_raw,
      candidate_key=coalesce(excluded.candidate_key,public.corridor_colophon_records.candidate_key),
      date_text=excluded.date_text,date_start=excluded.date_start,date_end=excluded.date_end,
      place_raw=excluded.place_raw,place_norm=excluded.place_norm,
      geography_class=excluded.geography_class,source_url=excluded.source_url,
      extraction_method=excluded.extraction_method,verification_status=excluded.verification_status,
      notes=excluded.notes,updated_at=now()
    returning id
    """
    n = 0
    with db() as con, con.cursor() as cur, path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r.setdefault("source_short", DEFAULT_SOURCE)
            for k in ["source_volume","page_no","person_name_raw","person_role","colophon_text",
                      "manuscript_ref_raw","candidate_key","date_text","date_start","date_end",
                      "place_raw","place_norm","geography_class","source_url","extraction_method","notes"]:
                r.setdefault(k, None)
            r.setdefault("verification_status", "unverified")
            try:
                validate_record(r)
            except Exception as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
            cur.execute(sql, r)
            colophon_id = cur.fetchone()[0]
            # Normalise a person only once a real name has been transcribed.
            if r.get("person_name_raw"):
                pk = person_key(r["person_name_raw"])
                role = r.get("person_role") or "unknown_person_role"
                cur.execute("""
                  insert into public.corridor_people(person_key,canonical_name,name_variants,role_tags,verification_status)
                  values(%s,%s,%s,%s,%s)
                  on conflict(person_key) do update set
                    name_variants=(select array(select distinct unnest(public.corridor_people.name_variants || excluded.name_variants))),
                    role_tags=(select array(select distinct unnest(public.corridor_people.role_tags || excluded.role_tags))),
                    updated_at=now()
                """, (pk, r["person_name_raw"], [r["person_name_raw"]], [role], r["verification_status"]))
                relationship = {
                    "scribe":"copied", "copyist":"copied", "illuminator":"illuminated",
                    "rubricator":"rubricated", "binder":"bound", "recipient":"received",
                    "commissioner":"commissioned", "owner":"owned", "corrector":"corrected",
                    "book_worker":"worked_on", "unknown_person_role":"uncertain"
                }.get(role, "uncertain")
                cur.execute("""
                  insert into public.corridor_person_manuscript_links(
                    person_key,colophon_record_id,candidate_key,manuscript_ref_raw,relationship,
                    evidence_text,evidence_source,evidence_url,verification_status
                  ) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                  on conflict(person_key,colophon_record_id,candidate_key,relationship) do nothing
                """, (pk,colophon_id,r.get("candidate_key"),r.get("manuscript_ref_raw"),relationship,
                      r.get("colophon_text"),f"{r['source_short']} v.{r.get('source_volume')} no.{r['entry_no']}",
                      r.get("source_url"),r["verification_status"]))
            n += 1
        con.commit()
    print(json.dumps({"ingested": n, "path": str(path)}, indent=2))


def show_entry(source: str, volume: int, entry: int) -> None:
    source_name = DEFAULT_SOURCE if source.lower() == "bouveret" else source
    with db() as con, con.cursor() as cur:
        cur.execute("""
          select * from public.corridor_colophon_records
          where source_short=%s and source_volume=%s and entry_no=%s
        """, (source_name, volume, entry))
        row = cur.fetchone()
        if not row:
            print(json.dumps({"found": False}))
            return
        cols = [d.name for d in cur.description]
        print(json.dumps(dict(zip(cols,row)), indent=2, default=str))


def unresolved_leads() -> None:
    with db() as con, con.cursor() as cur:
        cur.execute("""
          select source_short,source_volume,entry_no,person_name_raw,person_role,
                 manuscript_ref_raw,date_text,place_raw,geography_class,verification_status
          from public.corridor_colophon_records
          where candidate_key is null
          order by
            case geography_class when 'corridor_core' then 0 when 'corridor_buffer' then 1 else 2 end,
            source_volume,entry_no
        """)
        cols = [d.name for d in cur.description]
        print(json.dumps([dict(zip(cols,r)) for r in cur.fetchall()], indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("audit")
    p = sp.add_parser("ingest-jsonl"); p.add_argument("path", type=Path)
    p = sp.add_parser("show-entry"); p.add_argument("--source", default="bouveret"); p.add_argument("--volume", type=int, required=True); p.add_argument("--entry", type=int, required=True)
    sp.add_parser("unresolved-leads")
    a = ap.parse_args()
    if a.cmd == "audit": audit()
    elif a.cmd == "ingest-jsonl": ingest_jsonl(a.path)
    elif a.cmd == "show-entry": show_entry(a.source,a.volume,a.entry)
    elif a.cmd == "unresolved-leads": unresolved_leads()

if __name__ == "__main__":
    main()

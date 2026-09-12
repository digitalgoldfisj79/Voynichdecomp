#!/usr/bin/env python3
"""Historical Yiddish-German L recovery competition v0.1b.

This is a constrained repair wrapper around the frozen v0.1 runner. It changes only
ReF diplomatic extraction: all tok_dipl fragments are concatenated per CorA virtual
<token> before the literal a-z filter. All seven ReF XML hashes and resulting word
counts are hard-gated to the previously frozen Candidate-2 reference values.

No v0.1 solver output is read or reused. Voynich remains sealed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_RUNNER = HERE.parent / "yiddish_l_recovery_v01" / "run_l_recovery_v01.py"

spec = importlib.util.spec_from_file_location("lrec_v01_parent", PARENT_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load frozen parent runner: {PARENT_RUNNER}")
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)

REPAIR_VERSION = "yiddish_german_l_recovery_v01b_20260912"
SEED_NAMESPACE = "yiddish_german_l_recovery_v01_20260912"
PARENT_QUARANTINED_RUN_ID = 34714545374
PARENT_PROTOCOL_BLOB = "a7f350c6ef55f1c617dbf9cdfca173316f21b5bf"
CANDIDATE2_GATE_BLOB = "6344af22102d074208861f312b29607603d88124"

REF_EXPECTED = {
    "F014": {"sha256": "059211745db8c12b96d9ac4538cd94201c758f47cbc756b5fca70eeb59872f76", "words": 19097},
    "F015": {"sha256": "29b07d566cc8f3ceedbac16a5a0f98a9a4d8059b748c53d336d97aaffcd250f9", "words": 18162},
    "F016": {"sha256": "b898a98189fe8b3d20867dd347f71639643ee609513b76d5d85f19048bf80284", "words": 8124},
    "F018": {"sha256": "c694913a336563a5526b5447279359ac822da86e9ecf9ea50d1b30857de8368b", "words": 6270},
    "F034": {"sha256": "9d4e2da5fe4ac5f0438fd7599234749f4b3fe02333f60f19fed1e27215829135", "words": 5663},
    "F037": {"sha256": "8f7e855a96c3a6fcf03c4d91102197ddbd9b93fc79ab94393854f35385825f7d", "words": 15327},
    "F148": {"sha256": "4ca05d5794994980356dd3f9c932f8403a42d964745950dd1966d121d36c2802", "words": 20282},
}


def inherited_pseed(*parts) -> int:
    """Preserve the v0.1 public deterministic seed namespace exactly."""
    payload = "|".join([SEED_NAMESPACE, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def repaired_ref_words(root: Path, wid: str):
    """Return hash-pinned ReF words at the virtual-token unit.

    A valid word is formed by concatenating direct-child tok_dipl/@utf fragments
    inside one CorA <token>, then applying the parent's literal ASCII a-z filter.
    """
    if wid not in REF_EXPECTED:
        raise RuntimeError(f"unregistered ReF ID: {wid}")
    exp = REF_EXPECTED[wid]
    candidates = sorted(p for p in Path(root).rglob("*.xml") if p.name.lower() == f"{wid.lower()}.xml")
    exact = [p for p in candidates if core.sha_file(p) == exp["sha256"]]
    if len(exact) != 1:
        seen = [{"path": str(p), "sha256": core.sha_file(p)} for p in candidates]
        raise RuntimeError(f"{wid}: expected exactly one hash-pinned XML; matches={len(exact)} candidates={seen}")
    p = exact[0]
    xmlroot = ET.parse(p).getroot()
    out = []
    for tok in xmlroot.iter():
        if _localname(tok.tag) != "token":
            continue
        raw = "".join(
            (child.attrib.get("utf") or (child.text or ""))
            for child in list(tok)
            if _localname(child.tag) == "tok_dipl"
        )
        w = core.normalize_latin(raw)
        if w:
            out.append(w)
    if len(out) != exp["words"]:
        raise RuntimeError(f"{wid}: repaired extraction count {len(out)} != frozen Candidate-2 count {exp['words']}")
    return out, p


# Constrained monkeypatches. Everything else is inherited byte-for-byte from v0.1.
core.VERSION = REPAIR_VERSION
core.HERE = HERE
core.__file__ = str(Path(__file__).resolve())
core.pseed = inherited_pseed
core.ref_words = repaired_ref_words

_parent_preflight = core.preflight


def repaired_preflight(args):
    _parent_preflight(args)
    public = Path(args.out) / "public"
    manifest_path = public / "preflight_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    observed = manifest.get("source_hashes", {}).get("ref", {})
    for wid, exp in REF_EXPECTED.items():
        row = observed.get(wid)
        if row is None:
            raise RuntimeError(f"{wid}: missing from public preflight manifest")
        if row.get("sha256") != exp["sha256"] or row.get("words") != exp["words"]:
            raise RuntimeError(f"{wid}: public manifest fails frozen source-equivalence gate: {row}")

    manifest.update({
        "repair_version": REPAIR_VERSION,
        "parent_v01_quarantined_run_id": PARENT_QUARANTINED_RUN_ID,
        "parent_v01_solver_outputs_reused": False,
        "parent_v01_protocol_blob": PARENT_PROTOCOL_BLOB,
        "candidate2_gate_blob": CANDIDATE2_GATE_BLOB,
        "repair_scope": "REF_EXTRACTION_UNIT_ONLY__TOK_DIPL_FRAGMENTS_CONCATENATED_PER_VIRTUAL_TOKEN_BEFORE_LITERAL_AZ_FILTER",
        "public_seed_namespace_inherited": SEED_NAMESPACE,
        "ref_source_equivalence_all_seven": True,
        "ref_expected": REF_EXPECTED,
        "target_loaded": False,
        "voynich_access_allowed": False,
    })
    core.atomic_json(manifest, manifest_path)
    print(json.dumps({
        "status": "V01B_PREFLIGHT_SOURCE_EQUIVALENCE_OK",
        "all_seven_ref_hashes_and_counts_exact": True,
        "v01_solver_outputs_reused": False,
        "target_loaded": False,
    }, indent=2))


core.preflight = repaired_preflight

if __name__ == "__main__":
    core.main()

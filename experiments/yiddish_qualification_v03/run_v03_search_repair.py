#!/usr/bin/env python3
"""Yiddish qualification v03: bounded short-message search repair.

Scientific scope:
- Parent: v02 RECOVERY_UNQUALIFIED on fresh 512-word Lev Tov controls.
- Repair axis: SEARCH BUDGET ONLY. Objective/representation/thresholds unchanged.
- At most two candidates, chosen on already-consumed DEVELOPMENT works.
- Fresh 1501-1600 confirmation sources are not scored until the selected executable
  configuration is frozen to disk with hashes and a planting-root commitment.
- Voynich is never loaded. L and T are not run here.

This remains a secondary normalized Penn/YIVO-style Romanisation benchmark, not the
primary diplomatic Hebrew-script representation.
"""
from __future__ import annotations
import hashlib, hmac, json, math, os, pickle, random, secrets, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "yiddish_qualification_v1b"
sys.path.insert(0, str(PARENT))
import finite_panel_r_v02 as base  # noqa: E402
from encoder_v02 import encode_words  # noqa: E402
from independent_decoder_v02 import invert_permutation, decode_words, assert_roundtrip_non_erased  # noqa: E402

PROTOCOL_VERSION = "yiddish_qualification_v1_20260912"
RUN_VERSION = "yiddish_r_search_repair_v03_20260912"
OUT = HERE / "run_output_v03"
CHECKPOINT = OUT / "checkpoint.pkl"
DEV_ROWS = OUT / "development_rows.jsonl"
CONF_ROWS = OUT / "confirmation_rows.jsonl"
PRE_MANIFEST = OUT / "predevelopment_manifest.json"
FREEZE = OUT / "confirmation_freeze.json"
SUMMARY = OUT / "summary.json"
REVEAL = OUT / "plant_root_reveal.hex"

A = base.A
N_KEYS = 32
ERASURE_RATES = (0.0, 0.01)
BUFFER_WORDS = 32
ATOM_PASS = 0.90
WORD_PASS = 0.80
CELL_PASS = 29

# Already consumed in prior versions: safe for model/search development only.
DEVELOPMENT_WORKS = (
    "1507w-bovo.psd",
    "1648w-kine.psd",
    "1666w-messiah.psd",
    "1675e-ashkenaz-un-polak.psd",
)

# Fresh source IDs were frozen in v03_search_repair_spec.md before v03 outcomes.
CONFIRM_TRANSFER_SHORT = (
    "1588e-letters-cracow.psd",
    "1590e-sam-hayyim.psd",
)
CONFIRM_LATER_FULL = ("1834e-ukraine-2.psd",)

# Two and only two versioned candidates; same objective/move set as v02.
CANDIDATES = (
    {"id": "S1", "restarts": 8, "steps": 5000, "greedy_passes": 60},
    {"id": "S2", "restarts": 16, "steps": 6000, "greedy_passes": 80},
)

# Prior-consumed v02 BUILD pool, with v03 DEVELOPMENT works excluded below.
BUILD_POOL = (
    "1600e-magid-preface.psd","1600e-magid.psd","1600e-tsenerene.psd",
    "1619w-letters-prague.psd","1624e-magen.psd","1648w-kine.psd",
    "1666w-messiah.psd","1671e-vaad.psd","1675e-ashkenaz-un-polak.psd",
    "1677w-witzenhausen.psd","1692e-vilna.psd","1704e-ellush.psd",
    "1705w-glikl.psd","1712e-sarah.psd","1716e-duties.psd",
    "1717e-poznan.psd","1723w-simkhes.psd","1740w-drises.psd",
    "1743e-teshuat-preface.psd","1750w-moses.psd",
)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def atomic_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def append_jsonl(row, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())


def public_seed(*parts) -> int:
    payload = "|".join([PROTOCOL_VERSION, RUN_VERSION, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def secret_seed(root: bytes, *parts) -> int:
    return int.from_bytes(hmac.new(root, "|".join(map(str, parts)).encode(), hashlib.sha256).digest()[:8], "big")


def blind_solve_budget(fit_cipher, logp, train_uni, search_seed, cfg):
    """Same v02 bigram objective/moves; only budget parameters are candidates."""
    C, cuni = base.cipher_counts(fit_cipher)
    initial = base.frequency_initial(cuni, train_uni)
    rng = random.Random(search_seed)
    best_m = None; best_s = -1e100
    for r in range(cfg["restarts"]):
        m = initial.copy()
        rr = random.Random(public_seed("restart", cfg["id"], search_seed, r))
        for _ in range(8 * r):
            a, b = rr.sample(range(A), 2); m[a], m[b] = m[b], m[a]
        s = base.mapping_score(C, logp, m)
        if s > best_s: best_s = s; best_m = m.copy()
        for step in range(cfg["steps"]):
            a, b = rng.sample(range(A), 2)
            d = base.swap_delta(C, logp, m, a, b)
            frac = step / max(1, cfg["steps"] - 1)
            temp = .006 * (1 - frac) + .00003
            if d >= 0 or rng.random() < math.exp(max(-50.0, d / temp)):
                m[a], m[b] = m[b], m[a]; s += d
                if s > best_s: best_s = s; best_m = m.copy()
        m = best_m.copy(); s = best_s
        for _ in range(cfg["greedy_passes"]):
            bd = 0.0; bp = None
            for a in range(A):
                for b in range(a + 1, A):
                    d = base.swap_delta(C, logp, m, a, b)
                    if d > bd + 1e-12: bd = d; bp = (a, b)
            if bp is None: break
            a, b = bp; m[a], m[b] = m[b], m[a]; s += bd
            if s > best_s: best_s = s; best_m = m.copy()
    return best_m, best_s


def build_training(data: Path):
    files = [data / n for n in BUILD_POOL if n not in DEVELOPMENT_WORKS]
    words = []
    hashes = {}
    for p in files:
        ws = base.extract_words(p)
        words.extend(ws); hashes[p.name] = file_sha(p)
    if not words: raise RuntimeError("empty BUILD_SOLVER pool")
    return files, words, hashes


def make_trial(root, data, work, role, erasure, key_index, cfg, logp, train_uni, length=512):
    truth_all = base.extract_words(data / work)
    if len(truth_all) < 2 * length + BUFFER_WORDS:
        raise RuntimeError(f"{work} too short for {length}")
    fit_truth = truth_all[:length]
    audit_truth = truth_all[length + BUFFER_WORDS:length + BUFFER_WORDS + length]
    key = base.make_key(secret_seed(root, "plant", role, work, length, erasure, key_index))
    ef = random.Random(secret_seed(root, "erase_fit", role, work, length, erasure, key_index))
    ea = random.Random(secret_seed(root, "erase_audit", role, work, length, erasure, key_index))
    fit_cipher, _, _ = encode_words(fit_truth, key, erasure, ef)
    audit_cipher, _, _ = encode_words(audit_truth, key, erasure, ea)
    oracle = invert_permutation(key)
    assert_roundtrip_non_erased(fit_truth, decode_words(fit_cipher, oracle))
    assert_roundtrip_non_erased(audit_truth, decode_words(audit_cipher, oracle))
    search_seed = public_seed("search", cfg["id"], role, work, length, erasure, key_index)
    t0 = time.time()
    returned, ret_obj = blind_solve_budget(fit_cipher, logp, train_uni, search_seed, cfg)
    runtime = time.time() - t0
    C, _ = base.cipher_counts(fit_cipher)
    oracle_obj = base.mapping_score(C, logp, oracle)
    decoded = decode_words(audit_cipher, returned)
    rec = base.score_recovery(audit_truth, decoded)
    passed = rec["atom_recovery"] >= ATOM_PASS and rec["word_recovery"] >= WORD_PASS
    return {
        "candidate": cfg["id"], "role": role, "work": work, "length": length,
        "erasure_rate": erasure, "key_index": key_index, "pass": passed,
        "runtime_s": runtime, "search_seed": search_seed,
        "returned_objective": ret_obj, "oracle_objective": oracle_obj,
        "oracle_minus_returned": oracle_obj - ret_obj,
        "search_miss_witness": bool(oracle_obj > ret_obj + 1e-10), **rec,
    }


def summarize_rows(rows):
    cells = {}
    for r in rows:
        k = f"{r['work']}|{r['length']}|{r['erasure_rate']}"
        x = cells.setdefault(k, {"n": 0, "pass": 0, "search_miss": 0, "atom_sum": 0.0, "word_sum": 0.0})
        x["n"] += 1; x["pass"] += int(r["pass"]); x["search_miss"] += int(r["search_miss_witness"])
        x["atom_sum"] += r["atom_recovery"]; x["word_sum"] += r["word_recovery"]
    for x in cells.values():
        x["mean_atom_recovery"] = x.pop("atom_sum") / x["n"]
        x["mean_word_recovery"] = x.pop("word_sum") / x["n"]
        x["cell_pass"] = x["n"] == N_KEYS and x["pass"] >= CELL_PASS
    return cells


def candidate_pass(cells):
    expected = [f"{w}|512|{e}" for w in DEVELOPMENT_WORKS for e in ERASURE_RATES]
    return all(k in cells and cells[k]["cell_pass"] for k in expected)


def overlap_audit(data, build_names, dev_names, confirm_names):
    def ng(ws, n): return {tuple(ws[i:i+n]) for i in range(max(0, len(ws)-n+1))}
    refs = {n: base.extract_words(data / n) for n in confirm_names}
    ref8 = {n: ng(ws,8) for n,ws in refs.items()}; ref5 = {n: ng(ws,5) for n,ws in refs.items()}
    exact8=[]; near5=[]
    for n in list(build_names) + list(dev_names):
        ws=base.extract_words(data/n); a8=ng(ws,8); a5=ng(ws,5)
        for c in confirm_names:
            k8=len(a8 & ref8[c]); k5=len(a5 & ref5[c])
            if k8: exact8.append({"source":n,"confirmation":c,"shared_8gram_types":k8})
            if k5: near5.append({"source":n,"confirmation":c,"shared_5gram_types":k5})
    return exact8, near5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    repo, ppchy_commit = base.ensure_corpus(OUT / "sources")
    data = repo / "data"
    build_files, train_words, train_hashes = build_training(data)
    logp, train_uni = base.lm_from_words(train_words)

    cp = {}
    if CHECKPOINT.exists():
        with CHECKPOINT.open("rb") as f: cp = pickle.load(f)
    root = cp.get("plant_root") or secrets.token_bytes(32)
    cp.setdefault("plant_root", root); cp.setdefault("development", {}); cp.setdefault("confirmation", {})
    cp["state"] = "DEVELOPMENT_FROZEN"; atomic_pickle(cp, CHECKPOINT)

    source_hashes = {
        "run_v03_search_repair.py": file_sha(Path(__file__)),
        "parent_v02.py": file_sha(PARENT / "finite_panel_r_v02.py"),
        "encoder_v02.py": file_sha(PARENT / "encoder_v02.py"),
        "independent_decoder_v02.py": file_sha(PARENT / "independent_decoder_v02.py"),
        "spec": file_sha(PARENT / "v03_search_repair_spec.md"),
    }
    pre = {
        "protocol_version": PROTOCOL_VERSION, "run_version": RUN_VERSION,
        "parent_status": "v02_RECOVERY_UNQUALIFIED", "voynich_loaded": False,
        "target_access_allowed": False,
        "representation": "lossy Penn/YIVO-style Romanisation restricted to a-z; secondary normalized representation only",
        "repair_axis": "search budget only; v02 objective and recovery rule unchanged",
        "ppchy_commit": ppchy_commit, "source_code_hashes": source_hashes,
        "plant_root_commitment_sha256": sha_bytes(root),
        "development_works": DEVELOPMENT_WORKS,
        "fresh_confirmation_source_ids": {"transfer_short": CONFIRM_TRANSFER_SHORT, "later_full_robustness": CONFIRM_LATER_FULL},
        "candidates": CANDIDATES,
        "selection_rule": "select S1 iff all 8 development work/erasure cells pass >=29/32; else S2 iff all pass; else terminate without confirmation",
        "thresholds": {"atom": ATOM_PASS, "word": WORD_PASS, "cell_successes_required": CELL_PASS},
        "build_files": [p.name for p in build_files], "build_hashes": train_hashes,
        "build_word_count": len(train_words),
        "scope_limits": ["normalized representation only", "1501-1600 fresh confirmation supports 512 cells only", "later full work is robustness only", "no L", "no T"],
    }
    atomic_json(pre, PRE_MANIFEST)

    # Independent implementation fixture before any candidate trial.
    fixture=["abc","zebra","mish"]; fkey=list(range(A)); fkey=fkey[7:]+fkey[:7]
    enc,_,_=encode_words(fixture,fkey,0.0,random.Random(7)); dec=decode_words(enc,invert_permutation(fkey)); assert_roundtrip_non_erased(fixture,dec)

    selected = None; dev_by_candidate = {}
    for cfg in CANDIDATES:
        rows=[]
        for work in DEVELOPMENT_WORKS:
            for er in ERASURE_RATES:
                for k in range(N_KEYS):
                    tid=f"{cfg['id']}|{work}|512|{er}|{k}"
                    if tid in cp["development"]:
                        row=cp["development"][tid]
                    else:
                        row=make_trial(root,data,work,"development",er,k,cfg,logp,train_uni,512)
                        cp["development"][tid]=row; append_jsonl(row,DEV_ROWS); atomic_pickle(cp,CHECKPOINT)
                    rows.append(row)
        cells=summarize_rows(rows); dev_by_candidate[cfg["id"]] = cells
        if candidate_pass(cells): selected=cfg; break

    if selected is None:
        cp["state"]="RECOVERY_UNQUALIFIED_DEVELOPMENT"; atomic_pickle(cp,CHECKPOINT)
        atomic_json({
            "status":"RECOVERY_UNQUALIFIED_DEVELOPMENT","development":dev_by_candidate,
            "headline_bound":"exact 32-key conditional outcomes on named development works",
            "null_sd":"not applicable: exact recovery/search diagnostics, no effect-null gate",
            "voynich_inference":"NONE"
        },SUMMARY)
        print(SUMMARY.read_text()); return

    # Freeze selected executable/search configuration before confirmation plaintext is scored.
    confirm_names = list(CONFIRM_TRANSFER_SHORT + CONFIRM_LATER_FULL)
    exact8, near5 = overlap_audit(data,[p.name for p in build_files],DEVELOPMENT_WORKS,confirm_names)
    if exact8:
        cp["state"]="LEAKAGE_ABORT"; atomic_pickle(cp,CHECKPOINT)
        atomic_json({"status":"LEAKAGE_ABORT","exact_8gram_overlaps":exact8,"near_5gram":near5},SUMMARY)
        print(SUMMARY.read_text()); return
    freeze = {
        "state":"EXECUTION_FROZEN", "selected_candidate":selected,
        "source_code_hashes":source_hashes, "ppchy_commit":ppchy_commit,
        "plant_root_commitment_sha256":sha_bytes(root),
        "confirmation_source_ids":{"transfer_short":CONFIRM_TRANSFER_SHORT,"later_full_robustness":CONFIRM_LATER_FULL},
        "confirmation_hashes":{n:file_sha(data/n) for n in confirm_names},
        "leakage_audit":{"shared_8gram":exact8,"shared_5gram_diagnostics":near5},
        "note":"written after development candidate selection and before any confirmation solver trial",
    }
    atomic_json(freeze,FREEZE); cp["selected_candidate"]=selected; cp["state"]="EXECUTION_FROZEN"; atomic_pickle(cp,CHECKPOINT)

    conf_rows=[]
    roles=[("transfer_short",CONFIRM_TRANSFER_SHORT,(512,)),("later_full_robustness",CONFIRM_LATER_FULL,(512,2048))]
    for role,works,lengths in roles:
        for work in works:
            total_words=len(base.extract_words(data/work))
            for length in lengths:
                if total_words < 2*length+BUFFER_WORDS: continue
                for er in ERASURE_RATES:
                    for k in range(N_KEYS):
                        tid=f"{selected['id']}|{role}|{work}|{length}|{er}|{k}"
                        if tid in cp["confirmation"]:
                            row=cp["confirmation"][tid]
                        else:
                            row=make_trial(root,data,work,role,er,k,selected,logp,train_uni,length)
                            cp["confirmation"][tid]=row; append_jsonl(row,CONF_ROWS); atomic_pickle(cp,CHECKPOINT)
                        conf_rows.append(row)

    conf_cells=summarize_rows(conf_rows)
    transfer_expected=[f"{w}|512|{e}" for w in CONFIRM_TRANSFER_SHORT for e in ERASURE_RATES]
    transfer_short_pass=all(conf_cells.get(k,{}).get("cell_pass",False) for k in transfer_expected)
    later_expected=[f"{w}|{n}|{e}" for w in CONFIRM_LATER_FULL for n in (512,2048) for e in ERASURE_RATES]
    later_full_pass=all(conf_cells.get(k,{}).get("cell_pass",False) for k in later_expected)

    REVEAL.write_text(root.hex()+"\n",encoding="ascii")
    status = "FINITE_PANEL_TRANSFER_SHORT_R_PASS__LONG_HISTORICAL_CORPUS_LIMITED" if transfer_short_pass else "RECOVERY_UNQUALIFIED_FRESH_CONFIRMATION"
    summary={
        "status":status,"selected_candidate":selected,"development":dev_by_candidate,
        "confirmation_cells":conf_cells,"transfer_short_pass":transfer_short_pass,
        "later_full_robustness_pass":later_full_pass,
        "historical_long_cell_status":"CORPUS_LIMITED: no fresh 1501-1600 Penn work in this version supplies 4128 normalized words",
        "primary_representation_status":"UNQUALIFIED: this run is normalized Romanisation only",
        "L_status":"NOT_RUN","T_status":"NOT_RUN","voynich_inference":"NONE",
        "headline_bound":"exact 32-key outcomes conditional on named works; no population CI",
        "null_sd":"not applicable: R uses exact recovery thresholds rather than an effect/null-SD gate",
        "plant_root_commitment_sha256":sha_bytes(root),"plant_root_reveal_sha256":file_sha(REVEAL),
        "confirmation_freeze_sha256":file_sha(FREEZE),"predevelopment_manifest_sha256":file_sha(PRE_MANIFEST),
    }
    atomic_json(summary,SUMMARY); cp["state"]=status; atomic_pickle(cp,CHECKPOINT)
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__ == "__main__": main()

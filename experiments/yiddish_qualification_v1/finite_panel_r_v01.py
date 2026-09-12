#!/usr/bin/env python3
"""Yiddish qualification v1: NORMALIZED finite-panel M0 recovery benchmark.

This is deliberately NOT the primary diplomatic representation and NOT an L/T test.
It uses the Penn Parsed Corpus of Historical Yiddish's lossy Romanisation to test
whether a fixed blind solver can recover held-out historical-Yiddish messages under
uniform monoalphabetic substitution. Results may support R only for this declared
normalized transfer panel; they cannot admit Voynich target use.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

PROTOCOL_VERSION = "yiddish_qualification_v1_20260912"
RUN_VERSION = "finite_panel_r_normalized_v01"
PPCHY_URL = "https://github.com/beatrice57/penn-parsed-corpus-of-historical-yiddish.git"
ALPHABET = tuple("abcdefghijklmnopqrstuvwxyz")
A = len(ALPHABET)
BOUNDARY = A
BUFFER_WORDS = 32
LENGTHS = (512, 2048)
ERASURE_RATES = (0.0, 0.01)
N_KEYS = 32
ATOM_PASS = 0.90
WORD_PASS = 0.80
WORK_TRIAL_PASS = 29  # >=90% of 32
SEARCH_RESTARTS = 3
SEARCH_STEPS = 2500
GREEDY_PASSES = 40
ALPHA = 0.25

# Frozen source roles before outcomes. The 1507 item is development only.
DEVELOPMENT_WORKS = ("1507w-bovo.psd",)
CONFIRMATION_WORKS = ("1579e-shir.psd", "1589e-ester.psd")
# Training window is a declared transfer pool; fixed by filename date only.
TRAIN_MIN_YEAR = 1600
TRAIN_MAX_YEAR = 1750

HERE = Path(__file__).resolve().parent
OUT = HERE / "run_output"
CHECKPOINT = OUT / "checkpoint.pkl"
ROWS = OUT / "trial_rows.jsonl"
SUMMARY = OUT / "summary.json"
MANIFEST = OUT / "manifest.json"

LEAF_RE = re.compile(r"\(([A-Z][A-Z0-9$=*-]*)\s+([^()\s]+)\)")
YEAR_RE = re.compile(r"^(\d{4})")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def seed_for(*parts) -> int:
    payload = "|".join([PROTOCOL_VERSION, RUN_VERSION, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def normalize_leaf(raw: str) -> list[str]:
    if raw.startswith("*") or raw in {"0", "-NONE-"}:
        return []
    raw = raw.replace("@", "")
    raw = raw.split("^", 1)[0]
    out = []
    for part in raw.split("_"):
        w = "".join(ch for ch in part.lower() if "a" <= ch <= "z")
        if w:
            out.append(w)
    return out


def extract_words(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    words = []
    for tag, raw in LEAF_RE.findall(text):
        if tag.startswith("ID") or tag.startswith("CODE") or tag.startswith("PUNC"):
            continue
        words.extend(normalize_leaf(raw))
    return words


def year_from_name(name: str):
    m = YEAR_RE.match(name)
    return int(m.group(1)) if m else None


def ensure_corpus(root: Path) -> tuple[Path, str]:
    repo = root / "ppchy"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", PPCHY_URL, str(repo)], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return repo, commit


def build_training(data_dir: Path, excluded: set[str]):
    files = []
    words = []
    hashes = {}
    for p in sorted(data_dir.glob("*.psd")):
        y = year_from_name(p.name)
        if y is None or not (TRAIN_MIN_YEAR <= y <= TRAIN_MAX_YEAR):
            continue
        if p.name in excluded:
            continue
        ws = extract_words(p)
        if not ws:
            continue
        files.append(p.name); words.extend(ws); hashes[p.name] = file_sha(p)
    if not words:
        raise RuntimeError("No training words")
    return files, words, hashes


def lm_from_words(words: list[str]):
    # Dense conditional bigram log probabilities over 26 letters + fixed boundary.
    B = A + 1
    counts = [[ALPHA for _ in range(B)] for _ in range(B)]
    unis = [ALPHA for _ in range(A)]
    for w in words:
        ids = [ord(c) - 97 for c in w if c in ALPHABET]
        if not ids: continue
        for x in ids: unis[x] += 1.0
        seq = [BOUNDARY] + ids + [BOUNDARY]
        for x, y in zip(seq, seq[1:]): counts[x][y] += 1.0
    logp = []
    for row in counts:
        s = sum(row)
        logp.append([math.log(v/s) for v in row])
    return logp, unis


def cipher_counts(words: list[str]):
    B = A + 1
    C = [[0.0 for _ in range(B)] for _ in range(B)]
    uni = [0 for _ in range(A)]
    n_pairs = 0
    for w in words:
        ids = []
        for c in w:
            if c == "~": ids.append(None)
            elif "a" <= c <= "z":
                x = ord(c)-97; ids.append(x); uni[x] += 1
        seq = [BOUNDARY] + ids + [BOUNDARY]
        for x,y in zip(seq, seq[1:]):
            if x is None or y is None: continue
            C[x][y] += 1.0; n_pairs += 1
    if n_pairs == 0: raise RuntimeError("No eligible cipher bigrams")
    inv = 1.0/n_pairs
    for i in range(B):
        for j in range(B): C[i][j] *= inv
    return C, uni


def mapping_score(C, logp, m):
    B=A+1
    total=0.0
    for i in range(B):
        mi = BOUNDARY if i==BOUNDARY else m[i]
        for j in range(B):
            mj = BOUNDARY if j==BOUNDARY else m[j]
            total += C[i][j]*logp[mi][mj]
    return total


def swap_delta(C, logp, m, a, b):
    ma, mb = m[a], m[b]
    d=0.0
    for k in range(A+1):
        if k==a or k==b: continue
        mk = BOUNDARY if k==BOUNDARY else m[k]
        d += C[a][k]*(logp[mb][mk]-logp[ma][mk])
        d += C[b][k]*(logp[ma][mk]-logp[mb][mk])
        d += C[k][a]*(logp[mk][mb]-logp[mk][ma])
        d += C[k][b]*(logp[mk][ma]-logp[mk][mb])
    d += C[a][a]*(logp[mb][mb]-logp[ma][ma])
    d += C[b][b]*(logp[ma][ma]-logp[mb][mb])
    d += C[a][b]*(logp[mb][ma]-logp[ma][mb])
    d += C[b][a]*(logp[ma][mb]-logp[mb][ma])
    return d


def frequency_initial(cipher_uni, train_uni):
    cr = sorted(range(A), key=lambda x:(-cipher_uni[x], x))
    pr = sorted(range(A), key=lambda x:(-train_uni[x], x))
    m=[0]*A
    for c,p in zip(cr,pr): m[c]=p
    return m


def blind_solve(fit_cipher, logp, train_uni, search_seed):
    C, cuni = cipher_counts(fit_cipher)
    base = frequency_initial(cuni, train_uni)
    rng = random.Random(search_seed)
    best_m=None; best_s=-1e100
    for r in range(SEARCH_RESTARTS):
        m=base.copy()
        if r:
            rr=random.Random(seed_for("restart", search_seed, r))
            for _ in range(8*r):
                a,b=rr.sample(range(A),2); m[a],m[b]=m[b],m[a]
        s=mapping_score(C,logp,m)
        if s>best_s: best_s=s; best_m=m.copy()
        for step in range(SEARCH_STEPS):
            a,b=rng.sample(range(A),2)
            d=swap_delta(C,logp,m,a,b)
            frac=step/max(1,SEARCH_STEPS-1)
            temp=0.006*(1-frac)+0.00003
            if d>=0 or rng.random() < math.exp(max(-50.0,d/temp)):
                m[a],m[b]=m[b],m[a]; s += d
                if s>best_s: best_s=s; best_m=m.copy()
        # deterministic greedy polish
        m=best_m.copy(); s=best_s
        for _ in range(GREEDY_PASSES):
            bd=0.0; bp=None
            for a in range(A):
                for b in range(a+1,A):
                    d=swap_delta(C,logp,m,a,b)
                    if d>bd+1e-12: bd=d; bp=(a,b)
            if bp is None: break
            a,b=bp; m[a],m[b]=m[b],m[a]; s += bd
            if s>best_s: best_s=s; best_m=m.copy()
    return best_m,best_s


def make_key(seed):
    r=random.Random(seed); p=list(range(A)); r.shuffle(p)
    # plain -> cipher
    return p


def inverse_key(plain_to_cipher):
    inv=[0]*A
    for p,c in enumerate(plain_to_cipher): inv[c]=p
    return inv


def encode_words(words,key,erasure_rate,erase_seed):
    r=random.Random(erase_seed)
    out=[]; erased=0; total=0
    for w in words:
        z=[]
        for ch in w:
            p=ord(ch)-97
            total += 1
            if erasure_rate and r.random()<erasure_rate:
                z.append("~"); erased+=1
            else: z.append(chr(97+key[p]))
        out.append("".join(z))
    return out, erased, total


def decode_words(cipher_words, cipher_to_plain):
    out=[]
    for w in cipher_words:
        z=[]
        for ch in w:
            if ch=="~": z.append("~")
            else: z.append(chr(97+cipher_to_plain[ord(ch)-97]))
        out.append("".join(z))
    return out


def score_recovery(truth, decoded):
    correct=eligible=0; exact_words=eligible_words=0; total_atoms=0; erased_atoms=0
    for t,d in zip(truth,decoded):
        total_atoms += len(t)
        full=True
        for a,b in zip(t,d):
            if b=="~": erased_atoms+=1; full=False; continue
            eligible+=1
            if a==b: correct+=1
            else: full=False
        if "~" not in d:
            eligible_words += 1
            if d==t: exact_words += 1
    atom_acc=correct/eligible if eligible else None
    word_acc=exact_words/eligible_words if eligible_words else None
    coverage=eligible/total_atoms if total_atoms else None
    return dict(atom_recovery=atom_acc, word_recovery=word_acc, atom_coverage=coverage,
                eligible_atoms=eligible,total_atoms=total_atoms,eligible_words=eligible_words,
                total_words=len(truth),erased_atoms=erased_atoms)


def trial_id(role,work,n,erase,k): return f"{role}|{work}|{n}|{erase:.3f}|{k:02d}"


def load_done():
    if not CHECKPOINT.exists(): return {"done":{},"state":"IMPLEMENTED"}
    with CHECKPOINT.open("rb") as f: return pickle.load(f)


def append_row(row):
    ROWS.parent.mkdir(parents=True,exist_ok=True)
    with ROWS.open("a",encoding="utf-8") as f:
        f.write(json.dumps(row,sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    root=OUT/"sources"; root.mkdir(exist_ok=True)
    repo,commit=ensure_corpus(root)
    data=repo/"data"
    excluded=set(DEVELOPMENT_WORKS+CONFIRMATION_WORKS)
    # relationship safety: also exclude companion prefaces for held-out named works
    for n in list(excluded):
        stem=n[:-4]
        excluded.add(stem+"-preface.psd")
    train_files,train_words,train_hashes=build_training(data,excluded)
    logp,train_uni=lm_from_words(train_words)

    source_meta={}
    for role,names in (("development",DEVELOPMENT_WORKS),("confirmation",CONFIRMATION_WORKS)):
        for name in names:
            p=data/name
            ws=extract_words(p)
            source_meta[name]={"role":role,"words":len(ws),"sha256":file_sha(p)}
            need=max(LENGTHS)*2+BUFFER_WORDS
            if len(ws)<need:
                source_meta[name]["eligible_long_cell"]=False
            else: source_meta[name]["eligible_long_cell"]=True

    code_hash=file_sha(Path(__file__))
    manifest={
      "protocol_version":PROTOCOL_VERSION,"run_version":RUN_VERSION,
      "scope":"R_ONLY_NORMALIZED_TRANSFER_PANEL_NO_L_NO_T",
      "target_access_allowed":False,"voynich_loaded":False,
      "representation":"lossy Penn/YIVO-style Romanisation further restricted to a-z; secondary development representation only",
      "registered_alphabet":"abcdefghijklmnopqrstuvwxyz","word_boundaries_preserved":True,
      "ppchy_commit":commit,"code_sha256":code_hash,"training_files":train_files,
      "training_hashes":train_hashes,"training_word_count":len(train_words),"heldout":source_meta,
      "conditions":{"fit_words":list(LENGTHS),"audit_words":list(LENGTHS),"buffer_words":BUFFER_WORDS,
                    "erasure_rates":list(ERASURE_RATES),"keys_per_cell":N_KEYS},
      "recovery_thresholds":{"atom":ATOM_PASS,"word":WORD_PASS,"work_trials_required":WORK_TRIAL_PASS},
      "search":{"restarts":SEARCH_RESTARTS,"steps":SEARCH_STEPS,"greedy_passes":GREEDY_PASSES,
                "objective":"conditional character bigram cross-entropy under substitution"},
      "seed_derivation":"SHA256 domain-separated by protocol/run/role/work/condition/key/plant|erase|search",
      "known_limitations":["not diplomatic Hebrew-script atoms","no German/Hebrew language-discrimination arm","finite panel","training/heldout relationship audit limited to declared filename/source groups"]
    }
    atomic_json(manifest,MANIFEST)

    cp=load_done(); done=cp.setdefault("done",{})
    for role,names in (("development",DEVELOPMENT_WORKS),("confirmation",CONFIRMATION_WORKS)):
      for name in names:
        truth_all=extract_words(data/name)
        for n in LENGTHS:
          if len(truth_all)<2*n+BUFFER_WORDS: continue
          fit_truth=truth_all[:n]
          audit_truth=truth_all[n+BUFFER_WORDS:n+BUFFER_WORDS+n]
          for er in ERASURE_RATES:
            for k in range(N_KEYS):
              tid=trial_id(role,name,n,er,k)
              if tid in done: continue
              key_seed=seed_for("plant",role,name,n,er,k)
              erase_fit_seed=seed_for("erase_fit",role,name,n,er,k)
              erase_audit_seed=seed_for("erase_audit",role,name,n,er,k)
              search_seed=seed_for("search",role,name,n,er,k)
              key=make_key(key_seed)
              fit_cipher,_,_=encode_words(fit_truth,key,er,erase_fit_seed)
              audit_cipher,_,_=encode_words(audit_truth,key,er,erase_audit_seed)
              t0=time.time()
              returned,ret_obj=blind_solve(fit_cipher,logp,train_uni,search_seed)
              runtime=time.time()-t0
              # truth is revealed only after returned mapping commits
              oracle=inverse_key(key)
              C,_=cipher_counts(fit_cipher)
              oracle_obj=mapping_score(C,logp,oracle)
              decoded=decode_words(audit_cipher,returned)
              rec=score_recovery(audit_truth,decoded)
              passed=(rec["atom_recovery"] is not None and rec["word_recovery"] is not None and
                      rec["atom_recovery"]>=ATOM_PASS and rec["word_recovery"]>=WORD_PASS)
              row={"trial_id":tid,"role":role,"work":name,"fit_words":n,"audit_words":n,
                   "erasure_rate":er,"key_index":k,"pass":passed,"runtime_s":runtime,
                   "returned_objective":ret_obj,"oracle_objective":oracle_obj,
                   "oracle_minus_returned":oracle_obj-ret_obj,
                   "search_miss_witness":bool(oracle_obj>ret_obj+1e-10),**rec,
                   "seeds":{"plant":key_seed,"erase_fit":erase_fit_seed,"erase_audit":erase_audit_seed,"search":search_seed}}
              append_row(row); done[tid]=row
              cp.update({"state":"RUNNING_R_NORMALIZED","code_sha256":code_hash,"ppchy_commit":commit})
              atomic_pickle(cp,CHECKPOINT)

    rows=list(done.values())
    cells={}
    for r in rows:
      key=(r["role"],r["work"],r["fit_words"],r["erasure_rate"])
      x=cells.setdefault("|".join(map(str,key)),{"n":0,"pass":0,"atom":[],"word":[],"search_miss":0})
      x["n"]+=1; x["pass"]+=int(r["pass"]); x["atom"].append(r["atom_recovery"]); x["word"].append(r["word_recovery"]); x["search_miss"]+=int(r["search_miss_witness"])
    for x in cells.values():
      x["work_cell_pass"]=(x["n"]==N_KEYS and x["pass"]>=WORK_TRIAL_PASS)
      x["mean_atom_recovery"]=sum(x["atom"])/len(x["atom"]) if x["atom"] else None
      x["mean_word_recovery"]=sum(x["word"])/len(x["word"]) if x["word"] else None
      del x["atom"]; del x["word"]
    work_status={}
    for role,names in (("development",DEVELOPMENT_WORKS),("confirmation",CONFIRMATION_WORKS)):
      for name in names:
        required=[]
        for n in LENGTHS:
          if source_meta[name]["words"]>=2*n+BUFFER_WORDS:
            for er in ERASURE_RATES: required.append(cells.get("|".join(map(str,(role,name,n,er)))))
        work_status[name]={"role":role,"required_cells":len(required),"all_required_cells_pass":bool(required) and all(x and x["work_cell_pass"] for x in required)}
    summary={"status":"R_NORMALIZED_FINITE_PANEL_COMPLETE","scope":"NO_L_NO_T_NO_VOYNICH_INFERENCE",
             "manifest_sha256":file_sha(MANIFEST),"cells":cells,"works":work_status,
             "headline_bound":"FINITE_PANEL_ONLY; exact 32-key outcomes conditional on named works; no historical-population CI",
             "null_sd":"not_applicable_to_exact_recovery_rate; no effect-size/null comparison is used as the R gate"}
    atomic_json(summary,SUMMARY)
    cp["state"]="R_NORMALIZED_FINITE_PANEL_COMPLETE"; atomic_pickle(cp,CHECKPOINT)
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__": main()

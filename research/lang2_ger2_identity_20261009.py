#!/usr/bin/env python3
"""LANG2 full GER2-identity wrapper: only external target vocabulary changes.
No edits to frozen NeuroCipher solver/runtime, VMS folds, schedule or gates.
Runs LATIN_CI; archive-backed Padua MIXED is quarantined pending reproducible
transport of the exact frozen target vocabulary and label audit.
"""
import argparse
import collections
import gzip
import base64
import hashlib
import io
import json
import pickle
import re
import sys
import urllib.request
from pathlib import Path

REPO = "https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp"
SOLVER_COMMIT = "e36031a9a6b8b67fcebb4d6f4af1c3753fad4287"
RUNTIME_COMMIT = "98b7324fc65aa9b14242920f6931a8d1256064df"
CACHE_COMMIT = "a4a133f0635e0408266bc185a1697821ab79bd28"
CI_COMMIT = "67a73f80da2caefd6788de43c003701225833825"
CI_SHA = "377daa2a9c2403e6b8e10146a9c67f9e6b8144e959f232ec9096cdd0a4ae81f1"
VMS_FOLD_SHA = "e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888"
BASE_VMS_SHA = "26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
TARGET_K = 10000
TRAIN_K = 4103
SEEDS = [1234, 2026, 17, 73]
SCRAMBLE_SEEDS = [9001, 7001, 11003, 17011]
sha256 = lambda b: hashlib.sha256(b).hexdigest()

def download(rev, path):
    return urllib.request.urlopen(f"{REPO}/{rev}/{path}", timeout=300).read()

def bootstrap():
    basedir = Path("/tmp/lang2_ger2_frozen")
    basedir.mkdir(exist_ok=True)
    for file, rev in [
        ("neurodecipher_acl2019_refactor.py", SOLVER_COMMIT),
        ("vms_neurocipher_ger2_cached_runtime_20261006.py", RUNTIME_COMMIT)]:
        (basedir / file).write_bytes(download(rev, "research/" + file))
    payload = download(CACHE_COMMIT, "research/data/vms_neurocipher_ger2_cache_20261006.json.gz.b64")
    cache = json.loads(gzip.decompress(base64.b64decode(payload.strip())).decode())
    assert cache["schema"] == "VMS_NEUROCIPHER_GER2_CACHE_V2"
    assert cache["vms"]["audit"]["fold_sha"] == VMS_FOLD_SHA
    assert cache["solver_commit"] == SOLVER_COMMIT
    sys.path.insert(0, str(basedir))
    import vms_neurocipher_ger2_cached_runtime_20261006 as rt
    return cache, rt

def latin_ci():
    blob = download(CI_COMMIT, "Paper/Cipher_paper/ci_corpus_parsed.pkl")
    assert sha256(blob) == CI_SHA, "CI SOURCE SHA DRIFT: stop"
    src = pickle.loads(blob)
    freq = collections.Counter(str(w).lower() for w in src["all_words"]
                               if re.fullmatch(r"[a-z]+", str(w).lower()))
    words = [w for w, _ in freq.most_common()]
    assert len(words) >= TARGET_K
    return words[:TARGET_K], {"source":"CI Latin; single corpus, not independent-witness replication",
        "source_sha256":CI_SHA, "source_tokens":sum(freq.values()),
        "source_types":len(freq), "source_docs":1}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source", choices=["LATIN_CI"], default="LATIN_CI")
    ap.add_argument("--mode", choices=["real","scramble"], required=True)
    ap.add_argument("--seed", type=int, required=True, choices=SEEDS)
    ap.add_argument("--scramble-seed", type=int, default=9001, choices=SCRAMBLE_SEEDS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args=ap.parse_args()
    if not args.smoke and args.mode == "scramble":
        expected = SCRAMBLE_SEEDS[SEEDS.index(args.seed)]
        assert args.scramble_seed == expected, "Null seed mismatch: stop"
    cache,rt=bootstrap()
    words,meta=latin_ci()
    sha=sha256(json.dumps(words,ensure_ascii=False,separators=(",",":")).encode())
    assert sha=="e5b62f9d76bbbae21218b2d5c18249b0976c30aedaed42484e1d7e029490a364", "LANG2 target vocabulary drift: stop"
    cache["dialects"]={args.source:{"real":words,"docs":meta["source_docs"],
                                     "ref_types":meta["source_types"]}}
    path=Path("/tmp/lang2_ger2_frozen/lang2_cache.json")
    path.write_text(json.dumps(cache,ensure_ascii=False,separators=(",",":")))
    print("LANG2_INPUT_AUDIT="+json.dumps({"source":args.source,"mode":args.mode,
        "seed":args.seed,"scramble_seed":args.scramble_seed,
        "target_vocabulary_sha256":sha,"target_train":TRAIN_K,"target_full":TARGET_K,
        "frozen_fold_sha":VMS_FOLD_SHA,"frozen_solver_commit":SOLVER_COMMIT,
        "frozen_runtime_commit":RUNTIME_COMMIT,"frozen_cache_commit":CACHE_COMMIT,
        "source_details":meta,"scientific_change":"TARGET_VOCABULARY_ONLY"},
        sort_keys=True,separators=(",",":")),flush=True)
    class A: pass
    a=A()
    a.cache=str(path); a.dialect=args.source; a.mode=args.mode
    a.seed=args.seed; a.scramble_seed=args.scramble_seed
    a.cpu=args.cpu; a.smoke=args.smoke
    # EXACT full GER2 schedule; do not shorten to C-LANG.
    a.rounds=10; a.epochs=150; a.eval_every=10; a.log_every=10
    a.warm_up_steps=5; a.reg_hyper=.5; a.eval_batch=48
    assert rt.DISC_N==735 and rt.TRAIN_K==TRAIN_K and rt.FULL_K==TARGET_K
    assert rt.DISC_DEMAND==221
    rt.CachedRunner(a).run()

if __name__=="__main__":
    main()

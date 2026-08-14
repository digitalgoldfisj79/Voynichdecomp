from __future__ import annotations
import argparse, concurrent.futures, csv, hashlib, importlib, json, math, random, statistics, sys, time, unicodedata, urllib.request
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ALPHABET = "ABCDEFGHILMNOPQRSTUVXYZ"  # 23 letters after J->I, K->C, W->UU
STATES = ("alpha","beta1","beta2","beta3","gamma1","gamma2")
UNIGRAM_DECK = (0,1,2,3,4,5,0,1,2,3)
TRAIN_URLS = {
    "la": "https://www.gutenberg.org/cache/epub/218/pg218.txt",
    "it": "https://www.gutenberg.org/cache/epub/52484/pg52484.txt",
}
TEST_FILES = {
    "la": "input/examples/nathist_book16.txt",
    "it": "input/examples/divina_commedia.txt",
}
ITERATIONS = 700_000
RESTARTS = 50
N_TRIALS_PER_LANGUAGE = 10
LENGTH = 384


def stable_seed(*parts: object) -> int:
    b="|".join(map(str,parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(b).digest()[:8],"big") & 0x7FFFFFFFFFFFFFFF


def normalize(text: str) -> str:
    text=unicodedata.normalize("NFKD",text)
    text="".join(c for c in text if not unicodedata.combining(c)).upper()
    text=text.replace("W","UU").replace("J","I").replace("K","C")
    return "".join(c for c in text if c in ALPHABET)


def strip_gutenberg(text: str) -> str:
    lines=text.splitlines()
    starts=[i for i,x in enumerate(lines) if "*** START OF" in x.upper()]
    ends=[i for i,x in enumerate(lines) if "*** END OF" in x.upper()]
    if starts:
        lo=starts[0]+1
        hi=next((i for i in ends if i>lo),len(lines))
        return "\n".join(lines[lo:hi])
    return text


def fetch_text(url: str, cache: Path) -> str:
    if not cache.exists():
        req=urllib.request.Request(url,headers={"User-Agent":"VoynichFrontierU5/0.1"})
        with urllib.request.urlopen(req,timeout=60) as r:
            cache.write_bytes(r.read())
    return cache.read_text(encoding="utf-8",errors="ignore")


def build_language(train_text: str):
    s=normalize(strip_gutenberg(train_text))
    if len(s)<50_000:
        raise RuntimeError(f"training corpus unexpectedly short after normalization: {len(s)}")
    to_i={c:i for i,c in enumerate(ALPHABET)}
    stream=[to_i[c] for c in s]
    counts=np.bincount(np.asarray(stream,dtype=np.int64),minlength=len(ALPHABET)).astype(float)+0.15
    probs=(counts/counts.sum()).tolist()
    return SimpleNamespace(alphabet=list(ALPHABET),train_stream=stream,probabilities=probs),len(s)


def load_tables(path: Path):
    table={}
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f): table[r["code"]]=r["glyphs"]
    expected=3*6*23
    if len(table)!=expected:
        raise RuntimeError(f"expected {expected} Naibbe table entries, got {len(table)}")
    return table


def surface_realize(plain: str, perm: np.ndarray, tables: dict[str,str], seed: int):
    rng=random.Random(seed)
    uni_all={v for k,v in tables.items() if k.startswith("unigram_")}
    pieces=[]; singles=pairs=rejects=0; i=0
    while i<len(plain):
        width=1 if i==len(plain)-1 else rng.choice((1,2))
        if width==1:
            pidx=ALPHABET.index(plain[i]); enc=ALPHABET[int(perm[pidx])]
            state=STATES[rng.choice(UNIGRAM_DECK)]
            pieces.append(tables[f"unigram_{state}_{enc.lower()}"])
            singles+=1;i+=1;continue
        p1=ALPHABET.index(plain[i]);p2=ALPHABET.index(plain[i+1])
        e1=ALPHABET[int(perm[p1])].lower();e2=ALPHABET[int(perm[p2])].lower()
        while True:
            s1=STATES[rng.randrange(6)];s2=STATES[rng.randrange(6)]
            g=tables[f"prefix_{s1}_{e1}"]+tables[f"suffix_{s2}_{e2}"]
            if g not in uni_all: break
            rejects+=1
        pieces.append(g);pairs+=1;i+=2
    surface=" ".join(pieces)
    return {"surface_sha256":hashlib.sha256(surface.encode()).hexdigest(),"surface_tokens":len(pieces),"singles":singles,"pairs":pairs,"unambiguous_rejections":rejects,"surface_chars":sum(map(len,pieces))}


def deterministic_chunks(s: str, n: int, length: int):
    if len(s)<n*length*2:
        raise RuntimeError(f"locked source too short: {len(s)}")
    usable=len(s)-length
    starts=np.linspace(0,usable,n,dtype=int).tolist()
    chunks=[s[x:x+length] for x in starts]
    if any(len(x)!=length for x in chunks): raise RuntimeError("chunk length failure")
    return starts,chunks


def load_legacy_solver(legacy_repo: Path):
    d=legacy_repo/"experiments"/"recoverability_frontier_v0_5"
    sys.path.insert(0,str(d))
    return importlib.import_module("mono_solver_v051")


def trial_job(iso,idx,plain,language,model,mono,tables):
    started=time.perf_counter()
    seed=stable_seed("frontier-u5-a",iso,idx)
    rng=np.random.default_rng(seed)
    perm=np.arange(len(ALPHABET),dtype=np.int32);rng.shuffle(perm)
    to_i={c:i for i,c in enumerate(ALPHABET)}
    truth=[to_i[c] for c in plain]
    cipher=[int(perm[p]) for p in truth]
    surface=surface_realize(plain,perm,tables,stable_seed("frontier-u5-a-surface",iso,idx))
    initial=mono.frequency_key(cipher,language)
    carr=np.asarray(cipher,dtype=np.int32)
    baseline=initial[carr].astype(np.int32).tolist()
    baseline_accuracy=mono.fast_accuracy(truth,baseline)
    key,score=mono.anneal_mono(carr,initial,model[0],model[1],ITERATIONS,RESTARTS,seed)
    pred=key[carr].astype(np.int32).tolist()
    accuracy=mono.fast_accuracy(truth,pred)
    return {"language":iso,"trial":idx,"seed":seed,"length":len(truth),"accuracy":accuracy,"baseline_accuracy":baseline_accuracy,"pass_075":accuracy>=0.75,"exact":pred==truth,"score":float(score),"elapsed_seconds":time.perf_counter()-started,**surface}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--legacy-repo",type=Path,required=True)
    ap.add_argument("--naibbe-repo",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=20)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    mono=load_legacy_solver(a.legacy_repo)
    tables_path=a.naibbe_repo/"references"/"naibbe_tables.csv"
    tables=load_tables(tables_path)
    cache=a.out/"source_cache";cache.mkdir(exist_ok=True)

    languages={};models={};train_lengths={};locked={};chunk_starts={}
    for iso in ("la","it"):
        train=fetch_text(TRAIN_URLS[iso],cache/f"train_{iso}.txt")
        language,ntrain=build_language(train);languages[iso]=language;train_lengths[iso]=ntrain
        models[iso]=mono.build_language_model(language)
        test_path=a.naibbe_repo/TEST_FILES[iso]
        text=normalize(test_path.read_text(encoding="utf-8",errors="ignore"))
        starts,chunks=deterministic_chunks(text,N_TRIALS_PER_LANGUAGE,LENGTH)
        locked[iso]=chunks;chunk_starts[iso]=starts

    # Trigger numba compilation before parallel timing.
    lang=languages["la"];model=models["la"]
    mono.anneal_mono(np.asarray([0,1,0,1,0,1],dtype=np.int32),np.arange(23,dtype=np.int32),model[0],model[1],2,1,1)

    jobs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for iso in ("la","it"):
            for i,plain in enumerate(locked[iso]):
                jobs.append(ex.submit(trial_job,iso,i,plain,languages[iso],models[iso],mono,tables))
        rows=[]
        for n,f in enumerate(concurrent.futures.as_completed(jobs),1):
            r=f.result();rows.append(r)
            print("U5A_TRIAL",json.dumps(r,sort_keys=True),flush=True)
    rows.sort(key=lambda r:(r["language"],r["trial"]))
    mean=statistics.fmean(r["accuracy"] for r in rows)
    passed=sum(r["pass_075"] for r in rows)
    verdict="PASS_RECOVERY_CALIBRATION" if mean>=0.85 and passed>=16 else "FAIL_RECOVERY_CALIBRATION"
    result={
      "schema":"frontier-u5-a-v0.1","formal_verdict":verdict,"target_opened":False,"voynich_read":False,
      "family":"Greshko Naibbe positive control; fresh global 23-letter permutation; oracle role collapse",
      "iterations":ITERATIONS,"restarts":RESTARTS,"trials":len(rows),"length":LENGTH,
      "mean_accuracy":mean,"median_accuracy":statistics.median(r["accuracy"] for r in rows),
      "trials_ge_075":passed,"required_mean":0.85,"required_trials_ge_075":16,
      "baseline_mean":statistics.fmean(r["baseline_accuracy"] for r in rows),
      "train_sources":TRAIN_URLS,"train_normalized_lengths":train_lengths,
      "locked_test_files":TEST_FILES,"locked_chunk_starts":chunk_starts,
      "naibbe_table_sha256":hashlib.sha256(tables_path.read_bytes()).hexdigest(),
      "legacy_solver_path":"experiments/recoverability_frontier_v0_5/mono_solver_v051.py",
      "rows":rows,
      "consequence":"U5-B recognition may open; Voynich remains sealed" if verdict.startswith("PASS") else "U5 closes under v0.1; Voynich remains sealed",
    }
    (a.out/"U5A_RECOVERY_RESULT.json").write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")
    print("U5A_FINAL",json.dumps({k:result[k] for k in ("formal_verdict","mean_accuracy","trials_ge_075","baseline_mean","target_opened")},sort_keys=True),flush=True)

if __name__=="__main__":main()

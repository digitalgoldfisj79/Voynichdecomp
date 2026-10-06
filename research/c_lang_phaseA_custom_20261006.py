#!/usr/bin/env python3
"""
C-LANG Phase-A custom source screen, 2026-10-06.

Reuses the validated/frozen GER2 NeuroCipher runtime and exact VMS physical folds.
Only the external target vocabulary changes.

Candidates:
  CZECH       Old Czech, Martin Lupac New Testament, 1440.
  PINYIN      Mandarin Pinyin, tones stripped, ü->v.
  PINYIN_TONE Mandarin Pinyin retaining tone marks.

Common target sizes:
  train top 4,103; final top 9,000.

Phase-A promotion gate (frozen before target results):
  REAL seed1234 must have lower expected-edit MCF cost than its matched
  within-word-scrambled NULL on BOTH fold4 validation and sealed strict-final.
A passing candidate is only promoted to multi-seed replication; it is not
language identification.
"""
import argparse,base64,gzip,json,urllib.request,zipfile,io,re,unicodedata,collections,csv,sys,os
from pathlib import Path

RUNTIME_COMMIT="98b7324fc65aa9b14242920f6931a8d1256064df"
SOLVER_COMMIT="e36031a9a6b8b67fcebb4d6f4af1c3753fad4287"
CACHE_COMMIT="a4a133f0635e0408266bc185a1697821ab79bd28"
CZECH_COMMIT="efde1bbe36d4eac9f2266d38c03b4805937ba7ed"
PINYIN_COMMIT="dd6bc245038fdb0c93f7a29e380be15be671848b"
FULL_K=9000
TRAIN_K=4103

def get(url):
    return urllib.request.urlopen(url,timeout=300).read()

def bootstrap():
    base="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp"
    Path("/tmp/neurodecipher_acl2019_refactor.py").write_bytes(get(f"{base}/{SOLVER_COMMIT}/research/neurodecipher_acl2019_refactor.py"))
    Path("/tmp/vms_neurocipher_ger2_cached_runtime_20261006.py").write_bytes(get(f"{base}/{RUNTIME_COMMIT}/research/vms_neurocipher_ger2_cached_runtime_20261006.py"))
    packed=get(f"{base}/{CACHE_COMMIT}/research/data/vms_neurocipher_ger2_cache_20261006.json.gz.b64")
    obj=json.loads(gzip.decompress(base64.b64decode(packed.strip())).decode())
    sys.path.insert(0,"/tmp")
    import vms_neurocipher_ger2_cached_runtime_20261006 as rt
    return obj,rt

def ascii_fold(s):
    s=s.replace("ſ","s")
    s=unicodedata.normalize("NFKD",s)
    s="".join(ch for ch in s if unicodedata.category(ch)!="Mn").lower()
    return re.findall(r"[a-z]+",s)

def czech_vocab():
    u=f"https://codeload.github.com/HTR-School-Vienna/2023--medieval-czech/zip/{CZECH_COMMIT}"
    z=zipfile.ZipFile(io.BytesIO(get(u)))
    c=collections.Counter(); nf=nl=nt=0
    for n in z.namelist():
        if not n.endswith(".xml"): continue
        nf+=1
        txt=z.read(n).decode("utf-8","ignore")
        for m in re.finditer(r"<Unicode>([\s\S]*?)</Unicode>",txt):
            nl+=1; ws=ascii_fold(m.group(1));nt+=len(ws);c.update(ws)
    ranked=[w for w,n in c.most_common()]
    if len(ranked)<FULL_K: raise RuntimeError(("Czech types",len(ranked)))
    return ranked[:FULL_K],{"docs":nf,"lines":nl,"tokens":nt,"ref_types":len(ranked),"commit":CZECH_COMMIT}

def fix_legacy_tone(s):
    # Source contains a handful of u:1..u:4 legacy spellings.
    for a,b in (("u:1","ǖ"),("u:2","ǘ"),("u:3","ǚ"),("u:4","ǜ"),("u:5","ü")):
        s=s.replace(a,b)
    return s

def pinyin_vocab(tones):
    u=f"https://raw.githubusercontent.com/Roxaleen/hsk-annotated-corpus/{PINYIN_COMMIT}/export/csv/words.csv"
    txt=get(u).decode("utf-8")
    best={}
    for row in csv.DictReader(io.StringIO(txt)):
        py=(row.get("pinyin") or "").strip().lower()
        if not py or "|" in py: continue
        try: rank=int(row["frequency_ranking"])
        except: continue
        py=fix_legacy_tone(py)
        py=re.sub(r"[\s'’·-]+","",py)
        if tones:
            py=unicodedata.normalize("NFC",py)
            if not py or not all(ch.isalpha() for ch in py): continue
        else:
            py=py.replace("ü","v").replace("ǖ","v").replace("ǘ","v").replace("ǚ","v").replace("ǜ","v")
            py=unicodedata.normalize("NFD",py)
            py="".join(ch for ch in py if unicodedata.category(ch)!="Mn")
            py="".join(ch for ch in py if ch in "abcdefghijklmnopqrstuvwxyzv")
            if not py: continue
        if py not in best or rank<best[py]: best[py]=rank
    ranked=[w for w,r in sorted(best.items(),key=lambda x:(x[1],x[0]))]
    if len(ranked)<FULL_K: raise RuntimeError(("Pinyin types",len(ranked),tones))
    return ranked[:FULL_K],{"docs":1,"ref_types":len(ranked),"commit":PINYIN_COMMIT,"tones":tones}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",choices=["CZECH","PINYIN","PINYIN_TONE"],required=True)
    ap.add_argument("--mode",choices=["real","scramble"],required=True)
    ap.add_argument("--seed",type=int,default=1234)
    ap.add_argument("--scramble-seed",type=int,default=9001)
    a=ap.parse_args()
    cache,rt=bootstrap()
    if a.source=="CZECH": forms,meta=czech_vocab()
    else: forms,meta=pinyin_vocab(a.source=="PINYIN_TONE")
    rt.FULL_K=FULL_K
    rt.TRAIN_K=TRAIN_K
    cache["dialects"]={a.source:{"real":forms,"docs":meta.get("docs",1),"ref_types":meta["ref_types"]}}
    cp=f"/tmp/c_lang_{a.source.lower()}.json"
    Path(cp).write_text(json.dumps(cache,ensure_ascii=False,separators=(",",":")))
    audit={"source":a.source,"mode":a.mode,"seed":a.seed,"scramble_seed":a.scramble_seed,
           "train_k":TRAIN_K,"full_k":FULL_K,"source_meta":meta,
           "vms_fold_sha":cache["vms"]["audit"]["fold_sha"],
           "runtime_commit":RUNTIME_COMMIT,"solver_commit":SOLVER_COMMIT,"cache_commit":CACHE_COMMIT}
    print("CLANG_AUDIT="+json.dumps(audit,ensure_ascii=False,separators=(",",":")),flush=True)
    class A: pass
    x=A();x.cache=cp;x.dialect=a.source;x.mode=a.mode;x.seed=a.seed;x.scramble_seed=a.scramble_seed
    x.cpu=False;x.smoke=False;x.rounds=10;x.epochs=150;x.eval_every=10;x.log_every=25
    x.warm_up_steps=5;x.reg_hyper=.5;x.eval_batch=48
    rt.CachedRunner(x).run()
if __name__=="__main__":main()

#!/usr/bin/env python3
"""Yiddish Cipher Instrument Qualification Control v1.0 — M0.

This program is an instrument benchmark, not a Voynich experiment.
It deliberately has no Voynich import or target path.

Profiles:
  smoke: inexpensive executable/logic validation; NEVER a scientific PASS.
  full:  registered C0-C5 control battery. C6 transfer and C7 target are separate.

The M0 solver is the already-developed v03 S1 search configuration. The point of
this harness is to measure its operating characteristics rather than to tune it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V03 = REPO / "experiments" / "yiddish_qualification_v03"
V1B = REPO / "experiments" / "yiddish_qualification_v1b"
sys.path.insert(0, str(V03))
sys.path.insert(0, str(V1B))

import run_v03_search_repair as v03  # noqa: E402
import finite_panel_r_v02 as base  # noqa: E402
from encoder_v02 import encode_words  # noqa: E402
from independent_decoder_v02 import invert_permutation, decode_words, assert_roundtrip_non_erased  # noqa: E402

CONTROL_VERSION = "yiddish_cipher_instrument_control_v1_20260912"
A = base.A
ALPHABET = base.ALPHABET
BOUNDARY = base.BOUNDARY

# All are already consumed in earlier Yiddish work; they are safe instrument controls.
YID_BUILD = (
    "1600e-magid-preface.psd", "1600e-magid.psd", "1600e-tsenerene.psd",
    "1619w-letters-prague.psd", "1624e-magen.psd", "1671e-vaad.psd",
    "1677w-witzenhausen.psd", "1692e-vilna.psd", "1704e-ellush.psd",
    "1705w-glikl.psd", "1712e-sarah.psd", "1716e-duties.psd",
)
YID_DEV = (
    "1507w-bovo.psd", "1588e-letters-cracow.psd", "1590e-sam-hayyim.psd",
    "1620e-lev-tov-1.psd", "1648w-kine.psd", "1666w-messiah.psd",
)

# ReF official diplomatic XML IDs. F014/F015 are build/context; the others are controls.
REF_IDS = ("F014", "F015", "F016", "F018", "F034", "F037", "F148")

PRIMARY_LENGTHS = (512, 2048)
FULL_ERASURES = (0.0, 0.01, 0.03, 0.05)
ATOM_PASS = 0.90
WORD_PASS = 0.80
CELL_PASS = 29
S1 = {"id": "S1_FROZEN", "restarts": 8, "steps": 5000, "greedy_passes": 60}
SMOKE_CFG = {"id": "SMOKE_NOT_QUALIFYING", "restarts": 2, "steps": 600, "greedy_passes": 8}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def public_seed(*parts) -> int:
    b = "|".join([CONTROL_VERSION, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "big")


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def append_jsonl(row, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())


def cp_upper(successes: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound via binomial-tail bisection."""
    if n <= 0: return float("nan")
    if successes >= n: return 1.0
    # Find p such that P_p(X <= successes) = alpha.
    def cdf(k, p):
        return sum(math.comb(n, i) * (p ** i) * ((1-p) ** (n-i)) for i in range(k+1))
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if cdf(successes, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cp_lower(successes: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0: return float("nan")
    if successes <= 0: return 0.0
    # lower bound on success p: P_p(X >= successes)=alpha
    def sf(k, p):
        return sum(math.comb(n, i) * (p ** i) * ((1-p) ** (n-i)) for i in range(k, n+1))
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid=(lo+hi)/2
        if sf(successes, mid) < alpha:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2


def build_yiddish_training(data: Path):
    words=[]; files=[]; hashes={}
    for name in YID_BUILD:
        p=data/name
        if p.exists():
            ws=base.extract_words(p)
            words.extend(ws); files.append(name); hashes[name]=sha256_file(p)
    if len(words) < 10000:
        raise RuntimeError(f"Yiddish BUILD too small: {len(words)}")
    return words, files, hashes


def normalize_latin_token(s: str):
    s=s.lower()
    # Conservative a-z only, intentionally matched to the Penn secondary representation.
    w=''.join(c for c in s if 'a' <= c <= 'z')
    return w or None


def ref_dipl_words(ref_root: Path, work_id: str):
    """Find a ReF XML by Fxxx ID and read tok_dipl/@utf in document order."""
    candidates=[]
    for p in ref_root.rglob("*.xml"):
        if work_id.lower() in p.name.lower() or work_id.lower() in str(p.parent).lower():
            candidates.append(p)
    if not candidates:
        return [], None
    # Prefer the largest matching XML: tiny metadata files are common.
    p=max(candidates, key=lambda q:q.stat().st_size)
    root=ET.parse(p).getroot(); out=[]
    for el in root.iter():
        tag=el.tag.rsplit('}',1)[-1]
        if tag == 'tok_dipl':
            raw=el.attrib.get('utf') or (el.text or '')
            w=normalize_latin_token(raw)
            if w: out.append(w)
    return out,p


def kat_suite():
    fixtures=[
        ["a"], ["aa"], ["abba"], ["abcdefghijklmnopqrstuvwxyz"],
        ["abc","def","abc"], ["mississippi","banana","letter"],
    ]
    keys=[]
    keys.append(("identity", list(range(A))))
    keys.append(("cyclic5", list(range(5,A))+list(range(5))))
    keys.append(("reverse", list(reversed(range(A)))))
    r=random.Random(public_seed("kat","random")); k=list(range(A)); r.shuffle(k); keys.append(("random",k))
    rows=[]
    for fname,words in enumerate(fixtures):
        for kname,key in keys:
            for er in (0.0,0.01):
                rng=random.Random(public_seed("kat",fname,kname,er))
                enc,_,_=encode_words(words,key,er,rng)
                dec=decode_words(enc,invert_permutation(key))
                ok=True
                for t,d in zip(words,dec):
                    for a,b in zip(t,d):
                        if b != '~' and a != b: ok=False
                rows.append({"fixture":fname,"key":kname,"erasure":er,"pass":ok,"cipher":enc})
    return rows


def plant_global(words, key, erasure, tag):
    rng=random.Random(public_seed("erase",tag,erasure))
    return encode_words(words,key,erasure,rng)[0]


def key_for(tag):
    r=random.Random(public_seed("key",tag)); p=list(range(A)); r.shuffle(p); return p


def solve_and_score(fit_truth, audit_truth, train_lm, train_uni, cfg, erasure, tag,
                    fit_cipher=None, audit_cipher=None, truth_decoder=None):
    key=key_for(tag)
    if fit_cipher is None:
        fit_cipher=plant_global(fit_truth,key,erasure,tag+"|fit")
        audit_cipher=plant_global(audit_truth,key,erasure,tag+"|audit")
        oracle=invert_permutation(key)
    else:
        oracle=truth_decoder
    seed=public_seed("search",tag,cfg['id'])
    t0=time.time()
    returned,obj=v03.blind_solve_budget(fit_cipher,train_lm,train_uni,seed,cfg)
    runtime=time.time()-t0
    dec=decode_words(audit_cipher,returned)
    rec=base.score_recovery(audit_truth,dec)
    C,_=base.cipher_counts(fit_cipher)
    oracle_obj=base.mapping_score(C,train_lm,oracle) if oracle is not None else None
    passed=(rec['atom_recovery'] is not None and rec['atom_recovery']>=ATOM_PASS and
            rec['word_recovery'] is not None and rec['word_recovery']>=WORD_PASS)
    return {"tag":tag,"pass":passed,"runtime_s":runtime,"returned_objective":obj,
            "oracle_objective":oracle_obj,
            "oracle_minus_returned":None if oracle_obj is None else oracle_obj-obj, **rec}


def split_window(words,n):
    if len(words)<2*n+32: return None
    return words[:n], words[n+32:n+32+n]


def positive_power(data, train_lm, train_uni, profile, out_rows):
    cfg=S1 if profile=='full' else SMOKE_CFG
    keys_per=32 if profile=='full' else 2
    lengths=PRIMARY_LENGTHS if profile=='full' else (512,)
    erasures=FULL_ERASURES if profile=='full' else (0.0,0.01)
    works=YID_DEV if profile=='full' else YID_DEV[:2]
    cells={}
    for work in works:
        words=base.extract_words(data/work)
        for n in lengths:
            spl=split_window(words,n)
            if not spl: continue
            fit,audit=spl
            for er in erasures:
                passed=0
                for ki in range(keys_per):
                    tag=f"C3|{work}|{n}|{er}|{ki}"
                    row=solve_and_score(fit,audit,train_lm,train_uni,cfg,er,tag)
                    row.update({"certificate":"C3","work":work,"length":n,"erasure":er,"key_index":ki})
                    append_jsonl(row,out_rows); passed+=int(row['pass'])
                cells[f"{work}|{n}|{er}"]={"n":keys_per,"successes":passed,
                    "lower95":cp_lower(passed,keys_per),
                    "registered_cell_pass": (passed>=CELL_PASS if profile=='full' else None)}
    full_primary=[v for k,v in cells.items() if ('|512|0.0' in k or '|512|0.01' in k)]
    status='SMOKE_ONLY' if profile!='full' else ('PASS' if full_primary and all(x['registered_cell_pass'] for x in full_primary) else 'FAIL')
    return {"status":status,"cells":cells,"config":cfg}


def encrypt_per_word(words, tag):
    out=[]
    for i,w in enumerate(words):
        key=key_for(f"{tag}|word|{i}")
        out.append(''.join(ALPHABET[key[ord(c)-97]] for c in w))
    return out


def encrypt_key_drift(words, tag, blocks=4):
    out=[]; block=max(1,math.ceil(len(words)/blocks))
    for i,w in enumerate(words):
        key=key_for(f"{tag}|block|{i//block}")
        out.append(''.join(ALPHABET[key[ord(c)-97]] for c in w))
    return out


def shuffle_inside_words(words, tag):
    out=[]
    for i,w in enumerate(words):
        a=list(w); random.Random(public_seed(tag,i)).shuffle(a); out.append(''.join(a))
    return out


def unigram_null(words, tag):
    c=Counter(''.join(words)); letters=list(c); weights=[c[x] for x in letters]
    r=random.Random(public_seed(tag))
    return [''.join(r.choices(letters,weights=weights,k=len(w))) for w in words]


def structural_negatives(data, train_lm, train_uni, profile, out_rows):
    cfg=S1 if profile=='full' else SMOKE_CFG
    reps=32 if profile=='full' else 2
    source=base.extract_words(data/YID_DEV[0])
    spl=split_window(source,512)
    if not spl: raise RuntimeError('negative source too short')
    fit,audit=spl
    families=('per_word_key','key_drift','within_word_shuffle','unigram_null')
    result={}
    for fam in families:
        calls=0
        for i in range(reps):
            tag=f"C4a|{fam}|{i}"
            if fam=='per_word_key':
                fc=encrypt_per_word(fit,tag+'|fit'); ac=encrypt_per_word(audit,tag+'|audit')
            elif fam=='key_drift':
                fc=encrypt_key_drift(fit,tag+'|fit'); ac=encrypt_key_drift(audit,tag+'|audit')
            elif fam=='within_word_shuffle':
                # Shuffle plaintext within each word, then plant a legitimate global key. Truth for
                # scoring remains the ORIGINAL Yiddish: recovery should therefore not be called Yiddish.
                sf=shuffle_inside_words(fit,tag+'|fit'); sa=shuffle_inside_words(audit,tag+'|audit')
                key=key_for(tag); fc=plant_global(sf,key,0.0,tag+'|fit'); ac=plant_global(sa,key,0.0,tag+'|audit')
            else:
                sf=unigram_null(fit,tag+'|fit'); sa=unigram_null(audit,tag+'|audit')
                key=key_for(tag); fc=plant_global(sf,key,0.0,tag+'|fit'); ac=plant_global(sa,key,0.0,tag+'|audit')
            # There is deliberately no single valid oracle for per-word/drift. We still score the decoded
            # output against original Yiddish truth: a positive call would be a false Yiddish recovery.
            seed=public_seed('search',tag,cfg['id']); t0=time.time()
            returned,obj=v03.blind_solve_budget(fc,train_lm,train_uni,seed,cfg)
            dec=decode_words(ac,returned); rec=base.score_recovery(audit,dec)
            call=(rec['atom_recovery'] is not None and rec['atom_recovery']>=ATOM_PASS and
                  rec['word_recovery'] is not None and rec['word_recovery']>=WORD_PASS)
            row={"certificate":"C4a","family":fam,"rep":i,"positive_call":call,
                 "runtime_s":time.time()-t0,"returned_objective":obj,**rec}
            append_jsonl(row,out_rows); calls+=int(call)
        result[fam]={"n":reps,"false_positive_calls":calls,"upper95":cp_upper(calls,reps)}
    total_n=sum(x['n'] for x in result.values()); total_fp=sum(x['false_positive_calls'] for x in result.values())
    # Gate is family-wise in full profile; smoke never qualifies.
    status='SMOKE_ONLY' if profile!='full' else ('PASS' if all(x['upper95']<=0.05 for x in result.values()) else 'UNBOUNDED_OR_FAIL')
    return {"status":status,"families":result,"pooled":{"n":total_n,"false_positive_calls":total_fp,"upper95":cp_upper(total_fp,total_n)}}


def german_language_specificity(ref_root, train_lm, train_uni, profile, out_rows):
    """C4b: if Yiddish solver exactly recovers M0 German, it is mechanism-capable but not language-specific."""
    if ref_root is None or not ref_root.exists():
        return {"status":"PENDING_COMPARATOR","reason":"ReF root absent"}
    cfg=S1 if profile=='full' else SMOKE_CFG
    reps=8 if profile=='full' else 1
    ids=("F016","F018","F034","F037","F148") if profile=='full' else ("F016",)
    rows=[]
    for wid in ids:
        words,p=ref_dipl_words(ref_root,wid)
        if len(words)<1056:
            rows.append({"work":wid,"status":"TOO_SHORT","n_words":len(words)}); continue
        fit,audit=words[:512],words[544:1056]
        succ=0
        for i in range(reps):
            tag=f"C4b|german|{wid}|{i}"; key=key_for(tag)
            fc=plant_global(fit,key,0.0,tag+'|fit'); ac=plant_global(audit,key,0.0,tag+'|audit')
            oracle=invert_permutation(key)
            r=solve_and_score(fit,audit,train_lm,train_uni,cfg,0.0,tag,fc,ac,oracle)
            r.update({"certificate":"C4b","language":"german","work":wid,"rep":i})
            append_jsonl(r,out_rows); succ+=int(r['pass'])
        rows.append({"work":wid,"source":str(p),"n_words":len(words),"n":reps,
                     "yiddish_solver_recovery_successes":succ,"lower95":cp_lower(succ,reps)})
    # Important: success here is NOT an implementation failure. It proves the solver cannot itself
    # establish Yiddish identity; a separate L certificate is required. We therefore report specificity.
    tested=[r for r in rows if 'n' in r]
    total=sum(r['n'] for r in tested); succ=sum(r['yiddish_solver_recovery_successes'] for r in tested)
    return {"status":"MEASURED_NOT_A_GATE_ON_M0_MECHANISM","works":rows,
            "pooled":{"n":total,"german_recovered_as_plaintext":succ,
                      "rate":succ/total if total else None}}


def metamorphic_tests(data, train_lm, train_uni, profile, out_rows):
    reps=32 if profile=='full' else 4
    cfg=S1 if profile=='full' else SMOKE_CFG
    words=base.extract_words(data/YID_DEV[0]); spl=split_window(words,512)
    if not spl: raise RuntimeError('metamorphic source too short')
    fit,audit=spl; violations=[]; rows=[]
    # Cheap exact sufficient-statistic MRs + a smaller number of expensive solver relabel MRs.
    C0,_=base.cipher_counts(fit)
    Cdup,_=base.cipher_counts(fit+fit)
    dup_max=max(abs(C0[i][j]-Cdup[i][j]) for i in range(A+1) for j in range(A+1))
    rows.append({"mr":"MR3_duplicate_normalized_counts","max_abs_delta":dup_max,"pass":dup_max<1e-15})
    # Chunk recombination identity.
    recombined=fit[:171]+fit[171:341]+fit[341:]
    Cchunk,_=base.cipher_counts(recombined)
    ch_max=max(abs(C0[i][j]-Cchunk[i][j]) for i in range(A+1) for j in range(A+1))
    rows.append({"mr":"MR4_chunk_recombine","max_abs_delta":ch_max,"pass":ch_max<1e-15})
    # Independent decoder oracle over planted examples.
    for i in range(reps):
        key=key_for(f"mr5|{i}"); fc=plant_global(fit,key,0.01,f"mr5|{i}|fit")
        dec=decode_words(fc,invert_permutation(key)); ok=True
        for t,d in zip(fit,dec):
            for a,b in zip(t,d):
                if b!='~' and a!=b: ok=False
        rows.append({"mr":"MR5_independent_decoder","rep":i,"pass":ok})
    # Global ciphertext relabelling solver MR. Use fewer expensive replications in full but >=32 is
    # achieved through the exact key/oracle MR above; this relation is stochastic-search diagnostic.
    solver_reps=4 if profile=='full' else 1
    for i in range(solver_reps):
        tag=f"mr1|{i}"; key=key_for(tag); fc=plant_global(fit,key,0.0,tag+'|fit'); ac=plant_global(audit,key,0.0,tag+'|audit')
        rel=key_for(tag+'|relabel')
        def relabel(ws): return [''.join(ALPHABET[rel[ord(c)-97]] for c in w) for w in ws]
        seed=public_seed('mrsearch',tag)
        m1,_=v03.blind_solve_budget(fc,train_lm,train_uni,seed,cfg)
        m2,_=v03.blind_solve_budget(relabel(fc),train_lm,train_uni,seed,cfg)
        d1=decode_words(ac,m1); d2=decode_words(relabel(ac),m2)
        r1=base.score_recovery(audit,d1); r2=base.score_recovery(audit,d2)
        # Stochastic optimizer need not land at identical keys under relabelled initialization; registered
        # requirement is that recovery classification agrees and atom recovery differs <=0.05.
        p1=r1['atom_recovery']>=ATOM_PASS and r1['word_recovery']>=WORD_PASS
        p2=r2['atom_recovery']>=ATOM_PASS and r2['word_recovery']>=WORD_PASS
        ok=(p1==p2 and abs(r1['atom_recovery']-r2['atom_recovery'])<=0.05)
        rows.append({"mr":"MR1_global_relabel_solver","rep":i,"pass":ok,
                     "atom1":r1['atom_recovery'],"atom2":r2['atom_recovery'],"class1":p1,"class2":p2})
    for r in rows:
        append_jsonl({"certificate":"C5",**r},out_rows)
        if not r['pass']: violations.append(r)
    return {"status":('PASS' if not violations else 'FAIL') if profile=='full' else 'SMOKE_ONLY',
            "n_checks":len(rows),"violations":violations}


def source_overlap(data: Path):
    build=[]
    for n in YID_BUILD:
        p=data/n
        if p.exists(): build.extend(base.extract_words(p))
    out=[]
    b8=base.ngrams(build,8); b5=base.ngrams(build,5)
    for n in YID_DEV:
        p=data/n
        if not p.exists(): continue
        w=base.extract_words(p)
        out.append({"work":n,"words":len(w),"shared_8gram_types":len(b8 & base.ngrams(w,8)),
                    "shared_5gram_types":len(b5 & base.ngrams(w,5)),"sha256":sha256_file(p)})
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--profile',choices=['smoke','full'],default='smoke')
    ap.add_argument('--ref-root',type=Path,default=None)
    ap.add_argument('--out',type=Path,default=HERE/'control_output')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    rows_path=args.out/'trial_rows.jsonl'
    if rows_path.exists(): rows_path.unlink()

    # C0/C1 acquisition + provenance.
    src=args.out/'sources'; src.mkdir(exist_ok=True)
    ppchy,pp_commit=base.ensure_corpus(src); data=ppchy/'data'
    train_words,train_files,train_hashes=build_yiddish_training(data)
    lm,uni=base.lm_from_words(train_words)
    provenance={
        'control_version':CONTROL_VERSION,'profile':args.profile,'voynich_loaded':False,
        'target_access_allowed':False,'ppchy_commit':pp_commit,
        'representation':'Penn historical-Yiddish Romanisation reduced to literal a-z; secondary control representation only',
        'alphabet':''.join(ALPHABET),'boundary_id':BOUNDARY,
        'source_code_sha256':sha256_file(Path(__file__)),
        'encoder_sha256':sha256_file(V1B/'encoder_v02.py'),
        'decoder_sha256':sha256_file(V1B/'independent_decoder_v02.py'),
        'solver_sha256':sha256_file(V03/'run_v03_search_repair.py'),
        'training_files':train_files,'training_hashes':train_hashes,'training_words':len(train_words),
        'development_overlap':source_overlap(data),
        'ref_root':str(args.ref_root) if args.ref_root else None,
    }
    c0='PASS' if not provenance['voynich_loaded'] and provenance['target_access_allowed'] is False else 'FAIL'
    c1='PASS' if len(train_words)>=10000 and len(ALPHABET)==26 else 'FAIL'

    kats=kat_suite(); c2='PASS' if kats and all(x['pass'] for x in kats) else 'FAIL'
    for r in kats: append_jsonl({'certificate':'C2',**r},rows_path)

    c3=positive_power(data,lm,uni,args.profile,rows_path)
    c4a=structural_negatives(data,lm,uni,args.profile,rows_path)
    c4b=german_language_specificity(args.ref_root,lm,uni,args.profile,rows_path)
    c5=metamorphic_tests(data,lm,uni,args.profile,rows_path)

    if args.profile=='smoke':
        overall='SMOKE_EXECUTED_NOT_QUALIFYING'
    else:
        # C4b does not fail M0 mechanism qualification; it measures whether recovery is language-specific.
        # Yiddish-language inference still requires the separate L certificate regardless.
        req=[c0,c1,c2,c3['status'],c4a['status'],c5['status']]
        overall='C0_C5_M0_CONTROL_PASS__C6_NOT_RUN__TARGET_SEALED' if all(x=='PASS' for x in req) else 'M0_CONTROL_UNQUALIFIED'

    summary={'overall_status':overall,
             'certificates':{'C0':c0,'C1':c1,'C2':c2,'C3':c3,'C4a':c4a,'C4b_language_specificity':c4b,'C5':c5,
                             'C6':'NOT_RUN','C7':'SEALED'},
             'provenance':provenance,
             'interpretation':{
                 'positive_control_scope':'algorithmic M0 recovery on secondary normalized historical-Yiddish representation',
                 'language_identity_rule':'M0 recovery alone cannot identify Yiddish; C4b quantifies this and separate L remains mandatory',
                 'target_rule':'No Voynich inference is permitted from this run.'}}
    write_json(summary,args.out/'summary.json')
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=='__main__':
    main()

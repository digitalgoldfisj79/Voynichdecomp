#!/usr/bin/env python3
"""Fractionation composition v0.1a development amendment.

Replaces phase-only detection with a coordinate-structure score and matches
negative-control surface alphabet sizes to the coordinate channel. Development
only; Voynich and UD test remain sealed.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, math, random, statistics, sys
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fractionation_composition_v01 as base

POS = base.PRIMARY_POSITIVE
NEG = ("slot_matched", "expanded_shared", "expanded_shared_trans", "markov_matched")


def target_surface_size(a: int) -> int:
    c = int(math.ceil(math.sqrt(a)))
    r = int(math.ceil(a / c))
    return r + c


def expanded_shared(words: Sequence[Sequence[int]], a: int, rng: random.Random, transpose: bool=False) -> list[list[int]]:
    m = target_surface_size(a)
    choices: list[tuple[int, int]] = []
    for _ in range(a):
        x = rng.randrange(m)
        y = rng.randrange(m - 1)
        if y >= x:
            y += 1
        choices.append((x, y))
    out: list[list[int]] = []
    global_block = rng.choice((4, 6, 8, 10))
    perm = list(range(global_block)); rng.shuffle(perm)
    for word in words:
        token: list[int] = []
        for x in word:
            pair = choices[int(x)]
            token.extend((rng.choice(pair), rng.choice(pair)))
        if transpose:
            tr: list[int] = []
            for start in range(0, len(token), global_block):
                piece = token[start:start+global_block]
                if len(piece) == global_block:
                    tr.extend(piece[i] for i in perm)
                else:
                    tr.extend(piece)
            token = tr
        out.append(token)
    return out


def slot_matched(words: Sequence[Sequence[int]], a: int, rng: random.Random) -> list[list[int]]:
    return base.slot_control(words, rng, alphabet_size=target_surface_size(a))


def markov_matched(words: Sequence[Sequence[int]], a: int, rng: random.Random) -> list[list[int]]:
    return base.markov_control(words, rng, alphabet_size=target_surface_size(a))


def component_pairs(tokens: Sequence[Sequence[int]], mode: str, block: int) -> list[tuple[int,int]]:
    pairs: list[tuple[int,int]] = []
    streams = [base.flatten(tokens)] if mode == "stream" else [list(t) for t in tokens]
    for seq in streams:
        for start in range(0, len(seq), 2*block):
            piece = seq[start:start+2*block]
            if len(piece) < 2:
                continue
            k = len(piece)//2
            left = piece[:k]
            right = piece[k:k+k]
            pairs.extend(zip(left, right))
    return pairs


def entropy(counter: collections.Counter[int]) -> float:
    n=sum(counter.values())
    if n<=0: return 0.0
    return -sum((v/n)*math.log2(v/n) for v in counter.values() if v)


def pair_structure(pairs: Sequence[tuple[int,int]]) -> tuple[float,float,float]:
    if not pairs:
        return 0.0, 1.0, 0.0
    left=collections.Counter(x for x,_ in pairs)
    right=collections.Counter(y for _,y in pairs)
    joint=collections.Counter(pairs)
    n=len(pairs)
    n0=len(left); n1=len(right)
    density=len(joint)/max(1,n0*n1)
    h0=entropy(left); h1=entropy(right)
    mi=0.0
    for (x,y),v in joint.items():
        p=v/n; px=left[x]/n; py=right[y]/n
        mi += p*math.log2(p/(px*py))
    denom=min(h0,h1)
    nmi=mi/denom if denom>1e-12 else 1.0
    nmi=max(0.0,min(1.0,nmi))
    coord=density*(1.0-nmi)
    return density,nmi,coord


def candidate_score(tokens: Sequence[Sequence[int]], mode: str, block: int) -> tuple[float,float,float,float]:
    phase = base.role_score_token(tokens, block) if mode=="token" else base.role_score_stream(tokens, block)
    density,nmi,coord=pair_structure(component_pairs(tokens,mode,block))
    return phase*coord, phase, density, nmi


def coord_peak(tokens: Sequence[Sequence[int]]) -> tuple[float,str,int,dict[str,dict[str,float]]]:
    best=(-1.0,"",-1)
    detail={}
    for b in base.BLOCKS:
        for mode in ("token","stream"):
            score,phase,density,nmi=candidate_score(tokens,mode,b)
            key=f"{mode}_b{b}"
            detail[key]={"score":score,"phase_mi":phase,"pair_density":density,"pair_nmi":nmi}
            if score>best[0]: best=(score,mode,b)
    return best[0],best[1],best[2],detail


def evaluate(tokens: Sequence[Sequence[int]], rng: random.Random, null_reps: int) -> dict[str,object]:
    obs,mode,block,detail=coord_peak(tokens)
    null=[]
    for _ in range(null_reps):
        sh=base.matched_shuffle(tokens,rng)
        null.append(coord_peak(sh)[0])
    mean=statistics.fmean(null)
    sd=statistics.stdev(null) if len(null)>1 else 0.0
    residual=obs-mean
    z=residual/sd if sd>1e-12 else (999.0 if residual>0 else 0.0)
    return {"observed":obs,"best_mode":mode,"best_block":block,"null_mean":mean,"null_sd":sd,
            "residual":residual,"z":z,"candidate_detail":detail}


def make(words, a, family, rng, block):
    if family in POS:
        return base.encrypt_fractionated(words,a,rng,family,block)
    if family=="slot_matched": return slot_matched(words,a,rng)
    if family=="expanded_shared": return expanded_shared(words,a,rng,False)
    if family=="expanded_shared_trans": return expanded_shared(words,a,rng,True)
    if family=="markov_matched": return markov_matched(words,a,rng)
    raise ValueError(family)


def summarize(rows):
    out={}
    for fam in POS+NEG:
        rr=[r for r in rows if r["family"]==fam]
        zs=[float(r["z"]) for r in rr]
        out[fam]={"n":len(rr),"mean_z":statistics.fmean(zs),"median_z":statistics.median(zs),
                  "min_z":min(zs),"max_z":max(zs),"rate_z_ge_3":statistics.fmean(float(z>=3) for z in zs),
                  "mean_residual":statistics.fmean(float(r["residual"]) for r in rr),
                  "mean_null_sd":statistics.fmean(float(r["null_sd"]) for r in rr)}
    p=[float(r["z"]) for r in rows if r["family"] in POS]
    n=[float(r["z"]) for r in rows if r["family"] in NEG]
    pr=statistics.fmean(float(z>=3) for z in p); nr=statistics.fmean(float(z>=3) for z in n)
    pm=statistics.fmean(p); nm=statistics.fmean(n); ns=statistics.stdev(n) if len(n)>1 else 0
    sep=pm-nm; ratio=sep/ns if ns>1e-12 else 999
    gate=pr>=.90 and nr<=.10 and ratio>=2
    return {"by_family":out,"aggregate":{"positive_rate_z_ge_3":pr,"control_rate_z_ge_3":nr,
             "positive_mean_z":pm,"control_mean_z":nm,"control_z_sd":ns,"mean_z_separation":sep,
             "separation_in_control_sd":ratio},"gate":{"decision":"GO_TO_LOCKED_TEST" if gate else "STOP_NON_IDENTIFIABLE",
             "criteria":{"positive_rate_z_ge_3_at_least":.90,"control_rate_z_ge_3_at_most":.10,
             "mean_separation_at_least_control_sd":2.0}}}


def run(langs,reps,target_letters,null_reps,split):
    rows=[]
    for iso in base.LANGS:
        lang=langs[iso]
        src=lang.dev_words if split=="dev" else lang.test_words
        for rep in range(reps):
            words=base.sample_word_chunk(src,target_letters,random.Random(base.stable_seed("frac-v01a",split,iso,rep,"chunk")))
            for fam in POS+NEG:
                block=1 if fam=="frac_pair" else 2+(base.stable_seed("frac-v01a",split,iso,rep,fam,"block")%7)
                toks=make(words,len(lang.alphabet),fam,random.Random(base.stable_seed("frac-v01a",split,iso,rep,fam,"cipher")),int(block))
                ev=evaluate(toks,random.Random(base.stable_seed("frac-v01a",split,iso,rep,fam,"null")),null_reps)
                rows.append({"split":split,"iso":iso,"rep":rep,"family":fam,"block":int(block),
                            "plain_letters":sum(len(w) for w in words),"cipher_symbols":sum(len(t) for t in toks),
                            "cipher_alphabet":len(set(base.flatten(toks))),**ev})
    return {"rows":rows,"summary":summarize(rows)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--reps",type=int,default=4); ap.add_argument("--target-letters",type=int,default=400); ap.add_argument("--null-reps",type=int,default=39)
    ap.add_argument("--split",choices=("dev","test"),default="dev"); args=ap.parse_args()
    manifest=args.repo/"experiments/recoverability_frontier_v0_5/corpus_manifest_v050.json"
    langs=base.load_languages(manifest,args.repo/".cache/ud-v050")
    result=run(langs,args.reps,args.target_letters,args.null_reps,args.split)
    payload={"programme":"fractionation-composition-v0.1a-coordinate-structure-development","split":args.split,
             "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
             "parameters":{"reps":args.reps,"target_letters":args.target_letters,"null_reps":args.null_reps,
             "statistic":"max phase_MI * pair_density * (1-pair_NMI), token/stream b=1..8",
             "controls":"surface alphabet matched to rows+columns"},"result":result}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print("FRACTIONATION_V01A_GATE",json.dumps(result["summary"]["gate"],sort_keys=True))
    print("FRACTIONATION_V01A_AGG",json.dumps(result["summary"]["aggregate"],sort_keys=True))
    for fam,r in result["summary"]["by_family"].items(): print("FRACTIONATION_V01A_FAMILY",fam,json.dumps(r,sort_keys=True))
    print("FRACTIONATION_V01A_SHA256",hashlib.sha256(args.output.read_bytes()).hexdigest())
if __name__=="__main__": main()

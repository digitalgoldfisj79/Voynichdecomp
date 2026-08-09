# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26,<2.2",
#   "numba>=0.60,<0.62",
#   "Unidecode>=1.3,<2",
# ]
# ///
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import re
import statistics
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit
from unidecode import unidecode

NS = "CIPHERCLOSEV1"
PLAIN_ALPH = "abcdefghilmnopqrstu"
SURF_ALPH = "acdefghiklmnopqrsty"
K = 19
assert len(PLAIN_ALPH) == len(set(PLAIN_ALPH)) == K
assert len(SURF_ALPH) == len(set(SURF_ALPH)) == K
P2I = {c: i for i, c in enumerate(PLAIN_ALPH)}
S2I = {c: i for i, c in enumerate(SURF_ALPH)}

RF_URL = "https://www.voynich.nu/data/RF1b-er.txt"
RF_SHA = "eb857a1f353b18983fbc25b954e1bbce227a26d99cefabfda9206ff9b57644d2"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.voynich.nu/transcr.html",
}

LANGS = ["latin", "italian", "german", "french", "greek", "hebrew", "arabic", "spanish"]
LM_URLS = {
    "latin": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-train.conllu",
    "italian": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu",
    "german": "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu",
    "french": "https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master/fr_gsd-ud-train.conllu",
    "greek": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-train.conllu",
    "hebrew": "https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/master/he_htb-ud-train.conllu",
    "arabic": "https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-train.conllu",
    "spanish": "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-train.conllu",
}
TRAIN_RES = {0, 1, 3, 4, 6, 8}
CTRL_RES = {2, 5, 7, 9}

TQ_RULES = ["TQ_REV", "TQ_LAST_FIRST", "TQ_SWAP_ENDS", "TQ_OUTSIDE_L", "TQ_OUTSIDE_R"]
NQ_RULES = ["NQ_L0", "NQ_L1", "NQ_L2", "NQ_L3", "NQ_R0", "NQ_R1", "NQ_R2", "NQ_R3", "NQ_MID_FLOOR", "NQ_MID_CEIL"]
ALL_RULES = ["ID"] + TQ_RULES + NQ_RULES


def family(rule: str) -> str:
    if rule == "ID":
        return "M0"
    if rule.startswith("TQ_"):
        return "TQ"
    if rule.startswith("NQ_"):
        return "NQ"
    raise ValueError(rule)


def stable_seed(*parts: object) -> int:
    h = hashlib.sha256("::".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFF


def get_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def norm_word(raw: str) -> list[int]:
    s = unidecode(raw).lower()
    s = s.replace("j", "i").replace("v", "u").replace("w", "u").replace("y", "i").replace("x", "s").replace("z", "s")
    out = [P2I[c] for c in s if c in P2I]
    return out


def parse_conllu(raw: bytes) -> list[list[list[int]]]:
    sentences: list[list[list[int]]] = []
    cur: list[list[int]] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line:
            if cur:
                sentences.append(cur)
                cur = []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) >= 2 and cols[0].isdigit():
            w = norm_word(cols[1])
            if w:
                cur.append(w)
    if cur:
        sentences.append(cur)
    return sentences


@dataclass
class LM:
    name: str
    logT: np.ndarray
    logS: np.ndarray
    logE: np.ndarray
    logU: np.ndarray
    plain_freq: np.ndarray
    control_words: list[list[int]]
    meta: dict[str, Any]


def build_lm(name: str, sentences: list[list[list[int]]]) -> LM:
    tr = [s for i, s in enumerate(sentences) if i % 10 in TRAIN_RES]
    ctrl = [s for i, s in enumerate(sentences) if i % 10 in CTRL_RES]
    a = 0.25
    B = np.full((K, K), a, dtype=np.float64)
    S = np.full(K, a, dtype=np.float64)
    E = np.full(K, a, dtype=np.float64)
    U = np.full(K, a, dtype=np.float64)
    letters = 0
    words = 0
    for sent in tr:
        for w in sent:
            if not w:
                continue
            words += 1
            letters += len(w)
            S[w[0]] += 1
            E[w[-1]] += 1
            for x in w:
                U[x] += 1
            for x, y in zip(w, w[1:]):
                B[x, y] += 1
    T = B / B.sum(axis=1, keepdims=True)
    S /= S.sum()
    E /= E.sum()
    U /= U.sum()
    cwords = [w for sent in ctrl for w in sent if w]
    return LM(name, np.log(T), np.log(S), np.log(E), np.log(U), U.copy(), cwords,
              {"sentences": len(sentences), "train_words": words, "train_letters": letters, "control_words": len(cwords), "control_letters": sum(map(len, cwords))})


def load_lms(smoke: bool = False) -> dict[str, LM]:
    names = LANGS if not smoke else ["latin", "german"]
    out = {}
    for name in names:
        raw = get_bytes(LM_URLS[name])
        out[name] = build_lm(name, parse_conllu(raw))
        print("LOAD_LM", name, json.dumps(out[name].meta, sort_keys=True), flush=True)
    return out


def source_span(words: list[list[int]], tag: str, fit_letters: int, hold_letters: int) -> tuple[list[list[int]], list[list[int]]]:
    if not words:
        raise RuntimeError("empty control pool")
    start = stable_seed(NS, "span", tag) % len(words)
    fit: list[list[int]] = []
    hold: list[list[int]] = []
    nf = nh = 0
    i = 0
    phase = 0
    guard = 0
    while nh < hold_letters:
        w = words[(start + i) % len(words)]
        i += 1
        guard += 1
        if guard > len(words) * 20:
            raise RuntimeError(("span exhausted", tag, nf, nh))
        if phase == 0:
            fit.append(w)
            nf += len(w)
            if nf >= fit_letters:
                phase = 1
        else:
            hold.append(w)
            nh += len(w)
    return fit, hold


def perm_order(rule: str, n: int) -> list[int]:
    if n <= 1 or rule == "ID":
        return list(range(n))
    if rule == "TQ_REV":
        return list(range(n - 1, -1, -1))
    if rule == "TQ_LAST_FIRST":
        return [n - 1] + list(range(n - 1))
    if rule == "TQ_SWAP_ENDS":
        if n == 2:
            return [1, 0]
        return [n - 1] + list(range(1, n - 1)) + [0]
    if rule == "TQ_OUTSIDE_L":
        out = []
        l, r = 0, n - 1
        while l <= r:
            out.append(l)
            if r != l:
                out.append(r)
            l += 1
            r -= 1
        return out
    if rule == "TQ_OUTSIDE_R":
        out = []
        l, r = 0, n - 1
        while l <= r:
            out.append(r)
            if l != r:
                out.append(l)
            l += 1
            r -= 1
        return out
    raise ValueError(rule)


def nq_slot(rule: str, plain_n: int) -> int:
    # Slot is in 0..plain_n. For n>=2 source-described insertion is kept interior.
    if plain_n <= 1:
        return plain_n
    lo, hi = 1, plain_n - 1
    if rule.startswith("NQ_L"):
        k = int(rule[-1])
        return min(hi, 1 + k)
    if rule.startswith("NQ_R"):
        k = int(rule[-1])
        return max(lo, plain_n - 1 - k)
    if rule == "NQ_MID_FLOOR":
        return min(hi, max(lo, plain_n // 2))
    if rule == "NQ_MID_CEIL":
        return min(hi, max(lo, (plain_n + 1) // 2))
    raise ValueError(rule)


def forward_words(words: list[list[int]], rule: str, plain_to_cipher: np.ndarray, tag: str) -> list[list[int]]:
    rng = np.random.default_rng(stable_seed(NS, "null", tag, rule))
    out: list[list[int]] = []
    for w0 in words:
        w = list(w0)
        if rule.startswith("TQ_"):
            order = perm_order(rule, len(w))
            w = [w[i] for i in order]
        elif rule.startswith("NQ_"):
            slot = nq_slot(rule, len(w))
            null_letter = int(rng.integers(0, K))
            w = w[:slot] + [null_letter] + w[slot:]
        out.append([int(plain_to_cipher[x]) for x in w])
    return out


def inverse_words(words: list[list[int]], rule: str) -> list[list[int]]:
    if rule == "ID":
        return [list(w) for w in words]
    out: list[list[int]] = []
    if rule.startswith("TQ_"):
        for w in words:
            order = perm_order(rule, len(w))
            inv = [0] * len(w)
            for outpos, src in enumerate(order):
                inv[src] = w[outpos]
            out.append(inv)
        return out
    if rule.startswith("NQ_"):
        for w in words:
            if not w:
                continue
            plain_n = max(0, len(w) - 1)
            slot = nq_slot(rule, plain_n)
            if slot >= len(w):
                slot = len(w) - 1
            q = w[:slot] + w[slot + 1:]
            if q:
                out.append(q)
        return out
    raise ValueError(rule)


@dataclass
class Stats:
    B: np.ndarray
    S: np.ndarray
    E: np.ndarray
    F: np.ndarray
    chars: int
    words: int


def make_stats(words: list[list[int]]) -> Stats:
    B = np.zeros((K, K), dtype=np.int64)
    S = np.zeros(K, dtype=np.int64)
    E = np.zeros(K, dtype=np.int64)
    F = np.zeros(K, dtype=np.int64)
    chars = 0
    nw = 0
    for w in words:
        if not w:
            continue
        nw += 1
        chars += len(w)
        S[w[0]] += 1
        E[w[-1]] += 1
        for x in w:
            F[x] += 1
        for x, y in zip(w, w[1:]):
            B[x, y] += 1
    return Stats(B, S, E, F, chars, nw)


@njit(cache=False, nogil=True)
def total_score(B, S, E, F, key, logT, logS, logE, logU):
    z = 0.0
    for i in range(K):
        pi = key[i]
        z += S[i] * logS[pi] + E[i] * logE[pi] + 0.15 * F[i] * logU[pi]
        for j in range(K):
            z += B[i, j] * logT[pi, key[j]]
    return z


@njit(cache=False, nogil=True)
def delta_swap(B, S, E, F, key, a, b, logT, logS, logE, logU):
    ka = key[a]
    kb = key[b]
    old = S[a] * logS[ka] + E[a] * logE[ka] + 0.15 * F[a] * logU[ka]
    old += S[b] * logS[kb] + E[b] * logE[kb] + 0.15 * F[b] * logU[kb]
    new = S[a] * logS[kb] + E[a] * logE[kb] + 0.15 * F[a] * logU[kb]
    new += S[b] * logS[ka] + E[b] * logE[ka] + 0.15 * F[b] * logU[ka]
    for j in range(K):
        kj = key[j]
        if j == a:
            kj_new = kb
        elif j == b:
            kj_new = ka
        else:
            kj_new = kj
        old += B[a, j] * logT[ka, kj] + B[b, j] * logT[kb, kj]
        new += B[a, j] * logT[kb, kj_new] + B[b, j] * logT[ka, kj_new]
    for i in range(K):
        if i == a or i == b:
            continue
        ki = key[i]
        old += B[i, a] * logT[ki, ka] + B[i, b] * logT[ki, kb]
        new += B[i, a] * logT[ki, kb] + B[i, b] * logT[ki, ka]
    return new - old


@njit(cache=False, nogil=True)
def rng_step(state):
    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    return state * np.uint64(2685821657736338717)


@njit(cache=False, nogil=True)
def rng_int(state, upper):
    state = rng_step(state)
    return state, int(state % np.uint64(upper))


@njit(cache=False, nogil=True)
def rng_float(state):
    state = rng_step(state)
    v = float(state >> np.uint64(11)) * (1.0 / 9007199254740992.0)
    return state, v


@njit(cache=False, nogil=True)
def anneal_one(B, S, E, F, initial, logT, logS, logE, logU, proposals, seed):
    key = initial.copy()
    score = total_score(B, S, E, F, key, logT, logS, logE, logU)
    best = score
    bestk = key.copy()
    state = np.uint64(seed if seed > 0 else 1)
    mean_abs = 0.0
    ns = 0
    for _ in range(64):
        state, a = rng_int(state, K)
        state, b = rng_int(state, K)
        if a == b:
            continue
        d = delta_swap(B, S, E, F, key, a, b, logT, logS, logE, logU)
        mean_abs += abs(d)
        ns += 1
    mean_abs /= max(1, ns)
    t0 = max(0.05, 2.5 * mean_abs)
    tend = max(0.0005, 0.01 * mean_abs)
    cool = math.exp(math.log(tend / t0) / max(1, proposals))
    temp = t0
    stagnant = 0
    for _ in range(proposals):
        state, a = rng_int(state, K)
        state, b = rng_int(state, K)
        if a == b:
            temp *= cool
            continue
        d = delta_swap(B, S, E, F, key, a, b, logT, logS, logE, logU)
        accept = d >= 0.0
        if not accept:
            state, u = rng_float(state)
            accept = u < math.exp(d / max(temp, 1e-12))
        if accept:
            tmp = key[a]
            key[a] = key[b]
            key[b] = tmp
            score += d
            if score > best + 1e-10:
                best = score
                bestk = key.copy()
                stagnant = 0
            else:
                stagnant += 1
        else:
            stagnant += 1
        temp *= cool
        if stagnant > 5000:
            temp = max(temp, 0.15 * t0)
            stagnant = 0
    # Deterministic greedy polish.
    key = bestk.copy()
    score = best
    improved = True
    sweeps = 0
    while improved and sweeps < 8:
        improved = False
        sweeps += 1
        for a in range(K - 1):
            for b in range(a + 1, K):
                d = delta_swap(B, S, E, F, key, a, b, logT, logS, logE, logU)
                if d > 1e-9:
                    tmp = key[a]
                    key[a] = key[b]
                    key[b] = tmp
                    score += d
                    improved = True
    return key, score


def freq_key(stats: Stats, lm: LM) -> np.ndarray:
    cr = sorted(range(K), key=lambda i: (-int(stats.F[i]), i))
    pr = sorted(range(K), key=lambda i: (-float(lm.plain_freq[i]), i))
    key = np.empty(K, dtype=np.int32)
    for c, p in zip(cr, pr):
        key[c] = p
    return key


def perturb(key: np.ndarray, tag: str, swaps: int) -> np.ndarray:
    out = key.copy()
    rng = np.random.default_rng(stable_seed(NS, "perturb", tag))
    for _ in range(swaps):
        a, b = rng.integers(0, K, 2)
        out[a], out[b] = out[b], out[a]
    return out


def map_agreement(k1: np.ndarray, k2: np.ndarray, freq: np.ndarray) -> float:
    den = max(1, int(freq.sum()))
    return float(freq[k1 == k2].sum() / den)


def solve(stats: Stats, lm: LM, tag: str, smoke: bool = False) -> dict[str, Any]:
    proposals = 3000 if smoke else 60000
    max_restarts = 4 if smoke else 16
    batch = 2 if smoke else 4
    base = freq_key(stats, lm)
    best_keys = [None, None]
    best_scores = [-1e300, -1e300]
    used = [0, 0]
    converged = False
    agreement = 0.0
    for end in range(batch, max_restarts + 1, batch):
        for ens in (0, 1):
            for rr in range(used[ens], end):
                if rr == 0:
                    init = base.copy()
                elif rr % 3 == 1 and best_keys[ens] is not None:
                    init = perturb(best_keys[ens], f"{tag}:{ens}:{rr}", 2 + rr % 5)
                else:
                    rng = np.random.default_rng(stable_seed(NS, "init", tag, ens, rr))
                    init = rng.permutation(K).astype(np.int32)
                key, sc = anneal_one(stats.B, stats.S, stats.E, stats.F, init, lm.logT, lm.logS, lm.logE, lm.logU,
                                      proposals, stable_seed(NS, "anneal", tag, ens, rr))
                if sc > best_scores[ens]:
                    best_scores[ens] = float(sc)
                    best_keys[ens] = key.copy()
            used[ens] = end
        assert best_keys[0] is not None and best_keys[1] is not None
        agreement = map_agreement(best_keys[0], best_keys[1], stats.F)
        diff = abs(best_scores[0] - best_scores[1]) / max(1, stats.chars)
        if diff <= 1e-7 and agreement >= 0.95:
            converged = True
            break
    win = 0 if best_scores[0] >= best_scores[1] else 1
    return {
        "key": best_keys[win],
        "fit_score": best_scores[win] / max(1, stats.chars),
        "ensemble_scores": [x / max(1, stats.chars) for x in best_scores],
        "agreement": agreement,
        "score_diff": abs(best_scores[0] - best_scores[1]) / max(1, stats.chars),
        "converged": converged,
        "restarts_each": used[0],
        "proposals_per_restart": proposals,
    }


def score_fixed(stats: Stats, lm: LM, key: np.ndarray) -> float:
    return float(total_score(stats.B, stats.S, stats.E, stats.F, key, lm.logT, lm.logS, lm.logE, lm.logU) / max(1, stats.chars))


def decode_words(words: list[list[int]], key: np.ndarray) -> list[list[int]]:
    return [[int(key[x]) for x in w] for w in words]


def flat_acc(truth: list[list[int]], pred: list[list[int]]) -> float:
    a = [x for w in truth for x in w]
    b = [x for w in pred for x in w]
    if len(a) != len(b):
        return 0.0
    if not a:
        return 1.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def make_control(lm: LM, rule: str, rep: int, stage: str, fit_letters: int, hold_letters: int) -> dict[str, Any]:
    fit_plain, hold_plain = source_span(lm.control_words, f"{stage}:{lm.name}:{rule}:{rep}", fit_letters, hold_letters)
    rng = np.random.default_rng(stable_seed(NS, "key", stage, lm.name, rule, rep))
    p2c = rng.permutation(K).astype(np.int32)
    inv = np.empty(K, dtype=np.int32)
    for p, c in enumerate(p2c):
        inv[int(c)] = p
    fit_cipher = forward_words(fit_plain, rule, p2c, f"{stage}:{lm.name}:{rep}:fit")
    hold_cipher = forward_words(hold_plain, rule, p2c, f"{stage}:{lm.name}:{rep}:hold")
    return {"fit_plain": fit_plain, "hold_plain": hold_plain, "fit_cipher": fit_cipher, "hold_cipher": hold_cipher, "true_inverse": inv}


def evaluate_known_rule(control: dict[str, Any], rule: str, lm: LM, tag: str, smoke: bool = False) -> dict[str, Any]:
    fit_inv = inverse_words(control["fit_cipher"], rule)
    hold_inv = inverse_words(control["hold_cipher"], rule)
    sol = solve(make_stats(fit_inv), lm, tag, smoke=smoke)
    hs = make_stats(hold_inv)
    hold_score = score_fixed(hs, lm, sol["key"])
    acc = flat_acc(control["hold_plain"], decode_words(hold_inv, sol["key"]))
    map_acc = float(np.mean(sol["key"] == control["true_inverse"]))
    return {
        "hold_score": hold_score,
        "recovery": acc,
        "map_accuracy": map_acc,
        "agreement": sol["agreement"],
        "converged": sol["converged"],
        "restarts_each": sol["restarts_each"],
        "score_diff": sol["score_diff"],
    }


def percentile5(vals: list[float]) -> float:
    return float(np.quantile(np.asarray(vals, dtype=float), 0.05, method="linear"))


def q1_run(lms: dict[str, LM], smoke: bool) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rules = ALL_RULES if not smoke else ["ID", "TQ_REV", "NQ_L0"]
    rows = []
    for rule in rules:
        extra_pool = [x for x in lms if x != "latin"]
        extra = extra_pool[stable_seed(NS, "q1extra", rule) % len(extra_pool)] if extra_pool else "latin"
        trials = [("latin", 0), ("latin", 1), (extra, 2)] if not smoke else [("latin", 0)]
        for lang, rep in trials:
            ctl = make_control(lms[lang], rule, rep, "Q1", 768 if not smoke else 192, 768 if not smoke else 192)
            r = evaluate_known_rule(ctl, rule, lms[lang], f"Q1:{rule}:{lang}:{rep}", smoke=smoke)
            r.update(rule=rule, family=family(rule), language=lang, replicate=rep)
            rows.append(r)
            print("Q1_ROW", json.dumps(r, sort_keys=True), flush=True)
    fam_gate = {}
    for fam in ("M0", "TQ", "NQ"):
        z = [r for r in rows if r["family"] == fam]
        if not z:
            fam_gate[fam] = False
            continue
        byrule = collections.defaultdict(list)
        for r in z:
            byrule[r["rule"]].append(r)
        ok = True
        for rr, q in byrule.items():
            rec = [x["recovery"] for x in q]
            ok = ok and statistics.median(rec) >= 0.95 and min(rec) >= 0.85
            ok = ok and all(x["agreement"] >= 0.90 and x["converged"] for x in q)
        expected = 1 if fam == "M0" else (len(TQ_RULES) if fam == "TQ" else len(NQ_RULES))
        ok = ok and len(byrule) == (1 if smoke else expected)
        fam_gate[fam] = bool(ok)
    return rows, fam_gate


def q2_run(lms: dict[str, LM], fam_gate: dict[str, bool], smoke: bool, workers: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fams = [f for f in ("M0", "TQ", "NQ") if fam_gate.get(f)]
    if smoke:
        fams = [f for f in fams if f in ("M0", "TQ", "NQ")]
    jobs = []
    langs = list(lms)
    reps = range(1 if smoke else 3)
    for fam in fams:
        for li, lang in enumerate(langs):
            for rep in reps:
                if fam == "M0":
                    rule = "ID"
                elif fam == "TQ":
                    rule = TQ_RULES[(li * 3 + rep) % len(TQ_RULES)]
                else:
                    rule = NQ_RULES[(li * 3 + rep) % len(NQ_RULES)]
                jobs.append((fam, lang, rule, rep))

    def one(job):
        fam, lang, rule, rep = job
        ctl = make_control(lms[lang], rule, rep, "Q2", 768 if not smoke else 192, 768 if not smoke else 192)
        fit_inv = inverse_words(ctl["fit_cipher"], rule)
        hold_inv = inverse_words(ctl["hold_cipher"], rule)
        fs = make_stats(fit_inv)
        hs = make_stats(hold_inv)
        ranked = []
        sols = {}
        for cand in langs:
            sol = solve(fs, lms[cand], f"Q2:{fam}:{lang}:{rule}:{rep}:{cand}", smoke=smoke)
            sc = score_fixed(hs, lms[cand], sol["key"])
            ranked.append((cand, sc))
            sols[cand] = sol
        ranked.sort(key=lambda x: (-x[1], x[0]))
        true_sol = sols[lang]
        acc = flat_acc(ctl["hold_plain"], decode_words(hold_inv, true_sol["key"]))
        true_score = next(sc for la, sc in ranked if la == lang)
        rank = 1 + next(i for i, (la, _sc) in enumerate(ranked) if la == lang)
        runner = max(sc for la, sc in ranked if la != lang) if len(ranked) > 1 else -1e300
        return {"family": fam, "language": lang, "rule": rule, "replicate": rep, "true_score": true_score,
                "rank": rank, "margin": true_score - runner, "recovery": acc, "agreement": true_sol["agreement"],
                "converged": true_sol["converged"], "top_language": ranked[0][0], "top_score": ranked[0][1]}

    rows = []
    if workers <= 1:
        for j in jobs:
            r = one(j); rows.append(r); print("Q2_ROW", json.dumps(r, sort_keys=True), flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, j) for j in jobs]
            for fut in concurrent.futures.as_completed(futs):
                r = fut.result(); rows.append(r); print("Q2_ROW", json.dumps(r, sort_keys=True), flush=True)
    rows.sort(key=lambda r: (r["family"], r["language"], r["replicate"]))
    out = {}
    for fam in fams:
        z = [r for r in rows if r["family"] == fam]
        if smoke:
            pass_gate = bool(z) and all(r["converged"] for r in z)
        else:
            lang_correct = {la: sum(r["rank"] == 1 for r in z if r["language"] == la) for la in langs}
            pass_gate = (len(z) == 24 and all(r["converged"] for r in z) and
                         statistics.median(r["recovery"] for r in z) >= 0.95 and min(r["recovery"] for r in z) >= 0.80 and
                         sum(r["rank"] == 1 for r in z) >= 22 and all(v >= 2 for v in lang_correct.values()) and
                         statistics.median(r["margin"] for r in z) >= 0.05)
        floors = {}
        for la in langs:
            vals = [r["true_score"] for r in z if r["language"] == la]
            if vals:
                floors[la] = percentile5(vals)
        out[fam] = {"pass": bool(pass_gate), "floors": floors,
                    "median_recovery": statistics.median([r["recovery"] for r in z]) if z else None,
                    "minimum_recovery": min([r["recovery"] for r in z]) if z else None,
                    "correct_ranks": sum(r["rank"] == 1 for r in z), "trials": len(z),
                    "median_margin": statistics.median([r["margin"] for r in z]) if z else None}
    return rows, out


def parse_rf() -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    b = get_bytes(RF_URL, HEADERS)
    if hashlib.sha256(b).hexdigest() != RF_SHA:
        raise RuntimeError("RF hash mismatch")
    pages: dict[str, list[list[int]]] = collections.defaultdict(list)
    total_alpha = 0
    retained_alpha = 0
    raw_words = 0
    retained_words = 0
    uncertain_words = 0
    rare_words = 0
    for line in b.decode("utf-8", "replace").splitlines():
        if not line.startswith("<") or ">" not in line:
            continue
        lab, rhs = line.split(">", 1)
        if "." not in lab or "<!" in rhs:
            continue
        page = lab[1:].split(".", 1)[0]
        rhs = re.sub(r"<(?:-|~)>", ".", rhs)
        rhs = re.sub(r"<[^>]*>", ".", rhs)
        rhs = rhs.replace(",", "")
        for raww in rhs.split("."):
            raww = raww.strip()
            if not raww:
                continue
            raw_words += 1
            letters = [c for c in raww.lower() if "a" <= c <= "z"]
            total_alpha += len(letters)
            if "[" in raww or "]" in raww or "?" in raww:
                uncertain_words += 1
                continue
            clean = raww.replace("{", "").replace("}", "")
            chars = [c for c in clean.lower() if "a" <= c <= "z"]
            if not chars:
                continue
            if any(c not in S2I for c in chars):
                rare_words += 1
                continue
            w = [S2I[c] for c in chars]
            pages[page].append(w)
            retained_words += 1
            retained_alpha += len(w)
    meta = {"sha256": RF_SHA, "pages": len(pages), "raw_words": raw_words, "retained_words": retained_words,
            "total_alpha": total_alpha, "retained_alpha": retained_alpha, "coverage": retained_alpha / max(1, total_alpha),
            "uncertain_words": uncertain_words, "rare_words": rare_words}
    return dict(pages), meta


def split_pages(pages: dict[str, list[list[int]]]) -> tuple[list[str], list[str], list[str]]:
    fs = sorted(pages, key=lambda f: hashlib.sha256(f"{NS}split::{f}".encode()).digest())
    n = len(fs)
    nT = int(round(0.60 * n))
    nH = int(round(0.20 * n))
    return fs[:nT], fs[nT:nT+nH], fs[nT+nH:]


def combine_pages(pages: dict[str, list[list[int]]], fs: list[str]) -> list[list[int]]:
    return [w for f in fs for w in pages[f]]


def load_qual(url: str) -> dict[str, Any]:
    return json.loads(get_bytes(url).decode("utf-8"))


def target_run(lms: dict[str, LM], qual: dict[str, Any], smoke: bool, workers: int) -> dict[str, Any]:
    pages, meta = parse_rf()
    print("TARGET_CENSUS", json.dumps(meta, sort_keys=True), flush=True)
    T, H, C = split_pages(pages)
    if meta["coverage"] < 0.995:
        return {"verdict": "REPRESENTATION_COVERAGE_FAIL", "meta": meta, "T_folios": len(T), "H_folios": len(H), "C_folios": len(C)}
    tw = combine_pages(pages, T)
    hw = combine_pages(pages, H)
    cw = combine_pages(pages, C)
    permitted = []
    for fam in ("M0", "TQ", "NQ"):
        q1ok = bool(qual.get("q1_family_gate", {}).get(fam))
        q2ok = bool(qual.get("q2", {}).get(fam, {}).get("pass"))
        if q1ok and q2ok:
            permitted.append(fam)
    rules = [r for r in ALL_RULES if family(r) in permitted]
    if smoke:
        rules = [r for r in rules if r in ("ID", "TQ_REV", "NQ_L0")]
    langs = list(lms)
    jobs = [(rule, la) for rule in rules for la in langs]

    def one(job):
        rule, la = job
        ti = inverse_words(tw, rule)
        hi = inverse_words(hw, rule)
        sol = solve(make_stats(ti), lms[la], f"TARGET:{rule}:{la}", smoke=smoke)
        sc = score_fixed(make_stats(hi), lms[la], sol["key"])
        floor = qual["q2"][family(rule)]["floors"][la]
        return {"rule": rule, "family": family(rule), "language": la, "H_score": sc, "floor": floor,
                "evidence": sc - floor, "agreement": sol["agreement"], "converged": sol["converged"],
                "restarts_each": sol["restarts_each"], "score_diff": sol["score_diff"]}

    rows = []
    if workers <= 1:
        for j in jobs:
            r = one(j); rows.append(r); print("TARGET_ROW", json.dumps(r, sort_keys=True), flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, j) for j in jobs]
            for fut in concurrent.futures.as_completed(futs):
                r = fut.result(); rows.append(r); print("TARGET_ROW", json.dumps(r, sort_keys=True), flush=True)
    rows.sort(key=lambda r: (r["family"], -r["evidence"], r["rule"], r["language"]))
    decisions = {}
    for fam in permitted:
        z = [r for r in rows if r["family"] == fam]
        z.sort(key=lambda r: (-r["H_score"], r["rule"], r["language"]))
        best = z[0]
        if best["converged"] and best["H_score"] < best["floor"]:
            verdict = "CLOSED_NEGATIVE_INCOMPATIBLE_V1"
        elif not best["converged"]:
            verdict = "UNRESOLVED_SEARCH"
        else:
            verdict = "REACHES_POSITIVE_FLOOR_Q3_REQUIRED"
        decisions[fam] = {"verdict": verdict, "best": best,
                          "runner_up": z[1] if len(z) > 1 else None}
    return {"meta": meta, "T_folios": len(T), "H_folios": len(H), "C_folios": len(C),
            "T_chars": sum(len(w) for w in tw), "H_chars": sum(len(w) for w in hw), "C_chars": sum(len(w) for w in cw),
            "permitted_families": permitted, "decisions": decisions, "rows": rows}


def canonical_sha(payload: dict[str, Any]) -> str:
    x = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(x).hexdigest()


def precompile():
    B = np.ones((K, K), dtype=np.int64)
    S = E = F = np.ones(K, dtype=np.int64)
    key = np.arange(K, dtype=np.int32)
    lp = np.log(np.ones((K, K), dtype=np.float64) / K)
    lv = np.log(np.ones(K, dtype=np.float64) / K)
    anneal_one(B, S, E, F, key, lp, lv, lv, lv, 2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "qualify", "target"])
    ap.add_argument("--qual-url")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    smoke = args.mode == "smoke"
    lms = load_lms(smoke=smoke)
    precompile()
    if args.mode in ("smoke", "qualify"):
        q1, q1gate = q1_run(lms, smoke=smoke)
        q2rows, q2 = q2_run(lms, q1gate, smoke=smoke, workers=args.workers)
        payload = {"programme": "terminal-cipher-v1", "mode": args.mode, "namespace": NS,
                   "q1_rows": q1, "q1_family_gate": q1gate, "q2_rows": q2rows, "q2": q2}
        payload["scientific_sha256"] = canonical_sha(payload)
        print("TERMINAL_V1_RESULT=" + json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)
        return
    if not args.qual_url:
        raise SystemExit("--qual-url required")
    qual = load_qual(args.qual_url)
    payload = target_run(lms, qual, smoke=False, workers=args.workers)
    payload["programme"] = "terminal-cipher-v1"
    payload["mode"] = "target"
    payload["namespace"] = NS
    payload["scientific_sha256"] = canonical_sha(payload)
    print("TERMINAL_V1_TARGET=" + json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

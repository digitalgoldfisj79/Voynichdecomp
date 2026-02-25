#!/usr/bin/env python3
"""
FULL REPRODUCIBILITY SCRIPT
============================
Reproduces all generator hierarchy results from scratch.

Requirements:
    pip install numpy scipy

Data dependencies (auto-downloaded from GitHub):
    - enriched_records.pkl  (37,465 parsed VMS tokens)
    - p70c_full_spec_v1.json (6,750 PGCS quad entries)

Usage:
    python reproduce_all.py              # Full run (generates + scores + tables)
    python reproduce_all.py --skip-gen   # Skip generation, use cached results
    python reproduce_all.py --samples    # Also generate 200-word text samples

Outputs:
    results/hierarchy_85_results.pkl     # Complete results (pickle)
    results/vms_baseline_85metrics.pkl   # VMS baseline metrics (pickle)
    results/generator_samples_200.pkl    # 200-word samples per generator (pickle)
    results/running_results.md           # Human-readable summary
    results/S_TABLE_COMPLETE_METRICS.md  # Full supplement table

Author: Edward Bozzard, 2026
Computational assistant: Claude (Anthropic)
"""

import os, sys, argparse, pickle, json, time, urllib.request
import numpy as np
from collections import Counter, defaultdict

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = "data"
RESULTS_DIR = "results"
CACHE_DIR = "cache"

GITHUB_BASE = "https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main"
DATA_FILES = {
    "enriched_records.pkl": f"{GITHUB_BASE}/enriched_records.pkl",
    "p70c_full_spec_v1.json": f"{GITHUB_BASE}/Paper/p70c_full_spec_v1.json",
}

SEED = 42
N_SEEDS = 10       # independent seeds per generator
N_TARGET = 37465   # VMS token count


# ==============================================================================
# DATA ACQUISITION
# ==============================================================================

def ensure_dirs():
    for d in [DATA_DIR, RESULTS_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)


def fetch_data():
    """Download data files from GitHub if not present locally."""
    for fname, url in DATA_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            print(f"  Found: {path}")
            continue
        print(f"  Downloading: {fname}...")
        urllib.request.urlretrieve(url, path)
        print(f"  Saved: {path} ({os.path.getsize(path)} bytes)")


def load_data():
    """Load enriched records, P70C spec, and build VMS structures."""
    with open(os.path.join(DATA_DIR, "enriched_records.pkl"), "rb") as f:
        records = pickle.load(f)
    with open(os.path.join(DATA_DIR, "p70c_full_spec_v1.json")) as f:
        spec = json.load(f)

    tokens = [r["token"] for r in records]

    # Build VMS lines from folio/line_no
    lines = []
    cur_line = []
    cur_key = (records[0]["folio"], records[0]["line_no"])
    for r in records:
        key = (r["folio"], r["line_no"])
        if key != cur_key:
            if cur_line:
                lines.append(cur_line)
            cur_line = []
            cur_key = key
        cur_line.append(r["token"])
    if cur_line:
        lines.append(cur_line)

    # f1r seed tokens
    f1r_tokens = [r["token"] for r in records if r["folio"] == "f1r"]

    print(f"  VMS: {len(tokens)} tokens, {len(lines)} lines, f1r: {len(f1r_tokens)} tokens")
    return records, spec, tokens, lines, f1r_tokens


# ==============================================================================
# GENERATOR 1: Character Bigram  (cf. Timm & Schinner 2020)
# ==============================================================================

def build_char_bigram_model(tokens, seed_tokens):
    """Build char→char follower table from seed tokens."""
    followers = defaultdict(list)
    for tok in seed_tokens:
        for i in range(len(tok) - 1):
            followers[tok[i]].append(tok[i + 1])
        followers["$START"].append(tok[0])
        followers[tok[-1]].append("$END")
    return followers


def gen_char_bigram(followers, n_tokens, rng, min_len=1, max_len=12):
    """Generate tokens using character bigram model."""
    corpus = []
    for _ in range(n_tokens):
        tok = []
        ch = rng.choice(followers.get("$START", ["a"]))
        for _ in range(max_len):
            tok.append(ch)
            nexts = followers.get(ch, [])
            if not nexts:
                break
            ch = rng.choice(nexts)
            if ch == "$END":
                break
        word = "".join(tok)
        if len(word) >= min_len:
            corpus.append(word)
        else:
            corpus.append(rng.choice(list(followers.get("$START", ["a"]))))
    return corpus[:n_tokens]


# ==============================================================================
# GENERATOR 2: Scribal Ductus
# ==============================================================================

DUCTUS_GROUPS = {
    "bench": ["ch", "sh", "e", "ee", "eee"],
    "gallows": ["k", "t", "f", "p"],
    "loop": ["o", "a", "y"],
    "lig": ["d", "s", "r", "l", "n", "q", "i"],
}


def _tokenize_to_segments(tok):
    """Break EVA token into calligraphic segments."""
    segments = []
    i = 0
    while i < len(tok):
        found = False
        for length in [3, 2]:
            if i + length <= len(tok):
                seg = tok[i:i+length]
                if seg in ("eee", "ch", "sh", "ee", "qo"):
                    segments.append(seg)
                    i += length
                    found = True
                    break
        if not found:
            segments.append(tok[i])
            i += 1
    return segments


def build_ductus_model(tokens):
    """Build calligraphic-group bigram model."""
    char_to_group = {}
    for g, chars in DUCTUS_GROUPS.items():
        for ch in chars:
            char_to_group[ch] = g

    group_followers = defaultdict(list)
    char_exemplars = defaultdict(list)

    for tok in tokens:
        segments = _tokenize_to_segments(tok)
        for i in range(len(segments) - 1):
            g1 = char_to_group.get(segments[i], "lig")
            g2 = char_to_group.get(segments[i+1], "lig")
            group_followers[g1].append(g2)
        if segments:
            group_followers["$START"].append(char_to_group.get(segments[0], "lig"))
            group_followers[char_to_group.get(segments[-1], "lig")].append("$END")
        for s in segments:
            g = char_to_group.get(s, "lig")
            char_exemplars[g].append(s)

    return group_followers, char_exemplars


def gen_ductus(group_followers, char_exemplars, n_tokens, rng, max_segs=8):
    """Generate tokens using ductus group transitions."""
    corpus = []
    for _ in range(n_tokens):
        segs = []
        g = rng.choice(group_followers.get("$START", ["loop"]))
        for _ in range(max_segs):
            chars = char_exemplars.get(g, ["e"])
            segs.append(rng.choice(chars))
            nexts = group_followers.get(g, [])
            if not nexts:
                break
            g = rng.choice(nexts)
            if g == "$END":
                break
        word = "".join(segs)
        if word:
            corpus.append(word)
    return corpus[:n_tokens]


# ==============================================================================
# P70C LEDGER (shared by Generators 3–6)
# ==============================================================================

def build_p70c_ledger(spec, records):
    """Build the P70C generation ledger from spec and records."""
    entries = spec["entries"]

    prefixes = Counter(r["prefix"] for r in records)
    gallows = Counter(r["gallows"] for r in records)
    mcores = Counter(r["m_core"] for r in records)
    sfx_fams = Counter(r["sfx_fam"] for r in records)

    # Entry index
    entry_index = {}
    for e in entries:
        key = (e["prefix"], e["gallows"], e["m_core"], e["sfx_fam"])
        if key not in entry_index:
            entry_index[key] = e
        else:
            entry_index[key]["count"] += e["count"]

    # Per-section profiles (sorted for reproducibility)
    section_profiles = {}
    for sec in sorted(set(r["section"] for r in records)):
        sec_recs = [r for r in records if r["section"] == sec]
        section_profiles[sec] = {
            "n_tokens": len(sec_recs),
            "prefix_dist": Counter(r["prefix"] for r in sec_recs),
            "gallows_dist": Counter(r["gallows"] for r in sec_recs),
            "mcore_dist": Counter(r["m_core"] for r in sec_recs),
            "sfxfam_dist": Counter(r["sfx_fam"] for r in sec_recs),
        }

    # Currier A/B
    currier_A = {"Herbal-A", "Pharmaceutical", "Astronomical", "Rosettes"}
    currier_B = {"Herbal-B", "Stars", "Balneological", "Cosmological", "Zodiac"}

    currier_profiles = {}
    for label, sections in [("A", currier_A), ("B", currier_B)]:
        recs = [r for r in records if r["section"] in sections]
        currier_profiles[label] = {
            "n_tokens": len(recs),
            "prefix_dist": Counter(r["prefix"] for r in recs),
            "gallows_dist": Counter(r["gallows"] for r in recs),
            "mcore_dist": Counter(r["m_core"] for r in recs),
            "sfxfam_dist": Counter(r["sfx_fam"] for r in recs),
        }

    # Token reconstruction lookup
    slot_to_tokens = defaultdict(list)
    token_to_slots = {}
    for r in records:
        key = (r["prefix"], r["gallows"], r["m_core"], r["sfx_fam"])
        slot_to_tokens[key].append(r["token"])
        if r["token"] not in token_to_slots:
            token_to_slots[r["token"]] = key

    return {
        "entries": entries,
        "entry_index": entry_index,
        "prefixes": prefixes,
        "gallows": gallows,
        "mcores": mcores,
        "sfx_fams": sfx_fams,
        "section_profiles": section_profiles,
        "currier_profiles": currier_profiles,
        "slot_to_tokens": slot_to_tokens,
        "token_to_slots": token_to_slots,
        "all_tokens": [r["token"] for r in records],
    }


def _weighted_choice(dist, rng, concentration=1.0):
    items = list(dist.items())
    if not items:
        return ""
    weights = np.array([c ** concentration for _, c in items], dtype=float)
    weights /= weights.sum()
    idx = rng.choices(range(len(items)), weights=weights.tolist(), k=1)[0]
    return items[idx][0]


SUFFIX_MAP = {
    "Y": ["y", "ey", "dy", "edy", "eedy"],
    "N": ["n", "en", "an", "in", "ain", "aiin"],
    "DY": ["dy", "ody", "edy"],
    "EDY": ["edy", "eedy"],
    "IN": ["in", "iin", "iiin", "ain", "aiin"],
    "AIN": ["ain", "aiin", "aiiin"],
    "OL": ["ol", "al"],
    "EY": ["ey", "eey"],
    "AR": ["ar", "or"],
    "AL": ["al", "ol"],
    "AN": ["an", "am"],
    "AM": ["am"],
    "AIIN": ["aiin", "aiiin"],
    "AIIIN": ["aiiin"],
    "BARE": [""],
    "EE": ["ee", "eee"],
    "S": ["s"],
    "R": ["r"],
    "L": ["l"],
}


def _reconstruct_token(prefix, gallows_val, m_core, sfx_fam,
                       slot_to_tokens, rng, novelty_rate=0.13):
    key = (prefix, gallows_val, m_core, sfx_fam)
    candidates = slot_to_tokens.get(key, [])

    if candidates and rng.random() > novelty_rate:
        return rng.choice(candidates)

    sfx_options = SUFFIX_MAP.get(sfx_fam, [""])
    sfx = rng.choice(sfx_options)
    g = gallows_val if gallows_val != "∅" else ""
    p = prefix if prefix != "∅" else ""
    mc = m_core if m_core != "∅" else ""
    tok = p + g + mc + sfx
    return tok if tok else "daiin"


# ==============================================================================
# GENERATOR 3: P70C Single Ledger
# ==============================================================================

def gen_p70c_single(ledger, n_tokens, rng, copy_rate=0.20, modify_rate=0.50,
                    create_rate=0.30, seed_tokens=None):
    corpus = list(seed_tokens) if seed_tokens else []
    slot_to_tokens = ledger["slot_to_tokens"]
    entry_list = ledger["entries"]
    token_to_slots = ledger["token_to_slots"]

    entry_weights = [e["count"] for e in entry_list]
    total_w = sum(entry_weights)
    entry_probs = [w / total_w for w in entry_weights]

    slot_dists = {
        "prefix": ledger["prefixes"],
        "gallows": ledger["gallows"],
        "m_core": ledger["mcores"],
        "sfx_fam": ledger["sfx_fams"],
    }

    while len(corpus) < n_tokens:
        r = rng.random()
        if r < copy_rate and corpus:
            corpus.append(rng.choice(corpus))
        elif r < copy_rate + modify_rate and corpus:
            base = rng.choice(corpus[-200:]) if len(corpus) > 200 else rng.choice(corpus)
            slots = token_to_slots.get(base)
            if slots is None:
                idx = rng.choices(range(len(entry_list)), weights=entry_probs, k=1)[0]
                entry = entry_list[idx]
                slots = (entry["prefix"], entry["gallows"], entry["m_core"], entry["sfx_fam"])
            p, g, mc, sf = slots
            if rng.random() < 0.5:
                slot = rng.choice(["m_core", "sfx_fam"])
            else:
                slot = rng.choice(["prefix", "gallows"])
            new_val = _weighted_choice(slot_dists[slot], rng)
            new_p = new_val if slot == "prefix" else p
            new_g = new_val if slot == "gallows" else g
            new_mc = new_val if slot == "m_core" else mc
            new_sf = new_val if slot == "sfx_fam" else sf
            tok = _reconstruct_token(new_p, new_g, new_mc, new_sf, slot_to_tokens, rng)
            corpus.append(tok)
        else:
            idx = rng.choices(range(len(entry_list)), weights=entry_probs, k=1)[0]
            entry = entry_list[idx]
            tok = _reconstruct_token(
                entry["prefix"], entry["gallows"], entry["m_core"], entry["sfx_fam"],
                slot_to_tokens, rng)
            corpus.append(tok)

    return corpus[:n_tokens]


# ==============================================================================
# GENERATOR 4: P70C Dual A+B
# ==============================================================================

def gen_p70c_dual(ledger, n_tokens, rng, copy_rate=0.20, modify_rate=0.50,
                  create_rate=0.30):
    corpus = []
    slot_to_tokens = ledger["slot_to_tokens"]
    token_to_slots = ledger["token_to_slots"]
    entry_list = ledger["entries"]

    currier_A_secs = {"Herbal-A", "Pharmaceutical", "Astronomical", "Rosettes"}

    global_weights = np.array([e["count"] for e in entry_list], dtype=float)
    global_probs = global_weights / global_weights.sum()

    currier_entry_probs = {}
    currier_slot_dists = {}
    for currier_label in ["A", "B"]:
        cur_prof = ledger["currier_profiles"][currier_label]
        cur_total = cur_prof["n_tokens"]
        cur_weights = np.zeros(len(entry_list))
        for i, e in enumerate(entry_list):
            p_freq = cur_prof["prefix_dist"].get(e["prefix"], 0) / cur_total
            g_freq = cur_prof["gallows_dist"].get(e["gallows"], 0) / cur_total
            mc_freq = cur_prof["mcore_dist"].get(e["m_core"], 0) / cur_total
            sf_freq = cur_prof["sfxfam_dist"].get(e["sfx_fam"], 0) / cur_total
            cur_weights[i] = e["count"] * ((p_freq + 0.001) * (g_freq + 0.001) *
                                            (mc_freq + 0.001) * (sf_freq + 0.001)) ** 0.25
        cur_probs = cur_weights / cur_weights.sum()
        blended = 0.70 * global_probs + 0.30 * cur_probs
        blended /= blended.sum()
        currier_entry_probs[currier_label] = blended.tolist()
        currier_slot_dists[currier_label] = {
            "prefix": cur_prof["prefix_dist"],
            "gallows": cur_prof["gallows_dist"],
            "m_core": cur_prof["mcore_dist"],
            "sfx_fam": cur_prof["sfxfam_dist"],
        }

    sec_proportions = {}
    total = sum(p["n_tokens"] for p in ledger["section_profiles"].values())
    for sec, prof in ledger["section_profiles"].items():
        sec_proportions[sec] = prof["n_tokens"] / total

    for sec, prop in sorted(sec_proportions.items()):
        sec_n = int(prop * n_tokens)
        if sec_n < 10:
            continue
        currier = "A" if sec in currier_A_secs else "B"
        entry_probs = currier_entry_probs[currier]
        s_dists = currier_slot_dists[currier]

        sec_corpus = []
        while len(sec_corpus) < sec_n:
            r = rng.random()
            if r < copy_rate and sec_corpus:
                sec_corpus.append(rng.choice(sec_corpus))
            elif r < copy_rate + modify_rate and sec_corpus:
                base = rng.choice(sec_corpus[-200:]) if len(sec_corpus) > 200 else rng.choice(sec_corpus)
                slots = token_to_slots.get(base)
                if slots is None:
                    idx = rng.choices(range(len(entry_list)), weights=entry_probs, k=1)[0]
                    entry = entry_list[idx]
                    slots = (entry["prefix"], entry["gallows"], entry["m_core"], entry["sfx_fam"])
                p, g, mc, sf = slots
                slot = rng.choice(["prefix", "gallows", "m_core", "sfx_fam"])
                new_val = _weighted_choice(s_dists[slot], rng)
                new_p = new_val if slot == "prefix" else p
                new_g = new_val if slot == "gallows" else g
                new_mc = new_val if slot == "m_core" else mc
                new_sf = new_val if slot == "sfx_fam" else sf
                tok = _reconstruct_token(new_p, new_g, new_mc, new_sf, slot_to_tokens, rng)
                sec_corpus.append(tok)
            else:
                idx = rng.choices(range(len(entry_list)), weights=entry_probs, k=1)[0]
                entry = entry_list[idx]
                tok = _reconstruct_token(
                    entry["prefix"], entry["gallows"], entry["m_core"], entry["sfx_fam"],
                    slot_to_tokens, rng)
                sec_corpus.append(tok)
        corpus.extend(sec_corpus[:sec_n])

    while len(corpus) < n_tokens:
        corpus.append(rng.choice(corpus))
    rng.shuffle(corpus)
    return corpus[:n_tokens]


# ==============================================================================
# GENERATOR 5: P70C Section-Profiled
# ==============================================================================

SECTION_CONCENTRATION = {
    "Herbal-A": 1.0, "Herbal-B": 1.0, "Stars": 1.2,
    "Balneological": 1.5, "Pharmaceutical": 1.0,
    "Astronomical": 1.1, "Cosmological": 1.1,
    "Zodiac": 1.1, "Rosettes": 1.1,
}

SECTION_COPY_RATES = {
    "Herbal-A": 0.20, "Herbal-B": 0.20, "Stars": 0.20,
    "Balneological": 0.30, "Pharmaceutical": 0.20,
    "Astronomical": 0.20, "Cosmological": 0.20,
    "Zodiac": 0.20, "Rosettes": 0.20,
}

SECTION_FLAT_RATES = {
    "Herbal-A": 0.40, "Herbal-B": 0.40, "Stars": 0.35,
    "Balneological": 0.25, "Pharmaceutical": 0.45,
    "Astronomical": 0.40, "Cosmological": 0.35,
    "Zodiac": 0.35, "Rosettes": 0.40,
}


def gen_p70c_section_profiled(ledger, n_tokens, rng):
    slot_to_tokens = ledger["slot_to_tokens"]
    token_to_slots = ledger["token_to_slots"]
    entry_list = ledger["entries"]

    global_weights = np.array([e["count"] for e in entry_list], dtype=float)
    global_probs = global_weights / global_weights.sum()

    sec_proportions = {}
    total = sum(p["n_tokens"] for p in ledger["section_profiles"].values())
    for sec, prof in ledger["section_profiles"].items():
        sec_proportions[sec] = prof["n_tokens"] / total

    all_corpus = []
    for sec, prop in sorted(sec_proportions.items()):
        sec_n = int(prop * n_tokens)
        if sec_n < 10:
            continue
        conc = SECTION_CONCENTRATION.get(sec, 1.0)
        copy_r = SECTION_COPY_RATES.get(sec, 0.20)
        flat_r = SECTION_FLAT_RATES.get(sec, 0.40)
        modify_r = 1.0 - copy_r - flat_r

        sec_prof = ledger["section_profiles"][sec]
        sec_total = sec_prof["n_tokens"]

        sec_weights = np.zeros(len(entry_list))
        for i, e in enumerate(entry_list):
            p_f = (sec_prof["prefix_dist"].get(e["prefix"], 0) / sec_total) ** conc
            g_f = (sec_prof["gallows_dist"].get(e["gallows"], 0) / sec_total) ** conc
            mc_f = (sec_prof["mcore_dist"].get(e["m_core"], 0) / sec_total) ** conc
            sf_f = (sec_prof["sfxfam_dist"].get(e["sfx_fam"], 0) / sec_total) ** conc
            sec_weights[i] = e["count"] * ((p_f + 0.0001) * (g_f + 0.0001) *
                                            (mc_f + 0.0001) * (sf_f + 0.0001)) ** 0.25
        if sec_weights.sum() > 0:
            sec_probs = sec_weights / sec_weights.sum()
            blended = 0.60 * global_probs + 0.40 * sec_probs
            blended /= blended.sum()
            entry_probs = blended.tolist()
        else:
            entry_probs = global_probs.tolist()

        s_dists = {
            "prefix": sec_prof["prefix_dist"],
            "gallows": sec_prof["gallows_dist"],
            "m_core": sec_prof["mcore_dist"],
            "sfx_fam": sec_prof["sfxfam_dist"],
        }

        sec_corpus = []
        while len(sec_corpus) < sec_n:
            r = rng.random()
            if r < copy_r and sec_corpus:
                sec_corpus.append(rng.choice(sec_corpus))
            elif r < copy_r + modify_r and sec_corpus:
                base = rng.choice(sec_corpus[-200:]) if len(sec_corpus) > 200 else rng.choice(sec_corpus)
                slots = token_to_slots.get(base)
                if slots is None:
                    idx = rng.choices(range(len(entry_list)), weights=entry_probs, k=1)[0]
                    entry = entry_list[idx]
                    slots = (entry["prefix"], entry["gallows"], entry["m_core"], entry["sfx_fam"])
                p, g, mc, sf = slots
                slot = rng.choice(["prefix", "gallows", "m_core", "sfx_fam"])
                new_val = _weighted_choice(s_dists[slot], rng)
                new_p = new_val if slot == "prefix" else p
                new_g = new_val if slot == "gallows" else g
                new_mc = new_val if slot == "m_core" else mc
                new_sf = new_val if slot == "sfx_fam" else sf
                tok = _reconstruct_token(new_p, new_g, new_mc, new_sf, slot_to_tokens, rng)
                sec_corpus.append(tok)
            else:
                idx = rng.choices(range(len(entry_list)), weights=entry_probs, k=1)[0]
                entry = entry_list[idx]
                tok = _reconstruct_token(
                    entry["prefix"], entry["gallows"], entry["m_core"], entry["sfx_fam"],
                    slot_to_tokens, rng)
                sec_corpus.append(tok)
        all_corpus.extend(sec_corpus[:sec_n])

    while len(all_corpus) < n_tokens:
        all_corpus.append(rng.choice(all_corpus))
    return all_corpus[:n_tokens]


# ==============================================================================
# GENERATOR 6: Combined (Section + Folio Copy Pool)
# ==============================================================================

def gen_p70c_combined(ledger, n_tokens, rng):
    """Section-profiled + folio copy pool (impossibility proof model).

    Higher copy rate (0.35) restricted to folio-sized window. This creates
    wordlen_autocorr but destroys mattr_25 and rep_rate — the impossibility proof.
    """
    concentration = {
        "Herbal-A": 1.0, "Herbal-B": 1.0, "Stars": 1.2,
        "Balneological": 1.5, "Pharmaceutical": 1.0,
        "Astronomical": 1.1, "Cosmological": 1.1,
        "Zodiac": 1.1, "Rosettes": 1.1,
    }

    slot_to_tokens = ledger["slot_to_tokens"]
    token_to_slots = ledger.get("token_to_slots", {})
    entry_list = ledger["entries"]

    global_weights = np.array([e["count"] for e in entry_list], dtype=float)
    global_probs = global_weights / global_weights.sum()

    sec_proportions = {}
    total = sum(p["n_tokens"] for p in ledger["section_profiles"].values())
    for sec, prof in ledger["section_profiles"].items():
        sec_proportions[sec] = prof["n_tokens"] / total

    all_corpus = []
    folio_size = 350  # approximate tokens per folio

    for sec, prop in sorted(sec_proportions.items()):
        sec_n = int(prop * n_tokens)
        if sec_n < 10:
            continue

        conc = concentration.get(sec, 1.0)
        sec_prof = ledger["section_profiles"][sec]
        sec_total = sec_prof["n_tokens"]

        sec_weights = np.zeros(len(entry_list))
        for i, e in enumerate(entry_list):
            p_freq = (sec_prof["prefix_dist"].get(e["prefix"], 0) / sec_total) ** conc
            g_freq = (sec_prof["gallows_dist"].get(e["gallows"], 0) / sec_total) ** conc
            mc_freq = (sec_prof["mcore_dist"].get(e["m_core"], 0) / sec_total) ** conc
            sf_freq = (sec_prof["sfxfam_dist"].get(e["sfx_fam"], 0) / sec_total) ** conc
            sec_weights[i] = e["count"] * ((p_freq + 0.0001) * (g_freq + 0.0001) *
                                            (mc_freq + 0.0001) * (sf_freq + 0.0001)) ** 0.25

        if sec_weights.sum() > 0:
            sec_probs = sec_weights / sec_weights.sum()
            blended = 0.60 * global_probs + 0.40 * sec_probs
            blended /= blended.sum()
            entry_probs = blended.tolist()
        else:
            entry_probs = global_probs.tolist()

        slot_dists = {
            "prefix": sec_prof["prefix_dist"],
            "gallows": sec_prof["gallows_dist"],
            "m_core": sec_prof["mcore_dist"],
            "sfx_fam": sec_prof["sfxfam_dist"],
        }

        sec_corpus = []
        while len(sec_corpus) < sec_n:
            r = rng.random()

            # Folio-restricted COPY (higher rate: 0.35)
            if r < 0.35 and sec_corpus:
                folio_pool = sec_corpus[max(0, len(sec_corpus) - folio_size):]
                sec_corpus.append(rng.choice(folio_pool))
            elif r < 0.75 and sec_corpus:
                # MODIFY with section slot distributions
                base = rng.choice(sec_corpus[-200:]) if len(sec_corpus) > 200 else rng.choice(sec_corpus)
                slots = token_to_slots.get(base)
                if slots is None:
                    idx = rng.choices(range(len(entry_list)), weights=entry_probs, k=1)[0]
                    entry = entry_list[idx]
                    slots = (entry["prefix"], entry["gallows"], entry["m_core"], entry["sfx_fam"])

                p, g, mc, sf = slots
                slot = rng.choice(["prefix", "gallows", "m_core", "sfx_fam"])
                new_val = _weighted_choice(slot_dists[slot], rng)

                new_p = new_val if slot == "prefix" else p
                new_g = new_val if slot == "gallows" else g
                new_mc = new_val if slot == "m_core" else mc
                new_sf = new_val if slot == "sfx_fam" else sf

                tok = _reconstruct_token(new_p, new_g, new_mc, new_sf, slot_to_tokens, rng)
                sec_corpus.append(tok)
            else:
                # CREATE
                idx = rng.choices(range(len(entry_list)), weights=entry_probs, k=1)[0]
                entry = entry_list[idx]
                tok = _reconstruct_token(
                    entry["prefix"], entry["gallows"], entry["m_core"], entry["sfx_fam"],
                    slot_to_tokens, rng)
                sec_corpus.append(tok)

        all_corpus.extend(sec_corpus[:sec_n])

    while len(all_corpus) < n_tokens:
        all_corpus.append(rng.choice(all_corpus))
    return all_corpus[:n_tokens]


# ==============================================================================
# IMPOSSIBILITY METRICS
# ==============================================================================

def compute_impossibility_metrics(tokens):
    """Compute the impossibility proof metrics."""
    import math

    # Same-length same-word
    same_len_pairs = 0
    same_len_same_word = 0
    for i in range(len(tokens) - 1):
        if len(tokens[i]) == len(tokens[i+1]):
            same_len_pairs += 1
            if tokens[i] == tokens[i+1]:
                same_len_same_word += 1
    slsw_rate = same_len_same_word / same_len_pairs if same_len_pairs > 0 else 0

    # Entropy variation at different windows
    def _window_entropy(toks, w):
        entropies = []
        for start in range(0, len(toks) - w, w // 2):
            window = toks[start:start+w]
            freq = Counter(window)
            total = len(window)
            h = -sum((c/total) * math.log2(c/total) for c in freq.values())
            entropies.append(h)
        if len(entropies) < 2:
            return 0.0
        mean_h = np.mean(entropies)
        return float(np.std(entropies) / mean_h) if mean_h > 0 else 0.0

    ev25 = _window_entropy(tokens, 25)
    ev100 = _window_entropy(tokens, 100)
    ev500 = _window_entropy(tokens, 500)
    ev1000 = _window_entropy(tokens, 1000)
    ev_ratio = ev500 / ev25 if ev25 > 0 else 0.0

    # Word-length autocorrelation
    lengths = [len(t) for t in tokens]
    mean_l = np.mean(lengths)
    var_l = np.var(lengths)
    if var_l > 0:
        ac = np.mean([(lengths[i] - mean_l) * (lengths[i+1] - mean_l)
                       for i in range(len(lengths)-1)]) / var_l
    else:
        ac = 0.0

    # Repeated words and MATTR-25
    rep_count = sum(1 for i in range(len(tokens)-1) if tokens[i] == tokens[i+1])
    rep_rate = rep_count / (len(tokens) - 1) if len(tokens) > 1 else 0

    # MATTR-25
    w = 25
    ttrs = []
    for i in range(len(tokens) - w + 1):
        window = tokens[i:i+w]
        ttrs.append(len(set(window)) / w)
    mattr25 = np.mean(ttrs) if ttrs else 0

    return {
        "slsw_rate": slsw_rate,
        "same_len_pairs": same_len_pairs,
        "same_len_same_word": same_len_same_word,
        "ev25": ev25, "ev100": ev100, "ev500": ev500, "ev1000": ev1000,
        "ev_ratio_500_25": ev_ratio,
        "wordlen_autocorr": float(ac),
        "repeated_words": rep_rate,
        "mattr_25": float(mattr25),
    }


# ==============================================================================
# MULTI-SEED RUNNER
# ==============================================================================

def run_generator_multiseed(gen_func, gen_args, n_seeds, n_tokens, cache_path,
                            force=False):
    """Run a generator with multiple seeds, cache results."""
    from score_85_metrics import compute_metrics

    if os.path.exists(cache_path) and not force:
        print(f"    Loading cached: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    all_metrics = []
    all_impossibility = []

    for seed_offset in range(n_seeds):
        seed = SEED + seed_offset
        rng = __import__("random").Random(seed)
        args = gen_args + (rng,)
        corpus = gen_func(*args)[:n_tokens]

        # Build pseudo-lines (10 tokens each)
        lines = [corpus[i:i+10] for i in range(0, len(corpus), 10)]

        metrics = compute_metrics(corpus, lines=lines, seed=seed)
        imp = compute_impossibility_metrics(corpus)
        all_metrics.append(metrics)
        all_impossibility.append(imp)

    # Compute medians
    median_metrics = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
        if vals:
            median_metrics[key] = float(np.median(vals))

    median_imp = {}
    for key in all_impossibility[0]:
        vals = [m[key] for m in all_impossibility if isinstance(m[key], (int, float))]
        if vals:
            median_imp[key] = float(np.median(vals))

    result = {
        "median_metrics": median_metrics,
        "median_impossibility": median_imp,
        "all_metrics": all_metrics,
        "all_impossibility": all_impossibility,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"    Cached to: {cache_path}")

    return result


# ==============================================================================
# TABLE GENERATION
# ==============================================================================

def generate_complete_table(vms_baseline, results, scores):
    """Generate the complete S_TABLE markdown."""
    from score_85_metrics import CORE_15, TOLERANCES

    GROUPS = {
        "Word Length": ["wordlen_mean", "wordlen_std", "wordlen_skew",
                        "wordlen_unique_mean", "wordlen_unique_std", "wordlen_unique_skew",
                        "wordlen_autocorr"],
        "Entropy Hierarchy": ["H0_max_entropy", "H1_unigram", "H2_markov_cond",
                              "h2_joint_digraph", "h2_conditional", "h3_joint_trigraph", "h3_conditional"],
        "Character Distribution": ["char_evenness", "char_redundancy", "char_simpson_D", "char_yule_K"],
        "Digraph/Trigraph": ["digraph_unique", "digraph_coverage", "trigraph_unique"],
        "TTR / Lexical Diversity": ["ttr", "rttr", "cttr", "log_ttr", "maas_a2", "uber_index", "brunet_W",
                                     "msttr_25", "msttr_50", "msttr_100", "mattr_25", "mattr_50", "mattr_100"],
        "Hapax & Frequency": ["hapax_ratio_tokens", "hapax_ratio_types", "dis_ratio_tokens",
                              "dis_ratio_types", "sichel_S", "hapax_type_proportion",
                              "freq_spectrum_1", "freq_spectrum_2", "freq_spectrum_3", "freq_spectrum_gt10"],
        "Frequency Concentration": ["top10_share", "top50_share"],
        "Lexical Richness": ["word_yule_K", "honore_R"],
        "Autocorrelation": ["autocorr_wordlen", "autocorr_wordfreq", "autocorr_ttr_25", "autocorr_hapax_25"],
        "BG Subsampling — Word": ["wordunique_mean", "wordunique_std", "wordunique_skew",
                                   "wordchange_mean", "wordchange_std", "wordchange_skew",
                                   "worddist_max", "worddist_shape",
                                   "wordbias_mean", "wordbias_std", "wordbias_skew",
                                   "wordbias_lines_mean", "wordbias_lines_std", "wordbias_lines_skew"],
        "BG Subsampling — Character": ["chardist_max", "chardist_shape", "ngramdist_max", "ngramdist_shape",
                                        "charbias_mean", "charbias_std", "charbias_skew",
                                        "charbias_words_mean", "charbias_words_std", "charbias_words_skew"],
        "Counts": ["unique_words", "repeated_words", "tripled_words", "unique_chars",
                    "repeated_chars", "tripled_chars", "unique_ngrams"],
        "Global": ["entropy", "compression", "zipf", "zipf_alpha", "zipf_r2",
                    "flipped_pairs", "heaps_beta"],
    }

    models = ["Bigram", "Scribal", "P70C", "Dual", "Section", "Combined"]
    lines = ["# S_TABLE: Complete 90-Metric Results\n",
             "Medians across 10 seeds (42–51). ✓/✗ = within/outside calibrated tolerance.",
             "**Bold** = Core 15 metric.\n"]

    def _fmt(v):
        if v is None: return "—"
        if abs(v) > 100: return f"{v:.1f}"
        if abs(v) > 1: return f"{v:.3f}"
        return f"{v:.4f}"

    for group_name, metrics in GROUPS.items():
        lines.append(f"## {group_name}\n")
        hdr = f"| {'Metric':<25} | {'VMS':>8} |"
        for m in models: hdr += f" {m:>9} |"
        lines.append(hdr)
        sep = f"|{'-'*27}|{'-'*10}|" + "|".join(["-"*11]*len(models)) + "|"
        lines.append(sep)

        for metric in metrics:
            vms_val = vms_baseline.get(metric)
            name = f"**{metric}**" if metric in CORE_15 else metric
            passes_sets = {m: set(scores[m]["score_85"]["passes"]) for m in models}

            row = f"| {name:<25} | {_fmt(vms_val):>8} |"
            for model in models:
                gen_val = results[model]["median_metrics"].get(metric)
                pf = "✓" if metric in passes_sets[model] else "✗"
                row += f" {_fmt(gen_val):>7}{pf} |"
            lines.append(row)
        lines.append("")

    # Summary
    lines.append("## Summary\n")
    hdr = f"| {'':25} |" + "|".join([f" {m:>9} " for m in models]) + "|"
    lines.append(hdr)
    for label, key in [("Core 15 pass", "score_15"), ("Full suite pass", "score_85")]:
        row = f"| **{label}**{' '*(23-len(label))} |"
        for m in models:
            s = scores[m][key]
            row += f" {s['n_pass']:>2}/{s['n_total']:<6} |"
        lines.append(row)
    lines.append("")

    # Impossibility
    lines.append("## Impossibility Metrics\n")
    vms_imp = compute_impossibility_metrics(
        [r["token"] for r in pickle.load(open(os.path.join(DATA_DIR, "enriched_records.pkl"), "rb"))]
    )
    imp_rows = [
        ("Same-len same-word %", "slsw_rate"),
        ("EV ratio (500/25)", "ev_ratio_500_25"),
        ("Word-length AC(1)", "wordlen_autocorr"),
        ("Repeated-word rate", "repeated_words"),
        ("MATTR-25", "mattr_25"),
    ]
    hdr = f"| {'Metric':<25} | {'VMS':>8} |" + "|".join([f" {m:>9} " for m in models]) + "|"
    lines.append(hdr)
    lines.append(f"|{'-'*27}|{'-'*10}|" + "|".join(["-"*11]*len(models)) + "|")
    for label, key in imp_rows:
        row = f"| {label:<25} | {_fmt(vms_imp.get(key)):>8} |"
        for model in models:
            row += f" {_fmt(results[model]['median_impossibility'].get(key)):>9} |"
        lines.append(row)

    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reproduce VMS generator hierarchy results")
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation, use cached")
    parser.add_argument("--samples", action="store_true", help="Generate 200-word text samples")
    parser.add_argument("--force", action="store_true", help="Force regeneration (ignore cache)")
    args = parser.parse_args()

    print("=" * 70)
    print("VMS GENERATOR HIERARCHY — FULL REPRODUCTION")
    print("=" * 70)

    ensure_dirs()

    print("\n1. Fetching data...")
    fetch_data()

    print("\n2. Loading data...")
    records, spec, tokens, lines, f1r_tokens = load_data()

    print("\n3. Computing VMS baseline (90 metrics)...")
    from score_85_metrics import compute_metrics, score_against_vms, CORE_15, ALL_85, TOLERANCES

    vms_bl_path = os.path.join(RESULTS_DIR, "vms_baseline_85metrics.pkl")
    if os.path.exists(vms_bl_path) and not args.force:
        with open(vms_bl_path, "rb") as f:
            vms_baseline = pickle.load(f)
        print(f"  Loaded cached baseline ({len(vms_baseline)} metrics)")
    else:
        vms_baseline = compute_metrics(tokens, lines=lines)
        with open(vms_bl_path, "wb") as f:
            pickle.dump(vms_baseline, f)
        print(f"  Computed and cached ({len(vms_baseline)} metrics)")

    vms_imp = compute_impossibility_metrics(tokens)
    print(f"  VMS impossibility: SLSW={vms_imp['slsw_rate']:.4f}, "
          f"EV={vms_imp['ev_ratio_500_25']:.4f}, AC={vms_imp['wordlen_autocorr']:.4f}")

    print("\n4. Building models...")
    followers = build_char_bigram_model(tokens, f1r_tokens)
    group_followers, char_exemplars = build_ductus_model(tokens)
    ledger = build_p70c_ledger(spec, records)
    print(f"  Bigram model: {len(followers)} entries")
    print(f"  Ductus model: {len(group_followers)} groups")
    print(f"  P70C ledger: {len(ledger['entries'])} entries")

    if not args.skip_gen:
        print("\n5. Running generators (10 seeds each)...")
        generators = [
            ("Bigram",   gen_char_bigram,              (followers, N_TARGET)),
            ("Scribal",  gen_ductus,                   (group_followers, char_exemplars, N_TARGET)),
            ("P70C",     gen_p70c_single,              (ledger, N_TARGET)),
            ("Dual",     gen_p70c_dual,                (ledger, N_TARGET)),
            ("Section",  gen_p70c_section_profiled,    (ledger, N_TARGET)),
            ("Combined", gen_p70c_combined,            (ledger, N_TARGET)),
        ]

        results = {}
        for name, func, func_args in generators:
            print(f"  {name}...")
            t0 = time.time()
            cache_path = os.path.join(CACHE_DIR, f"cache_{name.lower()}.pkl")
            results[name] = run_generator_multiseed(
                func, func_args, N_SEEDS, N_TARGET, cache_path, force=args.force)
            elapsed = time.time() - t0
            print(f"    Done ({elapsed:.1f}s)")
    else:
        print("\n5. Loading cached results...")
        results = {}
        for name in ["Bigram", "Scribal", "P70C", "Dual", "Section", "Combined"]:
            cache_path = os.path.join(CACHE_DIR, f"cache_{name.lower()}.pkl")
            with open(cache_path, "rb") as f:
                results[name] = pickle.load(f)
            print(f"  Loaded: {name}")

    print("\n6. Scoring against VMS baseline...")
    scores = {}
    for name in results:
        s85 = score_against_vms(results[name]["median_metrics"], vms_baseline, ALL_85, TOLERANCES)
        s15 = score_against_vms(results[name]["median_metrics"], vms_baseline, CORE_15, TOLERANCES)
        scores[name] = {"score_85": s85, "score_15": s15}
        print(f"  {name:>12}: {s15['n_pass']}/{s15['n_total']} (core 15)  "
              f"{s85['n_pass']}/{s85['n_total']} (full)")

    # Save complete results
    full_results = {
        "results": results,
        "scores": scores,
        "vms_baseline": vms_baseline,
        "vms_impossibility": vms_imp,
    }
    with open(os.path.join(RESULTS_DIR, "hierarchy_85_results.pkl"), "wb") as f:
        pickle.dump(full_results, f)

    print("\n7. Generating tables...")
    table = generate_complete_table(vms_baseline, results, scores)
    with open(os.path.join(RESULTS_DIR, "S_TABLE_COMPLETE_METRICS.md"), "w") as f:
        f.write(table)
    print(f"  Written: {RESULTS_DIR}/S_TABLE_COMPLETE_METRICS.md")

    # Samples
    if args.samples:
        print("\n8. Generating 200-word samples...")
        import random
        samples = {"VMS (f1r)": tokens[:200]}
        sample_gens = [
            ("Bigram", gen_char_bigram, (followers, 200)),
            ("Scribal", gen_ductus, (group_followers, char_exemplars, 200)),
            ("P70C Single", gen_p70c_single, (ledger, 200)),
            ("P70C Section", gen_p70c_section_profiled, (ledger, 200)),
            ("P70C Combined", gen_p70c_combined, (ledger, 200)),
        ]
        for name, func, func_args in sample_gens:
            rng = random.Random(42)
            samples[name] = func(*func_args, rng)[:200]
        with open(os.path.join(RESULTS_DIR, "generator_samples_200.pkl"), "wb") as f:
            pickle.dump(samples, f)
        print(f"  Written: {RESULTS_DIR}/generator_samples_200.pkl")

    print("\n" + "=" * 70)
    print("DONE. All results in:", RESULTS_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()

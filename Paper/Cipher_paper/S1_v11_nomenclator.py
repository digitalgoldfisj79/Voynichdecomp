#!/usr/bin/env python3
"""
SUPPLEMENT S1: FORWARD CIPHER v11 WITH ABLATION STUDY
=======================================================
Two real inputs:
  1. enriched_records.pkl — VMS Herbal-A tokens (builds cell pools)
  2. ci_corpus_parsed.pkl — Circa Instans Latin (source text to encipher)

Everything else is inlined. No mystery pickles.

The cipher has two parts:
  PART A (lines ~50-140):  BABUINI ROUTING — Latin → grid cell assignment
  PART B (lines ~140-370): COPY-MUTATE SCRIBE — cell pool → token selection
  
Part A is the cipher-class contribution (this paper).
Part B is the scribal production model (Bozzard 2026a).

The ablation study (lines ~390+) systematically disables each component
and reports the effect on the 84-metric scoring battery:

  Configuration          n/84    C15   BG42   Δ
  ───────────────────────────────────────────────
  Full v11               62.0   12.3   33.3    —
  Minus nomenclator      60.7   11.0   33.3  -1.3
  Minus stickiness       65.3   13.3   34.7  +3.3
  Minus reuse            38.3   10.0   23.0  -23.7
  Minus avoidance        65.0   14.0   36.7  +3.0
  Architecture only      48.3    8.0   30.3  -13.7

Usage:
  python S1_v11_nomenclator.py              # Single run (default seed)
  python S1_v11_nomenclator.py 42           # Single run (seed 42)
  python S1_v11_nomenclator.py --ablation   # Full ablation study

Edward Bozzard · ORCID 0009-0002-4052-0994
DOI: 10.5281/zenodo.18812705
"""

import random, pickle, sys, os
from collections import Counter, defaultdict
import numpy as np

# ══════════════════════════════════════════════════════════════
# CONFIGURATION — all parameters inline
# ══════════════════════════════════════════════════════════════

SEED = 404
TARGET = 4033           # Herbal-A token count
VOCAB_CAP = 1430        # Herbal-A type count
AVOIDANCE = 15          # suffix-avoidance dampening factor
COPY_ALPHA = 1.3        # preferential reuse exponent (was 2.0; lowered to fix EC concentration)
P_STICKY = 0.22         # column stickiness: prob of staying in previous token's suffix family
REBAL_STR = 8.0         # suffix-family rebalancing strength
SEED_WEIGHT = 0.10      # weight for novel pool entries

# Provenance rates (from Bozzard 2026a corpus forensics)
FC_COPY_RATE = 0.133
FC_ED1_RATE = 0.387

# Herbal-A line lengths (613 lines, range 1-13)
LINE_LENGTHS = [5]*94 + [8]*93 + [6]*90 + [9]*83 + [7]*78 + [4]*52 + \
               [3]*35 + [10]*33 + [2]*19 + [11]*18 + [12]*9 + [1]*5 + [13]*4

# ══════════════════════════════════════════════════════════════
# PART A: BABUINI ROUTING
# This is the cipher-class contribution.
# Latin word → (row, suffix_family) cell address.
# ══════════════════════════════════════════════════════════════

# Grid: initial consonant → row (identity permutation, no keyword)
CONSONANT_TO_ROW = {}
for row, consonants in {
    'o': ['c','s','p'],
    'c': ['∅','v'],       # vowel-initial and v
    'e': ['f','d'],
    'a': ['m','l'],
    'd': ['r','q','h','n','g'],
    'l': ['t'],
    'r': ['b','z','x','j','k','w','y'],
}.items():
    for c in consonants:
        CONSONANT_TO_ROW[c] = row

# First vowel → suffix family
VOWEL_TO_FAMILY = {'a':'Y', 'e':'R', 'i':'N', 'o':'L', 'u':'BARE'}

# Suffix family target distribution (from Herbal-A)
FAMILY_TARGETS = {'Y':0.313, 'R':0.174, 'N':0.171, 'L':0.161, 'BARE':0.140, 'M':0.030}
FAMILIES = list(FAMILY_TARGETS.keys())

# Suffix members per family (from Herbal-A, with counts as weights)
SUFFIX_MEMBERS = {
    'BARE': [('', 563)],
    'Y':    [('y',204), ('ey',147), ('chy',234), ('dy',68), ('ody',96), ('eey',77), ('shy',18)],
    'N':    [('aiin',500), ('ain',92), ('iin',22), ('aiiin',22)],
    'L':    [('ol',493), ('al',101), ('l',26)],
    'R':    [('or',428), ('ar',169), ('r',62), ('ir',42)],
    'M':    [('am',61), ('m',16)],
}


# Primary grid cell contents (from actual_core_grid.pkl, inlined)
# These are the most common core strings per (row, family) cell.
# Used to CONSTRUCT novel tokens that don't appear in the manuscript.
GRID_PRIMARY = {
    ('a','BARE'):'a', ('a','L'):'aii', ('a','M'):'ai', ('a','N'):'a', ('a','R'):'a', ('a','Y'):'ar',
    ('c','BARE'):'cho', ('c','L'):'ch', ('c','M'):'ch', ('c','N'):'ch', ('c','R'):'ch', ('c','Y'):'ch',
    ('d','BARE'):'d', ('d','L'):'d', ('d','M'):'d', ('d','N'):'d', ('d','R'):'d', ('d','Y'):'d',
    ('e','BARE'):'eo', ('e','L'):'e', ('e','M'):'eo', ('e','N'):'e', ('e','R'):'e', ('e','Y'):'e',
    ('o','BARE'):'o', ('o','L'):'od', ('o','M'):'o', ('o','N'):'o', ('o','R'):'o', ('o','Y'):'ol',
}

# Rare tokens injected at random positions to maintain digraph coverage
RARE_TOKENS = ['oleeeb','oteeeb','choteeeb','okshodeeeb','oeeeb',
               'cheeeb','cheeb','tu','vor','zepchy']

PREFIXES = ['','ch','sh','d','o','qo','s','y']
GALLOWS_OPTIONS = ['','k','t']

VOWELS = set('aeiouàèìòùéêîôûäëïöü')
VOWEL_NORMALISE = {'à':'a','è':'e','é':'e','ê':'e','ì':'i','î':'i','ò':'o','ô':'o','ù':'u','û':'u'}

# Nomenclator: fixed family assignments for known function words
# Cross-validated on Ald.211 → CI → held-out VMS (p<0.0001)
NOMENCLATOR = {
    'et': 'Y', 'postea': 'Y',
    'in': 'N', 'cum': 'N', 'hoc': 'N',
    'de': 'L', 'habet': 'L', 'uel': 'L', 'vel': 'L',
    'que': 'L', 'supra': 'L', 'ad': 'L',
}

def classify_and_route(word, ec_words):
    """PART A: Route a Latin word to a grid cell.
    
    Returns: ('EC', suffix_family) or ('FC', row, suffix_family)
    """
    w = word.lower()
    
    # Check nomenclator first (fixed, cross-validated assignments)
    if w in NOMENCLATOR:
        return ('EC', NOMENCLATOR[w])
    
    # Get first vowel for heuristic
    first_vowel = 'a'
    for ch in w:
        if ch in VOWELS:
            first_vowel = VOWEL_NORMALISE.get(ch, ch)
            break
    family = VOWEL_TO_FAMILY.get(first_vowel, 'Y')
    
    # EC or FC?
    if w in ec_words:
        return ('EC', family)
    else:
        initial = w[0] if w[0] not in VOWELS else '∅'
        row = CONSONANT_TO_ROW.get(initial, 'c')
        return ('FC', row, family)

# ══════════════════════════════════════════════════════════════
# PART B: COPY-MUTATE SCRIBE
# This is from Bozzard (2026a).
# Given a cell address, select a token from the pool.
# ══════════════════════════════════════════════════════════════

def build_pools(ha_records):
    """Build cell pools from Herbal-A VMS tokens.
    
    This is the manuscript-derived proxy for the unknown grid contents.
    Each cell (row, family) contains the VMS tokens observed in that cell.
    """
    # Build character FSM from HA tokens (for validation)
    bigrams = set()
    starters = set()
    for r in ha_records:
        t = r['token']
        if t:
            starters.add(t[0])
            for j in range(len(t)-1):
                bigrams.add(t[j:j+2])
    
    def is_valid(tok):
        if not tok or tok[0] not in starters:
            return False
        return all(tok[i:i+2] in bigrams for i in range(len(tok)-1))
    
    # Group tokens by (row, family)
    pool = defaultdict(Counter)
    for r in ha_records:
        mc = r.get('m_core') or r.get('core') or ''
        sf = r.get('sfx_fam', 'BARE')
        row = mc[0] if mc and not r['empty_core'] else '∅'
        pool[(row, sf)][r['token']] += 1
    
    # Augment pools with CONSTRUCTED tokens from grid primary
    # This is the closest thing to what the real cipher would do:
    # assemble prefix + gallows + core_from_grid + suffix
    for row in ['o','c','e','a','d','l','r']:
        for fam in FAMILIES:
            core = GRID_PRIMARY.get((row, fam), GRID_PRIMARY.get((row, 'BARE'), None))
            if not core:
                continue
            for pfx in PREFIXES:
                for gal in GALLOWS_OPTIONS:
                    for sfx, _ in SUFFIX_MEMBERS.get(fam, SUFFIX_MEMBERS['Y']):
                        tok = pfx + gal + core + sfx
                        if tok not in pool[(row, fam)] and is_valid(tok):
                            pool[(row, fam)][tok] = SEED_WEIGHT
    
    # Prepare weighted sampling
    sampler = {}
    for cell, token_counts in pool.items():
        tokens = list(token_counts.keys())
        weights = np.array([token_counts[t] for t in tokens], dtype=float)
        weights /= weights.sum()
        sampler[cell] = (tokens, weights)
    
    return sampler, is_valid

def pick_token(sampler, cell, recent_tokens):
    """Select a token from the cell pool, avoiding recent tokens."""
    if cell not in sampler:
        return None
    tokens, weights = sampler[cell]
    adjusted = np.copy(weights)
    for j, t in enumerate(tokens):
        if t in recent_tokens:
            adjusted[j] /= AVOIDANCE
    s = adjusted.sum()
    if s > 0:
        adjusted /= s
        return tokens[np.random.choice(len(tokens), p=adjusted)]
    return tokens[np.random.choice(len(tokens), p=weights)]

def reuse_token(past_counts, sampler, cell=None):
    """Preferential reuse: pick a recent token weighted by frequency."""
    candidates = {}
    if cell and cell in sampler:
        pool_tokens, _ = sampler[cell]
        candidates = {t: past_counts[t] for t in pool_tokens if t in past_counts}
    if not candidates:
        candidates = dict(past_counts)
    if not candidates:
        return 'dy'
    tokens = list(candidates.keys())
    freqs = np.array([candidates[t]**COPY_ALPHA for t in tokens], dtype=float)
    freqs /= freqs.sum()
    return tokens[np.random.choice(len(tokens), p=freqs)]

def rebalance_family(base_family, family_counts, n_tokens):
    """Nudge suffix family toward target distribution when drifting."""
    if n_tokens < 30:
        return base_family
    current = {f: family_counts.get(f, 0) / n_tokens for f in FAMILIES}
    delta = {f: FAMILY_TARGETS.get(f, 0.02) - current.get(f, 0) for f in FAMILIES}
    if delta[base_family] < -0.03:
        best = max(FAMILIES, key=lambda f: delta[f])
        if delta[best] > 0.01:
            return best
    if delta[base_family] < -0.01:
        prob = min(0.8, abs(delta[base_family]) * REBAL_STR)
        if random.random() < prob:
            positive = [(f, d) for f, d in delta.items() if d > 0.01]
            if positive:
                fams, ds = zip(*positive)
                ds = np.array(ds); ds /= ds.sum()
                return fams[np.random.choice(len(fams), p=ds)]
    return base_family

# ══════════════════════════════════════════════════════════════
# MAIN: Run the forward cipher
# ══════════════════════════════════════════════════════════════

def run(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    
    # Load the two real inputs
    with open('enriched_records.pkl', 'rb') as f:
        records = pickle.load(f)
    with open('ci_corpus_parsed.pkl', 'rb') as f:
        ci = pickle.load(f)
    
    # Build pools from Herbal-A
    ha = [r for r in records if r.get('section') == 'Herbal-A']
    sampler, is_valid = build_pools(ha)
    
    # Latin source text
    ec_words = ci.get('ec_words', set())
    words = ci['all_words']
    start = random.randint(3000, 40000)
    words = words[start:] + words[:start]  # random starting point
    
    # Generate
    output = []
    produced = set()
    past_counts = Counter()
    family_counts = Counter()
    line_tokens = []
    line_target = random.choice(LINE_LENGTHS)
    
    # Pre-schedule rare token injection positions
    rare_positions = set(random.sample(range(100, TARGET-100), len(RARE_TOKENS)))
    rare_schedule = dict(zip(sorted(rare_positions), RARE_TOKENS))
    prev_family = 'Y'  # column stickiness: track previous token's suffix family
    
    i = 0
    while len(output) < TARGET and i < len(words):
        n = len(output)
        
        # Inject rare tokens at scheduled positions
        if n in rare_schedule:
            t = rare_schedule[n]
            output.append(('RC', t, '<rare>'))
            produced.add(t)
            past_counts[t] += 1
            family_counts['BARE'] = family_counts.get('BARE', 0) + 1
            prev_family = 'BARE'
            line_tokens.append(t)
            if len(line_tokens) >= line_target:
                line_tokens = []; line_target = random.choice(LINE_LENGTHS)
            continue
        
        word = words[i]; i += 1
        over_cap = len(produced) >= VOCAB_CAP
        
        # PART A: Route the Latin word
        route = classify_and_route(word, ec_words)
        
        if route[0] == 'EC':
            is_nom = word.lower() in NOMENCLATOR
            if is_nom:
                family = route[1]  # Fixed by cipher designer, never rebalanced
            else:
                family = rebalance_family(route[1], family_counts, n)
                # Column stickiness only for heuristic-routed words
                if random.random() < P_STICKY and prev_family in FAMILIES:
                    family = prev_family
            cell = ('∅', family)
            if over_cap:
                token = reuse_token(past_counts, sampler, cell)
            else:
                token = pick_token(sampler, cell, produced)
                if not token:
                    for alt in FAMILIES:
                        if alt != family:
                            token = pick_token(sampler, ('∅', alt), produced)
                            if token:
                                family = alt; break
                if not token:
                    token = 'dy'
        else:
            row, family = route[1], route[2]
            family = rebalance_family(family, family_counts, n)
            # Column stickiness: override to previous family if pool exists
            if random.random() < P_STICKY and prev_family in FAMILIES:
                if (row, prev_family) in sampler:
                    family = prev_family
            cell = (row, family)
            bf = 0.35 if (len(line_tokens) == 0 or len(line_tokens) >= line_target - 1) else 1.0
            if over_cap:
                token = reuse_token(past_counts, sampler, cell)
            else:
                if random.random() < (FC_COPY_RATE + FC_ED1_RATE) * bf:
                    token = pick_token(sampler, cell, produced)
                else:
                    alt_fams = [f for f in FAMILIES if f != family and (row, f) in sampler]
                    if alt_fams:
                        af = random.choice(alt_fams)
                        token = pick_token(sampler, (row, af), produced)
                        family = af
                    else:
                        token = pick_token(sampler, cell, produced)
                if not token:
                    token = pick_token(sampler, cell, produced) or 'dy'
        
        output.append((route[0], token, word))
        produced.add(token)
        past_counts[token] += 1
        family_counts[family] = family_counts.get(family, 0) + 1
        prev_family = family
        
        # Line breaks
        line_tokens.append(token)
        if len(line_tokens) >= line_target:
            line_tokens = []
            line_target = random.choice(LINE_LENGTHS)
    
    return output


# ══════════════════════════════════════════════════════════════
# ABLATION STUDY
# ══════════════════════════════════════════════════════════════
#
# Systematically disables each component to measure its
# contribution to the 84-metric scoring battery.
#
# Six configurations:
#   1. Full v11           — baseline (all components)
#   2. Minus nomenclator  — random family assignments (et→Y, in→N fixed)
#   3. Minus stickiness   — P_STICKY = 0.0
#   4. Minus reuse        — COPY_ALPHA = 0.0 (uniform weighting)
#   5. Minus avoidance    — AVOIDANCE = 1.0 (no penalty)
#   6. Architecture only  — S4 clean baseline (uniform random from pools)
#
# Results (seeds 42, 404, 501):
#
#   Configuration          n/84    C15   BG42       Δ
#   ────────────────────────────────────────────────────
#   Full v11               62.0   12.3   33.3      —
#   Minus nomenclator      60.7   11.0   33.3    -1.3
#   Minus stickiness       65.3   13.3   34.7    +3.3
#   Minus reuse            38.3   10.0   23.0   -23.7
#   Minus avoidance        65.0   14.0   36.7    +3.0
#   Architecture only      48.3    8.0   30.3   -13.7  (requires S4)
#
# Interpretation:
#
# Preferential reuse (COPY_ALPHA) is the dominant scribe component:
# removing it drops the score by 24 points. The two-table routing
# architecture provides the base (48.3/84); reuse raises it to 62.
#
# Column stickiness and suffix avoidance both reduce the general
# metric score by ~3 points each, but they target specific VMS
# properties. Stickiness targets the suffix-family bigram rate
# (sfx_bi = 0.252); avoidance targets vocabulary size (types =
# 1430). Without avoidance, types collapse to ~1120. These are
# refinement parameters that trade general metric performance
# for specific structural matches.
#
# The nomenclator contributes only 1.3 points to the metric score,
# confirming that the 84-metric battery validates the architecture,
# not the specific function-word assignments. The nomenclator is
# validated separately by bigram correlation (r = 0.96 training,
# r = 0.89 cross-validation on CI, p < 0.0001).
# ══════════════════════════════════════════════════════════════

def run_ablation(seeds=None):
    """Run the complete ablation study.
    
    Requires score_85_metrics.py and metric_defs.py in the same directory.
    Results are printed and saved to ablation_results.pkl.
    """
    global P_STICKY, COPY_ALPHA, AVOIDANCE, NOMENCLATOR
    
    if seeds is None:
        seeds = [42, 404, 501]
    
    try:
        sys.path.insert(0, '.')
        import score_85_metrics as _scorer
        import metric_defs as _mdefs
    except ImportError:
        print("ERROR: score_85_metrics.py and metric_defs.py required.")
        print("Place them in the same directory as this script.")
        return None
    
    # Load VMS reference
    with open('enriched_records.pkl', 'rb') as f:
        _records = pickle.load(f)
    _ha = [r for r in _records if r.get('section') == 'Herbal-A']
    _toks_vms = [r['token'] for r in _ha][:4032]
    _lines_vms = [_toks_vms[i:i+84] for i in range(0, 4032, 84)]
    print("Computing VMS reference metrics...")
    _VMS_M = _scorer.compute_metrics(
        _toks_vms, lines=_lines_vms,
        subset_iterations=30, seed=42, verbose=False)
    
    def _score(toks):
        gl = [toks[i:i+84] for i in range(0, len(toks), 84)]
        gm = _scorer.compute_metrics(
            toks, lines=gl, subset_iterations=30,
            seed=42, verbose=False)
        sr = _scorer.score_against_vms(gm, _VMS_M)
        c15 = sum(1 for m in _mdefs.CORE_15
                  if m in sr['details'] and sr['details'][m]['pass'])
        bg42 = sum(1 for m in _mdefs.BG_METRICS
                   if m in sr['details'] and sr['details'][m]['pass'])
        return sr['n_pass'], c15, bg42
    
    # Save originals
    _orig_sticky = P_STICKY
    _orig_alpha = COPY_ALPHA
    _orig_avoid = AVOIDANCE
    _orig_nom = dict(NOMENCLATOR)
    _families = ['Y', 'N', 'L', 'R', 'BARE', 'M']
    
    def _restore():
        global P_STICKY, COPY_ALPHA, AVOIDANCE, NOMENCLATOR
        P_STICKY = _orig_sticky
        COPY_ALPHA = _orig_alpha
        AVOIDANCE = _orig_avoid
        NOMENCLATOR.clear()
        NOMENCLATOR.update(_orig_nom)
    
    def _run_silent(seed):
        import io as _io
        old = sys.stdout; sys.stdout = _io.StringIO()
        try:
            out = run(seed=seed)
        finally:
            sys.stdout = old
        return [t[1] for t in out]
    
    configs = []
    
    # 1. Full v11
    def _cfg_full(seed):
        _restore()
        return _run_silent(seed)
    configs.append(("Full v11", _cfg_full))
    
    # 2. Minus nomenclator (random assignments, anchors fixed)
    def _cfg_no_nom(seed):
        global NOMENCLATOR
        _restore()
        rng = random.Random(seed + 10000)
        NOMENCLATOR.clear()
        NOMENCLATOR['et'] = 'Y'
        NOMENCLATOR['in'] = 'N'
        for w in ['postea','cum','hoc','de','habet','uel','vel','que','supra','ad']:
            NOMENCLATOR[w] = rng.choice(_families)
        return _run_silent(seed)
    configs.append(("Minus nomenclator", _cfg_no_nom))
    
    # 3. Minus stickiness
    def _cfg_no_sticky(seed):
        global P_STICKY
        _restore()
        P_STICKY = 0.0
        return _run_silent(seed)
    configs.append(("Minus stickiness", _cfg_no_sticky))
    
    # 4. Minus reuse
    def _cfg_no_reuse(seed):
        global COPY_ALPHA
        _restore()
        COPY_ALPHA = 0.0
        return _run_silent(seed)
    configs.append(("Minus reuse", _cfg_no_reuse))
    
    # 5. Minus avoidance
    def _cfg_no_avoid(seed):
        global AVOIDANCE
        _restore()
        AVOIDANCE = 1.0
        return _run_silent(seed)
    configs.append(("Minus avoidance", _cfg_no_avoid))
    
    # 6. Architecture only (S4 clean baseline if available,
    #    else v11 with scribe rules zeroed — note these differ because
    #    v11 retains rebalancing, vocab cap, and weighted pool sampling)
    def _cfg_arch(seed):
        try:
            import S4_forward_cipher_clean as _s4
            import importlib; importlib.reload(_s4)
            import io as _io2
            old = sys.stdout; sys.stdout = _io2.StringIO()
            try:
                out = _s4.run(seed=seed, n=4033)
            finally:
                sys.stdout = old
            return [t[1] for t in out]
        except ImportError:
            # Fallback: v11 with scribe rules disabled
            # NOTE: this scores ~64/84, not ~48/84, because v11 retains
            # rebalancing, vocab cap, and frequency-weighted pools.
            # Use S4_forward_cipher_clean.py for the true minimal baseline.
            global P_STICKY, COPY_ALPHA, AVOIDANCE
            _restore()
            P_STICKY = 0.0
            COPY_ALPHA = 0.0
            AVOIDANCE = 1.0
            return _run_silent(seed)
    configs.append(("Architecture only (S4)", _cfg_arch))
    
    # Run all configurations
    results = {}
    full_mean = None
    
    print("\n" + "=" * 65)
    print("ABLATION STUDY — v11 Forward Cipher")
    print(f"Seeds: {seeds}")
    print("=" * 65)
    
    for name, fn in configs:
        print(f"\n--- {name} ---")
        scores = []
        for seed in seeds:
            toks = fn(seed)
            n84, c15, bg42 = _score(toks)
            scores.append((n84, c15, bg42))
            print(f"  seed {seed}: {n84}/84, C15={c15}, BG42={bg42}, "
                  f"types={len(set(toks))}")
        
        n_mean = np.mean([s[0] for s in scores])
        c_mean = np.mean([s[1] for s in scores])
        b_mean = np.mean([s[2] for s in scores])
        
        if name == "Full v11":
            full_mean = n_mean
        
        delta = round(n_mean - full_mean, 1) if full_mean is not None else 0.0
        results[name] = {
            'scores': scores, 'seeds': seeds,
            'n84_mean': round(n_mean, 1),
            'c15_mean': round(c_mean, 1),
            'bg42_mean': round(b_mean, 1),
            'delta': delta,
        }
        print(f"  MEAN: {n_mean:.1f}/84, C15={c_mean:.1f}, "
              f"BG42={b_mean:.1f}, Δ={delta:+.1f}")
    
    # Restore originals
    _restore()
    
    # Summary table
    print("\n" + "=" * 65)
    print("ABLATION SUMMARY")
    print("=" * 65)
    print(f"\n{'Configuration':<25} {'n/84':>6} {'C15':>6} "
          f"{'BG42':>6} {'Δ':>8}")
    print("-" * 55)
    for name, _ in configs:
        r = results[name]
        print(f"{name:<25} {r['n84_mean']:>6.1f} {r['c15_mean']:>6.1f} "
              f"{r['bg42_mean']:>6.1f} {r['delta']:>+8.1f}")
    print("-" * 55)
    
    # Save
    with open('ablation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nSaved ablation_results.pkl")
    
    return results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if '--ablation' in sys.argv:
        run_ablation()
    else:
        seed = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != '--ablation' else SEED
        output = run(seed)
        tokens = [t[1] for t in output]
        
        print(f"Generated {len(tokens)} tokens, {len(set(tokens))} types")
        print(f"EC: {sum(1 for t in output if t[0]=='EC')}, FC: {sum(1 for t in output if t[0]=='FC')}")
        
        try:
            sys.path.insert(0, '.')
            from score_85_metrics import score_against_vms
            score = score_against_vms(tokens)
            print(f"Score: {score}/84")
        except ImportError:
            print("(score_85_metrics.py not found, skipping scoring)")

#!/usr/bin/env python3
"""
reproduce_s3.py — Definitive S3 Reproduction Script
=====================================================
Reproduces all 21 generator scores, ablation results, and impossibility
diagnostics reported in Supplement S3.

Requirements:
    pip install numpy scipy

Data dependencies (auto-located):
    - enriched_records.pkl   (37,465 parsed VMS tokens)
    - p70c_full_spec_v1.json (6,750 PGCS quad entries)
    - voynich_transcriptions_slim.json  (transcription, for zero-corpus generators)

Generator files (in Paper/Generators/):
    gen_f57v.py, gen_scribal_manual.py, gen_scribal_workshop.py,
    gen_template_v2.py through gen_template_v10.py,
    gen_template_ductus.py, gen_template_v4_tuned.py,
    gen_scribal_p70c.py

Also requires (in Paper/):
    score_85_metrics.py      (scoring module)
    reproduce_all.py         (contains the 6 BG22 generator functions:
                              Bigram, Scribal, P70C, Dual, Section-Profiled,
                              Combined — these are inline, not separate .py files)

Background:
    All 21 generators exist in this repo. 15 are standalone .py files in
    Paper/Generators/. The remaining 6 (BG22 family) are defined as
    functions inside reproduce_all.py, which despite its name only
    covers those 6. reproduce_session.py covers a further 4 (v5–v8).
    Neither script previously covered the full set.

    This script is the first to reproduce all 21 in a single run. It
    imports the BG22 functions from reproduce_all.py and runs all 15
    standalone generators directly via their .py files. It also adds
    ablation sweeps, impossibility diagnostics, and the self-consistency
    ceiling. reproduce_session.py and reproduce_all.py remain in the
    repo as historical records.

Usage:
    python reproduce_s3.py                  # Full run
    python reproduce_s3.py --skip-bg22      # Skip BG22 (use cached)
    python reproduce_s3.py --skip-template  # Skip template (use cached)
    python reproduce_s3.py --skip-ablation  # Skip ablation sweeps
    python reproduce_s3.py --resume         # Resume from last checkpoint
    python reproduce_s3.py --force          # Re-run everything

Outputs (all in results/s3/):
    s3_all_generators.pkl       Canonical 21-generator results
    s3_ablation_results.pkl     Three ablation sweeps
    s3_vms_baseline.pkl         VMS baseline (84 metrics + impossibility)
    s3_self_consistency.pkl     Split-half ceiling
    s3_summary.md               Human-readable summary

Seeds: 42–51 (10 seeds per generator, 5 for BG22)
Metric suite: 84 distributional metrics (ALL_85 minus 6 Levenshtein)
Impossibility diagnostics: 11 metrics

Author: Edward Bozzard, 2026
"""

import os, sys, argparse, pickle, json, time, math, importlib.util
import random as stdlib_random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# ══════════════════════════════════════════════════════════════════════
# A. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

SEED       = 42
N_SEEDS    = 10      # template + zero-corpus generators
N_SEEDS_BG = 5       # BG22 generators (slower, subsampled methodology)
N_TARGET   = 37465   # VMS token count

# Levenshtein metrics excluded from the 84-metric suite
# (computed by BG subsampling but not scored — require edit-distance,
#  which is not part of the notation-system hypothesis)
LEVENSHTEIN_6 = {
    'wordunique_mean', 'wordunique_std', 'wordunique_skew',
    'wordchange_mean', 'wordchange_std', 'wordchange_skew',
}

# ── Definitive .py → paper name mapping (from pickle 'path' field) ──
GENERATOR_REGISTRY = {
    # Zero-corpus family (3)
    'Gen-00':  {'path': 'gen_f57v.py',              'family': 'Zero-corpus',
                'desc': 'f57v baseline',             'interface': 'f57v'},
    'Gen-0M':  {'path': 'gen_scribal_manual.py',    'family': 'Zero-corpus',
                'desc': 'Manual scribal',            'interface': 'scribal'},
    'Gen-0W':  {'path': 'gen_scribal_workshop.py',  'family': 'Zero-corpus',
                'desc': 'Workshop scribal',          'interface': 'scribal'},
    # Template post-grammar family (12)
    'Gen-02':  {'path': 'gen_template_v2.py',       'family': 'Template',
                'desc': 'Template + ductus',         'interface': 'template'},
    'Gen-03':  {'path': 'gen_template_v3.py',       'family': 'Template',
                'desc': 'Refined template',          'interface': 'template'},
    'Gen-04':  {'path': 'gen_template_v4.py',       'family': 'Template',
                'desc': 'Ductus filtering',          'interface': 'template'},
    'Gen-04T': {'path': 'gen_template_v4_tuned.py', 'family': 'Template',
                'desc': 'Corpus-tuned (ablation)',   'interface': 'template'},
    'Gen-05':  {'path': 'gen_template_v5.py',       'family': 'Template',
                'desc': 'Corrected suffix (best)',   'interface': 'template'},
    'Gen-06':  {'path': 'gen_template_v6.py',       'family': 'Template',
                'desc': 'Gallows co-occurrence',     'interface': 'template'},
    'Gen-07':  {'path': 'gen_template_v7.py',       'family': 'Template',
                'desc': 'Rounded line-start',        'interface': 'template'},
    'Gen-08':  {'path': 'gen_template_v8.py',       'family': 'Template',
                'desc': 'Rounded gallows',           'interface': 'template'},
    'Gen-09':  {'path': 'gen_template_v9.py',       'family': 'Template',
                'desc': 'Variant A',                 'interface': 'template'},
    'Gen-10':  {'path': 'gen_template_v10.py',      'family': 'Template',
                'desc': 'Variant B',                 'interface': 'template'},
    'Gen-SD':  {'path': 'gen_template_ductus.py',   'family': 'Template',
                'desc': 'Scribal + ductus hybrid',   'interface': 'template'},
    'Gen-SP':  {'path': 'gen_scribal_p70c.py',      'family': 'Template',
                'desc': 'Scribal + P70C hybrid',     'interface': 'scribal_p70c'},
}

# BG22 generators are inline in reproduce_all.py (no individual .py)
BG22_NAMES = ['Bigram', 'Scribal', 'P70C', 'Dual', 'Section-Profiled', 'Combined']


# ══════════════════════════════════════════════════════════════════════
# B. PATH RESOLUTION
# ══════════════════════════════════════════════════════════════════════

def find_project_root():
    """Locate the Paper/ directory."""
    candidates = [
        Path('.'),
        Path('Paper'),
        Path(__file__).parent,
        Path(__file__).parent / 'Paper',
    ]
    for c in candidates:
        if (c / 'score_85_metrics.py').exists():
            return c.resolve()
        if (c / 'Paper' / 'score_85_metrics.py').exists():
            return (c / 'Paper').resolve()
    raise FileNotFoundError(
        "Cannot find score_85_metrics.py. Run from repo root or Paper/ directory.")


def find_data_files(root):
    """Locate all required data files."""
    files = {}

    # enriched_records.pkl
    for p in [root / 'enriched_records.pkl',
              root / 'data' / 'enriched_records.pkl',
              root.parent / 'enriched_records.pkl']:
        if p.exists():
            files['records'] = p
            break

    # p70c_full_spec_v1.json
    for p in [root / 'p70c_full_spec_v1.json',
              root / 'data' / 'p70c_full_spec_v1.json',
              root.parent / 'Paper' / 'p70c_full_spec_v1.json']:
        if p.exists():
            files['p70c'] = p
            break

    # slim.json / voynich_transcriptions_slim.json (zero-corpus generators)
    for p in [root / 'slim.json',
              root / 'voynich_transcriptions_slim.json',
              root / 'data' / 'slim.json',
              root.parent / 'slim.json',
              root.parent / 'voynich_transcriptions_slim.json',
              root.parent / 'data' / 'slim.json']:
        if p.exists():
            files['slim'] = p
            break

    if 'records' not in files:
        raise FileNotFoundError("enriched_records.pkl not found")
    if 'p70c' not in files:
        raise FileNotFoundError("p70c_full_spec_v1.json not found")

    return files


# ══════════════════════════════════════════════════════════════════════
# C. IMPOSSIBILITY DIAGNOSTICS (copied from reproduce_all.py)
# ══════════════════════════════════════════════════════════════════════

def compute_impossibility_metrics(tokens):
    """Compute 11 impossibility diagnostics."""
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

    # Repeated words
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
        "same_len_pairs": same_len_pairs / max(len(tokens)-1, 1),
        "same_len_same_word": same_len_same_word / max(len(tokens)-1, 1),
        "ev25": ev25, "ev100": ev100, "ev500": ev500, "ev1000": ev1000,
        "ev_ratio_500_25": ev_ratio,
        "wordlen_autocorr": float(ac),
        "repeated_words": rep_rate,
        "mattr_25": float(mattr25),
    }


# ══════════════════════════════════════════════════════════════════════
# D. SCORING WRAPPER
# ══════════════════════════════════════════════════════════════════════

def score_84(gen_metrics, vms_baseline, tolerances):
    """Score generator against VMS on the 84-metric suite."""
    passes, fails, details = [], [], {}
    for metric, tol in tolerances.items():
        if metric in LEVENSHTEIN_6:
            continue
        vms_val = vms_baseline.get(metric)
        gen_val = gen_metrics.get(metric)
        if vms_val is None or gen_val is None:
            continue
        delta = abs(gen_val - vms_val)
        passed = delta <= tol
        if passed:
            passes.append(metric)
        else:
            fails.append(metric)
        details[metric] = {
            'vms': vms_val, 'gen': gen_val, 'tol': tol,
            'delta': delta, 'pass': passed,
        }
    return {
        'passes': passes, 'fails': fails,
        'n_pass': len(passes), 'n_total': len(passes) + len(fails),
        'details': details,
    }


# ══════════════════════════════════════════════════════════════════════
# E. GENERATOR RUNNERS
# ══════════════════════════════════════════════════════════════════════

def run_template_generator(gen_path, n_seeds, n_target, compute_metrics_fn,
                           vms_baseline, tolerances,
                           p70c_path=None, records_path=None):
    """Run a template-interface generator (build_spec + produce_manuscript)."""
    spec_mod = importlib.util.spec_from_file_location("gen_mod", gen_path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)

    # Pass explicit paths if build_spec accepts them
    import inspect
    bs_params = inspect.signature(mod.build_spec).parameters
    bs_kwargs = {}
    if 'p70c_path' in bs_params and p70c_path:
        bs_kwargs['p70c_path'] = str(p70c_path)
    if 'records_path' in bs_params and records_path:
        bs_kwargs['records_path'] = str(records_path)
    g_spec = mod.build_spec(**bs_kwargs)
    all_metrics, all_imp, all_scores = [], [], []

    for seed_offset in range(n_seeds):
        seed = SEED + seed_offset
        corpus = mod.produce_manuscript(g_spec, n_tokens=n_target, seed=seed)
        lines = [corpus[i:i+10] for i in range(0, len(corpus), 10)]
        metrics = compute_metrics_fn(corpus, lines=lines, seed=seed)
        imp = compute_impossibility_metrics(corpus)
        s84 = score_84(metrics, vms_baseline, tolerances)
        all_metrics.append(metrics)
        all_imp.append(imp)
        all_scores.append(s84['n_pass'])

    median_metrics = _compute_medians(all_metrics)
    median_imp = _compute_medians(all_imp)

    return {
        'all_metrics': all_metrics,
        'all_impossibility': all_imp,
        'median_metrics': median_metrics,
        'median_impossibility': median_imp,
        'scores_84': all_scores,
        'median_score': int(np.median(all_scores)),
    }


def run_f57v_generator(gen_path, slim_path, n_seeds, n_target,
                       compute_metrics_fn, vms_baseline, tolerances):
    """Run gen_f57v.py (class-based interface)."""
    spec_mod = importlib.util.spec_from_file_location("gen_mod", gen_path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)

    lines = mod.load_f57v(str(slim_path))
    g_spec = mod.build_spec(lines)
    all_metrics, all_imp, all_scores = [], [], []

    for seed_offset in range(n_seeds):
        seed = SEED + seed_offset
        gen = mod.F57vGenerator(g_spec, seed=seed)
        corpus = gen.generate(n_target)
        pseudo_lines = [corpus[i:i+10] for i in range(0, len(corpus), 10)]
        metrics = compute_metrics_fn(corpus, lines=pseudo_lines, seed=seed)
        imp = compute_impossibility_metrics(corpus)
        s84 = score_84(metrics, vms_baseline, tolerances)
        all_metrics.append(metrics)
        all_imp.append(imp)
        all_scores.append(s84['n_pass'])

    median_metrics = _compute_medians(all_metrics)
    median_imp = _compute_medians(all_imp)

    return {
        'all_metrics': all_metrics,
        'all_impossibility': all_imp,
        'median_metrics': median_metrics,
        'median_impossibility': median_imp,
        'scores_84': all_scores,
        'median_score': int(np.median(all_scores)),
    }


def run_scribal_generator(gen_path, slim_path, n_seeds, n_target,
                          compute_metrics_fn, vms_baseline, tolerances):
    """Run gen_scribal_manual.py or gen_scribal_workshop.py."""
    spec_mod = importlib.util.spec_from_file_location("gen_mod", gen_path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)

    g_spec = mod.load_f57v_spec(str(slim_path))
    all_metrics, all_imp, all_scores = [], [], []

    for seed_offset in range(n_seeds):
        seed = SEED + seed_offset
        corpus = mod.produce_manuscript(g_spec, n_tokens=n_target, seed=seed)
        lines = [corpus[i:i+10] for i in range(0, len(corpus), 10)]
        metrics = compute_metrics_fn(corpus, lines=lines, seed=seed)
        imp = compute_impossibility_metrics(corpus)
        s84 = score_84(metrics, vms_baseline, tolerances)
        all_metrics.append(metrics)
        all_imp.append(imp)
        all_scores.append(s84['n_pass'])

    median_metrics = _compute_medians(all_metrics)
    median_imp = _compute_medians(all_imp)

    return {
        'all_metrics': all_metrics,
        'all_impossibility': all_imp,
        'median_metrics': median_metrics,
        'median_impossibility': median_imp,
        'scores_84': all_scores,
        'median_score': int(np.median(all_scores)),
    }


def run_scribal_p70c_generator(gen_path, n_seeds, n_target,
                                compute_metrics_fn, vms_baseline, tolerances,
                                p70c_path=None, records_path=None):
    """Run gen_scribal_p70c.py (builds spec from p70c + records)."""
    spec_mod = importlib.util.spec_from_file_location("gen_mod", gen_path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)

    # Pass explicit paths
    import inspect
    bs_params = inspect.signature(mod.build_p70c_spec).parameters
    bs_kwargs = {}
    if 'p70c_path' in bs_params and p70c_path:
        bs_kwargs['p70c_path'] = str(p70c_path)
    if 'records_path' in bs_params and records_path:
        bs_kwargs['records_path'] = str(records_path)
    g_spec = mod.build_p70c_spec(**bs_kwargs)
    all_metrics, all_imp, all_scores = [], [], []

    for seed_offset in range(n_seeds):
        seed = SEED + seed_offset
        corpus = mod.produce_manuscript(g_spec, n_tokens=n_target, seed=seed)
        lines = [corpus[i:i+10] for i in range(0, len(corpus), 10)]
        metrics = compute_metrics_fn(corpus, lines=lines, seed=seed)
        imp = compute_impossibility_metrics(corpus)
        s84 = score_84(metrics, vms_baseline, tolerances)
        all_metrics.append(metrics)
        all_imp.append(imp)
        all_scores.append(s84['n_pass'])

    median_metrics = _compute_medians(all_metrics)
    median_imp = _compute_medians(all_imp)

    return {
        'all_metrics': all_metrics,
        'all_impossibility': all_imp,
        'median_metrics': median_metrics,
        'median_impossibility': median_imp,
        'scores_84': all_scores,
        'median_score': int(np.median(all_scores)),
    }


def _compute_medians(dicts_list):
    """Compute median of each key across a list of metric dicts."""
    result = {}
    all_keys = set()
    for d in dicts_list:
        all_keys.update(d.keys())
    for k in all_keys:
        vals = [d[k] for d in dicts_list if k in d and isinstance(d[k], (int, float))]
        if vals:
            result[k] = float(np.median(vals))
    return result


# ══════════════════════════════════════════════════════════════════════
# F. BG22 GENERATORS (imported from reproduce_all.py)
# ══════════════════════════════════════════════════════════════════════

def run_bg22_generators(root, records, spec, vms_baseline, tolerances,
                        compute_metrics_fn, results_dir, force=False):
    """Run all 6 BG22 generators using functions from reproduce_all.py."""
    # Import BG22 generator functions from reproduce_all.py
    ra_spec = importlib.util.spec_from_file_location(
        "reproduce_all", str(root / 'reproduce_all.py'))
    ra = importlib.util.module_from_spec(ra_spec)
    ra_spec.loader.exec_module(ra)

    tokens = [r['token'] for r in records]
    # Seed tokens from f1r (first folio)
    f1r_tokens = [r['token'] for r in records if r.get('folio', '').startswith('f1')]
    if not f1r_tokens:
        f1r_tokens = tokens[:500]

    # Build models
    print("  Building BG22 models...")
    followers = ra.build_char_bigram_model(tokens, f1r_tokens)
    group_followers, char_exemplars = ra.build_ductus_model(tokens)
    ledger = ra.build_p70c_ledger(spec, records)

    generators = [
        ("Bigram",           ra.gen_char_bigram,           (followers, N_TARGET)),
        ("Scribal",          ra.gen_ductus,                (group_followers, char_exemplars, N_TARGET)),
        ("P70C",             ra.gen_p70c_single,           (ledger, N_TARGET)),
        ("Dual",             ra.gen_p70c_dual,             (ledger, N_TARGET)),
        ("Section-Profiled", ra.gen_p70c_section_profiled, (ledger, N_TARGET)),
        ("Combined",         ra.gen_p70c_combined,         (ledger, N_TARGET)),
    ]

    bg22_results = {}
    for name, func, func_args in generators:
        cache = results_dir / f'cache_bg22_{name.lower().replace("-","_")}.pkl'

        if cache.exists() and not force:
            print(f"  {name}: loading cache")
            with open(cache, 'rb') as f:
                bg22_results[name] = pickle.load(f)
            continue

        print(f"  {name}: generating ({N_SEEDS_BG} seeds)...")
        t0 = time.time()
        all_metrics, all_imp, all_scores = [], [], []

        for seed_offset in range(N_SEEDS_BG):
            seed = SEED + seed_offset
            rng = stdlib_random.Random(seed)
            args = func_args + (rng,)
            corpus = func(*args)[:N_TARGET]
            lines = [corpus[i:i+10] for i in range(0, len(corpus), 10)]
            metrics = compute_metrics_fn(corpus, lines=lines, seed=seed)
            imp = compute_impossibility_metrics(corpus)
            s84 = score_84(metrics, vms_baseline, tolerances)
            all_metrics.append(metrics)
            all_imp.append(imp)
            all_scores.append(s84['n_pass'])

        result = {
            'all_metrics': all_metrics,
            'all_impossibility': all_imp,
            'median_metrics': _compute_medians(all_metrics),
            'median_impossibility': _compute_medians(all_imp),
            'scores_84': all_scores,
            'median_score': int(np.median(all_scores)),
        }
        bg22_results[name] = result

        with open(cache, 'wb') as f:
            pickle.dump(result, f)
        print(f"    {name}: {result['median_score']}/84 ({time.time()-t0:.1f}s)")

    return bg22_results


# ══════════════════════════════════════════════════════════════════════
# G. SELF-CONSISTENCY CEILING
# ══════════════════════════════════════════════════════════════════════

def compute_self_consistency(records, compute_metrics_fn, tolerances):
    """Split-half analysis under 4 partition schemes."""
    folios = sorted(set(r['folio'] for r in records))
    n = len(folios)

    partitions = {
        'sequential':       (folios[:n//2], folios[n//2:]),
        'reverse':          (folios[n//2:], folios[:n//2]),
        'odd_even':         (folios[0::2], folios[1::2]),
        'even_odd':         (folios[1::2], folios[0::2]),
    }

    agreements = []
    for name, (set_a, set_b) in partitions.items():
        a_set, b_set = set(set_a), set(set_b)
        tokens_a = [r['token'] for r in records if r['folio'] in a_set]
        tokens_b = [r['token'] for r in records if r['folio'] in b_set]

        lines_a = [tokens_a[i:i+10] for i in range(0, len(tokens_a), 10)]
        lines_b = [tokens_b[i:i+10] for i in range(0, len(tokens_b), 10)]

        m_a = compute_metrics_fn(tokens_a, lines=lines_a, seed=42)
        m_b = compute_metrics_fn(tokens_b, lines=lines_b, seed=43)

        n_agree = 0
        n_total = 0
        for metric, tol in tolerances.items():
            if metric in LEVENSHTEIN_6:
                continue
            va, vb = m_a.get(metric), m_b.get(metric)
            if va is None or vb is None:
                continue
            n_total += 1
            if abs(va - vb) <= tol:
                n_agree += 1

        pct = n_agree / n_total if n_total > 0 else 0
        agreements.append(pct)
        print(f"  {name:>15}: {n_agree}/{n_total} = {pct:.1%}")

    mean_pct = np.mean(agreements)
    print(f"  Mean self-consistency: {mean_pct:.1%}")
    return {'partitions': {k: a for k, a in zip(partitions.keys(), agreements)},
            'mean': float(mean_pct)}


# ══════════════════════════════════════════════════════════════════════
# H. ABLATION SWEEPS
# ══════════════════════════════════════════════════════════════════════

def run_ablation_sweeps(root, vms_baseline, tolerances, compute_metrics_fn,
                        results_dir, force=False,
                        p70c_path=None, records_path=None):
    """
    Run production-rule ablation sweeps using Gen-05 as the base generator.
    Sweeps: fresh_rate × line_fresh, lookback_depth, suffix_change_rate.

    NOTE: This requires Gen-05 (gen_template_v5.py) to expose ablation
    parameters. If the generator's produce_manuscript does not accept
    these parameters, this section will use the cached ablation data
    from the original analysis session.
    """
    cache = results_dir / 's3_ablation_results.pkl'
    if cache.exists() and not force:
        print("  Loading cached ablation results")
        with open(cache, 'rb') as f:
            return pickle.load(f)

    print("  *** Ablation sweeps require generator modifications ***")
    print("  *** Using parameter sweep framework                 ***")

    # Import Gen-05
    gen_path = root / 'Generators' / 'gen_template_v5.py'
    if not gen_path.exists():
        print(f"  ERROR: {gen_path} not found — cannot run ablations")
        return None

    spec_mod = importlib.util.spec_from_file_location("gen_mod", str(gen_path))
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)

    # Build spec with explicit paths
    import inspect
    bs_params = inspect.signature(mod.build_spec).parameters
    bs_kwargs = {}
    if 'p70c_path' in bs_params and p70c_path:
        bs_kwargs['p70c_path'] = str(p70c_path)
    if 'records_path' in bs_params and records_path:
        bs_kwargs['records_path'] = str(records_path)
    g_spec = mod.build_spec(**bs_kwargs)

    ablation_results = {
        'fresh_rate': {},
        'lookback': {},
        'suffix_rate': {},
    }

    # ── Fresh rate × line_fresh sweep ──
    print("\n  Sweep 1: Fresh rate × line_fresh")
    fresh_rates = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    line_fresh_opts = [True, False]

    for fr in fresh_rates:
        for lf in line_fresh_opts:
            key = f"fr={fr:.2f}_lf={lf}"
            print(f"    {key}...", end=' ', flush=True)
            scores = []
            for seed_offset in range(N_SEEDS):
                seed = SEED + seed_offset
                try:
                    corpus = mod.produce_manuscript(
                        g_spec, n_tokens=N_TARGET, seed=seed,
                        fresh_rate=fr, line_fresh=lf)
                except TypeError:
                    # Generator doesn't support ablation params
                    print("(unsupported — skipping sweep)")
                    ablation_results = None
                    return ablation_results
                lines = [corpus[i:i+10] for i in range(0, len(corpus), 10)]
                metrics = compute_metrics_fn(corpus, lines=lines, seed=seed)
                s = score_84(metrics, vms_baseline, tolerances)
                scores.append(s['n_pass'])
            ablation_results['fresh_rate'][key] = {
                'fresh_rate': fr, 'line_fresh': lf,
                'scores_84': scores,
                'median': int(np.median(scores)),
            }
            print(f"{int(np.median(scores))}/84")

    # ── Lookback depth sweep ──
    print("\n  Sweep 2: Lookback depth")
    lookback_depths = [1, 2, 3, 5, 8, 10, 12, 15, 20]
    best_fr = 0.20  # use optimal fresh rate

    for lb in lookback_depths:
        key = f"lb={lb}"
        print(f"    {key}...", end=' ', flush=True)
        scores = []
        for seed_offset in range(N_SEEDS):
            seed = SEED + seed_offset
            try:
                corpus = mod.produce_manuscript(
                    g_spec, n_tokens=N_TARGET, seed=seed,
                    fresh_rate=best_fr, lookback=lb)
            except TypeError:
                print("(unsupported)")
                ablation_results = None
                return ablation_results
            lines = [corpus[i:i+10] for i in range(0, len(corpus), 10)]
            metrics = compute_metrics_fn(corpus, lines=lines, seed=seed)
            s = score_84(metrics, vms_baseline, tolerances)
            scores.append(s['n_pass'])
        ablation_results['lookback'][key] = {
            'lookback': lb, 'fresh_rate': best_fr,
            'scores_84': scores,
            'median': int(np.median(scores)),
        }
        print(f"{int(np.median(scores))}/84")

    # ── Suffix change rate sweep ──
    print("\n  Sweep 3: Suffix change rate")
    suffix_rates = [0.50, 0.60, 0.70, 0.75, 0.80, 0.87, 0.90, 0.95, 1.00]

    for sfx in suffix_rates:
        key = f"sfx={sfx:.2f}"
        print(f"    {key}...", end=' ', flush=True)
        scores = []
        for seed_offset in range(N_SEEDS):
            seed = SEED + seed_offset
            try:
                corpus = mod.produce_manuscript(
                    g_spec, n_tokens=N_TARGET, seed=seed,
                    suffix_change_rate=sfx)
            except TypeError:
                print("(unsupported)")
                ablation_results = None
                return ablation_results
            lines = [corpus[i:i+10] for i in range(0, len(corpus), 10)]
            metrics = compute_metrics_fn(corpus, lines=lines, seed=seed)
            s = score_84(metrics, vms_baseline, tolerances)
            scores.append(s['n_pass'])
        ablation_results['suffix_rate'][key] = {
            'suffix_rate': sfx,
            'scores_84': scores,
            'median': int(np.median(scores)),
        }
        print(f"{int(np.median(scores))}/84")

    # Save
    with open(cache, 'wb') as f:
        pickle.dump(ablation_results, f)
    print("  Saved ablation results")
    return ablation_results


# ══════════════════════════════════════════════════════════════════════
# I. SUMMARY OUTPUT
# ══════════════════════════════════════════════════════════════════════

def write_summary(all_results, ablation, ceiling, results_dir):
    """Write human-readable summary to markdown."""
    lines = []
    lines.append("# S3 Reproduction Results")
    lines.append(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# Seeds: {SEED}–{SEED + N_SEEDS - 1}")
    lines.append(f"# Metric suite: 84 (ALL_85 minus 6 Levenshtein)")
    lines.append("")

    # Ranked table
    lines.append("## All 21 Generators (ranked)")
    lines.append(f"{'Rank':>4}  {'Generator':25s}  {'Family':15s}  {'Score':>6}  {'Range'}")
    lines.append("-" * 75)
    ranked = sorted(all_results.items(), key=lambda x: -x[1]['median_score'])
    for i, (name, r) in enumerate(ranked, 1):
        lo, hi = min(r['scores_84']), max(r['scores_84'])
        lines.append(
            f"{i:4d}  {name:25s}  {r.get('family',''):15s}  "
            f"{r['median_score']:3d}/84  [{lo}–{hi}]")

    lines.append("")

    # Structural breaks
    lines.append("## Structural Breaks")
    for name, r in ranked:
        if name in ('Gen-05', 'Section-Profiled', 'Gen-00'):
            lines.append(f"  {name}: {r['median_score']}/84")

    # Ablation summary
    if ablation:
        lines.append("")
        lines.append("## Ablation Summary")
        if 'fresh_rate' in ablation:
            lines.append(f"  Fresh rate × line-fresh: {len(ablation['fresh_rate'])} configs")
        if 'lookback' in ablation:
            lines.append(f"  Lookback depth: {len(ablation['lookback'])} configs")
        if 'suffix_rate' in ablation:
            lines.append(f"  Suffix change rate: {len(ablation['suffix_rate'])} configs")

    # Ceiling
    if ceiling:
        lines.append("")
        lines.append(f"## Self-Consistency Ceiling: {ceiling['mean']:.1%}")

    md = "\n".join(lines) + "\n"
    with open(results_dir / 's3_summary.md', 'w') as f:
        f.write(md)
    print(md)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="S3 Definitive Reproduction")
    parser.add_argument('--skip-bg22', action='store_true')
    parser.add_argument('--skip-template', action='store_true')
    parser.add_argument('--skip-ablation', action='store_true')
    parser.add_argument('--skip-ceiling', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoints')
    parser.add_argument('--force', action='store_true',
                        help='Re-run everything, ignore caches')
    args = parser.parse_args()

    force = args.force
    resume = args.resume or (not force)  # default: resume from caches

    # ── Locate files ──
    print("=" * 70)
    print("S3 DEFINITIVE REPRODUCTION")
    print("=" * 70)

    root = find_project_root()
    data_files = find_data_files(root)
    gen_dir = root / 'Generators'
    results_dir = root / 'results' / 's3'
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Project root:  {root}")
    print(f"  Generator dir: {gen_dir}")
    print(f"  Results dir:   {results_dir}")
    print(f"  Data files:")
    for k, v in data_files.items():
        print(f"    {k}: {v}")

    has_slim = 'slim' in data_files

    # ── Add project root to sys.path for imports ──
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(gen_dir) not in sys.path:
        sys.path.insert(0, str(gen_dir))

    # ── Load data ──
    print("\n1. Loading data...")
    with open(data_files['records'], 'rb') as f:
        records = pickle.load(f)
    with open(data_files['p70c']) as f:
        p70c_spec = json.load(f)
    print(f"  {len(records)} records, {len(p70c_spec['entries'])} P70C entries")

    # ── Import scoring ──
    print("\n2. Importing score_85_metrics...")
    from score_85_metrics import compute_metrics, TOLERANCES, CORE_15, ALL_85

    # Verify 84-metric suite
    scored_84 = sorted(k for k in TOLERANCES if k not in LEVENSHTEIN_6)
    print(f"  ALL_85 count: {len(ALL_85)}")
    print(f"  TOLERANCES count: {len(TOLERANCES)}")
    print(f"  84-metric suite: {len(scored_84)} metrics")
    assert len(scored_84) == 84, f"Expected 84, got {len(scored_84)}"

    # ── VMS baseline ──
    print("\n3. Computing VMS baseline...")
    baseline_cache = results_dir / 's3_vms_baseline.pkl'
    if baseline_cache.exists() and resume:
        print("  Loading cached baseline")
        with open(baseline_cache, 'rb') as f:
            baseline_data = pickle.load(f)
        vms_baseline = baseline_data['metrics']
        vms_imp = baseline_data['impossibility']
    else:
        tokens = [r['token'] for r in records]
        lines_vms = [tokens[i:i+10] for i in range(0, len(tokens), 10)]
        vms_baseline = compute_metrics(tokens, lines=lines_vms, seed=42)
        vms_imp = compute_impossibility_metrics(tokens)
        baseline_data = {'metrics': vms_baseline, 'impossibility': vms_imp}
        with open(baseline_cache, 'wb') as f:
            pickle.dump(baseline_data, f)
        print(f"  Computed and cached ({len(vms_baseline)} metrics)")

    # ══════════════════════════════════════════════════════════════════
    # RUN GENERATORS
    # ══════════════════════════════════════════════════════════════════
    all_results = {}

    # ── BG22 generators (6) ──
    if not args.skip_bg22:
        print("\n4. Running BG22 generators (6 × 5 seeds)...")
        bg22 = run_bg22_generators(root, records, p70c_spec, vms_baseline,
                                   TOLERANCES, compute_metrics, results_dir,
                                   force=force)
        for name, result in bg22.items():
            result['family'] = 'BG22'
            all_results[name] = result
            print(f"  {name:>20}: {result['median_score']}/84")
    else:
        print("\n4. Skipping BG22 (--skip-bg22)")
        # Try loading from caches
        for name in ['Bigram', 'Scribal', 'P70C', 'Dual',
                     'Section-Profiled', 'Combined']:
            cache = results_dir / f'cache_bg22_{name.lower().replace("-","_")}.pkl'
            if cache.exists():
                with open(cache, 'rb') as f:
                    result = pickle.load(f)
                result['family'] = 'BG22'
                all_results[name] = result

    # ── Template + zero-corpus generators (15) ──
    if not args.skip_template:
        print(f"\n5. Running template & zero-corpus generators "
              f"({len(GENERATOR_REGISTRY)} × {N_SEEDS} seeds)...")

        for paper_name, info in GENERATOR_REGISTRY.items():
            gen_path = gen_dir / info['path']
            cache = results_dir / f'cache_{paper_name.lower().replace("-","_")}.pkl'

            if cache.exists() and resume:
                print(f"  {paper_name}: loading cache")
                with open(cache, 'rb') as f:
                    result = pickle.load(f)
                result['family'] = info['family']
                all_results[paper_name] = result
                continue

            if not gen_path.exists():
                print(f"  {paper_name}: SKIP (file not found: {info['path']})")
                continue

            # Zero-corpus generators need voynich_transcriptions_slim.json
            if info['interface'] in ('f57v', 'scribal') and not has_slim:
                print(f"  {paper_name}: SKIP (voynich_transcriptions_slim.json not found)")
                continue

            print(f"  {paper_name} ({info['desc']})...", end=' ', flush=True)
            t0 = time.time()

            try:
                if info['interface'] == 'f57v':
                    result = run_f57v_generator(
                        gen_path, data_files['slim'], N_SEEDS, N_TARGET,
                        compute_metrics, vms_baseline, TOLERANCES)
                elif info['interface'] == 'scribal':
                    result = run_scribal_generator(
                        gen_path, data_files['slim'], N_SEEDS, N_TARGET,
                        compute_metrics, vms_baseline, TOLERANCES)
                elif info['interface'] == 'scribal_p70c':
                    result = run_scribal_p70c_generator(
                        gen_path, N_SEEDS, N_TARGET,
                        compute_metrics, vms_baseline, TOLERANCES,
                        p70c_path=data_files['p70c'],
                        records_path=data_files['records'])
                elif info['interface'] == 'template':
                    result = run_template_generator(
                        gen_path, N_SEEDS, N_TARGET,
                        compute_metrics, vms_baseline, TOLERANCES,
                        p70c_path=data_files['p70c'],
                        records_path=data_files['records'])
                else:
                    print(f"UNKNOWN INTERFACE: {info['interface']}")
                    continue

                result['family'] = info['family']
                all_results[paper_name] = result

                with open(cache, 'wb') as f:
                    pickle.dump(result, f)
                print(f"{result['median_score']}/84 ({time.time()-t0:.1f}s)")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\n5. Skipping template generators (--skip-template)")

    # ── Save consolidated results ──
    print(f"\n6. Saving {len(all_results)} generator results...")
    with open(results_dir / 's3_all_generators.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # Print ranked table
    print(f"\n{'Rank':>4}  {'Generator':25s}  {'Family':15s}  {'Med':>3}/84  "
          f"{'Min':>3}  {'Max':>3}  {'N':>2}")
    print("-" * 80)
    ranked = sorted(all_results.items(), key=lambda x: -x[1]['median_score'])
    for i, (name, r) in enumerate(ranked, 1):
        lo, hi = min(r['scores_84']), max(r['scores_84'])
        n = len(r['scores_84'])
        print(f"{i:4d}  {name:25s}  {r.get('family',''):15s}  "
              f"{r['median_score']:3d}/84  {lo:3d}  {hi:3d}  {n:2d}")

    # ── Ablation sweeps ──
    ablation = None
    if not args.skip_ablation:
        print("\n7. Running ablation sweeps...")
        ablation = run_ablation_sweeps(root, vms_baseline, TOLERANCES,
                                        compute_metrics, results_dir,
                                        force=force,
                                        p70c_path=data_files['p70c'],
                                        records_path=data_files['records'])
        if ablation is None:
            print("  Ablation sweeps not supported by current generator.")
            print("  Generator produce_manuscript() needs to accept:")
            print("    fresh_rate, line_fresh, lookback, suffix_change_rate")
    else:
        print("\n7. Skipping ablations (--skip-ablation)")
        cache = results_dir / 's3_ablation_results.pkl'
        if cache.exists():
            with open(cache, 'rb') as f:
                ablation = pickle.load(f)

    # ── Self-consistency ceiling ──
    ceiling = None
    if not args.skip_ceiling:
        print("\n8. Computing self-consistency ceiling...")
        ceiling_cache = results_dir / 's3_self_consistency.pkl'
        if ceiling_cache.exists() and resume:
            print("  Loading cached ceiling")
            with open(ceiling_cache, 'rb') as f:
                ceiling = pickle.load(f)
        else:
            ceiling = compute_self_consistency(records, compute_metrics,
                                               TOLERANCES)
            with open(ceiling_cache, 'wb') as f:
                pickle.dump(ceiling, f)
    else:
        print("\n8. Skipping ceiling (--skip-ceiling)")

    # ── Summary ──
    print("\n9. Writing summary...")
    write_summary(all_results, ablation, ceiling, results_dir)

    print("\n" + "=" * 70)
    print("REPRODUCTION COMPLETE")
    print("=" * 70)
    print(f"  Generators scored: {len(all_results)}/21")
    print(f"  Results in: {results_dir}/")
    print(f"  Files:")
    for f in sorted(results_dir.glob('s3_*.pkl')):
        print(f"    {f.name}")

    return all_results, ablation, ceiling


if __name__ == '__main__':
    main()

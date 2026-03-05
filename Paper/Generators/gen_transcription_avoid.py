#!/usr/bin/env python3
"""
gen_transcription_avoid.py — Per-Triple All-History Avoidance Model

The final generator from Paper 1. Given the exact VMS triple stream
(prefix, gallows, core), selects suffix variants using frequency-weighted
sampling with a binary penalty (p=0.1) against any surface form previously
produced by the same triple.

Scores: 67–76/84 across 9 sections (mean 71.3), 98–99% type recovery,
13–14/15 CORE_15, 2,950 types vs VMS 2,982.

The model demonstrates that:
  1. The triple stream carries ~10 points of information beyond generative models
  2. Per-triple avoidance is the specific suffix-selection mechanism
  3. The avoidance operates as all-history (windowed K=3–15 fails at 43–50/84)
  4. This is consistent with an internalised production habit, not visual scanning

Usage:
  python gen_transcription_avoid.py                          # Score all 9 sections
  python gen_transcription_avoid.py --section Stars          # Score one section
  python gen_transcription_avoid.py --seed 43                # Different seed
  python gen_transcription_avoid.py --penalty 0.05           # Different penalty
  python gen_transcription_avoid.py --output generated.txt   # Save generated text

Requirements:
  - enriched_records.pkl (PGCS-parsed VMS corpus)
  - score_85_metrics-5.py (84-metric scoring framework)

Edward Bozzard, 2026. github.com/digitalgoldfisj79/Voynichdecomp
"""

import pickle, random, sys, os, argparse
from collections import Counter, defaultdict


# ─────────────────────────────────────────────────────────────────────
# Core model
# ─────────────────────────────────────────────────────────────────────

def build_section_data(records, section=None):
    """Extract line structure, triple stream, and suffix menus for a section."""
    if section:
        recs = [r for r in records if r['section'] == section]
    else:
        recs = list(records)

    if not recs:
        raise ValueError(f"No records found for section '{section}'")

    # Build line structure (preserving folio/line boundaries)
    lines = []        # list of lists of tokens
    line_triples = [] # list of lists of (prefix, gallows, m_core)
    cur_tok = []
    cur_tri = []
    cur_key = None

    for r in recs:
        key = (r['folio'], r['line_no'])
        triple = (r['prefix'], r['gallows'], r['m_core'])
        if key != cur_key:
            if cur_tok:
                lines.append(cur_tok)
                line_triples.append(cur_tri)
            cur_tok = [r['token']]
            cur_tri = [triple]
            cur_key = key
        else:
            cur_tok.append(r['token'])
            cur_tri.append(triple)
    if cur_tok:
        lines.append(cur_tok)
        line_triples.append(cur_tri)

    # Build per-triple suffix menus: triple → {surface_form: count}
    menus = defaultdict(Counter)
    for r in recs:
        triple = (r['prefix'], r['gallows'], r['m_core'])
        menus[triple][r['token']] += 1

    return lines, line_triples, dict(menus), recs


def generate_avoid(line_triples, menus, seed=42, penalty=0.1):
    """
    Per-triple all-history avoidance generator.

    For each token position:
      1. Look up the VMS triple (prefix, gallows, core)
      2. Get the frequency-weighted menu of surface forms for that triple
      3. Penalise any form already produced by this triple (weight *= penalty)
      4. Sample from the reweighted menu
      5. Add the chosen form to this triple's history

    Args:
        line_triples: list of lists of (prefix, gallows, m_core) — the VMS triple stream
        menus: dict mapping triple → {surface_form: count}
        seed: random seed
        penalty: weight multiplier for previously-used forms (default 0.1 = 10× penalty)

    Returns:
        gen_lines: list of lists of generated tokens
        gen_flat: flat list of all generated tokens
    """
    rng = random.Random(seed)
    triple_history = defaultdict(set)  # triple → set of previously produced forms
    gen_lines = []

    for lt in line_triples:
        row = []
        for triple in lt:
            menu = menus.get(triple, {})
            if not menu:
                row.append('∅')
                continue

            items = list(menu.keys())
            weights = []
            for tok in items:
                w = menu[tok]
                if tok in triple_history[triple]:
                    w *= penalty
                weights.append(w)

            total = sum(weights)
            if total == 0:
                # All forms penalised equally — fall back to raw frequencies
                weights = [menu[tok] for tok in items]
                total = sum(weights)

            weights = [w / total for w in weights]
            token = rng.choices(items, weights=weights, k=1)[0]
            row.append(token)
            triple_history[triple].add(token)

        gen_lines.append(row)

    gen_flat = [tok for line in gen_lines for tok in line]
    return gen_lines, gen_flat


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────

def load_scorer(scorer_path):
    """Import the 84-metric scorer as a module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("scorer", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    return scorer


def score_section(records, section, scorer, seed=42, penalty=0.1, verbose=True):
    """Score one section and return results dict."""
    lines, line_triples, menus, recs = build_section_data(records, section)
    vms_tokens = [r['token'] for r in recs]

    # Generate
    gen_lines, gen_flat = generate_avoid(line_triples, menus, seed=seed, penalty=penalty)

    # Compute metrics
    vms_metrics = scorer.compute_metrics(vms_tokens, lines=lines)
    gen_metrics = scorer.compute_metrics(gen_flat, lines=gen_lines)
    result = scorer.score_against_vms(gen_metrics, vms_metrics)

    # Summary stats
    vms_types = len(set(vms_tokens))
    gen_types = len(set(gen_flat))
    type_recovery = gen_types / vms_types if vms_types > 0 else 0
    exact_match = sum(1 for vl, gl in zip(lines, gen_lines)
                      for v, g in zip(vl, gl) if v == g) / len(gen_flat)

    # CORE_15 diagnostic
    core15_names = [
        'autocorr_wordlen', 'autocorr_wordfreq', 'autocorr_hapax_25',
        'charbias_mean', 'charbias_skew', 'H1_unigram',
        'h2_conditional', 'wordlen_mean', 'wordlen_unique_mean',
        'msttr_25', 'heaps_beta', 'chardist_max',
        'digraph_coverage', 'zipf_alpha', 'tripled_words'
    ]
    details = result.get('details', {})
    core15_pass = sum(1 for m in core15_names if details.get(m, {}).get('pass', False))

    info = {
        'section': section or 'Full corpus',
        'seed': seed,
        'penalty': penalty,
        'score': result['n_pass'],
        'total': result['n_total'],
        'core15': core15_pass,
        'vms_types': vms_types,
        'gen_types': gen_types,
        'type_recovery': type_recovery,
        'exact_match': exact_match,
        'n_tokens': len(gen_flat),
        'vms_metrics': vms_metrics,
        'gen_metrics': gen_metrics,
        'result': result,
    }

    if verbose:
        print(f"  {info['section']:<18s}  {info['score']}/{info['total']}  "
              f"C15={info['core15']}/15  Types={info['gen_types']}/{info['vms_types']} "
              f"({info['type_recovery']:.1%})  Match={info['exact_match']:.1%}")

    return info


# ─────────────────────────────────────────────────────────────────────
# Text output
# ─────────────────────────────────────────────────────────────────────

def write_comparison(records, section, outpath, seed=42, penalty=0.1):
    """Write side-by-side VMS vs generated text to a file."""
    lines, line_triples, menus, recs = build_section_data(records, section)
    gen_lines, gen_flat = generate_avoid(line_triples, menus, seed=seed, penalty=penalty)

    # Get folio/line keys
    folio_lines = []
    cur_key = None
    for r in recs:
        key = (r['folio'], r['line_no'])
        if key != cur_key:
            folio_lines.append(key)
            cur_key = key

    with open(outpath, 'w') as f:
        sec_label = section or 'Full corpus'
        f.write(f"GENERATED TEXT COMPARISON — {sec_label}\n")
        f.write(f"Model: per-triple all-history avoidance (penalty={penalty}, seed={seed})\n")
        f.write(f"{'=' * 80}\n\n")

        total = same = 0
        for i, (vline, gline) in enumerate(zip(lines, gen_lines)):
            if i < len(folio_lines):
                folio, lno = folio_lines[i]
            else:
                folio, lno = '?', '?'

            vms_str = '  '.join(vline)
            gen_str = '  '.join(gline)
            f.write(f"{folio} L{lno}:\n")
            f.write(f"  VMS: {vms_str}\n")
            f.write(f"  GEN: {gen_str}\n")

            diffs = sum(1 for v, g in zip(vline, gline) if v != g)
            total += len(vline)
            same += len(vline) - diffs
            if diffs:
                f.write(f"  [{diffs}/{len(vline)} differ]\n")
            else:
                f.write(f"  [IDENTICAL]\n")
            f.write("\n")

        f.write(f"{'=' * 80}\n")
        f.write(f"Exact token match: {same}/{total} = {same/total:.1%}\n")

    print(f"  Written to {outpath}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

SECTIONS = [
    'Stars', 'Balneological', 'Herbal-B', 'Herbal-A',
    'Pharmaceutical', 'Rosettes', 'Zodiac', 'Astronomical', 'Cosmological'
]


def find_file(name, search_paths):
    """Search for a file in multiple locations."""
    for base in search_paths:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Per-triple all-history avoidance transcription model (67–76/84)')
    parser.add_argument('--section', type=str, default=None,
                        help='Score a single section (default: all 9)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--penalty', type=float, default=0.1,
                        help='Avoidance penalty weight (default: 0.1)')
    parser.add_argument('--output', type=str, default=None,
                        help='Write generated text comparison to file')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to enriched_records.pkl')
    parser.add_argument('--scorer', type=str, default=None,
                        help='Path to score_85_metrics-5.py')
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds to run (default: 1)')
    parser.add_argument('--pickle', type=str, default=None,
                        help='Save results to pickle file')
    args = parser.parse_args()

    # Find data files
    search_paths = [
        '.', './handover/data', './data', './session_data', './repo',
        '/home/claude/repo', '/home/claude/session_data',
        '/home/claude/handover/data'
    ]

    data_path = args.data or find_file('enriched_records.pkl', search_paths)
    scorer_path = args.scorer or find_file('score_85_metrics-5.py', search_paths)

    if not data_path or not os.path.exists(data_path):
        print(f"ERROR: Cannot find enriched_records.pkl")
        print(f"  Searched: {search_paths}")
        print(f"  Use --data /path/to/enriched_records.pkl")
        sys.exit(1)

    if not scorer_path or not os.path.exists(scorer_path):
        print(f"ERROR: Cannot find score_85_metrics-5.py")
        print(f"  Use --scorer /path/to/score_85_metrics-5.py")
        sys.exit(1)

    # Load data
    print(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        records = pickle.load(f)
    print(f"  {len(records)} records loaded")

    scorer = load_scorer(scorer_path)

    # Text output mode
    if args.output:
        section = args.section or 'Stars'
        print(f"\nGenerating text for {section} (seed={args.seed}, penalty={args.penalty})")
        write_comparison(records, section, args.output,
                         seed=args.seed, penalty=args.penalty)
        return

    # Scoring mode
    sections = [args.section] if args.section else SECTIONS

    print(f"\n{'=' * 75}")
    print(f"Per-Triple All-History Avoidance Model")
    print(f"Penalty={args.penalty}  Seeds={args.seed}–{args.seed + args.seeds - 1}")
    print(f"{'=' * 75}")

    all_results = []
    for seed in range(args.seed, args.seed + args.seeds):
        print(f"\n--- Seed {seed} ---")
        print(f"  {'Section':<18s}  {'Score':>7s}  {'C15':>7s}  {'Types':>16s}  {'Match':>7s}")
        print(f"  {'-'*18}  {'-'*7}  {'-'*7}  {'-'*16}  {'-'*7}")

        seed_results = []
        for section in sections:
            info = score_section(records, section, scorer,
                                 seed=seed, penalty=args.penalty)
            seed_results.append(info)
            all_results.append(info)

        if len(sections) > 1:
            scores = [r['score'] for r in seed_results]
            c15s = [r['core15'] for r in seed_results]
            recoveries = [r['type_recovery'] for r in seed_results]
            print(f"\n  Mean: {sum(scores)/len(scores):.1f}/{seed_results[0]['total']}  "
                  f"C15={sum(c15s)/len(c15s):.1f}/15  "
                  f"TypeRec={sum(recoveries)/len(recoveries):.1%}")
            print(f"  Range: {min(scores)}–{max(scores)}")

    # Multi-seed summary
    if args.seeds > 1 and len(sections) > 1:
        print(f"\n{'=' * 75}")
        print(f"MULTI-SEED SUMMARY ({args.seeds} seeds × {len(sections)} sections)")
        print(f"{'=' * 75}")
        for section in sections:
            sec_results = [r for r in all_results if r['section'] == section]
            scores = [r['score'] for r in sec_results]
            print(f"  {section:<18s}  mean={sum(scores)/len(scores):.1f}  "
                  f"range={min(scores)}–{max(scores)}")

    # Save pickle
    if args.pickle:
        pickle.dump(all_results, open(args.pickle, 'wb'))
        print(f"\nResults saved to {args.pickle}")


if __name__ == '__main__':
    main()

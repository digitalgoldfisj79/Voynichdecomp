#!/usr/bin/env python3
"""
Supplement S3: CV Syllable Reader
===================================
Reads any VMS folio through the two-table architecture and computes
consonant-vowel enrichment against the Herbal-A baseline.

Usage:
    python S3_cv_folio_reader.py                    # all HA folios
    python S3_cv_folio_reader.py --folio f2r        # single folio
    python S3_cv_folio_reader.py --section Herbal-B # full section

Requires: enriched_records.pkl, ci_corpus_parsed.pkl
"""

import pickle, argparse, sys
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import hypergeom

# ══════════════════════════════════════════════════════════════
# GRID MAPPING (from Paper 2 §4)
# ══════════════════════════════════════════════════════════════

# m_core first character → consonant row
CORE_TO_ROW = {
    'o': 'o', 'c': 'c', 'e': 'e', 'a': 'a',
    'd': 'd', 'l': 'l', 'r': 'r'
}

# suffix family → vowel column
FAM_TO_VOWEL = {
    'Y': 'a', 'N': 'i', 'L': 'a', 'R': 'e',
    'BARE': 'u', 'M': 'a'
}

# Nomenclator assignments (Paper 2 §5.4)
NOMENCLATOR = {
    'et': 'Y', 'postea': 'Y',
    'in': 'N', 'cum': 'N', 'hoc': 'N',
    'de': 'L', 'ad': 'L', 'habet': 'L', 'uel': 'L',
    'que': 'L', 'supra': 'L',
    'eam': 'R'
}


def read_token(record):
    """Convert a VMS token to its CV reading."""
    if record['empty_core']:
        # EC token → function word family
        fam = record.get('sfx_fam', 'BARE')
        return ('EC', fam, None)
    else:
        # FC token → consonant-vowel pair
        m_core = record.get('m_core', '')
        fam = record.get('sfx_fam', 'BARE')
        if m_core:
            row = CORE_TO_ROW.get(m_core[0], '?')
        else:
            row = '?'
        vowel = FAM_TO_VOWEL.get(fam, '?')
        cv = row + vowel
        return ('FC', fam, cv)


def folio_cv_profile(records):
    """Build CV frequency profile for a set of records."""
    cv_counts = Counter()
    fc_total = 0
    for r in records:
        route, fam, cv = read_token(r)
        if route == 'FC' and cv and '?' not in cv:
            cv_counts[cv] += 1
            fc_total += 1
    return cv_counts, fc_total


def enrichment_test(folio_records, baseline_records, bonferroni_n=960):
    """
    Hypergeometric enrichment test for each CV pair.
    Tests whether a folio has more of a given CV than expected
    from the baseline distribution.
    """
    folio_cv, folio_fc = folio_cv_profile(folio_records)
    base_cv, base_fc = folio_cv_profile(baseline_records)

    results = []
    for cv in sorted(set(list(folio_cv.keys()) + list(base_cv.keys()))):
        k = folio_cv.get(cv, 0)       # successes in sample
        n = folio_fc                    # sample size
        K = base_cv.get(cv, 0)         # successes in population
        N = base_fc                     # population size

        if k == 0 or K == 0 or N == 0:
            continue

        # P(X >= k) under hypergeometric
        p = hypergeom.sf(k - 1, N, K, n)
        rate_folio = k / n if n > 0 else 0
        rate_base = K / N if N > 0 else 0
        enrichment = rate_folio / rate_base if rate_base > 0 else float('inf')

        results.append({
            'cv': cv,
            'count': k,
            'folio_fc': n,
            'rate': rate_folio,
            'base_rate': rate_base,
            'enrichment': enrichment,
            'p': p,
            'bonferroni': p < (0.05 / bonferroni_n)
        })

    results.sort(key=lambda x: x['p'])
    return results


def main():
    parser = argparse.ArgumentParser(description='CV Syllable Reader')
    parser.add_argument('--folio', type=str, default=None)
    parser.add_argument('--section', type=str, default=None)
    parser.add_argument('--data', type=str, default='enriched_records.pkl')
    args = parser.parse_args()

    with open(args.data, 'rb') as f:
        records = pickle.load(f)

    # Baseline: all Herbal-A
    ha = [r for r in records if r.get('section') == 'Herbal-A']
    print(f"Baseline: Herbal-A, {len(ha)} tokens")

    # Target folios
    if args.folio:
        target_folios = [args.folio]
    elif args.section:
        target_folios = sorted(set(r['folio'] for r in records
                                   if r.get('section') == args.section))
    else:
        target_folios = sorted(set(r['folio'] for r in ha))

    # Run enrichment for each folio
    print(f"\n{'Folio':<8} {'CV':<5} {'Count':>5} {'Rate':>7} {'Base':>7} "
          f"{'Enrich':>7} {'p-value':>12} {'Bonf':>5}")
    print("-" * 68)

    all_hits = []
    for fol in target_folios:
        fol_recs = [r for r in records if r['folio'] == fol]
        if len(fol_recs) < 10:
            continue
        results = enrichment_test(fol_recs, ha)
        for res in results:
            if res['p'] < 0.01:  # show only suggestive hits
                bf = '***' if res['bonferroni'] else ''
                print(f"{fol:<8} {res['cv']:<5} {res['count']:>5} "
                      f"{res['rate']:>6.1%} {res['base_rate']:>6.1%} "
                      f"{res['enrichment']:>6.1f}x {res['p']:>11.2e} {bf:>5}")
                all_hits.append((fol, res))

    # Summary
    bonf_hits = [(f, r) for f, r in all_hits if r['bonferroni']]
    print(f"\n{'='*68}")
    print(f"Total folios tested: {len(target_folios)}")
    print(f"Hits at p < 0.01: {len(all_hits)}")
    print(f"Bonferroni-surviving (p < {0.05/960:.1e}): {len(bonf_hits)}")
    if bonf_hits:
        print(f"\nBonferroni-surviving enrichments:")
        for fol, res in bonf_hits:
            print(f"  {fol}: {res['cv']} ({res['count']}/{res['folio_fc']} = "
                  f"{res['rate']:.1%}, base {res['base_rate']:.1%}, "
                  f"p = {res['p']:.2e})")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Supplement S4: Forward Cipher v11 — Clean Baseline
=====================================================
Architecture-only forward cipher WITHOUT scribe production rules.
Demonstrates what the two-table architecture produces before
copy-mutate, preferential reuse, suffix avoidance, and stickiness.

Compare output against S1 (full v11 with scribe rules) to measure
the contribution of each production layer.

Usage:
    python S4_forward_cipher_clean.py
    python S4_forward_cipher_clean.py --seed 42 --n 500

Requires: enriched_records.pkl, ci_corpus_parsed.pkl
"""

import pickle, random, argparse, re, sys
import numpy as np
from collections import Counter, defaultdict

# ══════════════════════════════════════════════════════════════
# PARAMETERS (architecture only — no scribe rules)
# ══════════════════════════════════════════════════════════════

SEED = 42
EC_THRESHOLD = 0.53  # top 53% of source tokens classified EC

# Nomenclator: function word → suffix family
NOMENCLATOR = {
    'et': 'Y', 'postea': 'Y',
    'in': 'N', 'cum': 'N', 'hoc': 'N',
    'de': 'L', 'ad': 'L', 'habet': 'L', 'uel': 'L',
    'que': 'L', 'supra': 'L',
    'eam': 'R'
}

# Grid mapping
VOWEL_TO_FAMILY = {'a': 'Y', 'e': 'R', 'i': 'N', 'o': 'L', 'u': 'BARE'}

# ══════════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════════

def classify_and_route(word, ec_words):
    """Route word through nomenclator (EC) or grid (FC)."""
    w = word.lower()
    if w in NOMENCLATOR:
        return ('EC', NOMENCLATOR[w])
    if w in ec_words:
        # High-frequency content word → EC via vowel heuristic
        for ch in w:
            if ch in 'aeiou':
                return ('EC', VOWEL_TO_FAMILY.get(ch, 'Y'))
        return ('EC', 'Y')
    else:
        # Content word → FC via grid
        first_consonant = ''
        first_vowel = ''
        for ch in w:
            if ch in 'aeiou':
                first_vowel = ch
                break
            else:
                first_consonant = ch
        family = VOWEL_TO_FAMILY.get(first_vowel, 'Y')
        return ('FC', family, first_consonant, first_vowel)


def build_pools(ha_records):
    """Build cell pools from VMS Herbal-A vocabulary."""
    pools = defaultdict(list)
    for r in ha_records:
        if r['empty_core']:
            fam = r.get('sfx_fam', 'BARE')
            pools[('EC', fam)].append(r['token'])
        else:
            m_core = r.get('m_core', '')
            fam = r.get('sfx_fam', 'BARE')
            row = m_core[0] if m_core else '?'
            pools[('FC', row, fam)].append(r['token'])
    return pools


def run(seed=SEED, n=None):
    """Run clean baseline forward cipher."""
    random.seed(seed)

    with open('enriched_records.pkl', 'rb') as f:
        records = pickle.load(f)
    with open('ci_corpus_parsed.pkl', 'rb') as f:
        ci = pickle.load(f)

    ha = [r for r in records if r.get('section') == 'Herbal-A']
    pools = build_pools(ha)

    # Build EC word set from CI
    freq = Counter(w.lower() for w in ci['all_words'])
    sorted_words = sorted(freq.keys(), key=lambda w: -freq[w])
    total = sum(freq.values())
    cumulative = 0
    ec_words = set()
    for w in sorted_words:
        cumulative += freq[w]
        ec_words.add(w)
        if cumulative / total >= EC_THRESHOLD:
            break

    source_words = ci['all_words']
    if n:
        source_words = source_words[:n]

    output = []
    for word in source_words:
        result = classify_and_route(word, ec_words)

        if result[0] == 'EC':
            fam = result[1]
            pool = pools.get(('EC', fam), [])
            if pool:
                # CLEAN: uniform random selection (no reuse, no avoidance)
                tok = random.choice(pool)
            else:
                tok = '???'
            output.append(('EC', tok, word))
        else:
            _, fam, cons, vowel = result
            # Map consonant to row
            from collections import defaultdict
            ROW_MAP = {
                'c': 'o', 's': 'o', 'p': 'o',
                '': 'c', 'v': 'c',
                'f': 'e', 'd': 'e',
                'm': 'a', 'l': 'a',
                'r': 'd', 'q': 'd', 'h': 'd', 'n': 'd', 'g': 'd',
                't': 'l',
                'b': 'r', 'z': 'r', 'x': 'r', 'j': 'r', 'k': 'r',
                'w': 'r', 'y': 'r',
            }
            row = ROW_MAP.get(cons, 'o')
            pool = pools.get(('FC', row, fam), [])
            if pool:
                tok = random.choice(pool)
            else:
                tok = '???'
            output.append(('FC', tok, word))

    return output


def main():
    parser = argparse.ArgumentParser(description='Clean baseline forward cipher')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()

    tokens = run(seed=args.seed, n=args.n)
    print(f"Generated {len(tokens)} tokens (seed={args.seed})")
    print(f"EC: {sum(1 for t in tokens if t[0]=='EC')}, "
          f"FC: {sum(1 for t in tokens if t[0]=='FC')}")

    vms_tokens = [t[1] for t in tokens]
    print(f"Unique types: {len(set(vms_tokens))}")
    print(f"\nFirst 20:")
    for route, tok, latin in tokens[:20]:
        print(f"  {route:<3} {tok:<15} ← {latin}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Roundtrip Validation: Latin → v11 Forward Cipher → PGCS Reading → Compare
============================================================================
Tests the two-table cipher architecture by enciphering Ald.211 Centaurea
minor text through v11, then reading the output tokens back through PGCS
to recover consonant and vowel information.

Pipeline:
  1. Extract Centaurea minor Latin words from Ald.211 HTR
  2. Encipher through v11 forward cipher (S1) with seed 42
  3. Parse each output VMS token back through PGCS (from enriched_records)
  4. Compare CV reading against Latin input's expected row and family
  5. Report row accuracy, family accuracy, individual consonant accuracy

Edward Bozzard · ORCID 0009-0002-4052-0994
"""

import pickle, random, re, sys, os
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# STEP 0: CONSTANTS (from S1 and S3)
# ══════════════════════════════════════════════════════════════

# Grid: initial consonant → row (identity permutation)
CONSONANT_TO_ROW = {}
for row, consonants in {
    'o': ['c','s','p'],
    'c': ['∅','v'],
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

# Reverse: suffix family → vowel (for comparison)
FAM_TO_VOWEL = {'Y':'a', 'R':'e', 'N':'i', 'L':'o', 'BARE':'u', 'M':'a'}

# Reverse: row → consonant set (for individual consonant accuracy)
ROW_TO_CONSONANTS = defaultdict(set)
for c, r in CONSONANT_TO_ROW.items():
    ROW_TO_CONSONANTS[r].add(c)

# Core row → row (from S3 — only 7 standard rows)
CORE_TO_ROW = {'o':'o', 'c':'c', 'e':'e', 'a':'a', 'd':'d', 'l':'l', 'r':'r'}

# Nomenclator
NOMENCLATOR = {
    'et': 'Y', 'postea': 'Y',
    'in': 'N', 'cum': 'N', 'hoc': 'N',
    'de': 'L', 'habet': 'L', 'uel': 'L', 'vel': 'L',
    'que': 'L', 'supra': 'L', 'ad': 'L',
}

VOWELS = set('aeiouàèìòùéêîôûäëïöü')
VOWEL_NORMALISE = {
    'à':'a','è':'e','é':'e','ê':'e','ì':'i','î':'i',
    'ò':'o','ô':'o','ù':'u','û':'u'
}

# ══════════════════════════════════════════════════════════════
# STEP 1: EXTRACT CENTAUREA MINOR LATIN TEXT
# ══════════════════════════════════════════════════════════════

def extract_centaurea_minor(filepath='ms_ald_211_htr_COMPLETE.md'):
    """Extract the Centaurea minor entry from Ald.211 HTR."""
    with open(filepath, 'r') as f:
        text = f.read()

    # Find the Centaurea minor section
    # Starts with "herba Centauzea minor" heading
    # Ends at next "---" separator or next "## Folio:" heading
    start_marker = '**herba Centauzea minor**'
    end_marker = '\n---\n'

    start_idx = text.find(start_marker)
    if start_idx == -1:
        raise ValueError("Could not find Centaurea minor section")

    # Move past the heading
    start_idx = start_idx + len(start_marker)
    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        end_idx = len(text)

    section = text[start_idx:end_idx]

    # Clean: remove editorial marks, pilcrows, parenthetical expansions
    section = re.sub(r'\*\*[A-Z]\*\*', '', section)  # Drop bold initial letters
    section = re.sub(r'¶', '', section)                # Drop pilcrows
    section = re.sub(r'\([\w?]*\)', '', section)       # Remove parenthetical expansions
    section = re.sub(r'\[.*?\]', '', section)           # Remove editorial brackets
    section = re.sub(r'\?', '', section)                # Remove question marks
    section = re.sub(r'-\n', '', section)               # Join hyphenated words
    section = re.sub(r'\n', ' ', section)               # Newlines to spaces
    section = re.sub(r'\s+', ' ', section)              # Normalize whitespace

    # Tokenise: extract Latin words
    words = re.findall(r'[a-zA-ZàèìòùéêîôûäëïöüÀÈÌÒÙ]+', section)
    words = [w.lower() for w in words if len(w) > 0]

    return words


# ══════════════════════════════════════════════════════════════
# STEP 2: ENCIPHER THROUGH v11
# ══════════════════════════════════════════════════════════════

def load_v11_machinery():
    """Load enriched records and CI corpus, build pools."""
    with open('enriched_records.pkl', 'rb') as f:
        records = pickle.load(f)
    with open('ci_corpus_parsed.pkl', 'rb') as f:
        ci = pickle.load(f)

    ha = [r for r in records if r.get('section') == 'Herbal-A']

    # Build token → PGCS lookup from ALL records (not just HA)
    tok_lookup = {}
    for r in records:
        t = r['token']
        if t not in tok_lookup:
            tok_lookup[t] = r

    ec_words = ci.get('ec_words', set())
    return records, ha, ci, ec_words, tok_lookup


def classify_and_route(word, ec_words):
    """Route a Latin word to a grid cell (from S1)."""
    w = word.lower()

    # Check nomenclator first
    if w in NOMENCLATOR:
        return ('EC', NOMENCLATOR[w], None, None)

    # Get first vowel for family
    first_vowel = 'a'
    for ch in w:
        if ch in VOWELS:
            first_vowel = VOWEL_NORMALISE.get(ch, ch)
            break
    family = VOWEL_TO_FAMILY.get(first_vowel, 'Y')

    # EC or FC?
    if w in ec_words:
        return ('EC', family, None, first_vowel)
    else:
        initial = w[0] if w[0] not in VOWELS else '∅'
        row = CONSONANT_TO_ROW.get(initial, 'c')
        return ('FC', family, row, first_vowel)


def encipher_words(words, ha, ec_words, seed=42):
    """Encipher Latin words through simplified v11 (no rebalancing/stickiness).

    Uses clean architecture-only approach for roundtrip transparency:
    route → pick from correct pool. No stochastic family reassignment.
    This isolates the architecture test from scribe-layer noise.

    Also runs FULL v11 (with scribe rules) for comparison.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Build pools from HA
    pool = defaultdict(list)
    for r in ha:
        mc = r.get('m_core') or r.get('core') or ''
        sf = r.get('sfx_fam', 'BARE')
        row = mc[0] if mc and not r['empty_core'] else '∅'
        pool[(row, sf)].append(r['token'])

    output_clean = []
    output_full = []

    # --- CLEAN PASS (architecture only, no stochastic reassignment) ---
    random.seed(seed)
    np.random.seed(seed)
    for word in words:
        route = classify_and_route(word, ec_words)
        route_type = route[0]

        if route_type == 'EC':
            family = route[1]
            cell = ('∅', family)
            p = pool.get(cell, [])
            tok = random.choice(p) if p else '???'
            output_clean.append({
                'latin': word, 'route': 'EC', 'token': tok,
                'intended_family': family, 'intended_row': None,
                'latin_consonant': None, 'latin_vowel': route[3],
                'is_nomenclator': word.lower() in NOMENCLATOR,
            })
        else:
            family = route[1]
            row = route[2]
            cell = (row, family)
            p = pool.get(cell, [])
            if not p:
                # Fallback: try any family for this row
                for alt_fam in ['Y','R','N','L','BARE','M']:
                    p = pool.get((row, alt_fam), [])
                    if p:
                        family = alt_fam
                        break
            tok = random.choice(p) if p else '???'
            initial = word[0] if word[0] not in VOWELS else '∅'
            output_clean.append({
                'latin': word, 'route': 'FC', 'token': tok,
                'intended_family': family, 'intended_row': row,
                'latin_consonant': initial, 'latin_vowel': route[3],
                'is_nomenclator': False,
            })

    # --- FULL v11 PASS (with rebalancing + stickiness) ---
    # Import and run S1's full machinery
    try:
        sys.path.insert(0, '.')
        import S1_v11_nomenclator as S1
        import importlib
        importlib.reload(S1)

        # Monkey-patch: replace CI source words with our Centaurea minor words
        # We'll call run() but need to intercept the word source
        # Instead, let's directly use S1's classify_and_route and build_pools

        random.seed(seed)
        np.random.seed(seed)

        sampler, is_valid = S1.build_pools(ha)
        prev_family = 'Y'
        family_counts = Counter()
        past_counts = Counter()
        produced = set()

        for idx, word in enumerate(words):
            n = idx
            route = S1.classify_and_route(word, ec_words)

            if route[0] == 'EC':
                is_nom = word.lower() in NOMENCLATOR
                if is_nom:
                    family = route[1]
                else:
                    family = S1.rebalance_family(route[1], family_counts, n)
                    if random.random() < S1.P_STICKY and prev_family in S1.FAMILIES:
                        family = prev_family
                cell = ('∅', family)
                over_cap = len(produced) >= S1.VOCAB_CAP
                if over_cap:
                    token = S1.reuse_token(past_counts, sampler, cell)
                else:
                    token = S1.pick_token(sampler, cell, produced)
                    if not token:
                        for alt in S1.FAMILIES:
                            if alt != family:
                                token = S1.pick_token(sampler, ('∅', alt), produced)
                                if token:
                                    family = alt; break
                    if not token:
                        token = 'dy'
                initial = None
            else:
                row, family = route[1], route[2]
                family = S1.rebalance_family(family, family_counts, n)
                if random.random() < S1.P_STICKY and prev_family in S1.FAMILIES:
                    if (row, prev_family) in sampler:
                        family = prev_family
                cell = (row, family)
                over_cap = len(produced) >= S1.VOCAB_CAP
                if over_cap:
                    token = S1.reuse_token(past_counts, sampler, cell)
                else:
                    token = S1.pick_token(sampler, cell, produced)
                    if not token:
                        token = S1.pick_token(sampler, cell, produced) or 'dy'
                initial = word[0] if word[0] not in VOWELS else '∅'

            produced.add(token)
            past_counts[token] += 1
            family_counts[family] = family_counts.get(family, 0) + 1
            prev_family = family

            first_vowel = 'a'
            for ch in word:
                if ch in VOWELS:
                    first_vowel = VOWEL_NORMALISE.get(ch, ch)
                    break

            output_full.append({
                'latin': word, 'route': route[0], 'token': token,
                'intended_family': family,
                'intended_row': route[1] if route[0] == 'FC' else None,
                'latin_consonant': initial if route[0] == 'FC' else None,
                'latin_vowel': first_vowel,
                'is_nomenclator': word.lower() in NOMENCLATOR,
            })

    except Exception as e:
        print(f"WARNING: Full v11 pass failed: {e}")
        output_full = output_clean  # fallback

    return output_clean, output_full


# ══════════════════════════════════════════════════════════════
# STEP 3: READ BACK THROUGH PGCS
# ══════════════════════════════════════════════════════════════

def read_back(output, tok_lookup):
    """Parse each VMS token back through PGCS to recover CV reading."""
    results = []
    for item in output:
        token = item['token']
        rec = tok_lookup.get(token)

        if rec is None:
            results.append({
                **item,
                'read_row': '?', 'read_family': '?',
                'read_empty_core': None, 'found_in_records': False,
            })
            continue

        m_core = rec.get('m_core', '∅')
        sfx_fam = rec.get('sfx_fam', '?')
        empty_core = rec.get('empty_core', True)

        if empty_core:
            read_row = '∅'
        else:
            if m_core and m_core != '∅':
                read_row = CORE_TO_ROW.get(m_core[0], m_core[0])
            else:
                read_row = '?'

        results.append({
            **item,
            'read_row': read_row,
            'read_family': sfx_fam,
            'read_empty_core': empty_core,
            'found_in_records': True,
            'read_m_core': m_core,
        })

    return results


# ══════════════════════════════════════════════════════════════
# STEP 4: COMPARE AND SCORE
# ══════════════════════════════════════════════════════════════

def score_roundtrip(results):
    """Score the roundtrip: row accuracy, family accuracy, consonant accuracy."""

    fc_results = [r for r in results if r['route'] == 'FC' and r['found_in_records']]
    ec_results = [r for r in results if r['route'] == 'EC' and r['found_in_records']]
    nom_results = [r for r in results if r.get('is_nomenclator', False)]

    scores = {}

    # --- FC ROW ACCURACY ---
    # Does the read-back row match the intended row?
    fc_row_correct = sum(1 for r in fc_results
                         if r['read_row'] == r['intended_row'])
    scores['fc_row_total'] = len(fc_results)
    scores['fc_row_correct'] = fc_row_correct
    scores['fc_row_accuracy'] = fc_row_correct / len(fc_results) if fc_results else 0

    # --- FC FAMILY ACCURACY ---
    # Does the read-back family match the intended family?
    fc_fam_correct = sum(1 for r in fc_results
                         if r['read_family'] == r['intended_family'])
    scores['fc_fam_total'] = len(fc_results)
    scores['fc_fam_correct'] = fc_fam_correct
    scores['fc_fam_accuracy'] = fc_fam_correct / len(fc_results) if fc_results else 0

    # --- FC FAMILY vs LATIN VOWEL ---
    # Does the read-back family match what VOWEL_TO_FAMILY predicts from the Latin vowel?
    fc_vowel_correct = sum(1 for r in fc_results
                           if FAM_TO_VOWEL.get(r['read_family'], '?') == r['latin_vowel'])
    scores['fc_vowel_total'] = len(fc_results)
    scores['fc_vowel_correct'] = fc_vowel_correct
    scores['fc_vowel_accuracy'] = fc_vowel_correct / len(fc_results) if fc_results else 0

    # --- INDIVIDUAL CONSONANT ACCURACY ---
    # For FC tokens, can we recover the specific Latin consonant from the row?
    # If row has N consonants, chance = 1/N. But for row 'l' (only 't'), it's 100%.
    # We score: does the Latin consonant belong to the row we read back?
    fc_cons_in_row = sum(1 for r in fc_results
                         if r['latin_consonant'] in ROW_TO_CONSONANTS.get(r['read_row'], set()))
    scores['fc_cons_in_row'] = fc_cons_in_row
    scores['fc_cons_in_row_accuracy'] = fc_cons_in_row / len(fc_results) if fc_results else 0

    # Expected individual consonant recovery (1/N for each row)
    individual_correct = 0
    individual_total = 0
    for r in fc_results:
        row = r['read_row']
        n_in_row = len(ROW_TO_CONSONANTS.get(row, set()))
        if n_in_row > 0:
            individual_correct += 1.0 / n_in_row
            individual_total += 1
    scores['fc_individual_consonant_expected'] = individual_correct / individual_total if individual_total else 0

    # --- EC FAMILY ACCURACY (nomenclator) ---
    nom_fam_correct = sum(1 for r in nom_results
                          if r['found_in_records'] and r['read_family'] == r['intended_family'])
    scores['nom_total'] = len(nom_results)
    scores['nom_correct'] = nom_fam_correct
    scores['nom_accuracy'] = nom_fam_correct / len(nom_results) if nom_results else 0

    # --- OVERALL EC FAMILY ---
    ec_fam_correct = sum(1 for r in ec_results
                         if r['read_family'] == r['intended_family'])
    scores['ec_total'] = len(ec_results)
    scores['ec_correct'] = ec_fam_correct
    scores['ec_accuracy'] = ec_fam_correct / len(ec_results) if ec_results else 0

    return scores


# ══════════════════════════════════════════════════════════════
# STEP 5: REPORT
# ══════════════════════════════════════════════════════════════

def print_report(results, scores, label=""):
    """Print detailed roundtrip report."""
    print(f"\n{'='*75}")
    print(f"ROUNDTRIP VALIDATION: {label}")
    print(f"{'='*75}")
    print(f"Total words: {len(results)}")

    fc = [r for r in results if r['route'] == 'FC']
    ec = [r for r in results if r['route'] == 'EC']
    print(f"FC tokens: {len(fc)}, EC tokens: {len(ec)}")

    # Token-level detail
    print(f"\n{'Word':<20} {'Route':<4} {'VMS Token':<18} "
          f"{'Intended':<12} {'ReadBack':<12} {'Row✓':>4} {'Fam✓':>4}")
    print("-" * 80)
    for r in results:
        if r['route'] == 'FC':
            intended = f"{r['intended_row']}/{r['intended_family']}"
            readback = f"{r['read_row']}/{r['read_family']}"
            row_ok = '✓' if r['read_row'] == r['intended_row'] else '✗'
            fam_ok = '✓' if r['read_family'] == r['intended_family'] else '✗'
        else:
            intended = f"∅/{r['intended_family']}"
            readback = f"∅/{r['read_family']}"
            row_ok = '—'
            fam_ok = '✓' if r['read_family'] == r['intended_family'] else '✗'
        found = '' if r.get('found_in_records', True) else ' [NOT FOUND]'
        print(f"{r['latin']:<20} {r['route']:<4} {r['token']:<18} "
              f"{intended:<12} {readback:<12} {row_ok:>4} {fam_ok:>4}{found}")

    # Summary scores
    print(f"\n{'='*75}")
    print("SCORES")
    print(f"{'='*75}")
    print(f"FC row accuracy:              {scores['fc_row_correct']}/{scores['fc_row_total']} "
          f"= {scores['fc_row_accuracy']:.1%}")
    print(f"FC family accuracy:           {scores['fc_fam_correct']}/{scores['fc_fam_total']} "
          f"= {scores['fc_fam_accuracy']:.1%}")
    print(f"FC vowel recovery:            {scores['fc_vowel_correct']}/{scores['fc_vowel_total']} "
          f"= {scores['fc_vowel_accuracy']:.1%}")
    print(f"FC consonant-in-row:          {scores['fc_cons_in_row']}/{scores['fc_row_total']} "
          f"= {scores['fc_cons_in_row_accuracy']:.1%}")
    print(f"FC individual consonant (E):  {scores['fc_individual_consonant_expected']:.1%} "
          f"(expected from row sizes)")
    print(f"Nomenclator family accuracy:  {scores['nom_correct']}/{scores['nom_total']} "
          f"= {scores['nom_accuracy']:.1%}")
    print(f"EC family accuracy (all):     {scores['ec_correct']}/{scores['ec_total']} "
          f"= {scores['ec_accuracy']:.1%}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main(seed=42):
    print("=" * 75)
    print("ROUNDTRIP VALIDATION")
    print(f"Latin → v11 Forward Cipher → PGCS Reading → Compare")
    print(f"Seed: {seed} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 75)

    # Step 1: Extract Centaurea minor
    print("\n[STEP 1] Extracting Centaurea minor from Ald.211...")
    words = extract_centaurea_minor()
    print(f"  Extracted {len(words)} Latin words")
    print(f"  First 10: {words[:10]}")
    print(f"  Last 10:  {words[-10:]}")
    pickle.dump(words, open('roundtrip_step1_words.pkl', 'wb'))

    # Step 2: Load machinery and encipher
    print("\n[STEP 2] Loading v11 machinery and enciphering...")
    records, ha, ci, ec_words, tok_lookup = load_v11_machinery()

    # Classify each word to show routing
    print("\n  Routing summary:")
    routes = []
    for w in words:
        r = classify_and_route(w, ec_words)
        routes.append((w, r))
    ec_count = sum(1 for _, r in routes if r[0] == 'EC')
    fc_count = sum(1 for _, r in routes if r[0] == 'FC')
    nom_count = sum(1 for w, r in routes if w.lower() in NOMENCLATOR)
    print(f"  EC: {ec_count} (nomenclator: {nom_count}, heuristic: {ec_count-nom_count})")
    print(f"  FC: {fc_count}")

    output_clean, output_full = encipher_words(words, ha, ec_words, seed=seed)
    pickle.dump({'clean': output_clean, 'full': output_full},
                open('roundtrip_step2_encipher.pkl', 'wb'))

    # Step 3: Read back
    print("\n[STEP 3] Reading VMS tokens back through PGCS...")
    results_clean = read_back(output_clean, tok_lookup)
    results_full = read_back(output_full, tok_lookup)
    pickle.dump({'clean': results_clean, 'full': results_full},
                open('roundtrip_step3_readback.pkl', 'wb'))

    not_found_clean = sum(1 for r in results_clean if not r.get('found_in_records', True))
    not_found_full = sum(1 for r in results_full if not r.get('found_in_records', True))
    print(f"  Tokens not found in records (clean): {not_found_clean}")
    print(f"  Tokens not found in records (full):  {not_found_full}")

    # Step 4: Score
    print("\n[STEP 4] Scoring roundtrip...")
    scores_clean = score_roundtrip(results_clean)
    scores_full = score_roundtrip(results_full)
    pickle.dump({'clean': scores_clean, 'full': scores_full},
                open('roundtrip_step4_scores.pkl', 'wb'))

    # Step 5: Report
    print_report(results_clean, scores_clean, label="CLEAN (architecture only)")
    print_report(results_full, scores_full, label="FULL v11 (with scribe rules)")

    # Comparison summary
    print(f"\n{'='*75}")
    print("COMPARISON: CLEAN vs FULL v11")
    print(f"{'='*75}")
    print(f"{'Metric':<35} {'Clean':>10} {'Full v11':>10}")
    print("-" * 57)
    for key in ['fc_row_accuracy', 'fc_fam_accuracy', 'fc_vowel_accuracy',
                'fc_cons_in_row_accuracy', 'fc_individual_consonant_expected',
                'nom_accuracy', 'ec_accuracy']:
        label = key.replace('_', ' ').replace('fc ', 'FC ').title()
        print(f"{label:<35} {scores_clean[key]:>9.1%} {scores_full[key]:>9.1%}")

    # Save full results
    full_results = {
        'seed': seed,
        'words': words,
        'results_clean': results_clean,
        'results_full': results_full,
        'scores_clean': scores_clean,
        'scores_full': scores_full,
        'timestamp': datetime.now().isoformat(),
    }
    pickle.dump(full_results, open('roundtrip_validation_results.pkl', 'wb'))
    print(f"\n✓ All results saved to roundtrip_validation_results.pkl")

    return full_results


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed=seed)


# ══════════════════════════════════════════════════════════════
# ADDENDUM: INDIVIDUAL CONSONANT RESOLUTION (raw core field)
# ══════════════════════════════════════════════════════════════
# CANONICAL consonant resolution from reproduce_all.py line 488.
# Uses EXACT matching against ROW_CONS lists, NOT prefix matching.
# CRITICAL: raw core 'cho/chod/chos/chol' → v (NOT ∅).
# Only 'ch' and 'che' → ∅. Prefix matching gets this wrong.

# Shipped ROW_CONS map (reproduce_all.py / reproduce_termux.py)
ROW_CONS_CANON = {
    'o': {'c': ['o'], 's': ['od', 'ol', 'ot', 'ok', 'os'],
          'p': ['or', 'octh', 'ockh', 'otch', 'op']},
    'e': {'d': ['e'],
          'f': ['ee', 'eo', 'ek', 'eod', 'es', 'et', 'ees']},
    'a': {'m': ['a', 'ar', 'ai'],
          'l': ['al', 'aii', 'air', 'aiin']},
    'c': {'∅': ['ch', 'che'],
          'v': ['cho', 'chod', 'chos', 'chol']},
    'd': {'r': ['d'], 'n': ['yd', 'da', 'dch']},
    'l': {'t': ['l', 'ld', 'ls']},
    'r': {'b': ['r', 'rch', 'ro']},
}
ROW_DEFAULTS_CANON = {'o': 'c', 'c': '∅', 'e': 'd', 'a': 'm',
                       'd': 'r', 'l': 't', 'r': 'b'}


def resolve_consonant_from_core(raw_core, row=None):
    """Resolve raw core to individual consonant using grid cracking.
    
    Canonical implementation from reproduce_all.py line 504.
    Uses EXACT matching against ROW_CONS lists.
    
    Args:
        raw_core: The raw 'core' field from enriched_records
        row: m_core[0] (if None, inferred from raw_core[0])
    """
    if not raw_core or raw_core == '∅':
        return None
    if row is None:
        row = raw_core[0] if raw_core[0] in 'oceadlr' else '?'
    if row not in ROW_CONS_CANON:
        return '?'
    for cons, cores in ROW_CONS_CANON[row].items():
        if raw_core in cores:
            return cons
    return ROW_DEFAULTS_CANON.get(row, '?')


def multi_seed_consonant_test(n_seeds=1000):
    """Monte Carlo over n_seeds to get expected individual consonant accuracy."""
    import random
    from collections import deque
    
    records, ha, ci, ec_words, tok_lookup = load_v11_machinery()
    
    # March 13 word list
    centaurea_text = ("ad uenenum puluis centauzee minoris in uino calido ueteri "
                      "aut suchus data et potata ualidissime resistit ueneno item "
                      "ad uentrem reducendum centauzea mayor et minor si eay suchus "
                      "potauis uentrem aduiat centaurea ut dicit platearius calida "
                      "et sicca est herba amarissima vnde fel terre appellatur uim "
                      "habet atractiuam et consumptiuam")
    words = centaurea_text.strip().split()
    
    pool = defaultdict(Counter)
    for r in ha:
        mc = r.get('m_core') or r.get('core') or ''
        sf = r.get('sfx_fam', 'BARE')
        row = mc[0] if mc and not r['empty_core'] else '∅'
        pool[(row, sf)][r['token']] += 1
    
    sampler = {}
    for cell, tc in pool.items():
        its = list(tc.keys())
        wt = np.array([tc[t] for t in its], dtype=float)
        wt /= wt.sum()
        sampler[cell] = (its, wt)
    
    def pick(cell, avoid):
        if cell not in sampler: return None
        tokens, weights = sampler[cell]
        adj = np.copy(weights)
        for j, t in enumerate(tokens):
            if t in avoid: adj[j] /= 15
        s = adj.sum()
        if s > 0: adj /= s; return tokens[np.random.choice(len(tokens), p=adj)]
        return tokens[np.random.choice(len(tokens), p=weights)]
    
    cons_accs = []
    for seed in range(n_seeds):
        random.seed(seed); np.random.seed(seed)
        recent = deque(maxlen=10)
        fc_ok = fc_tot = 0
        
        for w in words:
            w_low = w.lower()
            if w_low in NOMENCLATOR or w_low in ec_words:
                fv = 'a'
                for ch in w_low:
                    if ch in 'aeiou': fv = ch; break
                family = VOWEL_TO_FAMILY.get(fv, 'Y') if w_low not in NOMENCLATOR else NOMENCLATOR[w_low]
                tok = pick(('∅', family), set(recent))
                if tok: recent.append(tok)
                continue
            
            initial = w_low[0] if w_low[0] not in VOWELS else '∅'
            row = CONSONANT_TO_ROW.get(initial, 'c')
            fv = 'a'
            for ch in w_low:
                if ch in 'aeiou': fv = ch; break
            family = VOWEL_TO_FAMILY.get(fv, 'Y')
            tok = pick((row, family), set(recent))
            if not tok: tok = 'dy'
            recent.append(tok)
            
            rec = tok_lookup.get(tok)
            if rec and not rec['empty_core']:
                mc = rec.get('m_core') or rec.get('core') or ''
                read_row = mc[0] if mc else '?'
                raw_core = rec.get('core', '')
                resolved = resolve_consonant_from_core(raw_core, row=read_row)
                fc_tot += 1
                if resolved == initial: fc_ok += 1
        
        if fc_tot > 0:
            cons_accs.append(fc_ok / fc_tot)
    
    result = {
        'n_seeds': n_seeds,
        'mean': float(np.mean(cons_accs)),
        'sd': float(np.std(cons_accs)),
        'median': float(np.median(cons_accs)),
        'min': float(np.min(cons_accs)),
        'max': float(np.max(cons_accs)),
        'p25': float(np.percentile(cons_accs, 25)),
        'p75': float(np.percentile(cons_accs, 75)),
        'pct_ge_65': sum(1 for c in cons_accs if c >= 0.645) / len(cons_accs),
    }
    pickle.dump(result, open('roundtrip_consonant_monte_carlo.pkl', 'wb'))
    
    print(f"\nMonte Carlo individual consonant ({n_seeds} seeds):")
    print(f"  Mean: {result['mean']:.1%} (sd {result['sd']:.1%})")
    print(f"  Median: {result['median']:.1%}")
    print(f"  Range: [{result['min']:.1%}, {result['max']:.1%}]")
    print(f"  IQR: [{result['p25']:.1%}, {result['p75']:.1%}]")
    print(f"  Seeds ≥65%: {result['pct_ge_65']:.1%}")
    
    return result


if __name__ == '__main__' and '--monte-carlo' in sys.argv:
    multi_seed_consonant_test()

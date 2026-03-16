#!/usr/bin/env python3
"""
NOMENCLATOR OPTIMIZER
======================
Recovers EC function-word → suffix-family assignments by greedy
optimisation of EC-EC bigram correlation between an external Latin
pharmaceutical corpus and VMS Herbal-A.

Training corpus:  Ms.Ald.211 (or any pharmaceutical Latin text)
Validation corpus: Circa Instans (ci_corpus_parsed.pkl)
Target:           VMS Herbal-A EC-EC suffix-family bigrams

Method:
  1. Build VMS HA EC-EC bigram distribution (target)
  2. Build Latin corpus EC sequence under current assignment
  3. Greedy: for each unassigned EC function word, try all 6 families,
     pick the one that maximises Pearson r with VMS bigrams
  4. Validate on CI (independent corpus)
  5. Null model: 10,000 random assignments, compute p-value

Constraint: only genuine Latin function words are eligible for assignment.

Edward Bozzard · ORCID 0009-0002-4052-0994
"""

import pickle, re, sys
import numpy as np
from collections import Counter, defaultdict
import random

# ══════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════

with open('enriched_records.pkl', 'rb') as f:
    all_records = pickle.load(f)
with open('ci_corpus_parsed.pkl', 'rb') as f:
    ci = pickle.load(f)

ec_words = ci.get('ec_words', set())

# ══════════════════════════════════════════════════════════════
# PARSE ALD.211 (or substitute your own Latin pharma text)
# ══════════════════════════════════════════════════════════════

try:
    with open('/mnt/user-data/uploads/ms_ald_211_htr.md', 'r') as f:
        raw = f.read()
except FileNotFoundError:
    print("WARNING: Ald.211 not found. Using CI as training corpus (less ideal).")
    raw = None

if raw:
    text_lines = []
    in_notes = False
    for line in raw.split('\n'):
        if '## NOTES' in line:
            in_notes = True
        if in_notes:
            continue
        if line.startswith('#') or line.startswith('---'):
            continue
        if line.strip().startswith('[') and line.strip().endswith(']'):
            continue
        clean = line.replace('**', '')
        clean = re.sub(r'\[.*?\]', '', clean)
        clean = re.sub(r'\(\?\)', '', clean)
        clean = re.sub(r'\(([a-z]+)\)', r'\1', clean)
        if clean.strip():
            text_lines.append(clean.strip())
    text = ' '.join(text_lines)
    ENGLISH = {
        'the', 'and', 'or', 'with', 'in', 'of', 'a', 'is', 'for', 'from',
        'that', 'this', 'text', 'marked', 'bold', 'red', 'reading',
        'uncertain', 'word', 'after', 'line', 'breaks', 'follow',
        'manuscript', 'original', 'paragraph', 'mark', 'illegible',
        'abbreviations', 'expanded', 'parentheses', 'where', 'clear',
        'rubricated', 'pilcrow', 'plant', 'illustration', 'leaves',
        'stems', 'bearing', 'broad', 'lobed', 'flowering', 'cup',
        'shaped', 'seed', 'heads', 'three', 'continuing', 'previous',
        'folio', 'top', 'end', 'prev', 'entry', 'dr',
    }
    train_words = [
        w for w in re.findall(r'[a-zA-Zàèìòùéêîôûäëïöü]+', text.lower())
        if w not in ENGLISH and len(w) > 1
    ]
    print(f"Training corpus: Ald.211, {len(train_words)} words")
else:
    train_words = [w.lower() for w in ci['all_words']]
    print(f"Training corpus: CI (fallback), {len(train_words)} words")

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

VOWELS = set('aeiouàèìòùéêîôûäëïöü')
VNORM = {
    'à': 'a', 'è': 'e', 'é': 'e', 'ê': 'e', 'ì': 'i',
    'î': 'i', 'ò': 'o', 'ô': 'o', 'ù': 'u', 'û': 'u',
}
V2F = {'a': 'Y', 'e': 'R', 'i': 'N', 'o': 'L', 'u': 'BARE'}
FAMILIES = ['Y', 'R', 'N', 'L', 'BARE', 'M']

# Known fixed assignments (non-negotiable)
KNOWN = {'et': 'Y', 'in': 'N'}

# Genuine Latin function words eligible for assignment
REAL_FW = {
    # Prepositions
    'ad', 'cum', 'de', 'ex', 'in', 'per', 'pro', 'contra', 'supra',
    'super',
    # Conjunctions
    'et', 'aut', 'uel', 'vel', 'que', 'sed', 'non', 'nec', 'si',
    'ut', 'ne',
    # Pronouns/demonstratives
    'hoc', 'hic', 'ista', 'iste', 'eius', 'ei', 'eum', 'eam', 'ea',
    'qui', 'quod', 'qua', 'quam',
    # Common verbs
    'est', 'fiat', 'habet', 'sit',
    # Adverbs
    'item', 'postea', 'inde', 'idem', 'bene', 'sic',
}

# ══════════════════════════════════════════════════════════════
# VMS TARGET: EC-EC bigram distribution
# ══════════════════════════════════════════════════════════════

ha = [r for r in all_records if r.get('section') == 'Herbal-A']

vms_ec_seq = []
for r in ha:
    if r['empty_core']:
        vms_ec_seq.append(r.get('sfx_fam', 'BARE'))
    else:
        vms_ec_seq.append('_FC_')

vms_bg = Counter()
for i in range(len(vms_ec_seq) - 1):
    if vms_ec_seq[i] != '_FC_' and vms_ec_seq[i + 1] != '_FC_':
        vms_bg[(vms_ec_seq[i], vms_ec_seq[i + 1])] += 1

print(f"VMS HA EC-EC bigrams: {sum(vms_bg.values())} pairs, {len(vms_bg)} types")

# ══════════════════════════════════════════════════════════════
# ROUTING FUNCTION
# ══════════════════════════════════════════════════════════════

def get_family(w, assignment):
    """Route a Latin word to a suffix family using assignment dict,
    falling back to vowel heuristic for unassigned words."""
    if w in assignment:
        return assignment[w]
    for ch in w:
        if ch in VOWELS:
            v = VNORM.get(ch, ch)
            return V2F.get(v, 'Y')
    return 'Y'

# ══════════════════════════════════════════════════════════════
# OBJECTIVE: Pearson r between Latin EC-EC bigrams and VMS
# ══════════════════════════════════════════════════════════════

def build_bigrams(word_list, assignment):
    """Build EC-EC bigram distribution from a Latin word list."""
    ec_seq = [get_family(w, assignment) for w in word_list if w in ec_words]
    bg = Counter()
    for i in range(len(ec_seq) - 1):
        bg[(ec_seq[i], ec_seq[i + 1])] += 1
    return bg

def compute_r(assignment, word_list):
    """Pearson r between Latin bigrams (under assignment) and VMS bigrams."""
    pred_bg = build_bigrams(word_list, assignment)
    all_bgs = sorted(set(list(pred_bg.keys()) + list(vms_bg.keys())))
    total_p = sum(pred_bg.values())
    total_v = sum(vms_bg.values())
    if total_p == 0 or total_v == 0:
        return 0.0
    v_pred = np.array([pred_bg.get(k, 0) / total_p for k in all_bgs])
    v_vms = np.array([vms_bg.get(k, 0) / total_v for k in all_bgs])
    return float(np.corrcoef(v_pred, v_vms)[0, 1])

# ══════════════════════════════════════════════════════════════
# GREEDY OPTIMIZER
# ══════════════════════════════════════════════════════════════

def optimize(word_list, known=KNOWN, eligible=REAL_FW, min_freq=3,
             min_improvement=0.002):
    """Greedy optimisation: assign eligible function words to families."""
    # Count EC word frequencies in training corpus
    ec_freq = Counter(w for w in word_list if w in ec_words)
    candidates = [
        w for w, n in ec_freq.most_common()
        if w in eligible and n >= min_freq and w not in known
    ]

    print(f"\nEligible candidates ({len(candidates)} words, freq >= {min_freq}):")
    for w in candidates:
        print(f"  {w:<15} ×{ec_freq[w]:>4}")

    assignment = dict(known)
    baseline_r = compute_r(assignment, word_list)
    print(f"\nBaseline r (known only): {baseline_r:.4f}")
    print(f"\n{'Step':<5} {'Word':<15} {'Family':>7} {'r':>8} {'Δr':>8}")
    print("-" * 50)

    step = 0
    while candidates:
        best_word = None
        best_fam = None
        best_r = compute_r(assignment, word_list)

        for w in candidates:
            for f in FAMILIES:
                trial = dict(assignment)
                trial[w] = f
                r = compute_r(trial, word_list)
                if r > best_r:
                    best_r = r
                    best_word = w
                    best_fam = f

        if best_word is None or best_r - compute_r(assignment, word_list) < min_improvement:
            print(f"\n  Stopping: no word improves r by >= {min_improvement}")
            break

        step += 1
        prev_r = compute_r(assignment, word_list)
        assignment[best_word] = best_fam
        candidates.remove(best_word)
        print(f"  {step:<4} {best_word:<15} {best_fam:>7} {best_r:>8.4f} {best_r - prev_r:>+8.4f}")

    final_r = compute_r(assignment, word_list)
    return assignment, final_r

# ══════════════════════════════════════════════════════════════
# CROSS-VALIDATION ON CI
# ══════════════════════════════════════════════════════════════

def cross_validate(assignment):
    """Test assignment on CI (independent corpus)."""
    ci_words = [w.lower() for w in ci['all_words']]
    r_assigned = compute_r(assignment, ci_words)
    r_heuristic = compute_r({}, ci_words)
    print(f"\nCross-validation on CI:")
    print(f"  Heuristic r = {r_heuristic:.4f}")
    print(f"  Assigned r  = {r_assigned:.4f}")
    print(f"  Improvement = {r_assigned - r_heuristic:+.4f}")
    return r_assigned

# ══════════════════════════════════════════════════════════════
# HELD-OUT VMS TEST
# ══════════════════════════════════════════════════════════════

def held_out_vms_test(assignment, word_list, n_folds=3):
    """3-fold held-out VMS test."""
    folio_order = []
    seen = set()
    for r in ha:
        f = r['folio']
        if f not in seen:
            folio_order.append(f)
            seen.add(f)

    folds = defaultdict(list)
    for r in ha:
        fold = [i for i, f in enumerate(folio_order) if f == r['folio']][0] % n_folds
        folds[fold].append(r)

    print(f"\nHeld-out VMS test ({n_folds} folds):")
    for test_fold in range(n_folds):
        test_recs = folds[test_fold]
        test_ec_seq = []
        for r in test_recs:
            if r['empty_core']:
                test_ec_seq.append(r.get('sfx_fam', 'BARE'))
            else:
                test_ec_seq.append('_FC_')

        test_bg = Counter()
        for i in range(len(test_ec_seq) - 1):
            if test_ec_seq[i] != '_FC_' and test_ec_seq[i + 1] != '_FC_':
                test_bg[(test_ec_seq[i], test_ec_seq[i + 1])] += 1

        pred_bg = build_bigrams(word_list, assignment)
        all_bgs = sorted(set(list(pred_bg.keys()) + list(test_bg.keys())))
        tp = sum(pred_bg.values())
        tv = sum(test_bg.values())
        if tp == 0 or tv == 0:
            continue
        v_pred = np.array([pred_bg.get(k, 0) / tp for k in all_bgs])
        v_test = np.array([test_bg.get(k, 0) / tv for k in all_bgs])
        r = float(np.corrcoef(v_pred, v_test)[0, 1])
        print(f"  Fold {test_fold}: r = {r:.4f}")

# ══════════════════════════════════════════════════════════════
# NULL MODEL
# ══════════════════════════════════════════════════════════════

def null_model(assignment, word_list, n_trials=10000):
    """Random assignment null model."""
    our_r = compute_r(assignment, word_list)
    free_words = [w for w in assignment if w not in KNOWN]

    rng = random.Random(42)
    null_rs = []
    for _ in range(n_trials):
        rand = dict(KNOWN)
        for w in free_words:
            rand[w] = rng.choice(FAMILIES)
        null_rs.append(compute_r(rand, word_list))

    null_rs = np.array(null_rs)
    p = float(np.mean(null_rs >= our_r))
    z = float((our_r - np.mean(null_rs)) / np.std(null_rs)) if np.std(null_rs) > 0 else 0

    print(f"\nNull model ({n_trials} random assignments of {len(free_words)} words):")
    print(f"  Our r:     {our_r:.4f}")
    print(f"  Null mean: {np.mean(null_rs):.4f}")
    print(f"  Null max:  {np.max(null_rs):.4f}")
    print(f"  p-value:   {p:.5f} ({int(p * n_trials)}/{n_trials})")
    print(f"  Z-score:   {z:.1f}")
    return p

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("NOMENCLATOR OPTIMIZER")
    print("=" * 60)

    # Step 1: Optimize on training corpus
    assignment, final_r = optimize(train_words)

    print(f"\nFinal assignment ({len(assignment)} words, r = {final_r:.4f}):")
    fam_groups = defaultdict(list)
    for w, f in sorted(assignment.items()):
        fam_groups[f].append(w)
    for f in FAMILIES:
        if fam_groups[f]:
            print(f"  {f}: {', '.join(fam_groups[f])}")

    # Step 2: Cross-validate on CI
    cv_r = cross_validate(assignment)

    # Step 3: Held-out VMS test
    held_out_vms_test(assignment, train_words)

    # Step 4: Null model
    p = null_model(assignment, train_words)

    # Step 5: Save
    results = {
        'assignment': assignment,
        'final_r': final_r,
        'cv_r': cv_r,
        'null_p': p,
        'known_fixed': list(KNOWN.keys()),
        'training_corpus': 'Ald.211' if raw else 'CI',
    }

    with open('nomenclator_optimizer_result.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved: nomenclator_optimizer_result.pkl")

    # Print the dict for pasting into v11_nomenclator.py
    print(f"\n# Paste this into v11_nomenclator.py:")
    print("NOMENCLATOR = {")
    for w, f in sorted(assignment.items(), key=lambda x: (x[1], x[0])):
        print(f"    '{w}': '{f}',")
    print("}")

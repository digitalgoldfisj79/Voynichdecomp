#!/usr/bin/env python3
"""
S13: CORPUS REVERSAL TEST
===========================
Tests whether function-word assignments are corpus-independent by
reversing the training and validation corpora.

Run 1 (original): Ald.211 train → CI validate
Run 2 (reversed): CI train → Ald.211 validate

If the architecture is real, both runs should achieve comparable
cross-validation r. If the specific assignments are real, both runs
should recover the same word→family mappings.

Requires:
  enriched_records.pkl   (VMS PGCS-enriched tokens)
  ci_corpus_parsed.pkl   (Circa Instans parsed word list)
  ms_ald_211_htr.md      (Ms.Ald.211 HTR transcription)

Edward Bozzard · ORCID 0009-0002-4052-0994
"""

import pickle, re, sys
import numpy as np
from collections import Counter, defaultdict
import random
import json

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════

with open('enriched_records.pkl', 'rb') as f:
    all_records = pickle.load(f)
with open('ci_corpus_parsed.pkl', 'rb') as f:
    ci = pickle.load(f)

ec_words = ci.get('ec_words', set())

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

VOWELS = set('aeiouy')
VNORM = {'y': 'i', 'u': 'u'}
V2F = {'a': 'Y', 'e': 'R', 'i': 'N', 'o': 'L', 'u': 'BARE'}
FAMILIES = ['Y', 'N', 'L', 'R', 'BARE', 'M']
KNOWN = {'et': 'Y', 'in': 'N'}

REAL_FW = {
    'et', 'in', 'cum', 'de', 'ad', 'que', 'uel', 'vel', 'sed', 'si',
    'non', 'per', 'hoc', 'eius', 'habet', 'est', 'ex', 'supra',
    'quod', 'sunt', 'sit', 'fiat', 'ut', 'pro', 'post', 'ante',
    'qui', 'super', 'sub', 'inter', 'ab', 'aut',
    'eam', 'eo', 'ea', 'ibi',
    'item', 'postea', 'inde', 'idem', 'bene', 'sic',
}

# English words to strip from Ald.211 HTR
ENGLISH = {
    'the', 'and', 'or', 'with', 'in', 'of', 'a', 'is', 'for', 'from',
    'that', 'this', 'text', 'marked', 'bold', 'red', 'reading',
    'uncertain', 'word', 'after', 'line', 'breaks', 'follow',
    'manuscript', 'original', 'paragraph', 'mark', 'illegible',
    'abbreviations', 'expanded', 'parentheses', 'where', 'clear',
    'rubricated', 'pilcrow', 'plant', 'illustration', 'leaves',
    'stems', 'bearing', 'broad', 'lobed', 'flowering', 'cup',
    'shaped', 'seed', 'heads', 'three', 'continuing', 'previous',
}

# ══════════════════════════════════════════════════════════════
# VMS TARGET: EC-EC bigram distribution (Herbal-A)
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

# ══════════════════════════════════════════════════════════════
# ROUTING AND SCORING
# ══════════════════════════════════════════════════════════════

def get_family(w, assignment):
    """Route a Latin word to suffix family via assignment or vowel heuristic."""
    if w in assignment:
        return assignment[w]
    for ch in w:
        if ch in VOWELS:
            v = VNORM.get(ch, ch)
            return V2F.get(v, 'Y')
    return 'Y'


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

def optimize(word_list, label="", known=KNOWN, eligible=REAL_FW,
             min_freq=3, min_improvement=0.002):
    """Greedy assignment: try each eligible word in each family,
    keep the best improvement, repeat until no improvement."""
    ec_freq = Counter(w for w in word_list if w in ec_words)
    candidates = [
        w for w, n in ec_freq.most_common()
        if w in eligible and n >= min_freq and w not in known
    ]

    assignment = dict(known)
    current_r = compute_r(assignment, word_list)
    steps = []

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Eligible candidates: {len(candidates)} (freq >= {min_freq})")
    print(f"  Baseline r (anchors only): {current_r:.4f}")

    improved = True
    while improved:
        improved = False
        best_word, best_fam, best_r = None, None, current_r
        for w in candidates:
            if w in assignment:
                continue
            for fam in FAMILIES:
                trial = dict(assignment)
                trial[w] = fam
                r = compute_r(trial, word_list)
                if r > best_r + min_improvement:
                    best_word, best_fam, best_r = w, fam, r
        if best_word:
            assignment[best_word] = best_fam
            current_r = best_r
            steps.append((best_word, best_fam, best_r))
            print(f"  + {best_word:<15} → {best_fam:<6}  r = {best_r:.4f}")
            improved = True

    print(f"  Final: {len(assignment)} assignments, r = {current_r:.4f}")
    return assignment, current_r, steps

# ══════════════════════════════════════════════════════════════
# NULL MODEL
# ══════════════════════════════════════════════════════════════

def null_model(assignment, word_list, n_trials=10000):
    """Random assignment null model (et and in held fixed)."""
    our_r = compute_r(assignment, word_list)
    free_words = [w for w in assignment if w not in KNOWN]
    rng = random.Random(42)
    count_ge = 0
    null_rs = []
    for _ in range(n_trials):
        rand = dict(KNOWN)
        for w in free_words:
            rand[w] = rng.choice(FAMILIES)
        nr = compute_r(rand, word_list)
        null_rs.append(nr)
        if nr >= our_r:
            count_ge += 1
    p = count_ge / n_trials
    print(f"  Null model: {count_ge}/{n_trials} >= {our_r:.4f}, p = {p:.5f}")
    return p, null_rs

# ══════════════════════════════════════════════════════════════
# PARSE ALD.211
# ══════════════════════════════════════════════════════════════

with open('ms_ald_211_htr.md', 'r') as f:
    raw = f.read()

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
ald_words = [
    w.lower() for w in re.findall(r'[a-zA-Z]+', text)
    if len(w) > 1 and w.lower() not in ENGLISH
]

ci_words = [w.lower() for w in ci['all_words']]

# ══════════════════════════════════════════════════════════════
# RUN BOTH DIRECTIONS
# ══════════════════════════════════════════════════════════════

print(f"Ald.211: {len(ald_words)} words")
print(f"CI:      {len(ci_words)} words")
print(f"VMS HA EC-EC bigrams: {sum(vms_bg.values())} pairs, "
      f"{len(vms_bg)} types")

# Run 1: Original direction
a1, r1, s1 = optimize(ald_words, label="RUN 1: Ald.211 TRAIN → CI VALIDATE")
r1_cv = compute_r(a1, ci_words)
r1_heur = compute_r({}, ci_words)
print(f"  Cross-val on CI:     r = {r1_cv:.4f} (heuristic: {r1_heur:.4f})")
p1, _ = null_model(a1, ald_words)

# Run 2: Reversed direction
a2, r2, s2 = optimize(ci_words, label="RUN 2: CI TRAIN → Ald.211 VALIDATE")
r2_cv = compute_r(a2, ald_words)
r2_heur = compute_r({}, ald_words)
print(f"  Cross-val on Ald:    r = {r2_cv:.4f} (heuristic: {r2_heur:.4f})")
p2, _ = null_model(a2, ci_words)

# ══════════════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  COMPARISON")
print(f"{'='*60}")

all_words = sorted(set(list(a1.keys()) + list(a2.keys())))
match = 0
total = 0

print(f"\n  {'Word':<15} {'Ald→CI':<10} {'CI→Ald':<10} {'Match'}")
print(f"  {'-'*45}")
for w in all_words:
    f1 = a1.get(w, '—')
    f2 = a2.get(w, '—')
    m = '✓' if f1 == f2 else '✗'
    total += 1
    if f1 == f2:
        match += 1
    print(f"  {w:<15} {f1:<10} {f2:<10} {m}")

# Separate anchors from free
free_words = [w for w in all_words if w not in KNOWN]
free_both = [w for w in free_words if w in a1 and w in a2]
free_match = sum(1 for w in free_both if a1[w] == a2[w])

print(f"\n  Total agreement:     {match}/{total}")
print(f"  Anchor agreement:    {sum(1 for w in KNOWN if a1.get(w)==a2.get(w))}"
      f"/{len(KNOWN)} (fixed)")
print(f"  Free, both present:  {len(free_both)} words")
print(f"  Free, both agree:    {free_match}/{len(free_both)}")

# Corpus-independent assignments
corpus_independent = [w for w in all_words if a1.get(w) == a2.get(w)]
print(f"\n  Corpus-independent assignments:")
for w in corpus_independent:
    src = "anchor" if w in KNOWN else "free"
    print(f"    {w:<15} → {a1[w]:<6} ({src})")

# ══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  {'Metric':<30} {'Ald→CI':>10} {'CI→Ald':>10}")
print(f"  {'-'*50}")
print(f"  {'Training r':<30} {r1:>10.4f} {r2:>10.4f}")
print(f"  {'Cross-validation r':<30} {r1_cv:>10.4f} {r2_cv:>10.4f}")
print(f"  {'Heuristic r (CV corpus)':<30} {r1_heur:>10.4f} {r2_heur:>10.4f}")
print(f"  {'Improvement over heuristic':<30} {r1_cv-r1_heur:>+10.4f} "
      f"{r2_cv-r2_heur:>+10.4f}")
print(f"  {'Assignments (total)':<30} {len(a1):>10d} {len(a2):>10d}")
print(f"  {'Null model p':<30} {p1:>10.5f} {p2:>10.5f}")
print(f"  {'Corpus-independent':<30} {len(corpus_independent):>10d}")

# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════

results = {
    'run1': {
        'training_corpus': 'Ald.211',
        'validation_corpus': 'CI',
        'training_words': len(ald_words),
        'assignments': a1,
        'training_r': r1,
        'cv_r': r1_cv,
        'heuristic_r': r1_heur,
        'null_p': p1,
    },
    'run2': {
        'training_corpus': 'CI',
        'validation_corpus': 'Ald.211',
        'training_words': len(ci_words),
        'assignments': a2,
        'training_r': r2,
        'cv_r': r2_cv,
        'heuristic_r': r2_heur,
        'null_p': p2,
    },
    'corpus_independent': {w: a1[w] for w in corpus_independent},
    'vms_bigram_pairs': sum(vms_bg.values()),
    'vms_bigram_types': len(vms_bg),
}

with open('S13_reversal_test_result.pkl', 'wb') as f:
    pickle.dump(results, f)

# Also save human-readable markdown
with open('S13_reversal_test_result.md', 'w') as f:
    f.write("# S13: Corpus Reversal Test\n\n")
    f.write("## Setup\n\n")
    f.write(f"- Ald.211: {len(ald_words)} words\n")
    f.write(f"- CI: {len(ci_words)} words\n")
    f.write(f"- VMS target: {sum(vms_bg.values())} EC-EC bigram pairs, "
            f"{len(vms_bg)} types\n")
    f.write(f"- Anchors fixed: et→Y, in→N\n\n")
    f.write("## Results\n\n")
    f.write("| | Ald.211→CI | CI→Ald.211 |\n")
    f.write("|---|---|---|\n")
    f.write(f"| Training r | {r1:.4f} | {r2:.4f} |\n")
    f.write(f"| Cross-validation r | {r1_cv:.4f} | {r2_cv:.4f} |\n")
    f.write(f"| Heuristic r | {r1_heur:.4f} | {r2_heur:.4f} |\n")
    f.write(f"| Assignments | {len(a1)} | {len(a2)} |\n")
    f.write(f"| Null model p | {p1:.5f} | {p2:.5f} |\n\n")
    f.write("## Assignment comparison\n\n")
    f.write("| Word | Ald→CI | CI→Ald | Match |\n")
    f.write("|------|--------|--------|-------|\n")
    for w in all_words:
        f1 = a1.get(w, '—')
        f2 = a2.get(w, '—')
        m = '✓' if f1 == f2 else '✗'
        f.write(f"| {w} | {f1} | {f2} | {m} |\n")
    f.write(f"\n## Corpus-independent assignments\n\n")
    if corpus_independent:
        for w in corpus_independent:
            src = "anchor" if w in KNOWN else "free"
            f.write(f"- **{w}** → {a1[w]} ({src})\n")
    else:
        f.write("None. No assignment is stable across both training corpora.\n")

    f.write(f"\n## Summary statistics\n\n")
    f.write(f"- Cross-validation r: {r1_cv:.4f} (Ald→CI) vs {r2_cv:.4f} (CI→Ald)\n")
    f.write(f"- Improvement over heuristic: {r1_cv-r1_heur:+.4f} (Ald→CI) vs "
            f"{r2_cv-r2_heur:+.4f} (CI→Ald)\n")
    f.write(f"- Null model p: {p1:.5f} (Ald→CI) vs {p2:.5f} (CI→Ald)\n")
    f.write(f"- Corpus-independent assignments: {len(corpus_independent)} of "
            f"{total} total\n")
    f.write(f"- Free assignments in both runs: {len(free_both)}, "
            f"of which {free_match} agree\n")

print(f"\n✓ Saved: S13_reversal_test_result.pkl")
print(f"✓ Saved: S13_reversal_test_result.md")

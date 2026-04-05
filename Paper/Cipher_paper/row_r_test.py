#!/usr/bin/env python3
"""
ROW-R TEST: Does your source language fit the VMS grid shape?
=============================================================
Run this on any pharmaceutical corpus. It produces three numbers
that settle the language question.

Usage:
    python3 row_r_test.py your_wordlist.txt

Input: a text file with one word per line (content words only,
       function words removed). Or a plain text file — the script
       will tokenise it.

Output: initial consonant frequencies, 7-row grouping, shape
        distance to VMS.

Edward Bozzard · April 2026
"""

import sys
from collections import Counter

# ══════════════════════════════════════════════════════════════
# VMS REFERENCE (from enriched_records.pkl, Herbal-A FC, m_core[0])
# These are FIXED. Do not modify.
# ══════════════════════════════════════════════════════════════

VMS_SHAPE = [0.354, 0.289, 0.152, 0.093, 0.060, 0.039, 0.012]
VMS_ROW_R = 0.012  # 20/1621 tokens

# The 7 rows and their consonant assignments (from the grid)
CONSONANT_TO_ROW = {
    'c': 'o', 's': 'o', 'p': 'o',
    'v': 'c',  # vowel-initial also maps to 'c'
    'f': 'e', 'd': 'e',
    'm': 'a', 'l': 'a',
    'r': 'd', 'q': 'd', 'h': 'd', 'n': 'd', 'g': 'd',
    't': 'l',
    'b': 'r', 'z': 'r', 'x': 'r', 'j': 'r', 'k': 'r',
    'w': 'r', 'y': 'r',
}

# ══════════════════════════════════════════════════════════════
# LOAD AND PARSE
# ══════════════════════════════════════════════════════════════

if len(sys.argv) < 2:
    print("Usage: python3 row_r_test.py your_wordlist.txt")
    print("       One word per line, or plain text (will be tokenised).")
    sys.exit(1)

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    raw = f.read()

# Tokenise: lowercase, split on whitespace, keep only alpha, min 2 chars
words = [w.lower() for w in raw.split() if w.isalpha() and len(w) >= 2]
print(f"Loaded: {len(words)} words from {sys.argv[1]}")

# ══════════════════════════════════════════════════════════════
# TEST 1: INITIAL CONSONANT FREQUENCIES
# ══════════════════════════════════════════════════════════════

cons_freq = Counter()
VOWELS = set('aeiouäöüâêîôûæœ')
for w in words:
    first_char = w[0]
    if first_char in VOWELS:
        cons_freq['∅'] += 1
    else:
        cons_freq[first_char] += 1

total = sum(cons_freq.values())

print(f"\n{'='*50}")
print(f"INITIAL CONSONANT FREQUENCIES (n={total})")
print(f"{'='*50}")
for c, n in cons_freq.most_common():
    print(f"  {c:>3}: {n:>6} ({n/total*100:>5.1f}%)")

# ══════════════════════════════════════════════════════════════
# TEST 2: ROW 'r' (b, z, w, k, x, j, y)
# ══════════════════════════════════════════════════════════════

row_r_cons = set('bzwkxjy')
row_r_count = sum(n for c, n in cons_freq.items() if c in row_r_cons)
row_r_pct = row_r_count / total

print(f"\n{'='*50}")
print(f"ROW 'r' TEST")
print(f"{'='*50}")
print(f"  Your corpus:  {row_r_count}/{total} = {row_r_pct*100:.1f}%")
print(f"  VMS Herbal-A: 20/1621 = {VMS_ROW_R*100:.1f}%")
print(f"  Latin (CI):   567/24300 = 2.3%")
print(f"  Ratio (yours/VMS): {row_r_pct/VMS_ROW_R:.1f}×")

if row_r_pct > 0.10:
    print(f"  >> YOUR LANGUAGE IS INCOMPATIBLE with VMS row 'r'")
elif row_r_pct > 0.05:
    print(f"  >> YOUR LANGUAGE IS UNLIKELY to match VMS row 'r'")
elif row_r_pct < 0.04:
    print(f"  >> YOUR LANGUAGE IS COMPATIBLE with VMS row 'r'")

# Breakdown
print(f"\n  Row 'r' consonant breakdown:")
for c in 'bzwkxjy':
    n = cons_freq.get(c, 0)
    print(f"    {c}: {n} ({n/total*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# TEST 3: 7-ROW DISTRIBUTION (using Latin row assignments)
# ══════════════════════════════════════════════════════════════

row_counts = Counter()
for c, n in cons_freq.items():
    if c == '∅':
        row_counts['c'] += n  # vowel-initial → row 'c'
    elif c in CONSONANT_TO_ROW:
        row_counts[CONSONANT_TO_ROW[c]] += n
    else:
        # Characters not in the Latin mapping (ä,ö,ü,ß,sch...)
        # Map to nearest: sch→s→'o', ß→s→'o'
        row_counts['c'] += n  # default to vowel-initial row

print(f"\n{'='*50}")
print(f"7-ROW DISTRIBUTION (Latin row assignments)")
print(f"{'='*50}")
print(f"{'Row':<6} {'Consonants':<20} {'Yours':>8} {'VMS':>8} {'Latin':>8}")
print(f"{'-'*52}")

ROW_LABELS = {
    'o': 'c,s,p', 'c': '∅,v', 'e': 'f,d',
    'a': 'm,l', 'd': 'r,q,h,n,g', 'l': 't', 'r': 'b,z,x,j,k,w,y'
}
VMS_ROWS = {'o': 0.354, 'c': 0.289, 'e': 0.152, 'a': 0.093,
            'd': 0.060, 'l': 0.039, 'r': 0.012}
LAT_ROWS = {'o': 0.281, 'c': 0.264, 'e': 0.111, 'a': 0.119,
            'd': 0.159, 'l': 0.042, 'r': 0.023}

chi2_yours = 0
for row in ['o', 'c', 'e', 'a', 'd', 'l', 'r']:
    yours = row_counts.get(row, 0) / total
    vms = VMS_ROWS[row]
    lat = LAT_ROWS[row]
    if vms > 0:
        chi2_yours += (yours - vms) ** 2 / vms
    print(f"{row:<6} {ROW_LABELS[row]:<20} {yours*100:>7.1f}% {vms*100:>7.1f}% {lat*100:>7.1f}%")

print(f"\n  χ² (yours vs VMS): {chi2_yours:.4f}")
print(f"  χ² (Latin vs VMS): 0.025")

# ══════════════════════════════════════════════════════════════
# TEST 4: SHAPE COMPARISON (mapping-independent)
# ══════════════════════════════════════════════════════════════

# Optimal 7-grouping for YOUR language
# Sort consonants by frequency, chunk into 7 groups
sorted_cons = cons_freq.most_common()
n_per_group = max(1, len(sorted_cons) // 7)
your_groups = []
for i in range(7):
    start = i * n_per_group
    end = start + n_per_group if i < 6 else len(sorted_cons)
    if start >= len(sorted_cons):
        your_groups.append(0)
    else:
        your_groups.append(sum(n for _, n in sorted_cons[start:end]) / total)

your_shape = sorted(your_groups, reverse=True)
lat_shape = [0.281, 0.264, 0.159, 0.119, 0.111, 0.042, 0.023]

def shape_dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))

d_yours = shape_dist(VMS_SHAPE, your_shape)
d_latin = shape_dist(VMS_SHAPE, lat_shape)

print(f"\n{'='*50}")
print(f"SHAPE COMPARISON (mapping-independent)")
print(f"{'='*50}")
print(f"\nSorted frequency curves (7 groups):")
print(f"{'Rank':<6} {'VMS':>8} {'Latin':>8} {'Yours':>8}")
print(f"{'-'*32}")
for i in range(7):
    print(f"{i+1:<6} {VMS_SHAPE[i]*100:>7.1f}% {lat_shape[i]*100:>7.1f}% {your_shape[i]*100:>7.1f}%")

print(f"\n  Shape distance (yours vs VMS):  {d_yours:.6f}")
print(f"  Shape distance (Latin vs VMS):  {d_latin:.6f}")
if d_latin > 0:
    print(f"  Ratio: Latin is {d_yours/d_latin:.1f}× closer to VMS than yours")

print(f"\n  Top-2 concentration: VMS={sum(VMS_SHAPE[:2])*100:.0f}%, "
      f"Latin={sum(lat_shape[:2])*100:.0f}%, "
      f"Yours={sum(your_shape[:2])*100:.0f}%")

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"  Row 'r' (b,z,w,k...): yours={row_r_pct*100:.1f}%, VMS=1.2%, Latin=2.3%")
print(f"  χ² (Latin grouping):  yours={chi2_yours:.4f}, Latin=0.025")
print(f"  Shape distance:       yours={d_yours:.6f}, Latin={d_latin:.6f}")
print(f"\n  Send these three numbers to Ed.")

#!/usr/bin/env python3
"""
LANGUAGE BATTERY v2.2 — Greek Pharmaceutical Expansion
Date: 20 March 2026
Compares VMS Herbal-A m_core leading char distribution to Greek pharmaceutical corpora.

Key finding: Greek pharma with nomenclator filter (function words removed) scores
χ² = 0.138, beating CI Latin's 0.458 — the best score in the entire battery.

Requires: enriched_records.pkl (from Voynichdecomp repo)
          Greek XML texts from First1KGreek / OpenGreekAndLatin GitHub
"""

import re, unicodedata, pickle
from collections import Counter
import numpy as np
import pandas as pd

# =====================================================
# CONFIGURATION
# =====================================================
ROW_CHARS = ['o', 'c', 'e', 'a', 'l', 'd', 'r']

# Greek function words (nomenclator candidates)
# Under the two-table cipher, these encode as EC (empty-core) tokens
GREEK_FW = {
    'ο', 'η', 'το', 'τον', 'την', 'του', 'της', 'τω', 'τοι', 'ται',
    'των', 'τοις', 'ταις', 'τα', 'τους', 'τας',
    'και', 'δε', 'τε', 'γαρ', 'μεν', 'ουν', 'αν', 'ει', 'ως', 'αλλα',
    'ουτε', 'μητε', 'ητοι', 'εαν',
    'εν', 'εκ', 'εξ', 'εις', 'επι', 'προς', 'κατα', 'μετα', 'δια',
    'υπο', 'υπερ', 'παρα', 'περι', 'συν', 'αντι', 'απο', 'προ',
    'ουτος', 'αυτη', 'τουτο', 'ταυτα', 'τουτον', 'ταυτης',
    'αυτος', 'αυτη', 'αυτο', 'αυτου', 'αυτης', 'αυτων',
    'εστι', 'εστιν', 'ειναι',
    'ος', 'οι', 'ου', 'ουκ', 'ουχ', 'μη',
    'τις', 'τι', 'τινα', 'τινες', 'τινι', 'τινος', 'τινων',
}

# VMS row mapping for romanized Greek consonants
ROW_MAP = {
    's': 'o', 'p': 'o', 'c': 'o', 'k': 'o', 'g': 'o', 
    'ks': 'o', 'ps': 'o',
    'VOWEL': 'c', 'ch': 'c',
    'd': 'e', 'f': 'e', 'ph': 'e', 'b': 'e',
    'm': 'a', 'l': 'a',
    't': 'l', 'th': 'l',
    'n': 'd', 'z': 'd', 'r': 'r',
}

GREEK_ROMAN = {
    'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e',
    'ζ': 'z', 'η': 'e', 'θ': 'th', 'ι': 'i', 'κ': 'k',
    'λ': 'l', 'μ': 'm', 'ν': 'n', 'ξ': 'ks', 'ο': 'o',
    'π': 'p', 'ρ': 'r', 'σ': 's', 'ς': 's', 'τ': 't',
    'υ': 'u', 'φ': 'ph', 'χ': 'ch', 'ψ': 'ps', 'ω': 'o',
}

# =====================================================
# FUNCTIONS
# =====================================================
def extract_greek_from_xml(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)
    words = []
    for token in text.split():
        greek_chars = [c for c in token if '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF']
        if len(greek_chars) > len(token) * 0.5:
            clean = re.sub(r'[^\u0370-\u03FF\u1F00-\u1FFF]', '', token)
            if len(clean) >= 2:
                words.append(clean)
    return words

def strip_accents(w):
    w = unicodedata.normalize('NFD', w)
    return ''.join(c for c in w if unicodedata.category(c) != 'Mn').lower()

def classify_word(w):
    w_stripped = strip_accents(w)
    roman = ''.join(GREEK_ROMAN.get(c, c) for c in w_stripped)
    
    if roman.startswith('th'): cons = 'th'
    elif roman.startswith('ph'): cons = 'ph'
    elif roman.startswith('ch'): cons = 'ch'
    elif roman.startswith('ps'): cons = 'ps'
    elif roman.startswith('ks'): cons = 'ks'
    elif len(roman) > 0 and roman[0] in 'aeiou': cons = 'VOWEL'
    elif len(roman) > 0: cons = roman[0]
    else: return None, None, None
    
    row = ROW_MAP.get(cons, '?')
    is_son = cons in {'m', 'n', 'l', 'r', 'VOWEL'}
    return cons, is_son, row

def compute_battery(words, vms_dist, filter_fw=False):
    row_counts = Counter()
    total = 0
    son_count = 0
    
    for w in words:
        if filter_fw:
            if strip_accents(w) in GREEK_FW:
                continue
        cons, is_son, row = classify_word(w)
        if cons is None: continue
        row_counts[row] += 1
        total += 1
        if is_son: son_count += 1
    
    if total == 0:
        return None
    
    dist = np.array([row_counts.get(ch, 0) for ch in ROW_CHARS], dtype=float)
    dist = dist / dist.sum()
    chi2 = sum((dist[i] - vms_dist[i])**2 / vms_dist[i]
               for i in range(len(ROW_CHARS)) if vms_dist[i] > 0)
    
    return {
        'N': total, 'son_pct': son_count / total * 100,
        'chi2': chi2, 'dist': dict(zip(ROW_CHARS, dist))
    }

# =====================================================
# MAIN
# =====================================================
if __name__ == '__main__':
    # Load VMS baseline
    data = pickle.load(open('Voynichdecomp/enriched_records.pkl', 'rb'))
    df = pd.DataFrame(data)
    ha = df[(df['section'] == 'Herbal-A') & (~df['empty_core'])].copy()
    ha['lead'] = ha['m_core'].apply(lambda mc: mc[0] if isinstance(mc, str) and len(mc) > 0 and mc != '∅' else None)
    ha = ha[ha['lead'].notna()]
    vms_lead = Counter(ha['lead'])
    vms_dist = np.array([vms_lead.get(ch, 0) for ch in ROW_CHARS], dtype=float)
    vms_dist = vms_dist / vms_dist.sum()
    
    # Process Greek corpora
    files = {
        'Galen De Simpl.': 'galen_simpl_raw.xml',
        'Diosc. Euporista': 'dioscorides_tlg006.xml',
        'Hipp. Nat.Mul.': 'hippocrates_nat_mul.xml',
        'Hipp. De Morb.': 'hippocrates_morb.xml',
    }
    
    all_words = []
    for path in files.values():
        all_words.extend(extract_greek_from_xml(path))
    
    r_all = compute_battery(all_words, vms_dist, filter_fw=False)
    r_filtered = compute_battery(all_words, vms_dist, filter_fw=True)
    
    print(f"Greek (all):       N={r_all['N']:6d}  Son%={r_all['son_pct']:.0f}%  χ²={r_all['chi2']:.3f}")
    print(f"Greek (no FW):     N={r_filtered['N']:6d}  Son%={r_filtered['son_pct']:.0f}%  χ²={r_filtered['chi2']:.3f}")
    print(f"CI Latin baseline: N=24300   Son%=84%  χ²=0.458")

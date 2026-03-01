#!/usr/bin/env python3
"""
85-Metric Scoring Suite for VMS Generator Comparison
=====================================================
Full implementation matching Bowern-Gaskell (2022) methodology for all
shared metrics, plus 47 original extensions.

CRITICAL: The BG-shared metrics (wordlen, wordunique, wordchange, wordbias,
charbias, entropy, compression, zipf, etc.) are computed INSIDE a subsampling
loop exactly as in Gaskell & Bowern's stats.py:
  - 100 iterations (configurable)
  - ~200-word random subsamples drawn from random line positions
  - All BG metrics computed per subsample, then averaged

The 47 original metrics (entropy hierarchy, TTR variants, hapax spectrum,
autocorrelation, Heaps, etc.) are computed on the full corpus.

Dependencies:
  Required: numpy, scipy
  Optional: distance (pip install distance) — for Levenshtein metrics.
            Falls back to pure-Python implementation if unavailable.

Usage:
    from score_85_metrics import compute_metrics, score_against_vms, CORE_15, TOLERANCES

    metrics = compute_metrics(token_list, lines=line_list)
    results = score_against_vms(metrics, vms_baseline)

Reference:
    Gaskell, D.E. & Bowern, C.L. (2022). Gibberish after all? Voynichese
    is statistically similar to human-produced samples of meaningless text.
    CEUR Workshop Proceedings, ConfVM 2022, University of Malta.

Version: 2.0.0 (2026-03-01) — corrected BG methodology
"""

import math
import zlib
import random
import statistics
import numpy as np
from collections import Counter, defaultdict, deque
from scipy import stats as sp_stats
import scipy.stats

# ── Optional fast Levenshtein ─────────────────────────────────────────
try:
    import distance as _distance_lib
    def _levenshtein(a, b):
        return _distance_lib.levenshtein(a, b)
    _LEV_SOURCE = 'distance'
except ImportError:
    def _levenshtein(a, b):
        """Pure-Python Levenshtein distance (Wagner-Fischer)."""
        if len(a) < len(b):
            return _levenshtein(b, a)
        if len(b) == 0:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(
                    prev[j + 1] + 1,      # deletion
                    curr[j] + 1,           # insertion
                    prev[j] + (ca != cb)   # substitution
                ))
            prev = curr
        return prev[-1]
    _LEV_SOURCE = 'builtin'


# ======================================================================
# METRIC LISTS
# ======================================================================

CORE_15 = [
    'ttr', 'H1_unigram', 'H2_markov_cond', 'top10_share',
    'zipf_alpha', 'zipf_r2', 'heaps_beta',
    'wordlen_mean', 'wordlen_std', 'hapax_ratio_types',
    'repeated_words', 'mattr_25', 'mattr_100', 'mattr_50',
    'wordlen_autocorr'
]

# BG-methodology metrics (computed per subsample, averaged)
BG_METRICS = [
    'wordlen_mean', 'wordlen_std', 'wordlen_skew',
    'wordlen_unique_mean', 'wordlen_unique_std', 'wordlen_unique_skew',
    'wordlen_autocorr',
    'wordunique_mean', 'wordunique_std', 'wordunique_skew',
    'wordchange_mean', 'wordchange_std', 'wordchange_skew',
    'worddist_max', 'worddist_shape',
    'wordbias_mean', 'wordbias_std', 'wordbias_skew',
    'wordbias_lines_mean', 'wordbias_lines_std', 'wordbias_lines_skew',
    'chardist_max', 'chardist_shape',
    'ngramdist_max', 'ngramdist_shape',
    'charbias_mean', 'charbias_std', 'charbias_skew',
    'charbias_words_mean', 'charbias_words_std', 'charbias_words_skew',
    'unique_words', 'repeated_words', 'tripled_words',
    'unique_chars', 'repeated_chars', 'tripled_chars',
    'unique_ngrams',
    'entropy', 'compression', 'zipf_lmz', 'flipped_pairs',
]

# Original extensions (computed on full corpus)
ORIGINAL_METRICS = [
    'H0_max_entropy', 'H1_unigram', 'H2_markov_cond',
    'h2_joint_digraph', 'h2_conditional',
    'h3_joint_trigraph', 'h3_conditional',
    'char_evenness', 'char_redundancy',
    'char_simpson_D', 'char_yule_K',
    'digraph_unique', 'digraph_coverage', 'trigraph_unique',
    'ttr', 'rttr', 'cttr',
    'log_ttr', 'maas_a2', 'uber_index', 'brunet_W',
    'msttr_25', 'msttr_50', 'msttr_100',
    'mattr_25', 'mattr_50', 'mattr_100',
    'hapax_ratio_tokens', 'hapax_ratio_types',
    'dis_ratio_tokens', 'dis_ratio_types',
    'sichel_S', 'hapax_type_proportion',
    'freq_spectrum_1', 'freq_spectrum_2', 'freq_spectrum_3', 'freq_spectrum_gt10',
    'word_yule_K', 'honore_R',
    'autocorr_wordlen', 'autocorr_wordfreq',
    'autocorr_ttr_25', 'autocorr_hapax_25',
    'zipf_alpha', 'zipf_r2', 'heaps_beta',
    'top10_share', 'top50_share',
]

ALL_85 = sorted(set(BG_METRICS + ORIGINAL_METRICS))


# ======================================================================
# TOLERANCES
# ======================================================================

TOLERANCES = {
    # BG word length (subsampled means)
    'wordlen_mean': 0.50, 'wordlen_std': 0.20, 'wordlen_skew': 0.20,
    'wordlen_unique_mean': 0.50, 'wordlen_unique_std': 0.30, 'wordlen_unique_skew': 0.30,
    'wordlen_autocorr': 0.05,
    # BG Levenshtein-based
    'wordunique_mean': 0.10, 'wordunique_std': 0.05, 'wordunique_skew': 0.30,
    'wordchange_mean': 0.10, 'wordchange_std': 0.05, 'wordchange_skew': 0.30,
    # BG word distribution
    'worddist_max': 5.0, 'worddist_shape': 1.0,
    # BG word positional bias
    'wordbias_mean': 0.05, 'wordbias_std': 0.03, 'wordbias_skew': 0.50,
    'wordbias_lines_mean': 0.05, 'wordbias_lines_std': 0.03, 'wordbias_lines_skew': 0.50,
    # BG character distribution
    'chardist_max': 0.05, 'chardist_shape': 0.01,
    # BG ngram distribution
    'ngramdist_max': 20.0, 'ngramdist_shape': 1.0,
    # BG character positional bias
    'charbias_mean': 0.03, 'charbias_std': 0.02, 'charbias_skew': 0.50,
    'charbias_words_mean': 0.03, 'charbias_words_std': 0.02, 'charbias_words_skew': 0.50,
    # BG counts
    'unique_words': 20.0, 'repeated_words': 0.005, 'tripled_words': 0.001,
    'unique_chars': 3, 'repeated_chars': 0.02, 'tripled_chars': 0.005,
    'unique_ngrams': 50.0,
    # BG global
    'entropy': 0.30, 'compression': 0.05, 'zipf_lmz': 5.0, 'flipped_pairs': 0.01,
    # Entropy hierarchy (original)
    'H0_max_entropy': 0.30, 'H1_unigram': 0.15,
    'H2_markov_cond': 0.35, 'h2_conditional': 0.35,
    'h2_joint_digraph': 0.35,
    'h3_joint_trigraph': 0.50, 'h3_conditional': 0.35,
    # Character distribution (original)
    'char_evenness': 0.05, 'char_redundancy': 0.05,
    'char_simpson_D': 0.02, 'char_yule_K': 50,
    # Digraph/trigraph (original)
    'digraph_unique': 50, 'digraph_coverage': 0.10, 'trigraph_unique': 500,
    # TTR variants (original)
    'ttr': 0.03, 'rttr': 8.0, 'cttr': 4.0,
    'log_ttr': 0.02, 'maas_a2': 0.004, 'uber_index': 15, 'brunet_W': 2.0,
    'msttr_25': 0.05, 'msttr_50': 0.05, 'msttr_100': 0.05,
    'mattr_25': 0.05, 'mattr_50': 0.05, 'mattr_100': 0.05,
    # Hapax & frequency spectrum (original)
    'hapax_ratio_tokens': 0.03, 'hapax_ratio_types': 0.10,
    'dis_ratio_tokens': 0.01, 'dis_ratio_types': 0.03,
    'sichel_S': 0.03, 'hapax_type_proportion': 0.10,
    'freq_spectrum_1': 0.10, 'freq_spectrum_2': 0.03,
    'freq_spectrum_3': 0.02, 'freq_spectrum_gt10': 0.02,
    # Lexical richness (original)
    'word_yule_K': 15, 'honore_R': 500,
    # Autocorrelation (original)
    'autocorr_wordlen': 0.05, 'autocorr_wordfreq': 0.03,
    'autocorr_ttr_25': 0.10, 'autocorr_hapax_25': 0.10,
    # Zipf / Heaps (original)
    'zipf_alpha': 0.10, 'zipf_r2': 0.05, 'heaps_beta': 0.05,
    # Frequency concentration (original)
    'top10_share': 0.03, 'top50_share': 0.08,
}


# ======================================================================
# BG HELPER: Moran's I for 1D lattice
# ======================================================================

def _morans_i_1d(x):
    """
    Moran's I spatial autocorrelation for a 1D sequence.
    Equivalent to esda.Moran(x, lat2W(nrows=len(x), ncols=1)).I
    
    Uses binary contiguity weights (w_ij = 1 if |i-j| == 1).
    Formula: I = (N / S0) * (Σ_ij w_ij * z_i * z_j) / (Σ z_i^2)
    where z_i = x_i - mean(x), S0 = sum of all weights = 2*(N-1).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return 0.0
    z = x - x.mean()
    denom = np.sum(z ** 2)
    if denom == 0:
        return 0.0
    # For 1D binary contiguity: numerator = 2 * sum(z[i]*z[i+1])
    numer = 2.0 * np.sum(z[:-1] * z[1:])
    s0 = 2.0 * (n - 1)
    return float((n / s0) * numer / denom)


# ======================================================================
# BG HELPER: 2nd-order Markov character entropy
# ======================================================================

def _char_entropy_markov(text_lines):
    """
    Third-order Markov model character entropy rate.
    Exact reimplementation of BG's char_entropy function.
    
    Input: list of text strings (raw lines including spaces).
    Returns: entropy rate in bits per character.
    
    Based on Clement Pit-Claudel's implementation.
    """
    def _tokenize(lines):
        for line in lines:
            for ch in line.lower().strip() + " ":
                yield ch

    model_order = 2
    model = defaultdict(Counter)
    stats = Counter()
    buf = deque(maxlen=model_order)

    for token in _tokenize(text_lines):
        prefix = tuple(buf)
        buf.append(token)
        if len(prefix) == model_order:
            stats[prefix] += 1
            model[prefix][token] += 1

    def _entropy(counter, total):
        return -sum(
            (c / total) * math.log2(c / total)
            for c in counter.values()
        )

    total_contexts = sum(stats.values())
    if total_contexts == 0:
        return 0.0
    return sum(
        stats[prefix] * _entropy(model[prefix], stats[prefix])
        for prefix in stats
    ) / total_contexts


# ======================================================================
# BG SUBSAMPLED METRICS COMPUTATION
# ======================================================================

def _bg_subsample_iteration(lines_text, rng, subset_words, ngram_max_len,
                            do_levenshtein):
    """
    Run one iteration of BG subsampling.
    
    This is a faithful reimplementation of the inner loop of BG stats.py
    (lines 198-467). Returns a dict of per-iteration metric values.
    
    Args:
        lines_text: list of str — raw text lines (space-separated words)
        rng: random.Random instance
        subset_words: target word count per subsample
        ngram_max_len: max ngram length (BG uses 3)
        do_levenshtein: whether to compute Levenshtein metrics
    
    Returns: dict of metric_name -> value, or None if subsample failed
    """
    n_lines = len(lines_text)
    if n_lines == 0:
        return None

    # ── Pull random subset of lines (BG methodology) ──────────────
    num_words = 0
    lines_sub = []
    while num_words < subset_words:
        start = rng.randint(0, n_lines)  # BG uses randint(0, len(lines))
        for t in range(start, n_lines):
            lines_sub.append(lines_text[t])
            num_words += len(lines_text[t].split(' '))
            if num_words >= subset_words:
                break

    # ── Per-subsample data structures ─────────────────────────────
    docwords = []
    wordbank = {}
    wordlen_bank = []
    wordlen_unique_bank = []
    wordunique_bank = {}
    wordchange_bank = []
    word_heat = {}
    word_variation = {}
    word_heat_lines = {}
    word_lines_variation = {}
    charbank = {}
    ngram_bank = {}
    ngram_bank_unique = {}
    ngram_heat = {}          # 10-bin: position within LINE
    ngram_heat_words = {}    # 5-bin: position within WORD
    ngram_variation = {}     # CV of ngram_heat
    ngram_variation_words = {}  # CV of ngram_heat_words

    # ── Line-by-line stats (first pass) ───────────────────────────
    num_chars = 0
    for line_index, line in enumerate(lines_sub):
        words = line.split(' ')
        words = [w for w in words if w]  # remove blank words
        docwords.extend(words)

        for index, word in enumerate(words):
            if word not in wordbank:
                wordbank[word] = 0
                word_heat[word] = [0, 0, 0, 0, 0]
                word_heat_lines[word] = [0, 0, 0, 0, 0]
            wordbank[word] += 1

            # Position within line (5-bin heatmap)
            if len(words) > 0:
                word_heat[word][min(4, math.floor((index / len(words)) * 5))] += 1
            # Position of line within document (5-bin heatmap)
            if len(lines_sub) > 0:
                word_heat_lines[word][min(4, math.floor((line_index / len(lines_sub)) * 5))] += 1

            wordlen_bank.append(len(word))
            num_chars += len(word)

    if not docwords or num_chars == 0:
        return None

    # ── Whole-document sequential stats ───────────────────────────
    word_repeats = 0
    word_triples = 0
    char_repeats = 0
    char_triples = 0
    word_flips = 0
    last_word = ''
    last_word2 = ''

    for word in docwords:
        if word == last_word:
            word_repeats += 1
            if last_word == last_word2:
                word_triples += 1
        else:
            # Levenshtein distance to prior word (non-repeated only)
            if do_levenshtein and last_word:
                wordchange_bank.append(
                    _levenshtein(word, last_word) / len(word) if len(word) > 0 else 0
                )

        # Find reversed pairs (BG O(N^2) scan)
        prev2 = ''
        for word2 in docwords:
            if word2 == last_word and prev2 == word:
                word_flips += 1
                break
            prev2 = word2

        # Within-word character repetitions
        last_char = ''
        last_char2 = ''
        for char in word:
            if char not in charbank:
                charbank[char] = 0
            charbank[char] += 1

            if char == last_char:
                char_repeats += 1
            if char == last_char and last_char == last_char2:
                char_triples += 1

            last_char2 = last_char
            last_char = char

        last_word2 = last_word
        last_word = word

    # ── Word-by-word stats (type-level) ───────────────────────────
    for word in wordbank:
        word_len = len(word)
        wordlen_unique_bank.append(word_len)

        # Pairwise Levenshtein distance between all word types
        if do_levenshtein:
            for word2 in wordbank:
                key_rev = word2 + "_" + word
                if key_rev not in wordunique_bank:
                    wordunique_bank[word + "_" + word2] = (
                        _levenshtein(word, word2) / len(word) if len(word) > 0 else 0
                    )

        # Coefficient of variation for within-line position heatmap
        wh_mean = statistics.mean(word_heat[word])
        if wh_mean > 0:
            word_variation[word] = statistics.stdev(word_heat[word]) / wh_mean
        else:
            word_variation[word] = 0.0

        # Coefficient of variation for document-position heatmap
        whl_mean = statistics.mean(word_heat_lines[word])
        if whl_mean > 0:
            word_lines_variation[word] = statistics.stdev(word_heat_lines[word]) / whl_mean
        else:
            word_lines_variation[word] = 0.0

        # Ngrams (1 to ngram_max_len) within words
        for ngram_index in range(word_len):
            for ngram_len in range(1, ngram_max_len + 1):
                if ngram_index + ngram_len <= word_len:
                    ngram = word[ngram_index: ngram_index + ngram_len]
                    if ngram not in ngram_bank:
                        ngram_bank[ngram] = 0
                        ngram_bank_unique[ngram] = 0
                        ngram_heat[ngram] = [0] * 10  # line-position heatmap
                        ngram_heat_words[ngram] = [0] * 5  # word-position heatmap
                    ngram_bank[ngram] += wordbank[word]  # frequency-weighted!
                    ngram_bank_unique[ngram] += 1
                    # Word-position heatmap (5 bins)
                    denom_w = word_len - (ngram_len - 1)
                    if denom_w > 0:
                        bin_idx = round((ngram_index / denom_w) * 4)
                        ngram_heat_words[ngram][min(4, bin_idx)] += 1
                    else:
                        ngram_heat_words[ngram][0] += 1

    # ── Line-by-line stats (second pass: ngram line positions) ────
    for line in lines_sub:
        line_len = len(line)
        if line_len == 0:
            continue
        for ngram_index in range(line_len):
            for ngram_len in range(1, ngram_max_len + 1):
                if ngram_index + ngram_len <= line_len:
                    ngram = line[ngram_index: ngram_index + ngram_len]
                    if ' ' not in ngram and ngram in ngram_heat:
                        denom_l = line_len - (ngram_len - 1)
                        if denom_l > 0:
                            bin_idx = round((ngram_index / denom_l) * 9)
                            ngram_heat[ngram][min(9, bin_idx)] += 1
                        else:
                            ngram_heat[ngram][0] += 1

    # ── Ngram-by-ngram stats ──────────────────────────────────────
    for ngram in ngram_bank:
        nh_mean = statistics.mean(ngram_heat[ngram])
        nhw_mean = statistics.mean(ngram_heat_words[ngram])
        ngram_variation[ngram] = (
            statistics.stdev(ngram_heat[ngram]) / nh_mean if nh_mean > 0 else 0
        )
        ngram_variation_words[ngram] = (
            statistics.stdev(ngram_heat_words[ngram]) / nhw_mean if nhw_mean > 0 else 0
        )

    # ── Compile per-iteration metrics ─────────────────────────────
    result = {}

    # Word length
    if wordlen_bank:
        result['wordlen_mean'] = statistics.mean(wordlen_bank)
        result['wordlen_std'] = statistics.stdev(wordlen_bank) if len(wordlen_bank) > 1 else 0
        result['wordlen_skew'] = float(sp_stats.skew(wordlen_bank))
    if wordlen_unique_bank:
        result['wordlen_unique_mean'] = statistics.mean(wordlen_unique_bank)
        result['wordlen_unique_std'] = (
            statistics.stdev(wordlen_unique_bank) if len(wordlen_unique_bank) > 1 else 0
        )
        result['wordlen_unique_skew'] = float(sp_stats.skew(wordlen_unique_bank))

    # Moran's I autocorrelation
    result['wordlen_autocorr'] = _morans_i_1d(wordlen_bank)

    # Levenshtein metrics
    if do_levenshtein and wordunique_bank:
        wu_list = list(wordunique_bank.values())
        wu_mean_val = statistics.mean(wu_list)
        result['wordunique_mean'] = wu_mean_val
        result['wordunique_std'] = statistics.stdev(wu_list) if len(wu_list) > 1 else 0
        result['wordunique_skew'] = float(sp_stats.skew(wu_list)) if len(wu_list) > 2 else 0

        if wordchange_bank and wu_mean_val > 0:
            # Normalize wordchange to wordunique mean (BG line 386)
            wc_normalized = [v / wu_mean_val for v in wordchange_bank]
            result['wordchange_mean'] = statistics.mean(wc_normalized)
            result['wordchange_std'] = (
                statistics.stdev(wc_normalized) if len(wc_normalized) > 1 else 0
            )
            result['wordchange_skew'] = (
                float(sp_stats.skew(wc_normalized)) if len(wc_normalized) > 2 else 0
            )

    # Word frequency distribution (top-25 truncated, exponential fit)
    wordbank_sorted = sorted(wordbank.values(), reverse=True)
    if len(wordbank_sorted) > 25:
        wordbank_sorted = wordbank_sorted[:25]
    if wordbank_sorted:
        result['worddist_max'] = wordbank_sorted[0]  # BG: raw count, NOT normalized
        if len(wordbank_sorted) >= 2:
            try:
                _loc, scale = scipy.stats.expon.fit(wordbank_sorted)
                result['worddist_shape'] = scale
            except Exception:
                result['worddist_shape'] = 0

    # Word positional bias (frequency-weighted CV)
    def _freq_weighted_cv(variation_dict, freq_dict):
        """Duplicate CV values by word frequency, return flat list."""
        weighted = []
        for key, cv in variation_dict.items():
            count = freq_dict.get(key, 1)
            weighted.extend([cv] * count)
        return weighted

    wb_list = _freq_weighted_cv(word_variation, wordbank)
    if wb_list:
        result['wordbias_mean'] = statistics.mean(wb_list)
        result['wordbias_std'] = statistics.stdev(wb_list) if len(wb_list) > 1 else 0
        result['wordbias_skew'] = float(sp_stats.skew(wb_list)) if len(wb_list) > 2 else 0

    wbl_list = _freq_weighted_cv(word_lines_variation, wordbank)
    if wbl_list:
        result['wordbias_lines_mean'] = statistics.mean(wbl_list)
        result['wordbias_lines_std'] = statistics.stdev(wbl_list) if len(wbl_list) > 1 else 0
        result['wordbias_lines_skew'] = (
            float(sp_stats.skew(wbl_list)) if len(wbl_list) > 2 else 0
        )

    # Character distribution (normalized, exponential fit)
    charbank_sorted = sorted(charbank.values(), reverse=True)
    charbank_normed = [v / num_chars for v in charbank_sorted]
    if charbank_normed:
        result['chardist_max'] = charbank_normed[0]
        if len(charbank_normed) >= 2:
            try:
                _loc, scale = scipy.stats.expon.fit(charbank_normed)
                result['chardist_shape'] = scale
            except Exception:
                result['chardist_shape'] = 0

    # Ngram distribution (frequency-weighted, exponential fit)
    ngram_bank_sorted = sorted(ngram_bank.values(), reverse=True)
    if ngram_bank_sorted:
        result['ngramdist_max'] = ngram_bank_sorted[0]
        if len(ngram_bank_sorted) >= 2:
            try:
                _loc, scale = scipy.stats.expon.fit(ngram_bank_sorted)
                result['ngramdist_shape'] = scale
            except Exception:
                result['ngramdist_shape'] = 0

    # Character positional bias — LINE position (1-char ngrams only)
    char_variation = {k: v for k, v in ngram_variation.items() if len(k) == 1}
    cb_list = []
    for k, cv in char_variation.items():
        count = ngram_bank.get(k, 1)
        cb_list.extend([cv] * count)
    if cb_list:
        result['charbias_mean'] = statistics.mean(cb_list)
        result['charbias_std'] = statistics.stdev(cb_list) if len(cb_list) > 1 else 0
        result['charbias_skew'] = float(sp_stats.skew(cb_list)) if len(cb_list) > 2 else 0

    # Character positional bias — WORD position (1-char ngrams only)
    char_variation_words = {k: v for k, v in ngram_variation_words.items() if len(k) == 1}
    cbw_list = []
    for k, cv in char_variation_words.items():
        count = ngram_bank.get(k, 1)
        cbw_list.extend([cv] * count)
    if cbw_list:
        result['charbias_words_mean'] = statistics.mean(cbw_list)
        result['charbias_words_std'] = statistics.stdev(cbw_list) if len(cbw_list) > 1 else 0
        result['charbias_words_skew'] = (
            float(sp_stats.skew(cbw_list)) if len(cbw_list) > 2 else 0
        )

    # Counts
    result['unique_words'] = len(wordbank)
    result['repeated_words'] = word_repeats / num_words if num_words > 0 else 0
    result['tripled_words'] = word_triples / num_words if num_words > 0 else 0
    result['unique_chars'] = len(charbank)
    result['repeated_chars'] = char_repeats / num_chars if num_chars > 0 else 0
    result['tripled_chars'] = char_triples / num_chars if num_chars > 0 else 0
    result['unique_ngrams'] = len(ngram_bank_unique)

    # Entropy: 2nd-order Markov character entropy rate
    result['entropy'] = _char_entropy_markov(lines_sub)

    # Compression
    docwords_string = " ".join(docwords)
    compressed = zlib.compress(docwords_string.encode(), 9)
    result['compression'] = len(compressed) / len(docwords_string) if docwords_string else 0

    # Zipf: Urzúa (2000) LMZ test statistic
    # BG uses wordbank_sorted (already top-25 truncated)
    wbs_cut = wordbank_sorted[:max(1, len(wordbank_sorted) // 2)]
    n_cut = len(wbs_cut)
    if n_cut >= 2 and wbs_cut[-1] > 0:
        z1 = 1 - (1 / n_cut) * sum(math.log(xi / wbs_cut[-1]) for xi in wbs_cut)
        z2 = 0.5 - (1 / n_cut) * sum(wbs_cut[-1] / xi for xi in wbs_cut if xi > 0)
        result['zipf_lmz'] = 4 * n_cut * (z1 ** 2 + 6 * z1 * z2 + 12 * z2 ** 2)
    else:
        result['zipf_lmz'] = 0

    # Flipped pairs
    result['flipped_pairs'] = word_flips / num_words if num_words > 0 else 0

    return result


# ======================================================================
# MAIN METRIC COMPUTATION
# ======================================================================

def compute_metrics(tokens, lines=None, subset_iterations=100, subset_words=200,
                    seed=42, do_levenshtein=True, ngram_max_len=3,
                    verbose=True):
    """
    Compute all 85 metrics from a token list.

    BG-methodology metrics are computed per subsample and averaged.
    Original metrics are computed on the full corpus.

    Args:
        tokens: list of str — the full corpus as a flat token list
        lines: list of list of str — tokens grouped by line.
               If None, tokens are split into pseudo-lines of 10.
        subset_iterations: BG subsampling iterations (default: 100)
        subset_words: tokens per BG subsample (default: 200)
        seed: random seed for reproducibility
        do_levenshtein: compute Levenshtein metrics (slow, O(V^2))
        ngram_max_len: max ngram length for BG metrics (default: 3)
        verbose: print progress

    Returns: dict of metric_name -> float
    """
    rng = random.Random(seed)
    N = len(tokens)
    freq = Counter(tokens)
    V = len(freq)
    chars = list(''.join(tokens))
    C = len(chars)
    char_freq = Counter(chars)
    n_chars = len(char_freq)
    m = {}

    # Build lines if not provided
    if lines is None:
        lines = [tokens[i:i + 10] for i in range(0, N, 10)]

    # Convert lines to text strings for BG subsampling
    lines_text = [' '.join(line) for line in lines]

    if verbose:
        print(f"Computing 85 metrics: {N} tokens, {V} types, {n_chars} char types")
        print(f"  BG subsampling: {subset_iterations} iterations × {subset_words} words")
        print(f"  Levenshtein: {'ON' if do_levenshtein else 'OFF'} (source: {_LEV_SOURCE})")

    # ==================================================================
    # PART 1: BG SUBSAMPLED METRICS
    # ==================================================================
    bg_accumulators = defaultdict(list)

    for i in range(subset_iterations):
        if verbose and (i + 1) % 10 == 0:
            print(f"  BG iteration {i + 1}/{subset_iterations}")

        result = _bg_subsample_iteration(
            lines_text, rng, subset_words, ngram_max_len, do_levenshtein
        )
        if result is None:
            continue

        for key, val in result.items():
            bg_accumulators[key].append(val)

    # Average across iterations
    for key, vals in bg_accumulators.items():
        if vals:
            m[key] = statistics.mean(vals)
        else:
            m[key] = 0

    # ==================================================================
    # PART 2: ORIGINAL METRICS (full corpus)
    # ==================================================================

    # ── Entropy hierarchy ─────────────────────────────────────────
    char_N = sum(char_freq.values())
    m['H0_max_entropy'] = math.log2(n_chars) if n_chars > 0 else 0
    m['H1_unigram'] = -sum(
        (c / char_N) * math.log2(c / char_N) for c in char_freq.values()
    )

    # H2: character bigram
    bg_freq = Counter()
    for i in range(C - 1):
        bg_freq[(chars[i], chars[i + 1])] += 1
    bg_N = sum(bg_freq.values())
    h2_joint = -sum((c / bg_N) * math.log2(c / bg_N) for c in bg_freq.values()) if bg_N > 0 else 0
    m['h2_joint_digraph'] = h2_joint
    m['H2_markov_cond'] = h2_joint - m['H1_unigram']
    m['h2_conditional'] = m['H2_markov_cond']

    # H3: character trigram
    tg_freq = Counter()
    for i in range(C - 2):
        tg_freq[(chars[i], chars[i + 1], chars[i + 2])] += 1
    tg_N = sum(tg_freq.values())
    if tg_N > 0:
        h3_joint = -sum((c / tg_N) * math.log2(c / tg_N) for c in tg_freq.values())
        m['h3_joint_trigraph'] = h3_joint
        m['h3_conditional'] = h3_joint - h2_joint
    else:
        m['h3_joint_trigraph'] = 0
        m['h3_conditional'] = 0

    # ── Character distribution ────────────────────────────────────
    char_probs = np.array([c / char_N for c in char_freq.values()])
    m['char_evenness'] = (
        (-sum(p * math.log2(p) for p in char_probs)) / math.log2(n_chars)
        if n_chars > 1 else 0
    )
    m['char_redundancy'] = 1.0 - m['char_evenness']
    m['char_simpson_D'] = float(np.sum(char_probs ** 2))

    char_fs = Counter(char_freq.values())
    char_S2 = sum(i * i * fi for i, fi in char_fs.items())
    m['char_yule_K'] = (
        10000 * (char_S2 - char_N) / (char_N * (char_N - 1)) if char_N > 1 else 0
    )

    # ── Digraph / trigraph counts ─────────────────────────────────
    digraphs = Counter()
    for i in range(C - 1):
        digraphs[(chars[i], chars[i + 1])] += 1
    m['digraph_unique'] = len(digraphs)
    m['digraph_coverage'] = len(digraphs) / (n_chars ** 2) if n_chars > 0 else 0

    trigraphs = Counter()
    for i in range(C - 2):
        trigraphs[(chars[i], chars[i + 1], chars[i + 2])] += 1
    m['trigraph_unique'] = len(trigraphs)

    # ── TTR variants ──────────────────────────────────────────────
    m['ttr'] = V / N if N > 0 else 0
    m['rttr'] = V / math.sqrt(N) if N > 0 else 0
    m['cttr'] = V / (2 * math.sqrt(N)) if N > 0 else 0
    m['log_ttr'] = math.log(V) / math.log(N) if N > 1 and V > 0 else 0
    m['maas_a2'] = (
        (math.log(N) - math.log(V)) / (math.log(N) ** 2)
        if N > 1 and V > 1 else 0
    )
    m['uber_index'] = (
        (math.log(N) ** 2) / (math.log(N) - math.log(V))
        if V > 1 and N > 1 and math.log(N) != math.log(V) else 0
    )
    m['brunet_W'] = N ** (V ** -0.172) if N > 0 and V > 0 else 0

    for w in [25, 50, 100]:
        if N >= w:
            m[f'mattr_{w}'] = float(np.mean(
                [len(set(tokens[i:i + w])) / w for i in range(N - w + 1)]
            ))
        else:
            m[f'mattr_{w}'] = V / N if N > 0 else 0
        segs = [tokens[i:i + w] for i in range(0, N - w + 1, w)]
        full = [s for s in segs if len(s) == w]
        m[f'msttr_{w}'] = (
            float(np.mean([len(set(s)) / len(s) for s in full])) if full else 0
        )

    # ── Hapax & frequency spectrum ────────────────────────────────
    hapax = sum(1 for c in freq.values() if c == 1)
    dis = sum(1 for c in freq.values() if c == 2)
    m['hapax_ratio_tokens'] = hapax / N if N > 0 else 0
    m['hapax_ratio_types'] = hapax / V if V > 0 else 0
    m['hapax_type_proportion'] = m['hapax_ratio_types']
    m['dis_ratio_tokens'] = dis / N if N > 0 else 0
    m['dis_ratio_types'] = dis / V if V > 0 else 0
    m['sichel_S'] = dis / V if V > 0 else 0

    fs = Counter(freq.values())
    m['freq_spectrum_1'] = fs.get(1, 0) / V if V > 0 else 0
    m['freq_spectrum_2'] = fs.get(2, 0) / V if V > 0 else 0
    m['freq_spectrum_3'] = fs.get(3, 0) / V if V > 0 else 0
    m['freq_spectrum_gt10'] = sum(1 for c in freq.values() if c > 10) / V if V > 0 else 0

    # ── Lexical richness: Yule K, Honore R ────────────────────────
    S2 = sum(i * i * fi for i, fi in fs.items())
    m['word_yule_K'] = 10000 * (S2 - N) / (N * (N - 1)) if N > 1 else 0
    m['honore_R'] = (
        100 * math.log(N) / (1 - hapax / V)
        if hapax > 0 and V > 0 and (1 - hapax / V) > 0 else 0
    )

    # ── Heaps' law ────────────────────────────────────────────────
    checkpoints = [c for c in [100, 500, 1000, 2000, 5000, 10000, N] if c <= N]
    if len(checkpoints) >= 3:
        vocab_at = [len(set(tokens[:c])) for c in checkpoints]
        h_slope, _, _, _, _ = sp_stats.linregress(np.log(checkpoints), np.log(vocab_at))
        m['heaps_beta'] = h_slope
    else:
        m['heaps_beta'] = 0

    # ── Autocorrelation (full corpus) ─────────────────────────────
    wlens = np.array([len(t) for t in tokens], dtype=float)
    wl_c = wlens - wlens.mean()
    wl_denom = np.sum(wl_c ** 2)
    m['autocorr_wordlen'] = (
        float(np.sum(wl_c[:-1] * wl_c[1:]) / wl_denom) if wl_denom > 0 else 0
    )

    wfreqs = np.array([freq[t] for t in tokens], dtype=float)
    wf_c = wfreqs - wfreqs.mean()
    wf_denom = np.sum(wf_c ** 2)
    m['autocorr_wordfreq'] = (
        float(np.sum(wf_c[:-1] * wf_c[1:]) / wf_denom) if wf_denom > 0 else 0
    )

    if N >= 50:
        ttr_series = [len(set(tokens[i:i + 25])) / 25 for i in range(0, N - 25, 5)]
        if len(ttr_series) > 2:
            ts = np.array(ttr_series)
            ts_c = ts - ts.mean()
            ts_d = np.sum(ts_c ** 2)
            m['autocorr_ttr_25'] = (
                float(np.sum(ts_c[:-1] * ts_c[1:]) / ts_d) if ts_d > 0 else 0
            )
        else:
            m['autocorr_ttr_25'] = 0

        hapax_series = []
        for i in range(0, N - 25, 5):
            window = tokens[i:i + 25]
            wf = Counter(window)
            hapax_series.append(sum(1 for c in wf.values() if c == 1) / 25)
        if len(hapax_series) > 2:
            hs = np.array(hapax_series)
            hs_c = hs - hs.mean()
            hs_d = np.sum(hs_c ** 2)
            m['autocorr_hapax_25'] = (
                float(np.sum(hs_c[:-1] * hs_c[1:]) / hs_d) if hs_d > 0 else 0
            )
        else:
            m['autocorr_hapax_25'] = 0
    else:
        m['autocorr_ttr_25'] = 0
        m['autocorr_hapax_25'] = 0

    # ── Zipf (log-log regression — our original metric) ───────────
    sf = sorted(freq.values(), reverse=True)
    ranks = np.arange(1, len(sf) + 1)
    slope, _intercept, r_val, _, _ = sp_stats.linregress(np.log(ranks), np.log(sf))
    m['zipf_alpha'] = abs(slope)
    m['zipf_r2'] = r_val ** 2

    # ── Frequency concentration ───────────────────────────────────
    m['top10_share'] = sum(sf[:10]) / N if len(sf) >= 10 else 0
    m['top50_share'] = sum(sf[:50]) / N if len(sf) >= 50 else 0

    # ── Metadata ──────────────────────────────────────────────────
    m['n_tokens'] = N
    m['n_types'] = V

    if verbose:
        print(f"  Done. {len(m)} metrics computed.")

    return m


# ======================================================================
# SCORING
# ======================================================================

def score_against_vms(gen_metrics, vms_metrics, metric_list=None, tolerances=None):
    """
    Score generator metrics against VMS baseline.
    Returns dict: passes, fails, n_pass, n_total, details.
    """
    if metric_list is None:
        metric_list = list(set(ALL_85))
    if tolerances is None:
        tolerances = TOLERANCES

    passes, fails, details = [], [], {}
    for metric in metric_list:
        vms_val = vms_metrics.get(metric)
        gen_val = gen_metrics.get(metric)
        if vms_val is None or gen_val is None:
            continue
        tol = tolerances.get(metric, abs(vms_val) * 0.1 if vms_val != 0 else 0.1)
        delta = abs(gen_val - vms_val)
        passed = delta <= tol
        details[metric] = {
            'vms': vms_val, 'gen': gen_val,
            'delta': delta, 'tol': tol, 'pass': passed
        }
        (passes if passed else fails).append(metric)

    return {
        'passes': passes, 'fails': fails,
        'n_pass': len(passes), 'n_total': len(details),
        'details': details
    }


def print_comparison_table(vms_metrics, generator_dict, metric_list=None):
    """
    Print formatted comparison table.
    generator_dict: {label: metrics_dict}
    """
    if metric_list is None:
        metric_list = sorted(vms_metrics.keys())

    labels = list(generator_dict.keys())
    header = f"{'Metric':<28} {'VMS':>12}"
    for lab in labels:
        header += f" {lab:>14}"
    print(header)
    print("-" * (30 + 12 + 16 * len(labels)))

    for metric in metric_list:
        vms_val = vms_metrics.get(metric)
        if vms_val is None:
            continue
        row = f"{metric:<28} {vms_val:>12.4f}"
        for lab in labels:
            gen_val = generator_dict[lab].get(metric)
            if gen_val is not None:
                tol = TOLERANCES.get(metric, abs(vms_val) * 0.1 if vms_val != 0 else 0.1)
                mark = "✓" if abs(gen_val - vms_val) <= tol else "✗"
                row += f" {gen_val:>10.4f} {mark:>2}"
            else:
                row += f" {'N/A':>12}"
        print(row)


# ======================================================================
# SELF-TEST
# ======================================================================

if __name__ == '__main__':
    import pickle
    import sys
    import time

    # Try to load enriched_records.pkl
    pkl_paths = [
        'enriched_records.pkl',
        '../enriched_records.pkl',
        'Paper/enriched_records.pkl',
    ]
    records = None
    for p in pkl_paths:
        try:
            with open(p, 'rb') as f:
                records = pickle.load(f)
            print(f"Loaded {len(records)} records from {p}")
            break
        except FileNotFoundError:
            continue

    if records is None:
        print("ERROR: enriched_records.pkl not found. Provide path as argument.")
        sys.exit(1)

    tokens = [r['token'] for r in records]

    # Build lines from folio/line_no
    lines = []
    cur_line = []
    cur_key = (records[0]['folio'], records[0]['line_no'])
    for r in records:
        key = (r['folio'], r['line_no'])
        if key != cur_key:
            if cur_line:
                lines.append(cur_line)
            cur_line = []
            cur_key = key
        cur_line.append(r['token'])
    if cur_line:
        lines.append(cur_line)

    print(f"Corpus: {len(tokens)} tokens, {len(lines)} lines")
    print()

    t0 = time.time()
    m = compute_metrics(tokens, lines=lines, do_levenshtein=True)
    elapsed = time.time() - t0
    print(f"\nComputed in {elapsed:.1f}s")
    print(f"Total metrics: {len(m)} (target: 85+)")

    # Save baseline
    with open('vms_baseline_85metrics.pkl', 'wb') as f:
        pickle.dump(m, f)
    print("Saved: vms_baseline_85metrics.pkl")

    # Print all
    print("\n" + "=" * 60)
    for k in sorted(m.keys()):
        if k in ('n_tokens', 'n_types'):
            continue
        print(f"  {k:<30} = {m[k]:.6f}")

    # Self-score
    result = score_against_vms(m, m)
    print(f"\nSelf-score: {result['n_pass']}/{result['n_total']} (should be 100%)")

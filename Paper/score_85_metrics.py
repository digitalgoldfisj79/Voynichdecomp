#!/usr/bin/env python3
"""
85-Metric Scoring Suite for VMS Generator Comparison
=====================================================
Full implementation matching bartholomaeus_vs_vms_FINAL.xlsx metric suite.
Includes all Bowern-Gaskell subsampling metrics, entropy hierarchy,
character distribution, lexical richness, and autocorrelation metrics.

Usage:
    from score_85_metrics import compute_metrics, score_against_vms, CORE_15, TOLERANCES

    metrics = compute_metrics(token_list)
    results = score_against_vms(metrics, vms_baseline)

The subsampling metrics (wordunique, wordchange, worddist, wordbias, etc.)
follow BG methodology: draw random subsamples of `subset_words` tokens from
random lines, compute metric per subsample, report mean across iterations.
"""

import math, zlib, random, statistics
import numpy as np
from collections import Counter, defaultdict
from scipy import stats

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

ALL_85 = [
    # --- BG42 core: word length ---
    'wordlen_mean', 'wordlen_std', 'wordlen_skew',
    'wordlen_unique_mean', 'wordlen_unique_std', 'wordlen_unique_skew',
    'wordlen_autocorr',
    # --- BG42 subsampling: word-level ---
    'wordunique_mean', 'wordunique_std', 'wordunique_skew',
    'wordchange_mean', 'wordchange_std', 'wordchange_skew',
    'worddist_max', 'worddist_shape',
    'wordbias_mean', 'wordbias_std', 'wordbias_skew',
    'wordbias_lines_mean', 'wordbias_lines_std', 'wordbias_lines_skew',
    # --- BG42 subsampling: char-level ---
    'chardist_max', 'chardist_shape',
    'ngramdist_max', 'ngramdist_shape',
    'charbias_mean', 'charbias_std', 'charbias_skew',
    'charbias_words_mean', 'charbias_words_std', 'charbias_words_skew',
    # --- BG42 counts ---
    'unique_words', 'repeated_words', 'tripled_words',
    'unique_chars', 'repeated_chars', 'tripled_chars',
    'unique_ngrams',
    # --- BG42 global ---
    'entropy', 'compression', 'zipf', 'flipped_pairs',
    # --- Entropy hierarchy ---
    'H0_max_entropy', 'H1_unigram', 'H2_markov_cond',
    'h2_joint_digraph', 'h2_conditional',
    'h3_joint_trigraph', 'h3_conditional',
    # --- Character distribution ---
    'char_evenness', 'char_redundancy',
    'char_simpson_D', 'char_yule_K',
    # --- Digraph / trigraph ---
    'digraph_unique', 'digraph_coverage', 'trigraph_unique',
    # --- TTR variants ---
    'ttr', 'rttr', 'cttr',
    'log_ttr', 'maas_a2', 'uber_index', 'brunet_W',
    'msttr_25', 'msttr_50', 'msttr_100',
    'mattr_25', 'mattr_50', 'mattr_100',
    # --- Hapax & frequency spectrum ---
    'hapax_ratio_tokens', 'hapax_ratio_types',
    'dis_ratio_tokens', 'dis_ratio_types',
    'sichel_S', 'hapax_type_proportion',
    'freq_spectrum_1', 'freq_spectrum_2', 'freq_spectrum_3', 'freq_spectrum_gt10',
    # --- Lexical richness ---
    'word_yule_K', 'honore_R',
    # --- Autocorrelation ---
    'autocorr_wordlen', 'autocorr_wordfreq',
    'autocorr_ttr_25', 'autocorr_hapax_25',
    # --- Zipf / Heaps ---
    'zipf_alpha', 'zipf_r2', 'heaps_beta',
    # --- Frequency concentration ---
    'top10_share', 'top50_share',
]

# ======================================================================
# TOLERANCES
# ======================================================================

TOLERANCES = {
    # Word length
    'wordlen_mean': 0.50, 'wordlen_std': 0.20, 'wordlen_skew': 0.20,
    'wordlen_unique_mean': 0.50, 'wordlen_unique_std': 0.30, 'wordlen_unique_skew': 0.30,
    'wordlen_autocorr': 0.05,
    # BG subsampling word-level
    'wordunique_mean': 5.0, 'wordunique_std': 3.0, 'wordunique_skew': 0.30,
    'wordchange_mean': 0.10, 'wordchange_std': 0.05, 'wordchange_skew': 0.30,
    'worddist_max': 0.05, 'worddist_shape': 1.0,
    'wordbias_mean': 0.05, 'wordbias_std': 0.03, 'wordbias_skew': 0.50,
    'wordbias_lines_mean': 0.05, 'wordbias_lines_std': 0.03, 'wordbias_lines_skew': 0.50,
    # BG subsampling char-level
    'chardist_max': 0.05, 'chardist_shape': 1.0,
    'ngramdist_max': 0.02, 'ngramdist_shape': 1.0,
    'charbias_mean': 0.03, 'charbias_std': 0.02, 'charbias_skew': 0.50,
    'charbias_words_mean': 0.03, 'charbias_words_std': 0.02, 'charbias_words_skew': 0.50,
    # Counts (normalized)
    'unique_words': 0.03, 'repeated_words': 0.005, 'tripled_words': 0.001,
    'unique_chars': 3, 'repeated_chars': 0.02, 'tripled_chars': 0.005,
    'unique_ngrams': 0.05,
    # Global
    'entropy': 1.5, 'compression': 0.05, 'zipf': 0.15, 'flipped_pairs': 0.05,
    # Entropy hierarchy
    'H0_max_entropy': 0.30, 'H1_unigram': 0.15,
    'H2_markov_cond': 0.35, 'h2_conditional': 0.35,
    'h2_joint_digraph': 0.35,
    'h3_joint_trigraph': 0.50, 'h3_conditional': 0.35,
    # Character distribution
    'char_evenness': 0.05, 'char_redundancy': 0.05,
    'char_simpson_D': 0.02, 'char_yule_K': 50,
    # Digraph/trigraph
    'digraph_unique': 50, 'digraph_coverage': 0.10, 'trigraph_unique': 500,
    # TTR variants
    'ttr': 0.03, 'rttr': 8.0, 'cttr': 4.0,
    'log_ttr': 0.02, 'maas_a2': 0.004, 'uber_index': 15, 'brunet_W': 2.0,
    'msttr_25': 0.05, 'msttr_50': 0.05, 'msttr_100': 0.05,
    'mattr_25': 0.05, 'mattr_50': 0.05, 'mattr_100': 0.05,
    # Hapax & frequency spectrum
    'hapax_ratio_tokens': 0.03, 'hapax_ratio_types': 0.10,
    'dis_ratio_tokens': 0.01, 'dis_ratio_types': 0.03,
    'sichel_S': 0.03, 'hapax_type_proportion': 0.10,
    'freq_spectrum_1': 0.10, 'freq_spectrum_2': 0.03,
    'freq_spectrum_3': 0.02, 'freq_spectrum_gt10': 0.02,
    # Lexical richness
    'word_yule_K': 15, 'honore_R': 500,
    # Autocorrelation
    'autocorr_wordlen': 0.05, 'autocorr_wordfreq': 0.03,
    'autocorr_ttr_25': 0.10, 'autocorr_hapax_25': 0.10,
    # Zipf / Heaps
    'zipf_alpha': 0.10, 'zipf_r2': 0.05, 'heaps_beta': 0.05,
    # Frequency concentration
    'top10_share': 0.03, 'top50_share': 0.08,
}


# ======================================================================
# SUBSAMPLING HELPERS (BG methodology)
# ======================================================================

def _subsample_lines(tokens, lines, subset_words, rng):
    """Draw a contiguous block of lines totalling ~subset_words tokens."""
    if not lines or sum(len(l) for l in lines) < subset_words:
        return tokens[:subset_words]
    start = rng.randint(0, len(lines) - 1)
    collected = []
    idx = start
    while len(collected) < subset_words and idx < len(lines):
        collected.extend(lines[idx])
        idx += 1
    # Wrap if needed
    if len(collected) < subset_words:
        idx = 0
        while len(collected) < subset_words and idx < start:
            collected.extend(lines[idx])
            idx += 1
    return collected[:subset_words]


def _word_positional_bias(words, lines):
    """Compute how biased each word is toward specific positions within lines."""
    # For each word type, record normalized positions (0-1) within lines
    type_positions = defaultdict(list)
    for line in lines:
        n = len(line)
        if n < 2:
            continue
        for i, w in enumerate(line):
            type_positions[w].append(i / (n - 1))
    # Bias = std of positions (low = concentrated, high = spread)
    biases = []
    for w, positions in type_positions.items():
        if len(positions) >= 2:
            biases.append(np.std(positions))
    return biases


def _char_positional_bias_within_words(tokens):
    """Compute how biased each character is toward specific positions within words."""
    char_positions = defaultdict(list)
    for tok in tokens:
        n = len(tok)
        if n < 2:
            continue
        for i, ch in enumerate(tok):
            char_positions[ch].append(i / (n - 1))
    biases = []
    for ch, positions in char_positions.items():
        if len(positions) >= 2:
            biases.append(np.std(positions))
    return biases


def _word_change_rate(words, window=50):
    """Fraction of vocabulary that changes between consecutive windows."""
    changes = []
    for i in range(0, len(words) - 2 * window, window):
        w1 = set(words[i:i + window])
        w2 = set(words[i + window:i + 2 * window])
        if len(w1 | w2) > 0:
            changes.append(1.0 - len(w1 & w2) / len(w1 | w2))
    return changes


def _flipped_pairs(tokens):
    """Fraction of adjacent bigrams that appear in reverse elsewhere."""
    bigrams = set()
    for i in range(len(tokens) - 1):
        bigrams.add((tokens[i], tokens[i + 1]))
    flipped = sum(1 for a, b in bigrams if (b, a) in bigrams)
    return flipped / len(bigrams) if bigrams else 0


# ======================================================================
# MAIN METRIC COMPUTATION
# ======================================================================

def compute_metrics(tokens, lines=None, subset_iterations=50, subset_words=200, seed=42):
    """
    Compute all 85 metrics from a token list.
    
    Args:
        tokens: list of str — the corpus
        lines: list of list of str — tokens grouped by line (optional; 
               if None, splits into pseudo-lines of 10 tokens)
        subset_iterations: number of subsampling rounds for BG metrics
        subset_words: tokens per subsample
        seed: random seed for subsampling reproducibility
    
    Returns: dict of metric_name -> float
    """
    rng = random.Random(seed)
    N = len(tokens)
    V = len(set(tokens))
    freq = Counter(tokens)
    chars = list(''.join(tokens))
    C = len(chars)
    char_freq = Counter(chars)
    n_chars = len(char_freq)
    m = {}

    # Build lines if not provided
    if lines is None:
        lines = [tokens[i:i + 10] for i in range(0, N, 10)]

    # ============================================================
    # WORD LENGTH METRICS
    # ============================================================
    wlens = [len(t) for t in tokens]
    m['wordlen_mean'] = float(np.mean(wlens))
    m['wordlen_std'] = float(np.std(wlens))
    m['wordlen_skew'] = float(stats.skew(wlens))

    wlens_u = [len(t) for t in set(tokens)]
    m['wordlen_unique_mean'] = float(np.mean(wlens_u))
    m['wordlen_unique_std'] = float(np.std(wlens_u))
    m['wordlen_unique_skew'] = float(stats.skew(wlens_u))

    wl = np.array(wlens, dtype=float)
    wl_c = wl - wl.mean()
    denom = np.sum(wl_c ** 2)
    m['wordlen_autocorr'] = float(np.sum(wl_c[:-1] * wl_c[1:]) / denom) if denom > 0 else 0.0
    m['autocorr_wordlen'] = m['wordlen_autocorr']  # alias

    # ============================================================
    # BG SUBSAMPLING METRICS
    # ============================================================
    wordunique_vals = []
    wordchange_vals = []
    worddist_max_vals = []
    worddist_shape_vals = []
    wordbias_vals = []
    wordbias_lines_vals = []
    chardist_max_vals = []
    chardist_shape_vals = []
    ngramdist_max_vals = []
    ngramdist_shape_vals = []
    charbias_vals = []
    charbias_words_vals = []

    for _ in range(subset_iterations):
        sub = _subsample_lines(tokens, lines, subset_words, rng)
        if len(sub) < 20:
            continue
        sub_n = len(sub)
        sub_v = len(set(sub))
        sub_freq = Counter(sub)

        # Word unique count
        wordunique_vals.append(sub_v)

        # Word change rate
        wc = _word_change_rate(sub, window=max(10, sub_n // 5))
        if wc:
            wordchange_vals.extend(wc)

        # Word frequency distribution shape
        sf = sorted(sub_freq.values(), reverse=True)
        if sf:
            worddist_max_vals.append(sf[0] / sub_n)
            if len(sf) >= 3:
                log_sf = np.log(np.array(sf[:min(50, len(sf))]) + 1)
                worddist_shape_vals.append(float(stats.skew(log_sf)))

        # Word positional bias within subsample lines
        sub_lines = [sub[i:i + 10] for i in range(0, len(sub), 10)]
        wb = _word_positional_bias(sub, sub_lines)
        if wb:
            wordbias_vals.extend(wb)
        wbl = _word_positional_bias(sub, sub_lines)
        if wbl:
            wordbias_lines_vals.extend(wbl)

        # Character distribution in subsample
        sub_chars = list(''.join(sub))
        sub_char_freq = Counter(sub_chars)
        sub_char_n = sum(sub_char_freq.values())
        if sub_char_freq:
            scf_vals = sorted(sub_char_freq.values(), reverse=True)
            chardist_max_vals.append(scf_vals[0] / sub_char_n)
            if len(scf_vals) >= 3:
                log_scf = np.log(np.array(scf_vals) + 1)
                chardist_shape_vals.append(float(stats.skew(log_scf)))

        # Ngram distribution
        ngrams = Counter()
        for i in range(len(sub_chars) - 1):
            ngrams[(sub_chars[i], sub_chars[i + 1])] += 1
        if ngrams:
            ng_vals = sorted(ngrams.values(), reverse=True)
            ngramdist_max_vals.append(ng_vals[0] / sum(ng_vals))
            if len(ng_vals) >= 3:
                log_ng = np.log(np.array(ng_vals[:min(50, len(ng_vals))]) + 1)
                ngramdist_shape_vals.append(float(stats.skew(log_ng)))

        # Character positional bias within words
        cb = _char_positional_bias_within_words(sub)
        if cb:
            charbias_words_vals.extend(cb)
            charbias_vals.extend(cb)  # same computation, different context in BG

    # Aggregate subsampling results
    def _safe_stats(vals):
        if not vals:
            return 0, 0, 0
        return float(np.mean(vals)), float(np.std(vals)), float(stats.skew(vals)) if len(vals) > 2 else 0

    wu_m, wu_s, wu_sk = _safe_stats(wordunique_vals)
    m['wordunique_mean'] = wu_m
    m['wordunique_std'] = wu_s
    m['wordunique_skew'] = wu_sk

    wc_m, wc_s, wc_sk = _safe_stats(wordchange_vals)
    m['wordchange_mean'] = wc_m
    m['wordchange_std'] = wc_s
    m['wordchange_skew'] = wc_sk

    m['worddist_max'] = float(np.mean(worddist_max_vals)) if worddist_max_vals else 0
    m['worddist_shape'] = float(np.mean(worddist_shape_vals)) if worddist_shape_vals else 0

    wb_m, wb_s, wb_sk = _safe_stats(wordbias_vals)
    m['wordbias_mean'] = wb_m
    m['wordbias_std'] = wb_s
    m['wordbias_skew'] = wb_sk

    wbl_m, wbl_s, wbl_sk = _safe_stats(wordbias_lines_vals)
    m['wordbias_lines_mean'] = wbl_m
    m['wordbias_lines_std'] = wbl_s
    m['wordbias_lines_skew'] = wbl_sk

    m['chardist_max'] = float(np.mean(chardist_max_vals)) if chardist_max_vals else 0
    m['chardist_shape'] = float(np.mean(chardist_shape_vals)) if chardist_shape_vals else 0

    m['ngramdist_max'] = float(np.mean(ngramdist_max_vals)) if ngramdist_max_vals else 0
    m['ngramdist_shape'] = float(np.mean(ngramdist_shape_vals)) if ngramdist_shape_vals else 0

    cb_m, cb_s, cb_sk = _safe_stats(charbias_vals)
    m['charbias_mean'] = cb_m
    m['charbias_std'] = cb_s
    m['charbias_skew'] = cb_sk

    cbw_m, cbw_s, cbw_sk = _safe_stats(charbias_words_vals)
    m['charbias_words_mean'] = cbw_m
    m['charbias_words_std'] = cbw_s
    m['charbias_words_skew'] = cbw_sk

    # ============================================================
    # COUNTS (normalized per BG convention)
    # ============================================================
    m['unique_words'] = V / N
    m['repeated_words'] = sum(1 for i in range(N - 1) if tokens[i] == tokens[i + 1]) / N
    m['tripled_words'] = sum(1 for i in range(N - 2)
                             if tokens[i] == tokens[i + 1] == tokens[i + 2]) / N
    m['unique_chars'] = n_chars
    m['repeated_chars'] = sum(1 for i in range(C - 1) if chars[i] == chars[i + 1]) / C if C > 1 else 0
    m['tripled_chars'] = sum(1 for i in range(C - 2)
                             if chars[i] == chars[i + 1] == chars[i + 2]) / C if C > 2 else 0

    # Unique bigrams (ngrams)
    digraphs = Counter()
    for i in range(C - 1):
        digraphs[(chars[i], chars[i + 1])] += 1
    m['unique_ngrams'] = len(digraphs) / (n_chars ** 2) if n_chars > 0 else 0

    # ============================================================
    # GLOBAL: entropy, compression, zipf, flipped_pairs
    # ============================================================
    m['entropy'] = -sum((c / N) * math.log2(c / N) for c in freq.values())
    text_bytes = ' '.join(tokens).encode('utf-8')
    m['compression'] = len(zlib.compress(text_bytes)) / len(text_bytes)

    sf = sorted(freq.values(), reverse=True)
    ranks = np.arange(1, len(sf) + 1)
    slope, intercept, r_val, _, _ = stats.linregress(np.log(ranks), np.log(sf))
    m['zipf'] = abs(slope)  # BG convention name
    m['zipf_alpha'] = abs(slope)  # our convention
    m['zipf_r2'] = r_val ** 2

    m['flipped_pairs'] = _flipped_pairs(tokens)

    # ============================================================
    # ENTROPY HIERARCHY
    # ============================================================
    char_N = sum(char_freq.values())
    m['H0_max_entropy'] = math.log2(n_chars) if n_chars > 0 else 0
    m['H1_unigram'] = -sum((c / char_N) * math.log2(c / char_N) for c in char_freq.values())

    # H2: character bigram
    bg_freq = Counter()
    for i in range(C - 1):
        bg_freq[(chars[i], chars[i + 1])] += 1
    bg_N = sum(bg_freq.values())
    h2_joint = -sum((c / bg_N) * math.log2(c / bg_N) for c in bg_freq.values())
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

    # ============================================================
    # CHARACTER DISTRIBUTION METRICS
    # ============================================================
    char_probs = np.array([c / char_N for c in char_freq.values()])
    m['char_evenness'] = (-sum(p * math.log2(p) for p in char_probs)) / math.log2(n_chars) \
        if n_chars > 1 else 0  # Pielou's J
    m['char_redundancy'] = 1.0 - m['char_evenness']
    m['char_simpson_D'] = float(np.sum(char_probs ** 2))

    # Character Yule's K
    char_fs = Counter(char_freq.values())
    char_S2 = sum(i * i * fi for i, fi in char_fs.items())
    m['char_yule_K'] = 10000 * (char_S2 - char_N) / (char_N * (char_N - 1)) if char_N > 1 else 0

    # ============================================================
    # DIGRAPH / TRIGRAPH COUNTS
    # ============================================================
    m['digraph_unique'] = len(digraphs)
    m['digraph_coverage'] = len(digraphs) / (n_chars ** 2) if n_chars > 0 else 0

    trigraphs = Counter()
    for i in range(C - 2):
        trigraphs[(chars[i], chars[i + 1], chars[i + 2])] += 1
    m['trigraph_unique'] = len(trigraphs)

    # ============================================================
    # TTR VARIANTS
    # ============================================================
    m['ttr'] = V / N
    m['rttr'] = V / math.sqrt(N)
    m['cttr'] = V / (2 * math.sqrt(N))
    m['log_ttr'] = math.log(V) / math.log(N) if N > 1 else 0
    m['maas_a2'] = (math.log(N) - math.log(V)) / (math.log(N) ** 2) if N > 1 and V > 1 else 0
    m['uber_index'] = (math.log(N) ** 2) / (math.log(N) - math.log(V)) \
        if V > 1 and N > 1 and math.log(N) != math.log(V) else 0
    m['brunet_W'] = N ** (V ** -0.172) if N > 0 else 0

    for w in [25, 50, 100]:
        if N >= w:
            m[f'mattr_{w}'] = float(np.mean(
                [len(set(tokens[i:i + w])) / w for i in range(N - w + 1)]))
        else:
            m[f'mattr_{w}'] = V / N
        segs = [tokens[i:i + w] for i in range(0, N - w + 1, w)]
        full = [s for s in segs if len(s) == w]
        m[f'msttr_{w}'] = float(np.mean([len(set(s)) / len(s) for s in full])) if full else 0

    # ============================================================
    # HAPAX & FREQUENCY SPECTRUM
    # ============================================================
    hapax = sum(1 for c in freq.values() if c == 1)
    dis = sum(1 for c in freq.values() if c == 2)
    m['hapax_ratio_tokens'] = hapax / N
    m['hapax_ratio_types'] = hapax / V if V > 0 else 0
    m['hapax_type_proportion'] = m['hapax_ratio_types']  # alias
    m['dis_ratio_tokens'] = dis / N
    m['dis_ratio_types'] = dis / V if V > 0 else 0
    m['sichel_S'] = dis / V if V > 0 else 0

    fs = Counter(freq.values())
    m['freq_spectrum_1'] = fs.get(1, 0) / V if V > 0 else 0
    m['freq_spectrum_2'] = fs.get(2, 0) / V if V > 0 else 0
    m['freq_spectrum_3'] = fs.get(3, 0) / V if V > 0 else 0
    m['freq_spectrum_gt10'] = sum(1 for c in freq.values() if c > 10) / V if V > 0 else 0

    # ============================================================
    # LEXICAL RICHNESS: Yule K, Honore R
    # ============================================================
    S2 = sum(i * i * fi for i, fi in fs.items())
    m['word_yule_K'] = 10000 * (S2 - N) / (N * (N - 1)) if N > 1 else 0
    m['honore_R'] = 100 * math.log(N) / (1 - hapax / V) \
        if hapax > 0 and V > 0 and (1 - hapax / V) > 0 else 0

    # ============================================================
    # HEAPS
    # ============================================================
    checkpoints = [c for c in [100, 500, 1000, 2000, 5000, 10000, N] if c <= N]
    if len(checkpoints) >= 3:
        vocab_at = [len(set(tokens[:c])) for c in checkpoints]
        h_slope, _, _, _, _ = stats.linregress(np.log(checkpoints), np.log(vocab_at))
        m['heaps_beta'] = h_slope
    else:
        m['heaps_beta'] = 0

    # ============================================================
    # AUTOCORRELATION
    # ============================================================
    # Word frequency autocorrelation
    wfreqs = np.array([freq[t] for t in tokens], dtype=float)
    wf_c = wfreqs - wfreqs.mean()
    denom_f = np.sum(wf_c ** 2)
    m['autocorr_wordfreq'] = float(np.sum(wf_c[:-1] * wf_c[1:]) / denom_f) if denom_f > 0 else 0

    # TTR-25 autocorrelation (sliding window TTR series)
    if N >= 50:
        ttr_series = [len(set(tokens[i:i + 25])) / 25 for i in range(0, N - 25, 5)]
        if len(ttr_series) > 2:
            ts = np.array(ttr_series)
            ts_c = ts - ts.mean()
            ts_d = np.sum(ts_c ** 2)
            m['autocorr_ttr_25'] = float(np.sum(ts_c[:-1] * ts_c[1:]) / ts_d) if ts_d > 0 else 0
        else:
            m['autocorr_ttr_25'] = 0
    else:
        m['autocorr_ttr_25'] = 0

    # Hapax-25 autocorrelation
    if N >= 50:
        hapax_series = []
        for i in range(0, N - 25, 5):
            window = tokens[i:i + 25]
            wf = Counter(window)
            hapax_series.append(sum(1 for c in wf.values() if c == 1) / 25)
        if len(hapax_series) > 2:
            hs = np.array(hapax_series)
            hs_c = hs - hs.mean()
            hs_d = np.sum(hs_c ** 2)
            m['autocorr_hapax_25'] = float(np.sum(hs_c[:-1] * hs_c[1:]) / hs_d) if hs_d > 0 else 0
        else:
            m['autocorr_hapax_25'] = 0
    else:
        m['autocorr_hapax_25'] = 0

    # ============================================================
    # FREQUENCY CONCENTRATION
    # ============================================================
    m['top10_share'] = sum(sf[:10]) / N if len(sf) >= 10 else 0
    m['top50_share'] = sum(sf[:50]) / N if len(sf) >= 50 else 0

    # ============================================================
    # METADATA
    # ============================================================
    m['n_tokens'] = N
    m['n_types'] = V

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
        metric_list = list(set(ALL_85))  # deduplicated
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
    Print a formatted comparison table.
    generator_dict: {label: metrics_dict}
    """
    if metric_list is None:
        metric_list = sorted(vms_metrics.keys())

    labels = list(generator_dict.keys())
    header = f"{'Metric':<28} {'VMS':>10}"
    for lab in labels:
        header += f" {lab:>14}"
    print(header)
    print("-" * (30 + 12 + 16 * len(labels)))

    for metric in metric_list:
        vms_val = vms_metrics.get(metric)
        if vms_val is None:
            continue
        row = f"{metric:<28} {vms_val:>10.4f}"
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

    with open('enriched_records.pkl', 'rb') as f:
        records = pickle.load(f)
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

    print("Computing VMS 85-metric baseline...")
    m = compute_metrics(tokens, lines=lines)
    print(f"Total metrics: {len(m)} (target: 85+)")

    # Save
    with open('vms_baseline_85metrics.pkl', 'wb') as f:
        pickle.dump(m, f)

    # Print all
    for k in sorted(m.keys()):
        if k in ('n_tokens', 'n_types'):
            continue
        print(f"  {k:<30} = {m[k]:.6f}")

    # Self-score
    result = score_against_vms(m, m)
    print(f"\nSelf-score: {result['n_pass']}/{result['n_total']} (should be 100%)")

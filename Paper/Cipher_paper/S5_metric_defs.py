"""
Metric definitions for VMS scoring.
Import this alongside score_85_metrics.py to guarantee access to
CORE_15, BG_METRICS, and ALL_85 without dependency issues.

Usage:
    from metric_defs import CORE_15, BG_METRICS, ALL_85
"""

CORE_15 = [
    'autocorr_wordlen', 'autocorr_wordfreq', 'autocorr_hapax_25',
    'charbias_mean', 'charbias_skew',
    'H1_unigram', 'H2_markov_cond',
    'wordlen_mean', 'wordlen_unique_mean',
    'msttr_25', 'heaps_beta',
    'chardist_max', 'digraph_coverage',
    'zipf_lmz', 'tripled_words',
]

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

# Verify counts
assert len(CORE_15) == 15, f"CORE_15 has {len(CORE_15)}, expected 15"
assert len(BG_METRICS) == 42, f"BG_METRICS has {len(BG_METRICS)}, expected 42"
assert len(ALL_85) == 90, f"ALL_85 has {len(ALL_85)}, expected 90"

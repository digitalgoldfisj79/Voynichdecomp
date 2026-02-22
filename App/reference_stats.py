"""
Reference implementation of all statistics metrics.
Computes everything from scratch using standard Python/numpy.
Used to validate the JavaScript browser engine.
"""
import json
import math
import numpy as np
from collections import Counter

def load_data():
    with open('test_data.json') as f:
        return json.load(f)

def coef_var(arr):
    m = np.mean(arr)
    if m == 0: return 0
    return np.std(arr, ddof=1) / m if len(arr) > 1 else 0

def calc_stats(tokens, lines):
    N = len(tokens)
    if N < 5:
        return None
    
    results = {}
    
    # ── Basic counts ──
    freq = Counter(tokens)
    types = len(freq)
    freq_vals = sorted(freq.values(), reverse=True)
    wl = [len(t) for t in tokens]
    wl_unique = [len(t) for t in freq.keys()]
    all_chars = ''.join(tokens)
    char_freq = Counter(all_chars)
    n_chars = len(all_chars)
    unique_chars = len(char_freq)
    
    results['n_tokens'] = N
    results['n_types'] = types
    results['n_chars'] = n_chars
    results['n_unique_chars'] = unique_chars
    
    # ── Word length ──
    results['wordlen_mean'] = np.mean(wl)
    results['wordlen_std'] = np.std(wl, ddof=1)
    # Skewness: adjusted Fisher-Pearson (sample skewness)
    m = np.mean(wl)
    s = np.std(wl, ddof=1)
    n = len(wl)
    if s > 0 and n > 2:
        results['wordlen_skew'] = (n / ((n-1)*(n-2))) * sum(((x - m)/s)**3 for x in wl)
    else:
        results['wordlen_skew'] = 0
    
    results['wordlen_unique_mean'] = np.mean(wl_unique)
    results['wordlen_unique_std'] = np.std(wl_unique, ddof=1) if len(wl_unique) > 1 else 0
    m_u = np.mean(wl_unique)
    s_u = np.std(wl_unique, ddof=1) if len(wl_unique) > 1 else 0
    n_u = len(wl_unique)
    if s_u > 0 and n_u > 2:
        results['wordlen_unique_skew'] = (n_u / ((n_u-1)*(n_u-2))) * sum(((x - m_u)/s_u)**3 for x in wl_unique)
    else:
        results['wordlen_unique_skew'] = 0
    
    # Autocorrelation (lag-1)
    def autocorr(a):
        a = np.array(a, dtype=float)
        if len(a) < 3:
            return 0
        m = np.mean(a)
        den = np.sum((a - m)**2)
        if den == 0:
            return 0
        num = np.sum((a[:-1] - m) * (a[1:] - m))
        return num / den
    
    results['wordlen_autocorr'] = autocorr(wl)
    
    # ── Word distribution ──
    top25 = freq_vals[:min(25, len(freq_vals))]
    results['worddist_max'] = top25[0] if top25 else 0
    results['worddist_shape'] = np.mean(top25) - top25[-1] if len(top25) > 1 else 0
    
    # ── Word bias: positional heat within line (5-bin CV) ──
    word_heat = {}
    for line in lines:
        line_len = len(line)
        if line_len < 2:
            continue
        for i, t in enumerate(line):
            b = min(4, int((i / line_len) * 5))
            if t not in word_heat:
                word_heat[t] = [0,0,0,0,0]
            word_heat[t][b] += 1
    
    wb_list = []
    for w, h in word_heat.items():
        cv = coef_var(h)
        cnt = freq[w]
        wb_list.extend([cv] * cnt)
    
    results['wordbias_mean'] = np.mean(wb_list) if wb_list else 0
    results['wordbias_std'] = np.std(wb_list, ddof=1) if len(wb_list) > 1 else 0
    m_wb, s_wb, n_wb = np.mean(wb_list), (np.std(wb_list, ddof=1) if len(wb_list)>1 else 0), len(wb_list)
    if s_wb > 0 and n_wb > 2:
        results['wordbias_skew'] = (n_wb / ((n_wb-1)*(n_wb-2))) * sum(((x - m_wb)/s_wb)**3 for x in wb_list)
    else:
        results['wordbias_skew'] = 0
    
    # Word bias across lines (5-bin CV)
    word_heat_lines = {}
    for li, line in enumerate(lines):
        b = min(4, int((li / len(lines)) * 5))
        for t in line:
            if t not in word_heat_lines:
                word_heat_lines[t] = [0,0,0,0,0]
            word_heat_lines[t][b] += 1
    
    wbl_list = []
    for w, h in word_heat_lines.items():
        cv = coef_var(h)
        cnt = freq[w]
        wbl_list.extend([cv] * cnt)
    
    results['wordbias_lines_mean'] = np.mean(wbl_list) if wbl_list else 0
    results['wordbias_lines_std'] = np.std(wbl_list, ddof=1) if len(wbl_list) > 1 else 0
    m_wbl, s_wbl, n_wbl = np.mean(wbl_list), (np.std(wbl_list, ddof=1) if len(wbl_list)>1 else 0), len(wbl_list)
    if s_wbl > 0 and n_wbl > 2:
        results['wordbias_lines_skew'] = (n_wbl / ((n_wbl-1)*(n_wbl-2))) * sum(((x - m_wbl)/s_wbl)**3 for x in wbl_list)
    else:
        results['wordbias_lines_skew'] = 0
    
    # ── Character distribution ──
    char_probs = [v/n_chars for v in char_freq.values()]
    char_sorted = sorted(char_probs, reverse=True)
    results['chardist_max'] = char_sorted[0]
    results['chardist_shape'] = np.mean(char_sorted[:5]) - char_sorted[-1] if len(char_sorted) > 1 else 0
    
    # ── Char bias within word (5-bin CV) ──
    cbw = {}
    for t, cnt in freq.items():
        wlen = len(t)
        if wlen < 2:
            continue
        for i, c in enumerate(t):
            b = min(4, int((i / wlen) * 5))
            if c not in cbw:
                cbw[c] = [0,0,0,0,0]
            cbw[c][b] += cnt
    
    cbw_list = []
    for c, h in cbw.items():
        cv = coef_var(h)
        cnt = char_freq[c]
        cbw_list.extend([cv] * cnt)
    
    results['charbias_words_mean'] = np.mean(cbw_list) if cbw_list else 0
    results['charbias_words_std'] = np.std(cbw_list, ddof=1) if len(cbw_list) > 1 else 0
    m_cbw, s_cbw, n_cbw = np.mean(cbw_list), (np.std(cbw_list, ddof=1) if len(cbw_list)>1 else 0), len(cbw_list)
    if s_cbw > 0 and n_cbw > 2:
        results['charbias_words_skew'] = (n_cbw / ((n_cbw-1)*(n_cbw-2))) * sum(((x - m_cbw)/s_cbw)**3 for x in cbw_list)
    else:
        results['charbias_words_skew'] = 0
    
    # ── Char bias within line (10-bin CV) ──
    cbl = {}
    for line in lines:
        text = ' '.join(line)
        tlen = len(text)
        if tlen < 2:
            continue
        for i, c in enumerate(text):
            if c == ' ':
                continue
            b = min(9, int((i / tlen) * 10))
            if c not in cbl:
                cbl[c] = [0]*10
            cbl[c][b] += 1
    
    cbl_list = []
    for c, h in cbl.items():
        cv = coef_var(h)
        cnt = char_freq[c]
        cbl_list.extend([cv] * cnt)
    
    results['charbias_mean'] = np.mean(cbl_list) if cbl_list else 0
    results['charbias_std'] = np.std(cbl_list, ddof=1) if len(cbl_list) > 1 else 0
    m_cbl, s_cbl, n_cbl = np.mean(cbl_list), (np.std(cbl_list, ddof=1) if len(cbl_list)>1 else 0), len(cbl_list)
    if s_cbl > 0 and n_cbl > 2:
        results['charbias_skew'] = (n_cbl / ((n_cbl-1)*(n_cbl-2))) * sum(((x - m_cbl)/s_cbl)**3 for x in cbl_list)
    else:
        results['charbias_skew'] = 0
    
    # ── Entropy hierarchy ──
    H0 = math.log2(unique_chars)
    H1 = -sum(p * math.log2(p) for p in char_probs)
    results['H0_max_entropy'] = H0
    results['H1_unigram'] = H1
    results['char_evenness'] = H1 / H0 if H0 > 0 else 0
    results['char_redundancy'] = 1 - H1/H0 if H0 > 0 else 0
    results['char_simpson_D'] = sum(p**2 for p in char_probs)
    
    # Char Yule K
    char_fv = list(char_freq.values())
    m2_c = sum(f*f for f in char_fv)
    results['char_yule_K'] = 10000 * (m2_c - n_chars) / (n_chars * n_chars) if n_chars > 0 else 0
    
    # ── Digraph / trigraph (within words, not across spaces) ──
    digraphs = Counter()
    trigraphs = Counter()
    for t in tokens:
        for i in range(len(t) - 1):
            digraphs[t[i:i+2]] += 1
        for i in range(len(t) - 2):
            trigraphs[t[i:i+3]] += 1
    
    n_di = sum(digraphs.values())
    n_tri = sum(trigraphs.values())
    
    h2_joint = -sum((c/n_di) * math.log2(c/n_di) for c in digraphs.values()) if n_di > 0 else 0
    h2_cond = h2_joint - H1
    h3_joint = -sum((c/n_tri) * math.log2(c/n_tri) for c in trigraphs.values()) if n_tri > 0 else 0
    h3_cond = h3_joint - h2_joint
    
    results['h2_joint_digraph'] = h2_joint
    results['h2_conditional'] = h2_cond
    results['h3_joint_trigraph'] = h3_joint
    results['h3_conditional'] = h3_cond
    results['digraph_unique'] = len(digraphs)
    results['digraph_coverage'] = len(digraphs) / (unique_chars * unique_chars) if unique_chars > 0 else 0
    results['trigraph_unique'] = len(trigraphs)
    
    # ── 2nd-order Markov (H2 conditional) — over full text with spaces ──
    text = ' '.join(tokens)
    mk2 = {}
    mk2_total = {}
    for i in range(2, len(text)):
        pfx = text[i-2:i]
        ch = text[i]
        if pfx not in mk2:
            mk2[pfx] = Counter()
        mk2[pfx][ch] += 1
        mk2_total[pfx] = mk2_total.get(pfx, 0) + 1
    
    H2m = 0
    tc = 0
    for pfx, n in mk2_total.items():
        tc += n
        ent = 0
        for ch, cnt in mk2[pfx].items():
            p = cnt / n
            ent -= p * math.log2(p)
        H2m += n * ent
    H2m = H2m / tc if tc > 0 else 0
    results['H2_markov_cond'] = H2m
    results['entropy_approx'] = H2m
    
    # ── TTR variants ──
    results['ttr'] = types / N
    results['rttr'] = types / math.sqrt(N)
    results['cttr'] = types / math.sqrt(2 * N)
    results['log_ttr'] = math.log(types) / math.log(N)
    results['maas_a2'] = (math.log(N) - math.log(types)) / (math.log(N)**2)
    results['uber_index'] = (math.log(N)**2) / (math.log(N) - math.log(types)) if types > 1 else 0
    results['brunet_W'] = N ** (types ** (-0.172))
    
    # MSTTR
    def msttr(toks, w):
        if len(toks) < w:
            return 1
        segs = len(toks) // w
        s = 0
        for i in range(segs):
            s += len(set(toks[i*w:(i+1)*w])) / w
        return s / segs
    
    results['msttr_25'] = msttr(tokens, 25)
    results['msttr_50'] = msttr(tokens, 50)
    results['msttr_100'] = msttr(tokens, 100)
    
    # MATTR
    def mattr(toks, w):
        if len(toks) < w:
            return 1
        s = 0
        n_w = 0
        for i in range(len(toks) - w + 1):
            s += len(set(toks[i:i+w])) / w
            n_w += 1
        return s / n_w
    
    results['mattr_25'] = mattr(tokens, 25)
    results['mattr_50'] = mattr(tokens, 50)
    results['mattr_100'] = mattr(tokens, 100)
    
    # ── Hapax & frequency spectrum ──
    hapax = sum(1 for v in freq_vals if v == 1)
    dis = sum(1 for v in freq_vals if v == 2)
    results['hapax_ratio_tokens'] = hapax / N
    results['hapax_ratio_types'] = hapax / types
    results['dis_ratio_tokens'] = dis / N
    results['dis_ratio_types'] = dis / types
    results['sichel_S'] = dis / types
    results['honore_R'] = 100 * math.log(N) / (1 - hapax/types) if types > hapax and hapax > 0 else 0
    results['freq_spectrum_1'] = hapax / types
    results['freq_spectrum_2'] = dis / types
    results['freq_spectrum_3'] = sum(1 for v in freq_vals if v == 3) / types
    results['freq_spectrum_gt10'] = sum(1 for v in freq_vals if v > 10) / types
    
    # ── Lexical: Word Yule K ──
    m2_w = sum(f*f for f in freq_vals)
    results['word_yule_K'] = 10000 * (m2_w - N) / (N * N) if N > 0 else 0
    
    # ── Ngram distribution ──
    ngram_bank = Counter()
    for w, cnt in freq.items():
        for i in range(len(w)):
            for l in range(1, min(4, len(w) - i + 1)):
                ng = w[i:i+l]
                ngram_bank[ng] += cnt
    results['ngramdist_max'] = max(ngram_bank.values()) if ngram_bank else 0
    results['unique_ngrams'] = len(ngram_bank)
    
    # ── Repetition ──
    w_rep = sum(1 for i in range(1, N) if tokens[i] == tokens[i-1])
    w_trip = sum(1 for i in range(2, N) if tokens[i] == tokens[i-1] == tokens[i-2])
    c_rep = sum(1 for t in tokens for i in range(1, len(t)) if t[i] == t[i-1])
    c_trip = sum(1 for t in tokens for i in range(2, len(t)) if t[i] == t[i-1] == t[i-2])
    
    results['unique_words'] = types
    results['repeated_words'] = w_rep / N
    results['tripled_words'] = w_trip / N
    results['unique_chars'] = unique_chars
    results['repeated_chars'] = c_rep / n_chars
    results['tripled_chars'] = c_trip / n_chars
    
    # ── Flipped pairs (sampled) ──
    lim = min(N, 5000)
    pair_set = set()
    for i in range(1, lim):
        pair_set.add(f"{tokens[i-1]}|{tokens[i]}")
    flips = sum(1 for i in range(1, lim) if f"{tokens[i]}|{tokens[i-1]}" in pair_set)
    results['flipped_pairs'] = flips / lim
    
    # ── Autocorrelation: TTR and hapax in windows ──
    ttr_w = []
    for i in range(0, N - 25 + 1, 25):
        ttr_w.append(len(set(tokens[i:i+25])) / 25)
    hap_w = []
    for i in range(0, N - 25 + 1, 25):
        f = Counter(tokens[i:i+25])
        hap_w.append(sum(1 for v in f.values() if v == 1) / 25)
    
    results['autocorr_wordlen'] = autocorr(wl)
    results['autocorr_wordfreq'] = autocorr([freq[t] for t in tokens])
    results['autocorr_ttr_25'] = autocorr(ttr_w) if len(ttr_w) > 2 else 0
    results['autocorr_hapax_25'] = autocorr(hap_w) if len(hap_w) > 2 else 0
    
    # ── Zipf LMZ ──
    half = max(1, len(freq_vals) // 2)
    cut = freq_vals[:half]
    min_cut = cut[-1] if cut else 1
    z1 = 1 - (1/len(cut)) * sum(math.log(max(x/min_cut, 0.001)) for x in cut)
    z2 = 0.5 - (1/len(cut)) * sum(min_cut/x for x in cut)
    lmz = 4 * len(cut) * (z1**2 + 6*z1*z2 + 12*z2**2)
    results['zipf'] = lmz
    
    return results


if __name__ == '__main__':
    data = load_data()
    
    for scope in ['whole', 'herbal', 'f1r']:
        tokens = data[scope]['tokens']
        lines = data[scope]['lines']
        st = calc_stats(tokens, lines)
        if st is None:
            print(f"\n{scope}: insufficient data")
            continue
        json.dump(st, open(f'py_stats_{scope}.json', 'w'), indent=2)
        print(f"\n{scope}: {len(st)} metrics computed")
        for k in sorted(st.keys()):
            v = st[k]
            if isinstance(v, float):
                print(f"  {k:30s} = {v:.8f}")
            else:
                print(f"  {k:30s} = {v}")

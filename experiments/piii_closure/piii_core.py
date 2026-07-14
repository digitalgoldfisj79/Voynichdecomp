from __future__ import annotations

import csv
import hashlib
import math
import random
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260714
FEATURES = [
    'len_mean', 'len_sd', 'len_skew', 'tail9', 'type_gap', 'ttr', 'hapax',
    'top10', 'mattr25', 'mattr100', 'compression', 'h1', 'h2', 'h3',
    'charmax', 'bgcov', 'exact1', 'exact2', 'exactplat', 'exactbump',
    'ed1', 'ed2', 'edplat', 'edbump', 'pre_suf_mi', 'shape_mi', 'rank_mi',
    'first_delta', 'last_delta', 'within_mi', 'cross_mi', 'boundary_drop',
    'reuse50', 'ed1_graph'
]
CRITICAL = [
    'len_mean', 'len_sd', 'len_skew', 'type_gap', 'ttr', 'hapax',
    'mattr25', 'mattr100', 'h1', 'h2', 'h3', 'exact1', 'exactbump',
    'ed1', 'edbump', 'pre_suf_mi', 'boundary_drop', 'reuse50', 'ed1_graph'
]


def entropy(counts):
    a = np.array([v for v in counts if v > 0], dtype=float)
    if not len(a):
        return 0.0
    p = a / a.sum()
    return float(-(p * np.log2(p)).sum())


def conditional_entropy(seq, order):
    if len(seq) <= order:
        return 0.0
    d = defaultdict(Counter)
    for i in range(order, len(seq)):
        d[tuple(seq[i-order:i])][seq[i]] += 1
    n = sum(sum(c.values()) for c in d.values())
    if not n:
        return 0.0
    return sum(sum(c.values()) / n * entropy(c.values()) for c in d.values())


def mutual_information(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    joint = Counter(zip(a[:n], b[:n]))
    ca, cb = Counter(a[:n]), Counter(b[:n])
    return sum(c/n * math.log2((c*n) / (ca[x]*cb[y])) for (x, y), c in joint.items())


def skewness(values):
    a = np.array(values, dtype=float)
    if len(a) < 3 or a.std() == 0:
        return 0.0
    return float(np.mean(((a - a.mean()) / a.std()) ** 3))


def mattr(tokens, window):
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    return float(np.mean([
        len(set(tokens[i:i+window])) / window
        for i in range(len(tokens)-window+1)
    ]))


def token_shape(token):
    return token[0], token[-1], min(len(token), 9)


def lag_rate(tokens, lag, edit_distance=False):
    n = len(tokens) - lag
    if n <= 0:
        return 0.0
    if edit_distance:
        return sum(
            Levenshtein.distance(tokens[i-lag], tokens[i], score_cutoff=1) == 1
            for i in range(lag, len(tokens))
        ) / n
    return sum(tokens[i-lag] == tokens[i] for i in range(lag, len(tokens))) / n


def local_reuse(tokens, horizon=50):
    last, hits = {}, 0
    for i, token in enumerate(tokens):
        if token in last and i - last[token] <= horizon:
            hits += 1
        last[token] = i
    return hits / len(tokens) if tokens else 0.0


def edit1_graph_density(types, cap=1000):
    """Exact ED1 edge density using wildcard and deletion signatures."""
    items = sorted(types, key=lambda x: (-len(x), x))[:cap]
    index = {token: i for i, token in enumerate(items)}
    edges = set()
    wildcard = defaultdict(list)
    deletions = defaultdict(list)
    for i, token in enumerate(items):
        for pos in range(len(token)):
            wildcard[(len(token), token[:pos] + '*' + token[pos+1:])].append(i)
            deletions[token[:pos] + token[pos+1:]].append(i)
    for ids in wildcard.values():
        for a in range(len(ids)):
            for b in range(a+1, len(ids)):
                edges.add((min(ids[a], ids[b]), max(ids[a], ids[b])))
    for shorter, j in index.items():
        for i in deletions.get(shorter, []):
            if i != j:
                edges.add((min(i, j), max(i, j)))
    n = len(items)
    return 2 * len(edges) / (n * (n - 1)) if n > 1 else 0.0


def rank_bins(tokens, global_frequency):
    ordered = {
        token: i for i, (token, _) in enumerate(
            sorted(global_frequency.items(), key=lambda kv: (-kv[1], kv[0]))
        )
    }
    n = max(1, len(ordered))
    return [min(9, 10 * ordered[token] // n) for token in tokens]


def features(lines, global_frequency=None, graph_cap=300):
    tokens = [token for line in lines for token in line]
    global_frequency = global_frequency or Counter(tokens)
    types = set(tokens)
    lengths = [len(token) for token in tokens]
    type_lengths = [len(token) for token in types]
    chars_with_spaces = list(' '.join(tokens))
    chars = list(''.join(tokens))
    char_counts = Counter(chars)
    bigrams = Counter(zip(chars, chars[1:]))
    exact = {k: lag_rate(tokens, k) for k in [1, 2, 5, 6, 7, 8, 9, 10]}
    ed = {k: lag_rate(tokens, k, True) for k in [1, 2, 5, 6, 7, 8, 9, 10]}
    exact_plateau = float(np.mean([exact[k] for k in range(5, 11)]))
    ed_plateau = float(np.mean([ed[k] for k in range(5, 11)]))
    prefixes = [token[:2] for token in tokens]
    suffixes = [token[-2:] for token in tokens]
    shapes = [token_shape(token) for token in tokens]
    ranks = rank_bins(tokens, global_frequency)
    medial_lengths = [len(token) for line in lines for token in line[1:-1]]
    medial_mean = np.mean(medial_lengths) if medial_lengths else np.mean(lengths)

    within_a, within_b, cross_a, cross_b = [], [], [], []
    for line in lines:
        line_shapes = [token_shape(token) for token in line]
        within_a.extend(line_shapes[:-1])
        within_b.extend(line_shapes[1:])
    for first, second in zip(lines, lines[1:]):
        if first and second:
            cross_a.append(token_shape(first[-1]))
            cross_b.append(token_shape(second[0]))
    within_mi = mutual_information(within_a, within_b)
    cross_mi = mutual_information(cross_a, cross_b)
    raw = ' '.join(tokens).encode()

    return {
        'len_mean': float(np.mean(lengths)),
        'len_sd': float(np.std(lengths)),
        'len_skew': skewness(lengths),
        'tail9': sum(x >= 9 for x in lengths) / len(lengths),
        'type_gap': float(np.mean(type_lengths) - np.mean(lengths)),
        'ttr': len(types) / len(tokens),
        'hapax': sum(v == 1 for v in Counter(tokens).values()) / len(types),
        'top10': sum(v for _, v in Counter(tokens).most_common(10)) / len(tokens),
        'mattr25': mattr(tokens, 25),
        'mattr100': mattr(tokens, 100),
        'compression': len(zlib.compress(raw, 9)) / len(raw),
        'h1': entropy(char_counts.values()),
        'h2': conditional_entropy(chars_with_spaces, 1),
        'h3': conditional_entropy(chars_with_spaces, 2),
        'charmax': max(char_counts.values()) / len(chars),
        'bgcov': len(bigrams) / len(char_counts) ** 2,
        'exact1': exact[1],
        'exact2': exact[2],
        'exactplat': exact_plateau,
        'exactbump': exact[1] / exact_plateau if exact_plateau else 0.0,
        'ed1': ed[1],
        'ed2': ed[2],
        'edplat': ed_plateau,
        'edbump': ed[1] / ed_plateau if ed_plateau else 0.0,
        'pre_suf_mi': mutual_information(prefixes, suffixes),
        'shape_mi': mutual_information(shapes[:-1], shapes[1:]),
        'rank_mi': mutual_information(ranks[:-1], ranks[1:]),
        'first_delta': float(np.mean([len(line[0]) for line in lines]) - medial_mean),
        'last_delta': float(np.mean([len(line[-1]) for line in lines]) - medial_mean),
        'within_mi': within_mi,
        'cross_mi': cross_mi,
        'boundary_drop': within_mi - cross_mi,
        'reuse50': local_reuse(tokens),
        'ed1_graph': edit1_graph_density(types, graph_cap),
    }


def quire_for_folio(folio):
    match = re.match(r'f(\d+)', folio)
    number = int(match.group(1)) if match else 999
    cuts = [8, 16, 22, 32, 38, 42, 50, 58, 66, 73, 84, 86, 90, 96, 103, 116]
    for i, cut in enumerate(cuts):
        if number <= cut:
            return f'Q{i+1}'
    return 'UNK'


def chunks(corpus, target=120):
    global_frequency = Counter(token for row in corpus for token in row['t'])
    matrix, groups, sections = [], [], []
    i = 0
    while i < len(corpus):
        folio = corpus[i]['f']
        lines, count, section = [], 0, corpus[i]['s']
        while i < len(corpus) and corpus[i]['f'] == folio and (count < target or len(lines) < 2):
            lines.append(corpus[i]['t'])
            count += len(corpus[i]['t'])
            i += 1
        if count >= 40:
            f = features(lines, global_frequency, 150)
            matrix.append([f[key] for key in FEATURES])
            groups.append(quire_for_folio(folio))
            sections.append(section)
    return np.array(matrix), groups, sections


def grouped_auc(first, second):
    xa, ga, _ = chunks(first)
    xb, gb, _ = chunks(second)
    n = min(len(xa), len(xb))
    X = np.vstack([xa[:n], xb[:n]])
    y = np.array([0] * n + [1] * n)
    groups = np.array(ga[:n] + gb[:n])
    cv = GroupKFold(min(5, len(set(groups))))
    probabilities = np.zeros(len(y))
    for train, test in cv.split(X, y, groups):
        model = make_pipeline(
            SimpleImputer(strategy='median'), StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0)
        )
        model.fit(X[train], y[train])
        probabilities[test] = model.predict_proba(X[test])[:, 1]
    score = roc_auc_score(y, probabilities)
    return float(max(score, 1-score))


def random_label_auc(vms, seed):
    X, groups, _ = chunks(vms)
    y = np.array(([0, 1] * ((len(X)+1)//2))[:len(X)])
    np.random.default_rng(seed).shuffle(y)
    probabilities = np.zeros(len(y))
    cv = GroupKFold(min(5, len(set(groups))))
    for train, test in cv.split(X, y, groups):
        model = make_pipeline(
            SimpleImputer(strategy='median'), StandardScaler(),
            LogisticRegression(max_iter=2000)
        )
        model.fit(X[train], y[train])
        probabilities[test] = model.predict_proba(X[test])[:, 1]
    score = roc_auc_score(y, probabilities)
    return float(max(score, 1-score))


def policy_auc(corpora):
    classes = ['iid10', 'cycle131', 'line_fixed', 'permute10']
    matrices, labels, groups = [], [], []
    for arm, source, replicate, corpus in corpora:
        if arm not in classes:
            continue
        matrix, _, _ = chunks(corpus)
        matrices.append(matrix)
        labels.extend([arm] * len(matrix))
        groups.extend([f'{source}_{replicate}'] * len(matrix))
    X, y, groups = np.vstack(matrices), np.array(labels), np.array(groups)
    probabilities = np.zeros((len(y), len(classes)))
    predictions = np.empty(len(y), dtype=object)
    cv = StratifiedGroupKFold(4, shuffle=True, random_state=SEED)
    for train, test in cv.split(X, y, groups):
        model = make_pipeline(
            SimpleImputer(strategy='median'), StandardScaler(),
            LogisticRegression(max_iter=3000)
        )
        model.fit(X[train], y[train])
        p = model.predict_proba(X[test])
        index = {label: i for i, label in enumerate(model.classes_)}
        for j, label in enumerate(classes):
            probabilities[test, j] = p[:, index[label]]
        predictions[test] = model.predict(X[test])
    binary = np.column_stack([(y == label) for label in classes])
    return (
        float(roc_auc_score(binary, probabilities, average='macro')),
        float(accuracy_score(y, predictions)),
    )


def bootstrap_sd(vms, repetitions=200):
    by_quire = defaultdict(list)
    for row in vms:
        by_quire[quire_for_folio(row['f'])].append(row['t'])
    quires = sorted(by_quire)
    rng = random.Random(SEED)
    samples = {key: [] for key in CRITICAL}
    for _ in range(repetitions):
        lines = [
            line
            for quire in [rng.choice(quires) for _ in quires]
            for line in by_quire[quire]
        ]
        f = features(lines, graph_cap=1000)
        for key in samples:
            samples[key].append(f[key])
    return {key: float(np.std(values, ddof=1)) for key, values in samples.items()}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import requests
from lxml import etree
from scipy.special import logsumexp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260725
RNG = np.random.default_rng(SEED)

# Manuscripts listed in the CoReMA semantic-model navigation. Invalid or
# unavailable annotated-detail objects are audited and excluded, never guessed.
MANUSCRIPT_IDS = [
    "a1","b1","b2","b3","b4","b5","b6","br1","bs1","bs2","db1","ds1",
    "er1","er2","gr1","h1","h2","h3","h4","ha1","hi1","k1","ka1","ka2",
    "ka3","ko1","m1","m2","m3","m4","m5","m6","m7","m8","m9","m10",
    "m12","m13","m11","mi1","n1","n2","pa1","pr1","sb1","sb2","sb3",
    "so1","st1","ste1","w1","w2","w3","w4","wo1","wo2","wo3","wo4",
    "wo5","wo7","wo8","wo9","wo10","wo11","wol1","zu1"
]

SEMANTIC_TAGS = {
    "ingredient", "instruction", "tool", "time", "date", "dish", "title",
    "servingTip", "kitchenTip", "householdTip", "dietetics", "alternative",
    "opener", "closer", "household", "religion", "sp", "ref"
}

ROLE_ORDER = ["INGREDIENT", "ACTION", "TOOL", "TEMPORAL", "OUTPUT", "META", "OTHER"]
ROLE_TO_INT = {x: i for i, x in enumerate(ROLE_ORDER)}
TYPE_MAP = {
    "recipe": "RECIPE",
    "medicinal": "MEDICINAL",
    "dietetics": "MEDICINAL",
    "artTechnology": "TECHNICAL",
    "household": "TECHNICAL",
    "kitchenTip": "TIP",
}
TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*|\d+(?:[.,]\d+)?|[^\w\s]", re.UNICODE)


def localname(node) -> str:
    return etree.QName(node).localname if isinstance(node.tag, str) else ""


def is_xml_payload(content: bytes) -> bool:
    head = content.lstrip()[:200].lower()
    return head.startswith(b"<?xml") or head.startswith(b"<tei") or b"<tei" in head


def download_corema(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "VoynichRecoverabilityResearch/0.6 (+https://github.com/digitalgoldfisj79/Voynichdecomp)"})
    audit = {"attempted": len(MANUSCRIPT_IDS), "downloaded": [], "failed": []}
    for mid in MANUSCRIPT_IDS:
        dest = out_dir / f"{mid}.recipes.xml"
        if dest.exists() and dest.stat().st_size > 100:
            audit["downloaded"].append({"id": mid, "bytes": dest.stat().st_size, "cached": True})
            continue
        urls = [
            f"https://gams.uni-graz.at/o%3Acorema.{mid}.recipes/TEI_SOURCE",
            f"https://gams.uni-graz.at/o:corema.{mid}.recipes/TEI_SOURCE",
        ]
        ok = False
        errors = []
        for url in urls:
            for attempt in range(3):
                try:
                    r = session.get(url, timeout=60, allow_redirects=True)
                    if r.status_code == 200 and is_xml_payload(r.content):
                        try:
                            parser = etree.XMLParser(recover=True, huge_tree=True)
                            root = etree.fromstring(r.content, parser)
                            if root is None:
                                raise ValueError("lxml recovery returned no root")
                        except Exception as exc:
                            errors.append(f"parse:{exc}")
                            break
                        dest.write_bytes(r.content)
                        audit["downloaded"].append({"id": mid, "bytes": len(r.content), "url": url, "cached": False})
                        ok = True
                        break
                    errors.append(f"{r.status_code}:{r.headers.get('content-type')}:{len(r.content)}")
                except Exception as exc:
                    errors.append(type(exc).__name__ + ":" + str(exc)[:160])
                time.sleep(1.5 * (attempt + 1))
            if ok:
                break
        if not ok:
            audit["failed"].append({"id": mid, "errors": errors[-6:]})
    return audit


def role_from_stack(stack: Sequence[str]) -> str:
    s = set(stack)
    if "ingredient" in s:
        return "INGREDIENT"
    if "tool" in s:
        return "TOOL"
    if "time" in s or "date" in s:
        return "TEMPORAL"
    if "dish" in s or "title" in s:
        return "OUTPUT"
    if "instruction" in s:
        return "ACTION"
    if s.intersection({"servingTip", "kitchenTip", "householdTip", "dietetics", "alternative", "opener", "closer", "household", "religion", "sp", "ref"}):
        return "META"
    return "OTHER"


def tokenize_chunk(text: str) -> list[str]:
    return TOKEN_RE.findall(text or "")


def collect_ab_tokens(ab) -> list[dict]:
    out: list[dict] = []

    def add_text(text: str | None, stack: list[str]):
        if not text:
            return
        role = role_from_stack(stack)
        for tok in tokenize_chunk(text):
            if any(ch.isalnum() for ch in tok):
                out.append({"token": tok, "role": role, "semantic_stack": ">".join(stack)})

    def walk(node, stack: list[str], root=False):
        tag = localname(node)
        next_stack = stack + ([tag] if tag in SEMANTIC_TAGS else [])
        add_text(node.text, next_stack)
        for child in node:
            if not root and localname(child) == "ab":
                # Nested subrecipes are processed as their own sequence.
                add_text(child.tail, next_stack)
                continue
            if root and localname(child) == "ab":
                add_text(child.tail, next_stack)
                continue
            walk(child, next_stack, root=False)
            add_text(child.tail, next_stack)

    # Walk direct content while skipping nested <ab> descendants.
    tag = localname(ab)
    stack: list[str] = []
    add_text(ab.text, stack)
    for child in ab:
        if localname(child) == "ab":
            add_text(child.tail, stack)
            continue
        walk(child, stack)
        add_text(child.tail, stack)
    return out


def parse_corema(xml_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    token_rows = []
    recipe_rows = []
    audit = {"files": [], "parse_failures": [], "xml_recovery_issues": [], "types": Counter(), "roles": Counter()}
    for path in sorted(xml_dir.glob("*.recipes.xml")):
        mid = path.name.split(".", 1)[0]
        try:
            parser = etree.XMLParser(recover=True, huge_tree=True)
            root = etree.fromstring(path.read_bytes(), parser)
            if root is None:
                raise ValueError("lxml recovery returned no root")
            if parser.error_log:
                audit["xml_recovery_issues"].append({
                    "file": path.name,
                    "issues": [str(item) for item in parser.error_log],
                })
        except Exception as exc:
            audit["parse_failures"].append({"file": path.name, "error": str(exc)})
            continue
        abs_all = root.xpath("//*[local-name()='ab']")
        file_recipes = 0
        file_tokens = 0
        for ai, ab in enumerate(abs_all):
            typ = ab.get("type") or "unknown"
            # Include top-level units and explicit subrecipes; exclude wrappers with no text.
            seq = collect_ab_tokens(ab)
            if len(seq) < 3:
                continue
            rid = ab.get("{http://www.w3.org/XML/1998/namespace}id") or f"{mid}.{ai+1}"
            subtype = ab.get("subtype") or ""
            coarse_type = TYPE_MAP.get(typ, "OTHER_TYPE")
            audit["types"][typ] += 1
            role_counts = Counter(x["role"] for x in seq)
            audit["roles"].update(role_counts)
            n = len(seq)
            tokens = [x["token"] for x in seq]
            for i, item in enumerate(seq):
                token_rows.append({
                    "manuscript": mid,
                    "recipe_id": rid,
                    "recipe_type_raw": typ,
                    "recipe_type": coarse_type,
                    "subtype": subtype,
                    "position": i,
                    "n_tokens": n,
                    "token": item["token"],
                    "token_lower": item["token"].lower(),
                    "role": item["role"],
                    "semantic_stack": item["semantic_stack"],
                    "prev2": tokens[i-2] if i >= 2 else "<BOS2>",
                    "prev": tokens[i-1] if i >= 1 else "<BOS>",
                    "next": tokens[i+1] if i+1 < n else "<EOS>",
                    "next2": tokens[i+2] if i+2 < n else "<EOS2>",
                })
            recipe_rows.append({
                "manuscript": mid,
                "recipe_id": rid,
                "recipe_type_raw": typ,
                "recipe_type": coarse_type,
                "subtype": subtype,
                "n_tokens": n,
                "text": " ".join(tokens),
                **{f"role_{k.lower()}": role_counts.get(k, 0) for k in ROLE_ORDER},
            })
            file_recipes += 1
            file_tokens += n
        audit["files"].append({"manuscript": mid, "recipes": file_recipes, "tokens": file_tokens, "bytes": path.stat().st_size})
    audit["types"] = dict(audit["types"])
    audit["roles"] = dict(audit["roles"])
    tok = pd.DataFrame(token_rows)
    rec = pd.DataFrame(recipe_rows)
    if tok.empty:
        raise RuntimeError("No CoReMA tokens parsed")
    return tok, rec, audit


def pattern_signature(s: str) -> str:
    mp = {}
    nxt = 0
    out = []
    for ch in s.lower():
        if ch.isalpha():
            if ch not in mp:
                mp[ch] = chr(ord('a') + (nxt % 26))
                nxt += 1
            out.append(mp[ch])
        elif ch.isdigit():
            out.append('0')
        else:
            out.append(ch)
    return ''.join(out)


def enrich_token_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Manuscript-local token counts/ranks: identity-neutral but equality/frequency preserving.
    df["ms_count"] = df.groupby(["manuscript", "token_lower"])["token_lower"].transform("size")
    rank_map = {}
    for ms, d in df.groupby("manuscript"):
        counts = d["token_lower"].value_counts()
        for rank, token in enumerate(counts.index, 1):
            rank_map[(ms, token)] = rank
    df["ms_rank"] = [rank_map[(m, t)] for m, t in zip(df.manuscript, df.token_lower)]
    df["rank_bucket"] = df["ms_rank"].clip(upper=500).map(lambda x: f"r{int(x):03d}")
    for col in ["prev2", "prev", "next", "next2"]:
        low = df[col].str.lower()
        df[col + "_rank"] = [rank_map.get((m, t), 999) for m, t in zip(df.manuscript, low)]
        df[col + "_rank_bucket"] = df[col + "_rank"].clip(upper=999).map(lambda x: f"r{int(x):03d}")
    df["lex_context"] = (
        df["prev2"].str.lower() + " <P2> " + df["prev"].str.lower() + " <C> " +
        df["token_lower"] + " <N> " + df["next"].str.lower() + " <N2> " + df["next2"].str.lower()
    )
    df["rank_context"] = (
        df["prev2_rank_bucket"] + " " + df["prev_rank_bucket"] + " <C> " +
        df["rank_bucket"] + " " + df["next_rank_bucket"] + " " + df["next2_rank_bucket"]
    )
    df["pattern"] = df["token"].map(pattern_signature)
    df["prev_pattern"] = df["prev"].map(pattern_signature)
    df["next_pattern"] = df["next"].map(pattern_signature)
    df["pattern_context"] = df["prev_pattern"] + " <C> " + df["pattern"] + " <N> " + df["next_pattern"]
    return df


def structural_matrix(df: pd.DataFrame) -> np.ndarray:
    def length(s): return len(str(s))
    tok = df["token"].astype(str)
    prev = df["prev"].astype(str)
    nxt = df["next"].astype(str)
    X = np.column_stack([
        tok.map(length), prev.map(length), nxt.map(length),
        df["position"] / np.maximum(1, df["n_tokens"] - 1),
        np.log1p(df["ms_count"]), np.log1p(df["ms_rank"]),
        (tok.str.lower() == prev.str.lower()).astype(float),
        (tok.str.lower() == nxt.str.lower()).astype(float),
        tok.str[0].str.isupper().fillna(False).astype(float),
        tok.str.isdigit().astype(float),
        tok.str.contains(r"\d", regex=True).astype(float),
        tok.str.contains(r"[-’']", regex=True).astype(float),
        tok.map(lambda s: len(set(s.lower())) / max(1, len(s))),
        tok.map(lambda s: len(pattern_signature(s))),
        df["position"].eq(0).astype(float),
        (df["position"] + 1 == df["n_tokens"]).astype(float),
        (prev == "<BOS>").astype(float),
        (nxt == "<EOS>").astype(float),
    ]).astype(float)
    return X


def eligible_roles(df: pd.DataFrame) -> list[str]:
    support = df["role"].value_counts()
    groups = df.groupby("role")["manuscript"].nunique()
    return [r for r in ROLE_ORDER if support.get(r, 0) >= 100 and groups.get(r, 0) >= 3]


def metrics(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "macro_f1_eligible": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1_all": float(f1_score(y_true, y_pred, labels=ROLE_ORDER, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": classification_report(y_true, y_pred, labels=ROLE_ORDER, output_dict=True, zero_division=0),
    }


def fit_transition(y: Sequence[str], recipe_ids: Sequence[str], alpha=0.5):
    k = len(ROLE_ORDER)
    init = np.full(k, alpha)
    trans = np.full((k, k), alpha)
    prev_rid = None
    prev = None
    for lab, rid in zip(y, recipe_ids):
        j = ROLE_TO_INT[lab]
        if rid != prev_rid:
            init[j] += 1
        elif prev is not None:
            trans[prev, j] += 1
        prev_rid = rid
        prev = j
    init /= init.sum()
    trans /= trans.sum(axis=1, keepdims=True)
    return np.log(init), np.log(trans)


def viterbi(probs: np.ndarray, init_log: np.ndarray, trans_log: np.ndarray) -> np.ndarray:
    n, k = probs.shape
    emit = np.log(np.clip(probs, 1e-12, 1.0))
    dp = np.empty((n, k)); back = np.empty((n, k), dtype=int)
    dp[0] = init_log + emit[0]
    back[0] = -1
    for i in range(1, n):
        scores = dp[i-1][:, None] + trans_log
        back[i] = np.argmax(scores, axis=0)
        dp[i] = scores[back[i], np.arange(k)] + emit[i]
    path = np.empty(n, dtype=int); path[-1] = np.argmax(dp[-1])
    for i in range(n-2, -1, -1):
        path[i] = back[i+1, path[i+1]]
    return path


def hmm_decode_by_recipe(test_df: pd.DataFrame, probs: np.ndarray, init_log, trans_log) -> list[str]:
    pred = np.empty(len(test_df), dtype=object)
    for rid, inds in test_df.groupby("recipe_id", sort=False).indices.items():
        inds = np.array(sorted(inds, key=lambda j: int(test_df.iloc[j]["position"])), dtype=int)
        path = viterbi(probs[inds], init_log, trans_log)
        pred[inds] = [ROLE_ORDER[x] for x in path]
    return pred.tolist()


def run_token_cv(df: pd.DataFrame) -> dict:
    eligible = eligible_roles(df)
    groups = df["manuscript"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    y = df["role"].to_numpy()
    X_struct = structural_matrix(df)
    fold_rows = []
    pred_store = {m: np.empty(len(df), dtype=object) for m in ["majority", "lexical", "rank", "pattern", "structural", "structural_hmm"]}

    for fold, (tr, te) in enumerate(gkf.split(df, y, groups), 1):
        train = df.iloc[tr]; test = df.iloc[te]
        ytr = y[tr]; yte = y[te]
        majority = Counter(ytr).most_common(1)[0][0]
        pred_store["majority"][te] = majority

        lexical = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=80000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=2.0, n_jobs=1)),
        ])
        lexical.fit(train["lex_context"], ytr)
        pred_store["lexical"][te] = lexical.predict(test["lex_context"])

        rank = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, max_features=30000, token_pattern=r"[^ ]+")),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=1.0, n_jobs=1)),
        ])
        rank.fit(train["rank_context"], ytr)
        pred_store["rank"][te] = rank.predict(test["rank_context"])

        pattern = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2, max_features=40000)),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=1.0, n_jobs=1)),
        ])
        pattern.fit(train["pattern_context"], ytr)
        pred_store["pattern"][te] = pattern.predict(test["pattern_context"])

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=SEED + fold,
            n_jobs=-1, max_features="sqrt"
        )
        rf.fit(X_struct[tr], ytr)
        sp = rf.predict_proba(X_struct[te])
        # Expand model probabilities into the frozen global role order.
        expanded = np.full((len(te), len(ROLE_ORDER)), 1e-9)
        for j, cls in enumerate(rf.classes_):
            expanded[:, ROLE_TO_INT[cls]] = sp[:, j]
        expanded /= expanded.sum(axis=1, keepdims=True)
        pred_store["structural"][te] = [ROLE_ORDER[i] for i in expanded.argmax(axis=1)]
        init_log, trans_log = fit_transition(ytr, train["recipe_id"].to_numpy())
        pred_store["structural_hmm"][te] = hmm_decode_by_recipe(test.reset_index(drop=True), expanded, init_log, trans_log)

        for model in pred_store:
            p = pred_store[model][te]
            mm = metrics(yte, p, eligible)
            fold_rows.append({"fold": fold, "model": model, "test_manuscripts": sorted(set(groups[te])), **{k:v for k,v in mm.items() if k != "per_class"}})
        print(f"fold {fold}/{n_splits} complete: {len(te)} tokens", flush=True)

    summary = {}
    for model, pred in pred_store.items():
        summary[model] = metrics(y, pred.tolist(), eligible)
    return {"eligible_roles": eligible, "folds": fold_rows, "summary": summary, "predictions": {m: p.tolist() for m,p in pred_store.items()}}


def role_order_gain(df: pd.DataFrame) -> dict:
    groups = df["manuscript"].to_numpy()
    y = df["role"].to_numpy()
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    rows = []
    for fold, (tr, te) in enumerate(gkf.split(df, y, groups), 1):
        train = df.iloc[tr]; test = df.iloc[te]
        counts = Counter(y[tr]); total = sum(counts.values()); alpha = 0.5; k = len(ROLE_ORDER)
        uni = {r:(counts[r]+alpha)/(total+alpha*k) for r in ROLE_ORDER}
        init_log, trans_log = fit_transition(y[tr], train["recipe_id"].to_numpy(), alpha=alpha)
        trans = np.exp(trans_log)
        bits_iid = bits_markov = bits_shuffle = 0.0; n = 0
        for rid, d in test.groupby("recipe_id", sort=False):
            labs = d.sort_values("position")["role"].tolist()
            if not labs: continue
            bits_iid += sum(-math.log2(uni[x]) for x in labs)
            bits_markov += -math.log2(max(1e-12, math.exp(init_log[ROLE_TO_INT[labs[0]]])))
            for a,b in zip(labs[:-1], labs[1:]): bits_markov += -math.log2(max(1e-12, trans[ROLE_TO_INT[a], ROLE_TO_INT[b]]))
            sh = labs[:]; np.random.default_rng(SEED + fold + n).shuffle(sh)
            bits_shuffle += -math.log2(max(1e-12, math.exp(init_log[ROLE_TO_INT[sh[0]]])))
            for a,b in zip(sh[:-1], sh[1:]): bits_shuffle += -math.log2(max(1e-12, trans[ROLE_TO_INT[a], ROLE_TO_INT[b]]))
            n += len(labs)
        rows.append({"fold": fold, "n": n, "iid_bpt": bits_iid/n, "markov_bpt": bits_markov/n, "shuffle_markov_bpt": bits_shuffle/n, "order_gain_bpt": (bits_iid-bits_markov)/n, "real_vs_shuffle_bpt": (bits_shuffle-bits_markov)/n})
    return {"folds": rows, "mean_order_gain_bpt": float(np.mean([x["order_gain_bpt"] for x in rows])), "mean_real_vs_shuffle_bpt": float(np.mean([x["real_vs_shuffle_bpt"] for x in rows]))}


def recipe_structural_matrix(rec: pd.DataFrame) -> np.ndarray:
    role_cols = [f"role_{r.lower()}" for r in ROLE_ORDER]
    # Role counts are annotation-derived and therefore excluded. Surface-only features follow.
    text = rec["text"].astype(str)
    words = text.str.split()
    return np.column_stack([
        rec["n_tokens"],
        words.map(lambda x: np.mean([len(t) for t in x]) if x else 0),
        words.map(lambda x: np.std([len(t) for t in x]) if x else 0),
        words.map(lambda x: len(set(t.lower() for t in x)) / max(1,len(x))),
        words.map(lambda x: sum(t[:1].isupper() for t in x) / max(1,len(x))),
        words.map(lambda x: sum(any(c.isdigit() for c in t) for t in x) / max(1,len(x))),
        words.map(lambda x: sum(x[i].lower()==x[i-1].lower() for i in range(1,len(x))) / max(1,len(x)-1)),
    ]).astype(float)


def run_recipe_type_cv(rec: pd.DataFrame) -> dict:
    d = rec[rec["recipe_type"].isin(["RECIPE","MEDICINAL","TECHNICAL","TIP"])].copy()
    support = d["recipe_type"].value_counts()
    groups_per = d.groupby("recipe_type")["manuscript"].nunique()
    eligible = [x for x in ["RECIPE","MEDICINAL","TECHNICAL","TIP"] if support.get(x,0)>=20 and groups_per.get(x,0)>=3]
    d = d[d["recipe_type"].isin(eligible)].reset_index(drop=True)
    if len(eligible) < 2 or d["manuscript"].nunique() < 3:
        return {"admissible": False, "eligible_types": eligible, "support": support.to_dict()}
    groups = d["manuscript"].to_numpy(); y=d["recipe_type"].to_numpy(); gkf=GroupKFold(n_splits=min(5,len(np.unique(groups))))
    pred_lex=np.empty(len(d),dtype=object); pred_struct=np.empty(len(d),dtype=object)
    Xs=recipe_structural_matrix(d)
    folds=[]
    for fold,(tr,te) in enumerate(gkf.split(d,y,groups),1):
        lex=Pipeline([("tfidf",TfidfVectorizer(analyzer="char",ngram_range=(2,5),min_df=2,max_features=60000)),("clf",LogisticRegression(max_iter=500,class_weight="balanced",C=2.0))])
        lex.fit(d.iloc[tr]["text"],y[tr]);pred_lex[te]=lex.predict(d.iloc[te]["text"])
        rf=RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=3,class_weight="balanced_subsample",random_state=SEED+100+fold,n_jobs=-1)
        rf.fit(Xs[tr],y[tr]);pred_struct[te]=rf.predict(Xs[te])
        folds.append({"fold":fold,"test_manuscripts":sorted(set(groups[te]))})
    return {"admissible":True,"eligible_types":eligible,"support":support.to_dict(),"lexical":metrics_generic(y,pred_lex,eligible),"structural":metrics_generic(y,pred_struct,eligible),"folds":folds}


def metrics_generic(y,p,labels):
    return {"macro_f1":float(f1_score(y,p,labels=labels,average="macro",zero_division=0)),"weighted_f1":float(f1_score(y,p,average="weighted",zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"per_class":classification_report(y,p,labels=labels,output_dict=True,zero_division=0)}


def write_report(result: dict, path: Path):
    t = result["token_role_cv"]["summary"]
    gates = result["gates"]
    lines = [
        "# CoReMA procedural recoverability calibration v0.6",
        "",
        f"**Formal verdict:** **{result['formal_verdict']}**  ",
        f"**Manuscripts parsed:** {result['corpus']['manuscripts']}  ",
        f"**Procedural units:** {result['corpus']['recipes']}  ",
        f"**Labelled word tokens:** {result['corpus']['tokens']}  ",
        "",
        "## Frozen gates",
        "",
        f"- Lexical known-role recovery: **{'PASS' if gates['lexical_role'] else 'FAIL'}**",
        f"- Identity-neutral role recovery: **{'PASS' if gates['neutral_role'] else 'FAIL'}**",
        f"- Procedural sequence-order signal: **{'PASS' if gates['role_order'] else 'FAIL'}**",
        f"- CoReMA calibration admissible for target transfer: **{'YES' if gates['downstream_admissible'] else 'NO'}**",
        "",
        "## Token-role recovery",
        "",
        "| Model | Eligible-role macro-F1 | All-role macro-F1 | Weighted F1 | Balanced accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in ["majority","lexical","rank","pattern","structural","structural_hmm"]:
        x=t[m]
        lines.append(f"| {m} | {x['macro_f1_eligible']:.4f} | {x['macro_f1_all']:.4f} | {x['weighted_f1']:.4f} | {x['balanced_accuracy']:.4f} |")
    og=result["role_order"]
    lines += [
        "",
        "## Role sequence structure",
        "",
        f"Mean first-order Markov gain over IID: **{og['mean_order_gain_bpt']:.4f} bits/token**.  ",
        f"Mean real-order advantage over within-recipe shuffling: **{og['mean_real_vs_shuffle_bpt']:.4f} bits/token**.",
        "",
        "## Interpretation",
        "",
    ]
    if gates["downstream_admissible"]:
        lines.append("The real medieval procedural corpus passes the frozen calibration and authorises the sealed Voynich transfer stage. Passing means that operational roles can be recovered across manuscripts; it does not imply that Voynich contains those roles.")
    else:
        lines.append("The corpus does not pass the frozen recoverability calibration. The Voynich target must remain sealed for this route; no nearest-role narrative is admissible.")
    lines += ["", "## Provenance", "", "CoReMA TEI/XML was retrieved from the University of Graz GAMS public endpoints. The semantic model supplies explicit ingredient, instruction, tool, time, dish and advisory annotations. Manuscripts are the cross-validation groups."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data-dir",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    acquisition=download_corema(args.data_dir)
    tok,rec,parse_audit=parse_corema(args.data_dir)
    tok=enrich_token_features(tok)
    if tok["manuscript"].nunique()<8 or len(rec)<100 or len(tok)<5000:
        raise RuntimeError(f"Insufficient corpus: {tok.manuscript.nunique()} manuscripts, {len(rec)} units, {len(tok)} tokens")
    cv=run_token_cv(tok)
    order=role_order_gain(tok)
    recipe_cv=run_recipe_type_cv(rec)
    eligible=cv["eligible_roles"]
    lex=cv["summary"]["lexical"]; neutral=cv["summary"]["structural_hmm"]; maj=cv["summary"]["majority"]
    strong_classes=sum(lex["per_class"].get(r,{}).get("f1-score",0)>=0.40 for r in eligible)
    gates={
        "lexical_role": bool(lex["macro_f1_eligible"]>=0.60 and strong_classes>=min(3,len(eligible))),
        "neutral_role": bool(neutral["macro_f1_eligible"]>=0.35 and neutral["macro_f1_eligible"]-maj["macro_f1_eligible"]>=0.10),
        "role_order": bool(order["mean_real_vs_shuffle_bpt"]>=0.05 and all(x["real_vs_shuffle_bpt"]>0 for x in order["folds"])),
    }
    gates["downstream_admissible"]=bool(gates["lexical_role"] and gates["neutral_role"] and gates["role_order"])
    formal="CALIBRATION_PASS" if gates["downstream_admissible"] else "CALIBRATION_FAILURE"
    result={
        "schema":"corema-procedural-recoverability-v0.6",
        "formal_verdict":formal,
        "corpus":{"manuscripts":int(tok.manuscript.nunique()),"recipes":int(len(rec)),"tokens":int(len(tok)),"eligible_roles":eligible,"role_support":tok.role.value_counts().to_dict(),"type_support":rec.recipe_type_raw.value_counts().to_dict()},
        "acquisition":acquisition,"parse_audit":parse_audit,"token_role_cv":{k:v for k,v in cv.items() if k!="predictions"},"role_order":order,"recipe_type_cv":recipe_cv,"gates":gates,
    }
    # Save predictions separately for audit without bloating the primary JSON.
    pred=pd.DataFrame({"manuscript":tok.manuscript,"recipe_id":tok.recipe_id,"position":tok.position,"token":tok.token,"gold_role":tok.role,**{f"pred_{m}":p for m,p in cv["predictions"].items()}})
    pred.to_csv(args.out/"corema_role_cv_predictions_v0_6.csv.gz",index=False,compression="gzip")
    tok.drop(columns=["lex_context","rank_context","pattern_context"],errors="ignore").to_csv(args.out/"corema_token_rows_v0_6.csv.gz",index=False,compression="gzip")
    rec.to_csv(args.out/"corema_recipe_rows_v0_6.csv.gz",index=False,compression="gzip")
    (args.out/"corema_recoverability_results_v0_6.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    write_report(result,args.out/"COREMA_RECOVERABILITY_REPORT_v0_6.md")
    manifest=[]
    for p in sorted(args.out.iterdir()):
        if p.is_file():manifest.append({"path":p.name,"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    (args.out/"SHA256_MANIFEST_v0_6.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"formal_verdict":formal,"corpus":result["corpus"],"gates":gates,"metrics":{m:{k:v for k,v in cv["summary"][m].items() if k!="per_class"} for m in cv["summary"]},"role_order":order},indent=2))

if __name__=="__main__":main()

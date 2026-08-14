from __future__ import annotations
import argparse, csv, hashlib, json, math, pickle, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

CANON_SHA256 = "dbf87cf5525e065da881b06a26c9d411543ff8ef3f5f8e15a9e4b557808f1174"
FEATURE_COLUMNS = [
    "TEXT_ORDER::adjacent_mi",
    "TEXT_ENTROPY::red1",
    "TEXT_ENTROPY::red2",
    "TEXT_EDIT::ed1_density",
    "TEXT_PERSIST::midfix_lag1",
    "TEXT_PERSIST::suffix_lag1",
    "LEXICAL::hapax",
    "LEXICAL::type_token",
    "PAGE::between_page_overlap",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def folio_number(x: str):
    m = re.match(r"^f(\d+)", str(x))
    return int(m.group(1)) if m else None


def load_manifest(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj["rows"]
    by_num, folds = {}, {}
    for r in rows:
        b = r["bifolium"]
        folds[b] = int(r["fold"])
        m = re.fullmatch(r"B(\d{3})_(\d{3})", b)
        if not m:
            raise ValueError(f"bad bifolium id {b}")
        for n in map(int, m.groups()):
            if n in by_num and by_num[n] != b:
                raise ValueError(f"folio {n} assigned twice")
            by_num[n] = b
    return rows, by_num, folds


def line_key(r):
    return (str(r.get("folio", "")), str(r.get("line_no", "")))


def entropy(counter: Counter) -> float:
    n = sum(counter.values())
    if not n:
        return float("nan")
    return -sum((v/n) * math.log2(v/n) for v in counter.values() if v)


def conditional_entropy(lines, order: int) -> float:
    joint, ctx = Counter(), Counter()
    for seq in lines:
        if len(seq) <= order:
            continue
        for i in range(order, len(seq)):
            c = tuple(seq[i-order:i])
            joint[c + (seq[i],)] += 1
            ctx[c] += 1
    n = sum(joint.values())
    if not n:
        return float("nan")
    out = 0.0
    for key, v in joint.items():
        c = key[:-1]
        out -= (v/n) * math.log2(v / ctx[c])
    return out


def red_features(token_lines):
    char_lines = []
    uni = Counter()
    for toks in token_lines:
        seq = list(" ".join(toks))
        if seq:
            char_lines.append(seq)
            uni.update(seq)
    h0 = entropy(uni)
    h1 = conditional_entropy(char_lines, 1)
    h2 = conditional_entropy(char_lines, 2)
    red1 = (h0-h1)/h0 if h0 and math.isfinite(h0) and math.isfinite(h1) else float("nan")
    red2 = (h1-h2)/h1 if h1 and math.isfinite(h1) and math.isfinite(h2) else float("nan")
    return red1, red2, sum(map(len, char_lines))


def mutual_information(pairs):
    if not pairs:
        return float("nan")
    xy, x, y = Counter(pairs), Counter(a for a,_ in pairs), Counter(b for _,b in pairs)
    n = len(pairs)
    mi = 0.0
    for (a,b), v in xy.items():
        pxy = v/n
        mi += pxy * math.log2(pxy / ((x[a]/n)*(y[b]/n)))
    return mi


def corrected_adjacent_mi(token_lines, vocab, seed, shuffles=64):
    mapped = [[t if t in vocab else "<OTHER>" for t in line] for line in token_lines]
    obs_pairs = [(a,b) for line in mapped for a,b in zip(line, line[1:])]
    obs = mutual_information(obs_pairs)
    if not math.isfinite(obs):
        return float("nan"), 0
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(shuffles):
        pairs = []
        for line in mapped:
            if len(line) < 2:
                continue
            z = list(line)
            rng.shuffle(z)
            pairs.extend(zip(z, z[1:]))
        null.append(mutual_information(pairs))
    null = [x for x in null if math.isfinite(x)]
    return obs - float(np.mean(null)), len(obs_pairs)


def lev1(a: str, b: str) -> bool:
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la-lb) > 1:
        return False
    if la == lb:
        mism = 0
        for x,y in zip(a,b):
            if x != y:
                mism += 1
                if mism > 1:
                    return False
        return mism == 1
    if la > lb:
        a,b,la,lb = b,a,lb,la
    i=j=0; edits=0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1; j += 1
        else:
            edits += 1; j += 1
            if edits > 1:
                return False
    edits += (lb-j)
    return edits == 1


def ed1_density(page_tokens):
    num = den = 0
    for toks in page_tokens.values():
        n = len(toks)
        den_p = n*(n-1)//2
        if den_p <= 0:
            continue
        c = Counter(toks)
        types = sorted(c)
        num_p = 0
        for i,a in enumerate(types):
            ca = c[a]
            for b in types[i+1:]:
                if abs(len(a)-len(b)) <= 1 and lev1(a,b):
                    num_p += ca*c[b]
        num += num_p; den += den_p
    return (num/den if den else float("nan")), den


def page_overlap(page_tokens):
    sets = [set(v) for v in page_tokens.values() if v]
    vals=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            u=sets[i] | sets[j]
            if u:
                vals.append(len(sets[i] & sets[j]) / len(u))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--fold-manifest", type=Path, required=True)
    ap.add_argument("--persistence", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a=ap.parse_args()

    got=sha256(a.records)
    if got != CANON_SHA256:
        raise SystemExit(f"canonical record SHA mismatch: {got}")
    with a.records.open("rb") as f:
        records=pickle.load(f)
    if len(records) != 37465:
        raise SystemExit(f"expected 37465 records, got {len(records)}")

    manifest, by_num, folds = load_manifest(a.fold_manifest)
    persist={}
    with a.persistence.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f): persist[r["bifolium"]]=r

    # Global top-100 token vocabulary fixed once for all physical units.
    global_counts=Counter(str(r.get("token", "")) for r in records if r.get("token"))
    vocab={t for t,_ in global_counts.most_common(100)}

    grouped=defaultdict(list); unassigned=Counter()
    for r in records:
        fol=str(r.get("folio", "")); n=folio_number(fol)
        if fol == "ros": b="B085_086"  # rosette foldout belongs to the 85/86 physical unit
        else: b=by_num.get(n)
        if b is None:
            unassigned[fol]+=1
            continue
        grouped[b].append(r)

    outrows=[]; audits=[]
    for ordinal,mr in enumerate(manifest):
        b=mr["bifolium"]; recs=grouped.get(b,[])
        recs=sorted(recs,key=lambda r:(str(r.get("folio","")), int(r.get("line_no",0) or 0), int(r.get("pos",0) or 0)))
        lines=defaultdict(list); pages=defaultdict(list)
        for r in recs:
            tok=str(r.get("token", ""))
            if not tok: continue
            lines[line_key(r)].append(tok)
            pages[str(r.get("folio",""))].append(tok)
        token_lines=list(lines.values()); toks=[t for z in token_lines for t in z]
        ami, adj_n=corrected_adjacent_mi(token_lines,vocab,20260814+ordinal)
        red1,red2,nchars=red_features(token_lines)
        ed1,ed1den=ed1_density(pages)
        ov,ov_n=page_overlap(pages)
        c=Counter(toks); ntypes=len(c)
        hapax=sum(v==1 for v in c.values())/ntypes if ntypes else float("nan")
        ttr=ntypes/len(toks) if toks else float("nan")
        pr=persist.get(b,{})
        def fv(k):
            x=pr.get(k,"")
            return float(x) if x not in ("",None) else float("nan")
        row={
            "bifolium":b,"fold":int(mr["fold"]),"quire":"SEALED","currier":"SEALED","hand":"SEALED","section":"SEALED",
            "TEXT_ORDER::adjacent_mi":ami,"TEXT_ENTROPY::red1":red1,"TEXT_ENTROPY::red2":red2,
            "TEXT_EDIT::ed1_density":ed1,"TEXT_PERSIST::midfix_lag1":fv("midfix_lag1"),"TEXT_PERSIST::suffix_lag1":fv("suffix_lag1"),
            "LEXICAL::hapax":hapax,"LEXICAL::type_token":ttr,"PAGE::between_page_overlap":ov,
        }
        outrows.append(row)
        audits.append({"bifolium":b,"fold":int(mr["fold"]),"n_tokens":len(toks),"n_chars":nchars,"n_pages":len(pages),"adjacent_pairs":adj_n,"ed1_pair_denominator":ed1den,"page_overlap_pairs":ov_n,"midfix_n":int(pr.get("midfix_n") or 0),"suffix_n":int(pr.get("suffix_n") or 0)})

    a.out.mkdir(parents=True,exist_ok=True)
    matrix=a.out/"U3_FEATURE_MATRIX.csv"
    fields=["bifolium","fold","quire","currier","hand","section"]+FEATURE_COLUMNS
    with matrix.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in outrows:
            w.writerow({k:("" if isinstance(r[k],float) and not math.isfinite(r[k]) else r[k]) for k in fields})
    with (a.out/"U3_FEATURE_AUDIT.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(audits[0]));w.writeheader();w.writerows(audits)
    audit={"formal_status":"FEATURE_BUILD_ONLY_TARGET_NOT_MODELLED","canonical_sha256":got,"records":len(records),"bifolia":len(outrows),"feature_columns":FEATURE_COLUMNS,"global_top100":sorted(vocab),"unassigned_records":sum(unassigned.values()),"unassigned_folios":dict(sorted(unassigned.items()))}
    (a.out/"U3_FEATURE_BUILD.json").write_text(json.dumps(audit,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ("formal_status","canonical_sha256","records","bifolia","unassigned_records")},indent=2))

if __name__=="__main__": main()

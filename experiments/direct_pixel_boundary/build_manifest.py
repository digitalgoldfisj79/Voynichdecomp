#!/usr/bin/env python3
"""Build label-blind direct-pixel boundary manifests from public Voynich sources.

Scientific separation:
- this stage may see ZL separator labels to write the sealed key;
- the blind manifest contains no separator label or token string;
- the downstream measurement job receives only the blind artifact.

Coordinates are Voynichese.com 636x900 runtime word boxes registered onto a
2500-pixel-wide Yale IIIF derivative with SIFT + CLAHE + USAC_MAGSAC.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, re, sys, time, urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

VOYNICHESE = "https://www.voynichese.com/1/data/folio"
ZL_URL = "https://www.voynich.nu/data/previous/ZL_ivtff_2b.txt"
YALE_MANIFEST = "https://collections.library.yale.edu/manifests/2002046"
REG_WIDTH = 2500
UA = "Voynich-boundary-scale-replication/0.1 (+https://github.com/digitalgoldfisj79/Voynichdecomp)"


def get(url: str, retries: int = 4, timeout: int = 90) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def base_label(folio: str) -> str:
    s = folio[1:] if folio.startswith("f") else folio
    s = re.sub(r"([rv])[1-9]$", r"\1", s)
    return s


def label_text(x) -> str:
    if isinstance(x, dict):
        for k in ("none", "en"):
            if k in x and x[k]:
                return str(x[k][0])
        for v in x.values():
            if v:
                return str(v[0])
    return str(x or "")


def yale_index(manifest: dict):
    out = defaultdict(list)
    for c in manifest.get("items", []):
        lab = label_text(c.get("label"))
        try:
            body = c["items"][0]["items"][0]["body"]
            service = body.get("service", [{}])[0].get("@id") or body.get("service", [{}])[0].get("id")
            if not service:
                continue
            out[lab].append({
                "label": lab,
                "service": service.rstrip("/"),
                "source_width": int(body.get("width", c.get("width", 0))),
                "source_height": int(body.get("height", c.get("height", 0))),
            })
        except Exception:
            continue
    return out


def candidate_canvases(idx, folio: str):
    b = base_label(folio)
    cands = list(idx.get(b, []))
    if cands:
        return cands
    for lab, rows in idx.items():
        if re.search(rf"(^|\D){re.escape(b)}($|\D)", lab):
            cands.extend(rows)
    return cands


def parse_runtime_js(raw: bytes):
    obj = json.loads(raw.decode("utf-8"))
    words, boxes = obj[0], obj[1]
    vocab = [str(r[0]) for r in words]
    out = []
    for j, b in enumerate(boxes):
        wi, x, y, w, h = b[:5]
        out.append({"box_index": j, "word": vocab[int(wi)], "x": float(x), "y": float(y),
                    "w": float(w), "h": float(h)})
    return out


def clean_token(tok: str) -> str:
    prev = None
    while prev != tok:
        prev = tok
        tok = re.sub(r"\[([^:\]]+):[^\]]*\]", r"\1", tok)
    tok = re.sub(r"<![^>]*>", "", tok)
    tok = re.sub(r"<[^>]*>", "", tok)
    tok = tok.replace("<%>", "").replace("<$>", "")
    tok = re.sub(r"\{([^}]*)\}", lambda m: re.sub(r"[^A-Za-z]", "", m.group(1)), tok)
    tok = re.sub(r"@\d+;", "", tok)
    tok = tok.replace("'", "")
    tok = re.sub(r"[^A-Za-z?]", "", tok)
    return tok.lower()


def parse_zl(text: str):
    """Return per-folio flattened running-text tokens and labelled intra-line .,/ boundaries."""
    pages = defaultdict(lambda: {"tokens": [], "boundaries": [], "meta": {}})
    header_re = re.compile(r"^<(?P<folio>f[^>\s]+)>\s*<!\s*(?P<meta>[^>]*)>")
    line_re = re.compile(r"^<(?P<loc>f[^.>]+\.(?P<line>[^,>]+)),(?P<role>[^>]+)>\s*(?P<text>.*)$")
    for rawline in text.splitlines():
        hm = header_re.match(rawline)
        if hm:
            fol = hm.group("folio")
            meta = dict(re.findall(r"\$(\w)=([^\s>]+)", hm.group("meta")))
            pages[fol]["meta"].update(meta)
            continue
        m = line_re.match(rawline)
        if not m:
            continue
        fol = m.group("loc").split(".", 1)[0]
        role = m.group("role")
        if "P" not in role:
            continue
        s = m.group("text")
        s = s.replace("<%>", "").replace("<$>", "")
        s = s.replace("<->", "§").replace("<~>", "§")
        s = re.sub(r"<![^>]*>", "", s)
        seq = []
        for q in re.split(r"([.,§])", s):
            if q in (".", ",", "§"):
                seq.append(("sep", q))
            else:
                t = clean_token(q.strip())
                if t:
                    seq.append(("tok", t))
        toks_local = []
        bound_local = []
        pending_sep = None
        for kind, val in seq:
            if kind == "tok":
                if toks_local and pending_sep in (".", ","):
                    bound_local.append((len(toks_local) - 1, pending_sep))
                toks_local.append(val)
                pending_sep = None
            else:
                pending_sep = val
        if not toks_local:
            continue
        start = len(pages[fol]["tokens"])
        pages[fol]["tokens"].extend(toks_local)
        for li, sep in bound_local:
            pages[fol]["boundaries"].append({
                "zleft": start + li,
                "zright": start + li + 1,
                "line": str(m.group("line")),
                "label": sep,
            })
    return pages


def prep_gray(im: Image.Image):
    a = np.array(im.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(a)


def register(legacy: Image.Image, target: Image.Image):
    a, b = prep_gray(legacy), prep_gray(target)
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(da, db, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0, maxIters=10000, confidence=0.999)
    if H is None or mask is None:
        return None
    inl = mask.ravel().astype(bool)
    nin = int(inl.sum())
    if nin < 10:
        return None
    pred = cv2.perspectiveTransform(src[inl], H)
    err = np.sqrt(((pred - dst[inl]) ** 2).sum(axis=2)).ravel()
    corners = np.float32([[[0, 0], [legacy.width, 0], [legacy.width, legacy.height], [0, legacy.height]]])
    tc = cv2.perspectiveTransform(corners, H)[0]
    if not np.isfinite(tc).all():
        return None
    return {
        "H": H,
        "matches": len(good),
        "inliers": nin,
        "inlier_ratio": nin / len(good),
        "median_reproj": float(np.median(err)),
        "p90_reproj": float(np.quantile(err, .9)),
        "corners": tc.tolist(),
    }


def transform_box(H, b):
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    pts = np.float32([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]])
    q = cv2.perspectiveTransform(pts, H)[0]
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def align_boundaries(zpage, boxes):
    zw = zpage["tokens"]
    bw = [b["word"].lower() for b in boxes]
    sm = SequenceMatcher(None, zw, bw, autojunk=False)
    mapping, block_id = {}, {}
    for bi, block in enumerate(sm.get_matching_blocks()):
        if block.size < 2:
            continue
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
            block_id[block.a + k] = bi
    rows = []
    for ev in zpage["boundaries"]:
        zl, zr = ev["zleft"], ev["zright"]
        if zl not in mapping or zr not in mapping or block_id.get(zl) != block_id.get(zr):
            continue
        bl, br = mapping[zl], mapping[zr]
        if br != bl + 1:
            continue
        l, r = boxes[bl], boxes[br]
        if r["x"] <= l["x"]:
            continue
        lc, rc = l["y"] + .5 * l["h"], r["y"] + .5 * r["h"]
        if abs(lc - rc) > 0.80 * (l["h"] + r["h"]):
            continue
        rows.append((ev, l, r, bl, br))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    zl_b = get(ZL_URL)
    zl = parse_zl(zl_b.decode("utf-8", errors="replace"))
    ym_b = get(YALE_MANIFEST)
    ym = json.loads(ym_b)
    yi = yale_index(ym)
    folios = sorted(zl.keys(), key=lambda s: (int(re.search(r"\d+", s).group()), s))
    folios = [f for i, f in enumerate(folios) if i % args.nshards == args.shard]

    blind_rows, key_rows, reg_rows = [], [], []
    counts = defaultdict(int)
    counts["zl_sha256"] = sha256_bytes(zl_b)
    counts["yale_manifest_sha256"] = sha256_bytes(ym_b)

    for fol in folios:
        counts["raw_boundaries"] += len(zl[fol]["boundaries"])
        counts["raw_uncertain"] += sum(x["label"] == "," for x in zl[fol]["boundaries"])
        try:
            js_b = get(f"{VOYNICHESE}/script/{fol}.js")
            boxes = parse_runtime_js(js_b)
            if not boxes:
                raise RuntimeError("no runtime boxes")
            legacy_b = get(f"{VOYNICHESE}/image/glance/color/large/{fol}.jpg")
            legacy_path = cache / f"legacy_{fol}.jpg"
            legacy_path.write_bytes(legacy_b)
            legacy = Image.open(legacy_path).convert("RGB")
            cands = candidate_canvases(yi, fol)
            if not cands:
                raise RuntimeError(f"no Yale canvas candidate for {base_label(fol)}")
            best = None
            for c in cands:
                svc = c["service"]
                safe = re.sub(r"\D", "", svc.split("/")[-1]) or hashlib.sha1(svc.encode()).hexdigest()[:12]
                tp = cache / f"yale_{safe}_{REG_WIDTH}.jpg"
                if not tp.exists():
                    tp.write_bytes(get(f"{svc}/full/{REG_WIDTH},/0/default.jpg"))
                target = Image.open(tp).convert("RGB")
                rr = register(legacy, target)
                if rr is None:
                    continue
                score = (rr["inlier_ratio"], -rr["median_reproj"], rr["inliers"])
                if best is None or score > best[0]:
                    best = (score, c, target.size, rr)
            if best is None:
                raise RuntimeError("registration failed all candidates")
            _, c, (tw, th), rr = best
            passed = rr["inliers"] >= 12 and rr["inlier_ratio"] >= .35 and rr["median_reproj"] <= 4.0
            reg_rows.append({
                "folio": fol,
                "base_label": base_label(fol),
                "yale_label": c["label"],
                "iiif_service": c["service"],
                "target_w": tw,
                "target_h": th,
                "matches": rr["matches"],
                "inliers": rr["inliers"],
                "inlier_ratio": rr["inlier_ratio"],
                "median_reproj": rr["median_reproj"],
                "p90_reproj": rr["p90_reproj"],
                "passed": passed,
                "H_json": json.dumps(np.asarray(rr["H"]).tolist(), separators=(",", ":")),
            })
            if not passed:
                continue
            aligned = align_boundaries(zl[fol], boxes)
            counts["aligned_boundaries"] += len(aligned)
            counts["aligned_uncertain"] += sum(ev["label"] == "," for ev, *_ in aligned)
            for ev, lb, rb, bli, bri in aligned:
                lx0, ly0, lx1, ly1 = transform_box(rr["H"], lb)
                rx0, ry0, rx1, ry1 = transform_box(rr["H"], rb)
                raw_id = f"v0.1|{fol}|{ev['line']}|{ev['zleft']}|{bli}|{bri}"
                blind_id = hashlib.sha256(raw_id.encode()).hexdigest()[:20]
                blind_rows.append({
                    "blind_id": blind_id,
                    "folio": fol,
                    "line": ev["line"],
                    "iiif_service": c["service"],
                    "reg_width": tw,
                    "reg_height": th,
                    "lx0": lx0,
                    "lx1": lx1,
                    "ly0": ly0,
                    "ly1": ly1,
                    "rx0": rx0,
                    "rx1": rx1,
                    "ry0": ry0,
                    "ry1": ry1,
                    "reg_inlier_ratio": rr["inlier_ratio"],
                    "reg_median_reproj": rr["median_reproj"],
                })
                meta = zl[fol].get("meta", {})
                key_rows.append({
                    "blind_id": blind_id,
                    "folio": fol,
                    "line": ev["line"],
                    "label": ev["label"],
                    "left_token": lb["word"],
                    "right_token": rb["word"],
                    "box_left_index": bli,
                    "box_right_index": bri,
                    "quire": meta.get("Q", ""),
                    "hand": meta.get("H", ""),
                })
        except Exception as e:
            reg_rows.append({"folio": fol, "base_label": base_label(fol), "passed": False, "error": repr(e)})
            print(f"ERROR {fol}: {e}", file=sys.stderr, flush=True)

    blind_path = out / f"blind_manifest_s{args.shard}.csv"
    key_path = out / f"sealed_key_s{args.shard}.csv"
    reg_path = out / f"registration_s{args.shard}.csv"
    blind_cols = [
        "blind_id", "folio", "line", "iiif_service", "reg_width", "reg_height",
        "lx0", "lx1", "ly0", "ly1", "rx0", "rx1", "ry0", "ry1",
        "reg_inlier_ratio", "reg_median_reproj",
    ]
    with blind_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=blind_cols)
        w.writeheader()
        w.writerows(blind_rows)
    key_cols = [
        "blind_id", "folio", "line", "label", "left_token", "right_token",
        "box_left_index", "box_right_index", "quire", "hand",
    ]
    with key_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=key_cols)
        w.writeheader()
        w.writerows(key_rows)
    reg_cols = sorted(set().union(*(r.keys() for r in reg_rows))) if reg_rows else ["folio"]
    with reg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=reg_cols)
        w.writeheader()
        w.writerows(reg_rows)

    header = blind_path.read_text(encoding="utf-8").splitlines()[0].lower() if blind_rows else ",".join(blind_cols)
    forbidden = [x for x in ("label", "certain", "uncertain", "left_token", "right_token", "group") if x in header]
    if forbidden:
        raise SystemExit(f"BLINDING FAILURE: forbidden manifest columns {forbidden}")
    freeze = {
        "schema": "direct-pixel-boundary-freeze-v0.1",
        "shard": args.shard,
        "nshards": args.nshards,
        "blind_manifest_sha256": sha256_bytes(blind_path.read_bytes()),
        "sealed_key_sha256": sha256_bytes(key_path.read_bytes()),
        "registration_sha256": sha256_bytes(reg_path.read_bytes()),
        "blind_rows": len(blind_rows),
        "key_rows": len(key_rows),
        "counts": dict(counts),
        "registration_gate": {"min_inliers": 12, "min_inlier_ratio": .35, "max_median_reproj_px": 4.0},
        "registration_width": REG_WIDTH,
    }
    (out / f"freeze_s{args.shard}.json").write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

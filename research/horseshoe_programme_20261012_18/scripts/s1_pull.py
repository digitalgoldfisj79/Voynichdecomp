#!/usr/bin/env python3
"""Stage 1: pull all v3 Atlas word embeddings (public tiles, no login) and join metadata.

Output (pickled, atomic): data/s1_words.pkl = dict(ids, X[float32 n x 768], meta DataFrame).
"""
import json, os, pickle, sys, urllib.request
import numpy as np, pandas as pd, pyarrow.feather as feather

ROOT = sys.argv[1]                      # e.g. /home/claude/horseshoe
NOMIC = sys.argv[2]                     # dir with manifest.feather + atlas_v3_full.parquet
LINKS = sys.argv[3]                     # vdino3_word_iiif_links.csv
TILE_DIR = os.path.join(ROOT, "data", "tiles_emb")
OUT = os.path.join(ROOT, "data", "s1_words.pkl")
os.makedirs(TILE_DIR, exist_ok=True)

P = "aa6efade-3e8d-4026-a931-1887cff58b81"
J = "e0400e7c-1d16-4f6a-a60a-671bbf691c74"
B = f"https://api-atlas.nomic.ai/v1/project/{P}/index/projection/{J}/quadtree"


def atomic_pickle(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


if os.path.exists(OUT):
    print("stage 1 already done:", OUT); sys.exit(0)

man = feather.read_table(os.path.join(NOMIC, "manifest.feather"))
side = json.loads(man.schema.metadata[b"sidecars"])
ids, vecs = [], []
for key in man["key"].to_pylist():
    k = key.replace("/", "_")
    p_emb = os.path.join(TILE_DIR, f"{k}.emb.feather")
    p_id = os.path.join(TILE_DIR, f"{k}.id.feather")
    if not os.path.exists(p_emb):
        urllib.request.urlretrieve(f"{B}/{key}.{side['_embeddings']}.feather", p_emb)
    if not os.path.exists(p_id):
        urllib.request.urlretrieve(f"{B}/{key}.{side['id']}.feather", p_id)
    e = feather.read_table(p_emb).column(0)
    i = feather.read_table(p_id).column(0).to_pylist()
    arr = np.asarray(e.combine_chunks().flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
    arr = arr.reshape(len(i), 768)
    assert len(i) == arr.shape[0]
    ids.extend(i); vecs.append(arr)
X = np.concatenate(vecs, 0)
assert len(ids) == len(set(ids)) == X.shape[0] == 37886, (len(ids), X.shape)

atl = pd.read_parquet(os.path.join(NOMIC, "atlas_v3_full.parquet"))
atl = atl[["id", "x", "y", "folio", "word_index", "line_num", "section_fixed", "currier",
           "scribe_davis", "quire"]].rename(columns={"x": "umap_x", "y": "umap_y"})
lk = pd.read_csv(LINKS, usecols=["word_id", "word", "w", "h", "pad_w", "pad_h"])
meta = pd.DataFrame({"id": ids}).merge(atl, on="id", how="left").merge(
    lk, left_on="id", right_on="word_id", how="left")
assert meta["folio"].notna().all() and meta["w"].notna().all(), "join gap"
assert (meta["id"].values == np.array(ids)).all()
norms = np.linalg.norm(X, axis=1)
print("X", X.shape, "norm min/max", norms.min().round(4), norms.max().round(4))
print("folios", meta.folio.nunique(), "| sections", meta.section_fixed.value_counts().to_dict())
print("currier", meta.currier.value_counts(dropna=False).to_dict())
atomic_pickle(dict(ids=ids, X=X, meta=meta), OUT)
print("wrote", OUT)

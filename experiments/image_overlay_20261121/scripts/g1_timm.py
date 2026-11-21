"""Glyph stage 1: parse Timm's Archetype dump -> graph boxes in Yale full-res image pixels (y top-down).
Usage: python3 -I g1_timm.py <dl_dir> <out_dir>   -> out/g1_timm.pkl
"""
import json, os, pickle, sys, collections
DL, OUT = sys.argv[1], sys.argv[2]
SQL = os.path.join(DL, "timm", "proj", "digipal_project", "archetype.sql")
lines = open(SQL, encoding="utf-8", errors="replace").read().split("\n")
def load(table):
    for i, l in enumerate(lines):
        if l.startswith("COPY %s (" % table):
            cols = [c.strip().strip('"') for c in l[l.index("(") + 1:l.rindex(")")].split(",")]
            rows = []
            for l2 in lines[i + 1:]:
                if l2 == "\\.": break
                rows.append(dict(zip(cols, l2.split("\t"))))
            return rows
    raise KeyError(table)
ann, graph, idio, allo, img = (load("digipal_" + t) for t in ("annotation", "graph", "idiograph", "allograph", "image"))
allo_n = {a["id"]: a["name"] for a in allo}
idio_a = {i["id"]: allo_n.get(i["allograph_id"], "?") for i in idio}
g_allo = {g["id"]: idio_a.get(g["idiograph_id"], "?") for g in graph}
im = {i["id"]: i for i in img}
out = []
for a in ann:
    if a["graph_id"] in ("\\N", ""): continue
    try:
        gj = json.loads(a["geo_json"].replace("'", '"'))
    except Exception:
        continue
    geom = gj.get("geometry", gj); ring = geom["coordinates"][0]
    xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
    I = im[a["image_id"]]; W, H = int(I["width"]), int(I["height"])
    out.append(dict(allograph=g_allo.get(a["graph_id"], "?"), locus=I["locus"], label=I["display_label"], W=W, H=H,
                    x0=min(xs), x1=max(xs), y0=H - max(ys), y1=H - min(ys), created=a["created"][:10]))
pickle.dump(out, open(os.path.join(OUT, "g1_timm.pkl"), "wb"))
print("graphs", len(out), collections.Counter(o["allograph"] for o in out))
print("loci sample", sorted({o["locus"] for o in out})[:25], "n loci", len({o["locus"] for o in out}))
print("example", out[0])

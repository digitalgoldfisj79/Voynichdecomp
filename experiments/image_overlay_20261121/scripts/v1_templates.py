"""v4 step 1: collect template samples from UNFLAGGED glyphs (no labels used) on drawn, non-circular folios.
Usage: python3 -I v1_templates.py <dl_dir> <out_dir> k n   -> out/v4/tmpl_part_k.pkl  (atomic per part)"""
import json, os, pickle, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkglyph as IG
from k3_lib import readings_for
DL, OUT = sys.argv[1], sys.argv[2]; K, N = int(sys.argv[3]), int(sys.argv[4])
V = os.path.join(OUT, "v4"); os.makedirs(V, exist_ok=True)
J = json.load(open(os.path.join(OUT, "glyph_width_table.json")))
SP = json.load(open(os.path.join(OUT, "split_v4.json")))
fos = sorted(SP["dev"] + SP["test"])[K::N]; samples = {}
for fo in fos:
    words, im = readings_for(DL, OUT, fo)
    ctx = IG.analyse_page(im, [b for _, b in words])
    for (wd, _), mem in zip(words, ctx["members"]):
        IG.word_glyphs(ctx, mem, wd, J["widths"], J["default"], collect=samples)
    print(fo, sum(len(v) for v in samples.values()), flush=True)
pickle.dump(samples, open(os.path.join(V, "tmpl_part_%d.pkl" % K), "wb"))

"""Step 2: draw Ed's 30-glyph spot-check (seed 20261123) from the 240 judged items; sheets carry no verdicts.
Usage: python3 -I m3_spotcheck.py <out_dir>  -> out/step2/spotcheck_ids.json, out/step2/spotcheck_1.jpg .. _4.jpg"""
import json, os, random, sys
from PIL import Image
OUT = sys.argv[1]; S = os.path.join(OUT, "step2")
items = json.load(open(os.path.join(S, "step2_sample.json")))["items"]
ids = sorted(random.Random(20261123).sample([it["id"] for it in items], 30))
json.dump(ids, open(os.path.join(S, "spotcheck_ids.json"), "w"))
cells = []
for i in ids:
    sheet = Image.open(os.path.join(S, "sheet_%02d.jpg" % (i // 8))); j = i % 8
    cells.append(sheet.crop(((j % 2) * 700, (j // 2) * 190, (j % 2) * 700 + 700, (j // 2) * 190 + 190)))
for p in range(0, 30, 8):
    out = Image.new("RGB", (1400, 760), (200, 200, 200))
    for k, c in enumerate(cells[p:p + 8]): out.paste(c, ((k % 2) * 700, (k // 2) * 190))
    out.save(os.path.join(S, "spotcheck_%d.jpg" % (p // 8 + 1)), quality=90)
print(ids)

"""Stage 5: draw arm-B boxes on the cached Yale derivatives for eyeballing.
Usage: python3 -I s5_render_check.py <dl_dir> <out_dir> folio [folio ...]"""
import json, os, pickle, sys
from PIL import Image, ImageDraw
DL, OUT = sys.argv[1], sys.argv[2]
ARMS = pickle.load(open(os.path.join(OUT, "s3_arms.pkl"), "rb"))
S2 = pickle.load(open(os.path.join(OUT, "s2_mapped.pkl"), "rb"))["mapped"]
reg = {json.loads(l)["selected"]["folio"]: json.loads(l)["selected"] for l in open(os.path.join(DL, "reg_full_225.jsonl")) if l.strip()}
for fo in sys.argv[3:]:
    im = Image.open(os.path.join(DL, "img1000", reg[fo]["service_id"].rsplit("/", 1)[1] + ".jpg")).convert("RGB")
    k = max(im.size) / max(S2[fo]["full_wh"]); d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in ARMS[fo]["B"] * k:
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=1)
    b = ARMS[fo]["B"] * k
    pad = 30
    box = (max(0, int(b[:, 0].min()) - pad), max(0, int(b[:, 1].min()) - pad), min(im.size[0], int(b[:, 2].max()) + pad), min(im.size[1], int(b[:, 3].max()) + pad))
    im.crop(box).save(os.path.join(OUT, "check_%s.png" % fo)); print(fo, im.size, box)

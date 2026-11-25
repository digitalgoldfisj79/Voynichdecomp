"""Share pack: Voynichese Project word boxes re-projected onto the Yale IIIF canvases, as a flat CSV in
full-resolution canvas pixels, plus README / NOTICE / LICENSE.
Usage: python3 -I x1_share_pack.py <dl_dir> <out_dir>  -> out/share/voynichese_yale_word_boxes_v1/"""
import csv, json, os, pickle, shutil, sys
DL, OUT = sys.argv[1], sys.argv[2]
P = os.path.join(OUT, "share", "voynichese_yale_word_boxes_v1"); os.makedirs(P, exist_ok=True)
S1 = pickle.load(open(os.path.join(OUT, "s1_boxes.pkl"), "rb"))
S2 = pickle.load(open(os.path.join(OUT, "s2_mapped.pkl"), "rb"))["mapped"]
ARMS = pickle.load(open(os.path.join(OUT, "s3_arms.pkl"), "rb"))
AL = pickle.load(open(os.path.join(OUT, "s6_align.pkl"), "rb"))
PK = {r["folio"]: r for r in csv.DictReader(open(os.path.join(OUT, "g2b_peak_by_folio_1000.csv")))}
reg = {json.loads(l)["selected"]["folio"]: json.loads(l)["selected"] for l in open(os.path.join(DL, "reg_full_225.jsonl")) if l.strip()}
cols = ["folio", "voynichese_index", "voynichese_word", "zl_line", "yale_canvas_label", "iiif_service", "canvas_w", "canvas_h",
        "x", "y", "w", "h", "iiif_crop_url", "folio_placement_check", "folio_reg_median_err_px"]
rows = []; dropped = {"outside_registered_area": 0, "zero_size_after_clip": 0}; folios = []
for fo in S1["folios"]:
    X = S1["xml_boxes"][fo]; B = ARMS[fo]["B"]; fw, fh = S2[fo]["full_wh"]; r = reg[fo]; sid = r["service_id"]
    chk = PK[fo]["verdict_B"].split(" (")[0].replace("Inconclusive at this power", "unchecked")
    err = round(r["median_reproj_px"] / r["deriv_scale"], 1); n0 = len(rows)
    for i, x in enumerate(X):
        if not AL[fo]["placed"][i]: dropped["outside_registered_area"] += 1; continue
        x0, y0, x1, y1 = max(0.0, B[i][0]), max(0.0, B[i][1]), min(fw, B[i][2]), min(fh, B[i][3])
        if x1 - x0 < 1 or y1 - y0 < 1: dropped["zero_size_after_clip"] += 1; continue
        xi, yi, wi, hi = int(x0), int(y0), int(round(x1 - x0)), int(round(y1 - y0))
        ln = AL[fo]["line"][i]
        rows.append(dict(folio=fo, voynichese_index=x[0], voynichese_word=x[1], zl_line=ln if ln else "", yale_canvas_label=r["canvas_label"],
                         iiif_service=sid, canvas_w=fw, canvas_h=fh, x=xi, y=yi, w=wi, h=hi,
                         iiif_crop_url="%s/%d,%d,%d,%d/full/0/default.jpg" % (sid, xi, yi, wi, hi),
                         folio_placement_check=chk, folio_reg_median_err_px=err))
    folios.append(dict(folio=fo, boxes=len(rows) - n0, check=chk, err=err, canvas=r["canvas_label"]))
with open(os.path.join(P, "voynichese_yale_word_boxes.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
with open(os.path.join(P, "folio_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(folios[0])); w.writeheader(); w.writerows(folios)
shutil.copy(os.path.join(DL, "reg_full_225.jsonl"), os.path.join(P, "registration_homographies.jsonl"))
import collections
cc = collections.Counter(f["check"] for f in folios)
print("rows", len(rows), "folios", len(folios), "dropped", dropped, "checks", dict(cc))
json.dump(dict(rows=len(rows), folios=len(folios), dropped=dropped, checks=dict(cc)), open(os.path.join(P, "_stats.json"), "w"))

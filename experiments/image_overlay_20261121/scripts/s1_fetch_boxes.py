"""Stage 1: fetch voynichese runtime boxes (636x900 frame) + parse the XML zip (1090x1500 frame).

Atomic: one cached .js per folio; result pickled to out/s1_boxes.pkl.
Usage: python3 -I s1_fetch_boxes.py <dl_dir> <out_dir>
"""
import io, json, os, pickle, sys, time, urllib.request, zipfile
import xml.etree.ElementTree as ET

DL, OUT = sys.argv[1], sys.argv[2]
SCRIPT_URL = "https://www.voynichese.com/1/data/folio/script/{folio}.js"
ZIP_PATH = os.path.join(DL, "voynichese_data.zip.out")
JS_DIR = os.path.join(DL, "js")
os.makedirs(JS_DIR, exist_ok=True)

z = zipfile.ZipFile(ZIP_PATH)
folios = sorted(n[len("data/"):-4] for n in z.namelist() if n.endswith(".xml"))
print("folios in zip:", len(folios))

xml_boxes, xml_frame, js_boxes, errors = {}, {}, {}, {}
for fo in folios:
    root = ET.fromstring(z.read(f"data/{fo}.xml"))
    xml_frame[fo] = (int(root.get("width")), int(root.get("height")))
    xml_boxes[fo] = [(int(w.get("index")), (w.text or "").strip(), int(w.get("x")), int(w.get("y")),
                      int(w.get("width")), int(w.get("height"))) for w in root.findall("word")]
    p = os.path.join(JS_DIR, fo + ".js")
    if not os.path.exists(p):
        try:
            req = urllib.request.Request(SCRIPT_URL.format(folio=fo), headers={"User-Agent": "voynichstats-overlay/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                open(p, "wb").write(r.read())
            time.sleep(0.15)
        except Exception as e:
            errors[fo] = str(e)
            continue
    data = json.loads(open(p, "rb").read().decode("utf-8", "replace"))
    table, rects = data[0], data[1]
    js_boxes[fo] = [(j, table[r[0]][0], int(r[1]), int(r[2]), int(r[3]), int(r[4])) for j, r in enumerate(rects)]

pickle.dump(dict(folios=folios, xml_boxes=xml_boxes, xml_frame=xml_frame, js_boxes=js_boxes, errors=errors),
            open(os.path.join(OUT, "s1_boxes.pkl"), "wb"))
print("js folios:", len(js_boxes), "errors:", errors)
print("js words:", sum(len(v) for v in js_boxes.values()), "xml words:", sum(len(v) for v in xml_boxes.values()))
print("xml frames:", sorted(set(xml_frame.values()))[:6], "n distinct", len(set(xml_frame.values())))

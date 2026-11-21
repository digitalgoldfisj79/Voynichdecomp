"""Glyph stage 0: cache one Yale derivative (long side 2000 px) per registered image service.
Usage: python3 -I g0_fetch_images.py <dl_dir>"""
import json, os, sys, time, urllib.request
DL = sys.argv[1]; D = os.path.join(DL, "img2000"); os.makedirs(D, exist_ok=True)
sids = sorted({json.loads(l)["selected"]["service_id"] for l in open(os.path.join(DL, "reg_full_225.jsonl")) if l.strip()})
from concurrent.futures import ThreadPoolExecutor
def get(s):
    p = os.path.join(D, s.rsplit("/", 1)[1] + ".jpg")
    if os.path.exists(p) and os.path.getsize(p) > 10000: return 0
    for attempt in range(3):
        try:
            req = urllib.request.Request(s + "/full/!2000,2000/0/default.jpg", headers={"User-Agent": "voynichstats-overlay/1.0"})
            b = urllib.request.urlopen(req, timeout=120).read(); open(p + ".tmp", "wb").write(b); os.replace(p + ".tmp", p); return 1
        except Exception as e:
            time.sleep(3)
    return 0
with ThreadPoolExecutor(5) as ex: n = sum(ex.map(get, sids))
print("fetched", n, "have", len([f for f in os.listdir(D) if f.endswith(".jpg")]), "of", len(sids))

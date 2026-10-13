"""
Rebuilds assets.js from what is in the image folders. Run it after adding or removing PNGs:

    python tools/make_asset_list.py

  plants/*.png      plant cut-outs, named after their folio:  f1v.png, f44r.png ...
  creatures/*.png   the start of the name says what it does:
                      nymph_...  stands in the pond, dives when you come close
                      fish_...   jumps out of the pond now and then
                      hop_... / creature_...  hops across the path
                      fly_...    flies across the sky
                      guide_...  can be chosen as the guide in walk.config.js
  sky/sun.png, sky/moon.png   optional

The folio number in a file name (f44r, nymph_75r_2) is used to link each picture
to Yale's full-page scan when you click it (see tools/folio_iiif.json).
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IIIF = json.load(open(os.path.join(ROOT, 'tools', 'folio_iiif.json')))

def pngs(folder):
    d = os.path.join(ROOT, folder)
    return sorted(f for f in os.listdir(d) if f.lower().endswith('.png')) if os.path.isdir(d) else []

def folio_of(name):
    m = re.search(r'(\d+[rv]\d?(?:-\d+[rv])?p?)', name)
    return m.group(1) if m else None

plants = []
for f in pngs('plants'):
    fo = folio_of(f[1:] if f.startswith('f') else f)
    info = IIIF.get(fo) or {}
    plants.append({"file": f"plants/{f}", "folio": fo, "section": info.get("section"), "iiif": info.get("iiif")})

ROLE = [('guide', 'guide', 1.5), ('nymph', 'pond', 2.0), ('fish', 'fish', 0.5), ('hop', 'hop', 0.9), ('creature', 'hop', 0.9), ('fly', 'fly', 1.0)]
creatures = []
for f in pngs('creatures'):
    for prefix, role, h in ROLE:
        if f.lower().startswith(prefix):
            fo = folio_of(f)
            info = IIIF.get(fo) or {}
            creatures.append({"file": f"creatures/{f}", "role": role, "height": h, "folio": fo, "section": info.get("section"), "iiif": info.get("iiif")}); break
    else:
        print(f"  (skipped creatures/{f}: name should start with nymph_, fish_, hop_ or fly_)")

sky = pngs('sky')
out = {"plants": plants, "creatures": creatures,
       "sky": {"sun": "sky/sun.png" if "sun.png" in sky else None, "moon": "sky/moon.png" if "moon.png" in sky else None}}
with open(os.path.join(ROOT, 'assets.js'), 'w', encoding='utf-8') as fh:
    fh.write("/* The list of pictures. Made by tools/make_asset_list.py - run it again after adding PNGs.\n"
             "   Pictures cut from Beinecke MS 408 (Beinecke Rare Book & Manuscript Library, Yale University). */\n\n")
    fh.write("export const ASSETS = " + json.dumps(out, indent=1) + ";\n")
print(f"assets.js: {len(plants)} plants, {len(creatures)} creatures")

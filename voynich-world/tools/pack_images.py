"""Pack the cut-out pictures so the world works by double-clicking index.html.

    python tools/pack_images.py

Why: Chrome will not let a page opened straight from your disk (file://) use PNG
files as textures - it blocks them. Pictures written *inside* a .js file are
allowed. So this reads

    plants/*.png    ->  images/plants.js
    nymphs/*.png    ->  images/nymphs.js
    animals/*.png   ->  images/animals.js

shrinking each picture (longest side --size pixels, default 640; this copy uses --size 768) and storing it
as WebP, which keeps the transparency and is about 5x smaller than PNG.

After adding, removing or editing a PNG, run this again and reload the page.
Needs Python 3 and Pillow (pip install pillow).

Options
  --size 640              longest side for every picture
  --big f93r_plant=1024   a sharper copy of one picture (e.g. the giant landmark); repeat for more
"""
import base64, io, json, os, sys
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def args():
    size, big = 640, {}
    a = sys.argv[1:]
    for i, x in enumerate(a):
        if x == '--size': size = int(a[i + 1])
        if x == '--big': k, v = a[i + 1].split('='); big[k] = int(v)
    return size, big


def pack(folder, var, size, big):
    src = os.path.join(ROOT, folder)
    out, info = {}, {}
    for name in sorted(os.listdir(src)):
        if not name.lower().endswith('.png'): continue
        pid = name[:-4]
        im = Image.open(os.path.join(src, name)).convert('RGBA')
        side = big.get(pid, size)
        s = min(1.0, side / max(im.size))
        if s < 1: im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        b = io.BytesIO(); im.save(b, 'WEBP', quality=85, method=4)          # alpha is stored losslessly
        out[pid] = {'w': im.width, 'h': im.height, 'src': 'data:image/webp;base64,' + base64.b64encode(b.getvalue()).decode()}
        info[pid] = round(im.width / im.height, 3)
    os.makedirs(os.path.join(ROOT, 'images'), exist_ok=True)
    path = os.path.join(ROOT, 'images', folder + '.js')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'// Made by tools/pack_images.py from {folder}/*.png - do not edit, run the tool again.\n')
        f.write(f'window.{var} = ' + json.dumps(out) + ';\n')
    print(f'{folder}: {len(out)} pictures -> images/{folder}.js ({os.path.getsize(path) / 1e6:.1f} MB)')
    return info


if __name__ == '__main__':
    size, big = args()
    pack('plants', 'VOYNICH_PLANTS', size, big)
    pack('nymphs', 'VOYNICH_NYMPHS', size, big)
    pack('animals', 'VOYNICH_ANIMALS', size, big)

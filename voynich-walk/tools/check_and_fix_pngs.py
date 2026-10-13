"""
Checks cut-out PNGs and fixes the common problems, so they look clean in the world.

    python tools/check_and_fix_pngs.py plants            (checks and fixes everything in plants/)
    python tools/check_and_fix_pngs.py creatures --halo 2

For each picture it checks:
  - is the transparency real?   (a "transparent" PNG that is really a white or checkerboard
                                   background is the most common problem)
  - are the edges clean?         (a pale or dark fringe, "halo", round the outside)
  - is there empty space round it, is it bigger than needed?

Fixed copies go into a 'fixed' folder next to the originals. Your originals are never
changed unless you add --in-place. Look at the fixed ones, then move them over yourself.

Options:
  --halo N        shave N pixels off the outer edge (default 1; 0 = don't)
  --size N        longest side after fixing (default 1024; 0 = don't resize)
  --key-out       if a picture has NO transparency, try to remove a flat background colour
  --in-place      overwrite the originals (careful)

Needs Python 3 with Pillow and numpy:   pip install pillow numpy
"""
import argparse, os, sys
import numpy as np
from PIL import Image, ImageFilter

ap = argparse.ArgumentParser()
ap.add_argument('folder')
ap.add_argument('--halo', type=int, default=1)
ap.add_argument('--size', type=int, default=1024)
ap.add_argument('--key-out', action='store_true')
ap.add_argument('--in-place', action='store_true')
a = ap.parse_args()

src = a.folder
out = src if a.in_place else os.path.join(src, 'fixed')
os.makedirs(out, exist_ok=True)
files = sorted(f for f in os.listdir(src) if f.lower().endswith('.png'))
if not files: sys.exit(f'No PNG files in {src}')

def key_out(img):
    """Remove a flat background: the colour of the corners, where it touches the border."""
    rgb = np.asarray(img.convert('RGB')).astype(int)
    corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    bg = np.median(corners, axis=0)
    close = (np.abs(rgb - bg).sum(axis=2) < 60)
    from scipy import ndimage                                   # only needed for --key-out
    lbl, _ = ndimage.label(close)
    border = set(np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))) - {0}
    alpha = np.where(np.isin(lbl, list(border)), 0, 255).astype('uint8')
    im = img.convert('RGBA'); im.putalpha(Image.fromarray(alpha)); return im

print(f"{'file':34} {'size':>11}  {'see-through':>11}  {'soft edge':>9}  result")
for f in files:
    img = Image.open(os.path.join(src, f))
    notes = []
    has_alpha = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)
    im = img.convert('RGBA')
    al = np.asarray(im)[..., 3]
    clear = (al == 0).mean()
    soft = ((al > 0) & (al < 255)).mean()
    if not has_alpha or clear < 0.01:
        notes.append('NO REAL TRANSPARENCY')
        if a.key_out:
            im = key_out(img); al = np.asarray(im)[..., 3]; notes.append('background keyed out - check it!')
        else:
            print(f"{f:34} {str(img.size):>11}  {clear:>10.0%}  {soft:>8.1%}  {'; '.join(notes)} (try --key-out)"); continue
    # halo: shave the outer edge, then soften by a hair so it isn't jagged
    if a.halo > 0:
        m = Image.fromarray(al).filter(ImageFilter.MinFilter(2 * a.halo + 1)).filter(ImageFilter.GaussianBlur(0.7))
        im.putalpha(m); notes.append(f'edge -{a.halo}px')
    # crop away empty space
    bbox = im.getchannel('A').point(lambda v: 255 if v > 8 else 0).getbbox()
    if bbox:
        pad = 4; bbox = (max(bbox[0] - pad, 0), max(bbox[1] - pad, 0), min(bbox[2] + pad, im.width), min(bbox[3] + pad, im.height))
        if bbox != (0, 0, im.width, im.height): im = im.crop(bbox); notes.append('cropped')
    if a.size and max(im.size) > a.size:
        s = a.size / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS); notes.append(f'resized to {im.size[0]}x{im.size[1]}')
    im.save(os.path.join(out, f), optimize=True)
    print(f"{f:34} {str(img.size):>11}  {clear:>10.0%}  {soft:>8.1%}  {', '.join(notes) or 'ok'}")
print(f"\nFixed copies are in: {out}")

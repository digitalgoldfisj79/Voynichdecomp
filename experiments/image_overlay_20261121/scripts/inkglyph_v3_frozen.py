"""Ink-based glyph assignment (v3): page image + word boxes + EVA strings -> per-glyph ink masks and tight boxes.

Per page:  darkness map -> ink mask with hysteresis -> connected components -> each component to one word box.
Per word:  find the x-height band (densest rows); cut the band's ink at thin joins and dips ("atoms");
           group atoms left-to-right into the word's characters by dynamic programming (expected widths +
           "has a tall riser above the band = gallows" cue); ink above and below the band (gallows tops,
           plumes, tails) is then attached to whichever character it physically touches.
No learning and no recognition: the transliteration decides how many glyphs there are and their order.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from PIL import ImageFilter

GALLOWS = set("ktpf")
S8 = np.ones((3, 3), int)
P0 = dict(thr_lo=0.09, thr_hi=0.30, weak=0.5, speck=0.02, valley=1.4, split_min_w=0.7, dip=0.22, tall=0.6,
          alpha=1.0, beta=0.6, gamma=3.0, empty=1.2, max_atoms=40)


def darkness(img_L, radius):
    g = np.asarray(img_L, np.float32)
    bg = np.asarray(img_L.filter(ImageFilter.GaussianBlur(radius)), np.float32)
    return np.clip(1.0 - g / np.maximum(bg, 1.0), 0, 1)


def otsu(v, lo=0.0, hi=0.8, bins=160):
    h, e = np.histogram(v, bins=bins, range=(lo, hi)); h = h.astype(float); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0; m0 = np.cumsum(h * c); mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (mt * w0 - m0 * w0[-1]) ** 2 / (w0 * w1)
    var[~np.isfinite(var)] = 0
    return float(c[int(np.argmax(var))])


def analyse_page(img_L, boxes, P=P0):
    boxes = [tuple(float(v) for v in b) for b in boxes]
    hmed = float(np.median([b[3] - b[1] for b in boxes])) if boxes else 40.0
    d = darkness(img_L, radius=max(8, int(1.5 * hmed)))
    Hh, Ww = d.shape
    ib = [(max(0, int(np.floor(b[0]))), max(0, int(np.floor(b[1]))), min(Ww, int(np.ceil(b[2]))), min(Hh, int(np.ceil(b[3])))) for b in boxes]
    sample = np.concatenate([d[y0:y1:2, x0:x1:2].ravel() for x0, y0, x1, y1 in ib if x1 > x0 and y1 > y0] or [np.zeros(1)])
    thr = float(np.clip(otsu(sample), P["thr_lo"], P["thr_hi"]))
    lab, n = ndi.label(d > P["weak"] * thr, structure=S8)          # hysteresis: weak ink only if joined to strong ink
    if n:
        strong = ndi.maximum(d, lab, index=np.arange(1, n + 1)) >= thr
        lut = np.zeros(n + 1, np.int32); lut[1:][strong] = np.arange(1, int(strong.sum()) + 1)
        lab = lut[lab]; n = int(strong.sum())
    sl = ndi.find_objects(lab)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    inside = {}
    for bi, (x0, y0, x1, y1) in enumerate(ib):
        if x1 <= x0 or y1 <= y0: continue
        sub = lab[y0:y1, x0:x1]; ids, cnt = np.unique(sub[sub > 0], return_counts=True)
        for i, c in zip(ids.tolist(), cnt.tolist()): inside.setdefault(i, []).append((c / area[i], bi))
    members = [[] for _ in boxes]
    for i, lst in inside.items():
        s = sl[i - 1]; cy = (s[0].start + s[0].stop) / 2
        best = max(lst, key=lambda t: (round(t[0], 2), -abs(cy - (boxes[t[1]][1] + boxes[t[1]][3]) / 2)))
        if best[0] >= 0.5: members[best[1]].append(i)
    hs = np.array([sl[i - 1][0].stop - sl[i - 1][0].start for m in members for i in m if area[i] >= (0.1 * hmed) ** 2] or [hmed / 2])
    xh = float(np.median(hs))
    dts = []
    for m in members[:: max(1, len(members) // 60)]:
        for i in m:
            mm = np.pad(lab[sl[i - 1]] == i, 1); dts.append(ndi.distance_transform_edt(mm)[mm])
    t = 4.0 * float(np.median(np.concatenate(dts))) if dts else max(2.0, 0.12 * xh)
    keep = [[i for i in m if area[i] >= P["speck"] * xh * xh and (sl[i - 1][0].stop - sl[i - 1][0].start) <= 6 * xh
             and (sl[i - 1][1].stop - sl[i - 1][1].start) <= 14 * xh] for m in members]
    return dict(lab=lab, sl=sl, area=area, members=keep, xh=xh, t=t, thr=thr, boxes=boxes, d=d)


def _cut_columns(m, xh, t, P):
    """cut positions (columns) for one band component mask m."""
    w = m.shape[1]; cuts = []
    if w <= P["split_min_w"] * xh: return cuts
    proj = m.sum(0).astype(float); ps = np.convolve(proj, np.ones(3) / 3, "same"); low = ps <= P["valley"] * t; c = 0
    while c < w:
        if low[c]:
            e = c
            while e + 1 < w and low[e + 1]: e += 1
            if c > 0 and e < w - 1: cuts.append((c + e + 1) // 2)
            c = e + 1
        else: c += 1
    k = max(3, int(round(0.15 * xh)) | 1); pw = np.convolve(proj, np.ones(k) / k, "same")
    dips, _ = find_peaks(-pw, prominence=P["dip"] * pw.max(), distance=max(1, int(0.3 * xh)))
    for dp in dips.tolist():
        if 0.25 * xh < dp < w - 0.25 * xh and all(abs(dp - c_) >= 0.3 * xh for c_ in cuts): cuts.append(int(dp))
    return sorted(cuts)


def _atom(mask, asc=False):
    cols = np.where(mask.any(0))[0]; ar = int(mask.sum())
    return dict(mask=mask, x0=int(cols[0]), x1=int(cols[-1]) + 1, area=ar, cx=float((mask.sum(0) * np.arange(mask.shape[1])).sum() / ar), asc=asc)


def _dp(atoms, word, table, default, P):
    n, m = len(word), len(atoms)
    A = min(a["x0"] for a in atoms); B = max(a["x1"] for a in atoms)
    w = np.array([max(table.get(ch, default), 1e-3) for ch in word], float)
    e = A + (B - A) * np.concatenate([[0.0], np.cumsum(w)]) / w.sum(); mu = (e[:-1] + e[1:]) / 2; Wd = np.diff(e); s = max((B - A) / n, 1.0)
    INF = 1e18; D = np.full((n + 1, m + 1), INF); D[0, 0] = 0.0; bk = np.zeros((n + 1, m + 1), int)
    for i in range(1, n + 1):
        gal = word[i - 1] in GALLOWS
        for k in range(m + 1):
            best, arg = D[i - 1, k] + P["empty"], k
            gx0, gx1, anyasc = INF, -INF, False
            for j in range(k - 1, -1, -1):
                a = atoms[j]; gx0 = min(gx0, a["x0"]); gx1 = max(gx1, a["x1"]); anyasc = anyasc or a["asc"]
                if D[i - 1, j] >= INF: continue
                c = (P["alpha"] * ((gx0 + gx1) / 2 - mu[i - 1]) ** 2 + P["beta"] * ((gx1 - gx0) - Wd[i - 1]) ** 2) / (s * s) + (P["gamma"] if gal != anyasc else 0.0)
                if D[i - 1, j] + c < best: best, arg = D[i - 1, j] + c, j
            D[i, k] = best; bk[i, k] = arg
    groups = [[] for _ in word]; k = m
    for i in range(n, 0, -1):
        j = bk[i, k]; groups[i - 1] = atoms[j:k]; k = j
    return groups, w


def word_glyphs(ctx, mem, word, table, default, P=P0):
    """Returns per-character list of dict(mask, x, y, bbox, est) or None; coordinates are page pixels."""
    n = len(word)
    if not mem or n == 0: return [None] * n
    sl, xh, t = ctx["sl"], ctx["xh"], ctx["t"]
    Y0 = min(sl[i - 1][0].start for i in mem); Y1 = max(sl[i - 1][0].stop for i in mem)
    X0 = min(sl[i - 1][1].start for i in mem); X1 = max(sl[i - 1][1].stop for i in mem)
    M = np.isin(ctx["lab"][Y0:Y1, X0:X1], mem); h = M.shape[0]; win = max(2, int(round(xh)))
    if h <= win: a, b = 0, h
    else:
        cs = np.concatenate([[0.0], np.cumsum(M.sum(1))]); top = int(np.argmax(cs[win:] - cs[:-win]))
        a, b = max(0, top - int(0.15 * xh)), min(h, top + win + int(0.2 * xh))
    band = np.zeros_like(M); band[a:b] = M[a:b]
    lm, nl = ndi.label(band, structure=S8); atoms = []
    for c, s in enumerate(ndi.find_objects(lm), 1):
        cm = lm[s] == c
        cuts = [0] + _cut_columns(cm, xh, t, P) + [cm.shape[1]]
        for u, v in zip(cuts[:-1], cuts[1:]):
            part = np.zeros_like(cm); part[:, u:v] = cm[:, u:v]
            if not part.any(): continue
            full = np.zeros_like(M); full[s] = part; atoms.append(_atom(full))
    if not atoms: return [None] * n
    # ink above / below the band: components and what they touch
    zones = []
    for z0, z1, edge_row, band_row in ((0, a, a - 1, a), (b, h, b, b - 1)):
        if z1 <= z0: continue
        zm = np.zeros_like(M); zm[z0:z1] = M[z0:z1]; lz, nz = ndi.label(zm, structure=S8)
        for c, s in enumerate(ndi.find_objects(lz), 1):
            comp = lz == c; cols = np.where(comp[edge_row])[0]
            touch = np.zeros(M.shape[1], bool)
            for dc in (-1, 0, 1):
                cc_ = cols + dc; touch[cc_[(cc_ >= 0) & (cc_ < M.shape[1])]] = True
            rise = (a - s[0].start) if z0 == 0 else 0
            zones.append(dict(mask=comp, touch=touch, band_row=band_row, tall=rise >= P["tall"] * xh, x0=s[1].start, x1=s[1].stop))
    for z in zones:
        if not z["tall"]: continue
        hit = [k for k, at in enumerate(atoms) if (at["mask"][z["band_row"]] & z["touch"]).any()]
        if not hit: continue
        fused = atoms[hit[0]]["mask"].copy()
        for k in hit[1:]: fused |= atoms[k]["mask"]          # both legs of one gallows: fuse
        atoms = [at for k, at in enumerate(atoms) if k not in hit] + [_atom(fused, asc=True)]
    atoms = sorted(atoms, key=lambda at: at["cx"])[: P["max_atoms"]]
    groups, w = _dp(atoms, word, table, default, P)
    cmask = [np.zeros_like(M) for _ in word]; est = [False] * n
    for i, g in enumerate(groups):
        for at in g: cmask[i] |= at["mask"]
    # characters left without ink: share the neighbouring character's band ink by column, by expected width
    def ext(q):
        cols = np.where(cmask[q].any(0))[0]; return (cols[-1] + 1 - cols[0]) / w[q]
    i = 0
    while i < n:
        if cmask[i].any(): i += 1; continue
        j = i
        while j < n and not cmask[j].any(): j += 1
        cand = [q for q in (i - 1, j) if 0 <= q < n and cmask[q].any()]
        if not cand: break
        donor = max(cand, key=ext); idx = sorted(list(range(i, j)) + [donor]); dm = cmask[donor].copy()
        cols = np.where(dm.any(0))[0]; gx0, gx1 = cols[0], cols[-1] + 1
        ee = np.rint(gx0 + (gx1 - gx0) * np.concatenate([[0.0], np.cumsum(w[idx])]) / w[idx].sum()).astype(int)
        for q, lo, hi in zip(idx, ee[:-1], ee[1:]):
            cmask[q] = np.zeros_like(M); cmask[q][:, lo:hi] = dm[:, lo:hi]; est[q] = True
        i = j
    # attach ink above / below the band to the character it touches (else the one it overlaps most in x)
    for z in zones:
        cont = [int((cmask[q][z["band_row"]] & z["touch"]).sum()) for q in range(n)]
        if max(cont) > 0: q = int(np.argmax(cont))
        else:
            ov = []
            for q in range(n):
                cols = np.where(cmask[q].any(0))[0]
                ov.append(0 if not len(cols) else max(0, min(cols[-1] + 1, z["x1"]) - max(cols[0], z["x0"])) + (0.5 if (z["tall"] and word[q] in GALLOWS) else 0))
            if max(ov) <= 0: continue
            q = int(np.argmax(ov))
        cmask[q] |= z["mask"]
    out = []
    for q in range(n):
        if not cmask[q].any(): out.append(None); continue
        rows = np.where(cmask[q].any(1))[0]; cols = np.where(cmask[q].any(0))[0]
        out.append(dict(mask=cmask[q], x=X0, y=Y0, est=est[q], bbox=(X0 + int(cols[0]), Y0 + int(rows[0]), X0 + int(cols[-1]) + 1, Y0 + int(rows[-1]) + 1)))
    return out


def page_glyphs(img_L, words, table, default, P=P0):
    """words: list of (string, (x0,y0,x1,y1)). Returns (ctx, per-word list of per-character glyph dicts or None)."""
    ctx = analyse_page(img_L, [b for _, b in words], P)
    return ctx, [word_glyphs(ctx, mem, wd, table, default, P) for (wd, _), mem in zip(words, ctx["members"])]


def paint(rgb, glyphs_per_word, words, colour_of, alpha=0.8):
    out = rgb.astype(np.float32).copy()
    for (wd, _), gl in zip(words, glyphs_per_word):
        for ch, g in zip(wd, gl):
            if g is None: continue
            m = g["mask"]; region = out[g["y"]:g["y"] + m.shape[0], g["x"]:g["x"] + m.shape[1]]
            region[m] = (1 - alpha) * region[m] + alpha * np.array(colour_of(ch), np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

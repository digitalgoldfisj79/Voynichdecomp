// ============================================================================
// placeholders.js  -  stand-in plants drawn by code, so Stage 1 needs no images.
//
// Each one is a flat cut-out with a transparent background and its root at the
// bottom centre - exactly how your real manuscript PNGs will behave.
// In Stage 2 these get replaced by your real images from the plants/ folder;
// the rest of the code does not care where a texture comes from.
// ============================================================================

const PAL = {
  green: '#86ab7c', teal: '#5fa597', olive: '#a8a85e', ochre: '#dcaa48', red: '#bf5a46',
  blue: '#6384b5', violet: '#8f70ad', brown: '#92704a', rose: '#cf8a86',
};
const INK = '#3a2a1c';

// ---- tiny drawing helpers (canvas y points DOWN, root is at the bottom) -----
function pStem(ctx, pts, width, color) {
  const trace = () => {
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length - 1; i++) {
      ctx.quadraticCurveTo(pts[i][0], pts[i][1], (pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2);
    }
    const l = pts[pts.length - 1]; ctx.lineTo(l[0], l[1]);
  };
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  trace(); ctx.strokeStyle = INK; ctx.lineWidth = width + 6; ctx.stroke();   // ink outline
  trace(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();     // coloured wash
}
function pLeaf(ctx, x, y, ang, len, wid, fill) {
  ctx.save(); ctx.translate(x, y); ctx.rotate(ang);
  ctx.beginPath(); ctx.moveTo(0, 0);
  ctx.quadraticCurveTo(len * 0.5, -wid, len, 0); ctx.quadraticCurveTo(len * 0.5, wid, 0, 0);
  ctx.fillStyle = fill; ctx.fill(); ctx.lineWidth = 4; ctx.strokeStyle = INK; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(len * 0.88, 0); ctx.lineWidth = 2; ctx.stroke();
  ctx.restore();
}
function pBlob(ctx, x, y, rx, ry, fill) {
  ctx.beginPath(); ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = fill; ctx.fill(); ctx.lineWidth = 4; ctx.strokeStyle = INK; ctx.stroke();
}
function pRoots(ctx, cx, b) {
  ctx.strokeStyle = INK; ctx.lineWidth = 3; ctx.lineCap = 'round';
  for (let k = -2; k <= 2; k++) { ctx.beginPath(); ctx.moveTo(cx + k * 4, b - 4); ctx.lineTo(cx + k * 12, b + 2); ctx.stroke(); }
}

// ---- the twelve placeholder plants -------------------------------------------
// aspect = width / height of the picture. draw() paints it.
const PLACEHOLDER_KINDS = [
  { id: 'stalk', aspect: 0.5, draw(ctx, w, h) {
    const cx = w / 2, b = h - 10;
    pStem(ctx, [[cx, b], [cx - 6, h * 0.65], [cx + 5, h * 0.35], [cx, 50]], 9, PAL.green);
    for (let k = 0; k < 5; k++) {
      const y = b - 80 - k * 70, L = 100 - k * 12, Wd = 26 - k * 3;
      pLeaf(ctx, cx, y, -0.65, L, Wd, PAL.green); pLeaf(ctx, cx, y - 30, Math.PI + 0.65, L, Wd, PAL.teal);
    }
    pLeaf(ctx, cx, 56, -Math.PI / 2, 70, 22, PAL.green); pRoots(ctx, cx, b);
  } },
  { id: 'bell', aspect: 0.5, draw(ctx, w, h) {
    const cx = w / 2, b = h - 10;
    pStem(ctx, [[cx, b], [cx - 8, h * 0.6], [cx + 6, h * 0.3], [cx + 30, 100]], 8, PAL.green);
    pLeaf(ctx, cx, b - 20, -0.9, 120, 30, PAL.teal); pLeaf(ctx, cx, b - 30, Math.PI + 0.9, 120, 30, PAL.green);
    pBlob(ctx, cx + 30, 140, 38, 52, PAL.red);
    for (let k = -1; k <= 1; k++) { ctx.beginPath(); ctx.moveTo(cx + 30 + k * 14, 100); ctx.lineTo(cx + 30 + k * 20, 180); ctx.lineWidth = 2; ctx.strokeStyle = INK; ctx.stroke(); }
    pRoots(ctx, cx, b);
  } },
  { id: 'tuft', aspect: 1.1, draw(ctx, w, h) {
    const cx = w / 2, b = h - 10;
    for (let k = 0; k < 9; k++) {
      const a = -Math.PI + 0.35 + k * (Math.PI - 0.7) / 8, len = 150 + 120 * Math.sin(k / 8 * Math.PI);
      pLeaf(ctx, cx, b, a, len, 24, k % 2 ? PAL.teal : PAL.green);
    }
    pRoots(ctx, cx, b);
  } },
  { id: 'lollipop', aspect: 0.7, draw(ctx, w, h) {
    const cx = w / 2, b = h - 10;
    pStem(ctx, [[cx, b], [cx + 6, h * 0.7], [cx - 6, h * 0.5], [cx, 230]], 18, PAL.brown);
    const circles = [[0, 0, 80], [-70, 30, 60], [70, 30, 60], [-40, -60, 55], [45, -55, 55], [0, 55, 60]];
    ctx.lineWidth = 8; ctx.strokeStyle = INK;
    for (const [dx, dy, r] of circles) { ctx.beginPath(); ctx.arc(cx + dx, 150 + dy, r, 0, Math.PI * 2); ctx.stroke(); }
    ctx.fillStyle = PAL.ochre;
    for (const [dx, dy, r] of circles) { ctx.beginPath(); ctx.arc(cx + dx, 150 + dy, r - 1, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = PAL.red;
    for (let k = 0; k < 9; k++) { ctx.beginPath(); ctx.arc(cx - 80 + (k * 53) % 170, 100 + (k * 37) % 110, 7, 0, Math.PI * 2); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = INK; ctx.stroke(); }
    pRoots(ctx, cx, b);
  } },
  { id: 'grass', aspect: 0.6, draw(ctx, w, h) {
    const b = h - 8, r = mulberry32(7);
    for (let k = 0; k < 14; k++) {
      const bx = w / 2 - 70 + k * 10, tx = bx + (r() - 0.5) * 120, ty = b - 200 - r() * 220, curve = (r() - 0.5) * 60;
      ctx.beginPath(); ctx.moveTo(bx - 7, b); ctx.quadraticCurveTo(bx + curve, (b + ty) / 2, tx, ty);
      ctx.quadraticCurveTo(bx + curve + 12, (b + ty) / 2, bx + 7, b);
      ctx.fillStyle = k % 3 ? PAL.olive : PAL.green; ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = INK; ctx.stroke();
    }
  } },
  { id: 'branching', aspect: 0.8, draw(ctx, w, h) {
    const rec = (x, y, ang, len, depth) => {
      const x2 = x + Math.cos(ang) * len, y2 = y + Math.sin(ang) * len;
      pStem(ctx, [[x, y], [(x + x2) / 2 + 4, (y + y2) / 2], [x2, y2]], 3 + depth * 2, depth > 1 ? PAL.brown : PAL.green);
      if (depth === 0) { pLeaf(ctx, x2, y2, ang, 40, 14, PAL.teal); pLeaf(ctx, x2, y2, ang - 0.8, 30, 11, PAL.green); return; }
      rec(x2, y2, ang - 0.5, len * 0.72, depth - 1); rec(x2, y2, ang + 0.45, len * 0.7, depth - 1);
    };
    rec(w / 2, h - 10, -Math.PI / 2, 120, 4);
  } },
  { id: 'thistle', aspect: 0.45, draw(ctx, w, h) {
    const cx = w / 2, b = h - 10;
    pStem(ctx, [[cx, b], [cx + 6, h * 0.6], [cx - 4, h * 0.3], [cx, 190]], 8, PAL.green);
    pLeaf(ctx, cx, b - 60, -0.8, 90, 22, PAL.olive); pLeaf(ctx, cx, b - 120, Math.PI + 0.8, 90, 22, PAL.olive);
    ctx.strokeStyle = INK; ctx.lineWidth = 3;
    for (let k = 0; k < 20; k++) { const a = -Math.PI + k * Math.PI / 19; ctx.beginPath(); ctx.moveTo(cx, 150); ctx.lineTo(cx + Math.cos(a) * 62, 150 + Math.sin(a) * 62); ctx.stroke(); }
    pBlob(ctx, cx, 150, 34, 30, PAL.violet); pBlob(ctx, cx, 190, 30, 24, PAL.green);
  } },
  { id: 'bigleaf', aspect: 1.0, draw(ctx, w, h) {
    const cx = w / 2, b = h - 10;
    pLeaf(ctx, cx, b, -Math.PI / 2 - 0.65, 320, 90, PAL.teal);
    pLeaf(ctx, cx, b, -Math.PI / 2 + 0.65, 320, 90, PAL.green);
    pLeaf(ctx, cx, b, -Math.PI / 2, 440, 100, PAL.green);
    pRoots(ctx, cx, b);
  } },
  { id: 'reeds', aspect: 0.5, draw(ctx, w, h) {
    const b = h - 8;
    [[-60, -35, 380], [-30, -10, 450], [0, 14, 400], [30, 40, 460], [58, 62, 360]].forEach(([bx, tx, ty], k) => {
      const x0 = w / 2 + bx, x1 = w / 2 + tx;
      pStem(ctx, [[x0, b], [(x0 + x1) / 2 + 6, b - ty / 2], [x1, b - ty]], 6, PAL.olive);
      pBlob(ctx, x1, b - ty + 40, 11, 38, PAL.brown);
      pLeaf(ctx, x0, b - 30, bx < 0 ? Math.PI + 1.2 : -1.2, 130, 14, PAL.green);
    });
  } },
  { id: 'bulb', aspect: 0.6, draw(ctx, w, h) {
    const cx = w / 2, b = h - 10;
    pLeaf(ctx, cx, b - 70, -Math.PI / 2 - 0.35, 340, 40, PAL.green); pLeaf(ctx, cx, b - 70, -Math.PI / 2 + 0.35, 340, 40, PAL.teal);
    pLeaf(ctx, cx, b - 70, -Math.PI / 2, 400, 34, PAL.olive);
    pBlob(ctx, cx, b - 48, 56, 44, PAL.red);
    ctx.strokeStyle = INK; ctx.lineWidth = 2;
    for (let k = -2; k <= 2; k++) { ctx.beginPath(); ctx.moveTo(cx + k * 14, b - 90); ctx.quadraticCurveTo(cx + k * 22, b - 48, cx + k * 12, b - 6); ctx.stroke(); }
    pRoots(ctx, cx, b);
  } },
  { id: 'star', aspect: 0.5, draw(ctx, w, h) {
    const cx = w / 2, b = h - 10;
    pStem(ctx, [[cx, b], [cx - 6, h * 0.6], [cx + 6, h * 0.3], [cx, 100]], 8, PAL.green);
    pLeaf(ctx, cx, b - 70, -0.7, 100, 24, PAL.green); pLeaf(ctx, cx, b - 130, Math.PI + 0.7, 100, 24, PAL.teal);
    for (let k = 0; k < 7; k++) pLeaf(ctx, cx, 90, k * Math.PI * 2 / 7, 62, 15, PAL.ochre);
    pBlob(ctx, cx, 90, 14, 14, PAL.red);
    pRoots(ctx, cx, b);
  } },
  { id: 'fern', aspect: 0.55, draw(ctx, w, h) {
    const cx = w / 2, b = h - 10, P0 = [cx, b], P1 = [cx - 30, h * 0.45], P2 = [cx + 50, 50];
    const at = (t) => [(1 - t) * (1 - t) * P0[0] + 2 * (1 - t) * t * P1[0] + t * t * P2[0], (1 - t) * (1 - t) * P0[1] + 2 * (1 - t) * t * P1[1] + t * t * P2[1]];
    pStem(ctx, [at(0), at(0.3), at(0.6), at(1)], 7, PAL.brown);
    for (let t = 0.08; t < 0.98; t += 0.065) {
      const p = at(t), q = at(t + 0.02), ta = Math.atan2(q[1] - p[1], q[0] - p[0]), len = 95 * (1 - t) + 18;
      pLeaf(ctx, p[0], p[1], ta - 1.1, len, 11, PAL.green); pLeaf(ctx, p[0], p[1], ta + 1.1, len, 11, PAL.teal);
    }
    pRoots(ctx, cx, b);
  } },
];

// Make a texture for every placeholder. Returns [{id, aspect, texture}].
// (Stage 2 will return the same shape of list, built from your PNG files.)
function makePlaceholderKinds() {
  return PLACEHOLDER_KINDS.map((k) => {
    const h = 512, w = Math.round(h * k.aspect);
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    k.draw(ctx, w, h);
    const texture = new THREE.CanvasTexture(canvas);
    texture.anisotropy = 4;
    return { id: k.id, aspect: w / h, texture };
  });
}

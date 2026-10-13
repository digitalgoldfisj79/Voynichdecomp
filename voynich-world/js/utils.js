// ============================================================================
// utils.js  -  small helpers used everywhere. You normally never need to edit this.
// ============================================================================

// ---- Small math helpers ----------------------------------------------------
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
function lerp(a, b, t) { return a + (b - a) * t; }
// smoothstep: 0 below e0, 1 above e1, a smooth S-curve in between.
function smoothstep(e0, e1, x) {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}
// Wrap an angle difference into -PI..PI (so we always turn the short way round).
function wrapPi(a) {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
}

// ---- Seeded random numbers --------------------------------------------------
// "Seeded" means: same seed in -> exactly the same random numbers out.
// That is what lets you recreate a scene you liked.
function hashString(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// makeRng(seed, 'plants') gives an independent random stream for one job.
// Separate streams mean: changing the plant count does NOT change the hills.
function makeRng(seed, salt) {
  const next = mulberry32((hashString(String(salt)) ^ Math.imul(seed | 0, 2654435761)) >>> 0);
  return {
    next,                                              // 0..1
    range: (a, b) => a + (b - a) * next(),             // a..b
    int: (a, b) => Math.floor(a + (b - a + 1) * next()),
    chance: (p) => next() < p,                         // true with probability p
    pick: (arr) => arr[Math.floor(next() * arr.length)],
    gauss() {                                          // bell-curve random (mean 0, spread 1)
      let u = 0;
      while (u === 0) u = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
    },
  };
}

// ---- Smooth 2D noise (used for hills) --------------------------------------
// noise(x, y) -> 0..1, smooth and seeded.  noise.fbm(x, y, octaves) layers
// several sizes of bumps on top of each other, which looks natural.
function makeNoise2D(rng) {
  const vals = new Float32Array(256 * 256);
  for (let i = 0; i < vals.length; i++) vals[i] = rng.next();
  const lat = (ix, iy) => vals[((iy & 255) << 8) | (ix & 255)];
  const noise = (x, y) => {
    const ix = Math.floor(x), iy = Math.floor(y);
    const fx = x - ix, fy = y - iy;
    const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
    const a = lat(ix, iy), b = lat(ix + 1, iy), c = lat(ix, iy + 1), d = lat(ix + 1, iy + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  };
  noise.fbm = (x, y, octaves) => {
    let sum = 0, amp = 0.5, f = 1, norm = 0;
    for (let i = 0; i < (octaves || 3); i++) {
      sum += amp * noise(x * f, y * f);
      norm += amp; amp *= 0.5; f *= 2.03;
    }
    return sum / norm;
  };
  return noise;
}

// ---- Reading/writing settings by a text path ("plants.count") ---------------
function getPath(obj, path) {
  return path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj);
}
function setPath(obj, path, value) {
  const keys = path.split('.');
  const last = keys.pop();
  const target = keys.reduce((o, k) => o[k], obj);
  target[last] = value;
}

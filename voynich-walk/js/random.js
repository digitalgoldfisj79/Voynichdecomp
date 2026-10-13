/* Seeded randomness.
   Math.random() gives different numbers every time. These functions give the
   SAME sequence for the same seed, which is what lets you rebuild a world you liked. */

// A small, fast random number generator ("mulberry32"). Returns a function that
// gives a new number between 0 and 1 each time you call it.
export function makeRng(seed) {
  let a = (seed >>> 0) || 1;
  const rnd = () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  rnd.range = (lo, hi) => lo + (hi - lo) * rnd();          // number between lo and hi
  rnd.int = (lo, hi) => Math.floor(rnd.range(lo, hi + 1));  // whole number lo..hi
  rnd.pick = arr => arr[Math.floor(rnd() * arr.length)];    // one item from a list
  rnd.shuffle = arr => { for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; };
  return rnd;
}

// Smooth "value noise": gives gently rolling numbers across the ground,
// used for hills and colour blotches. Same seed = same landscape.
export function makeNoise(seed) {
  const rnd = makeRng(seed);
  const N = 256, perm = new Uint16Array(N * 2), val = new Float32Array(N);
  for (let i = 0; i < N; i++) { perm[i] = i; val[i] = rnd(); }
  for (let i = N - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  for (let i = 0; i < N; i++) perm[N + i] = perm[i];
  const fade = t => t * t * (3 - 2 * t);
  const v = (ix, iz) => val[perm[(ix & 255) + perm[iz & 255]]];
  function noise(x, z) {                       // 0..1
    const ix = Math.floor(x), iz = Math.floor(z), fx = fade(x - ix), fz = fade(z - iz);
    const a = v(ix, iz), b = v(ix + 1, iz), c = v(ix, iz + 1), d = v(ix + 1, iz + 1);
    return a + (b - a) * fx + (c - a) * fz + (a - b - c + d) * fx * fz;
  }
  // several layers of noise added together = more natural ("fractal") shapes
  noise.fbm = (x, z, octaves = 4) => {
    let sum = 0, amp = 0.5, f = 1, norm = 0;
    for (let o = 0; o < octaves; o++) { sum += amp * noise(x * f, z * f); norm += amp; amp *= 0.5; f *= 2.03; }
    return sum / norm;
  };
  return noise;
}

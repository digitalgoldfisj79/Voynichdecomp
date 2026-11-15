// Voynich journey: a five-minute walk up a valley and out into a clearing.
// Every drawing is a cut-out from Beinecke MS 408; nothing is redrawn. The land, water and sky are plain colour.
//
// Address options:  ?t=120   start at 120 s        ?clean=1  hide the controls (for screen recording)
//                   ?q=hi    1024 px drawings       ?q=lo     512 px drawings (default on phones)
// For frame-by-frame export, call  window.journey.renderAt(seconds)  and capture the canvas.

import * as THREE from 'three';
import { Reflector } from './lib/Reflector.js';

const Q = new URLSearchParams(location.search);
const PHONE = matchMedia('(pointer: coarse)').matches;
const TEX = (Q.get('q') || (PHONE ? 'lo' : 'hi')) === 'hi' ? 1024 : 512;
if (Q.get('clean')) document.body.classList.add('clean');

// ------------------------------------------------------------------ the tour
export const DURATION = 300;                 // seconds
const VALLEY_END = 172;                      // where the valley opens out (metres along z)
const CLEAR = { x: 0, z: 218, r: 46 };       // the clearing
const POND = { x: 13, z: 224, r: 11 };
const WATER_Y = -0.35;
const LANDMARK = { x: -20, z: 252 };         // the giant f93r plant
const ARCHER = { x: -24, z: 214 };
const EYE = 1.65;

const sstep = (a, b, v) => { const t = Math.min(1, Math.max(0, (v - a) / (b - a))); return t * t * (3 - 2 * t); };
const pathX = z => 6 * Math.sin(z / 24) + 2.2 * Math.sin(z / 9.5 + 1.3);

function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function noise(x, z) {                       // smooth value noise, 0..1
  const xi = Math.floor(x), zi = Math.floor(z), xf = x - xi, zf = z - zi;
  const h = (a, b) => { const s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return s - Math.floor(s); };
  const u = xf * xf * (3 - 2 * xf), v = zf * zf * (3 - 2 * zf);
  return (h(xi, zi) * (1 - u) + h(xi + 1, zi) * u) * (1 - v) + (h(xi, zi + 1) * (1 - u) + h(xi + 1, zi + 1) * u) * v;
}

// Distance outside the open ground (valley floor or clearing); 0 inside it.
function outside(x, z) {
  const floorHalf = 2.6 + 1.2 * noise(z * 0.07, 3.1);
  const dValley = z < VALLEY_END + 10 ? Math.abs(x - pathX(z)) - floorHalf : 1e9;
  const dClear = Math.hypot(x - CLEAR.x, z - CLEAR.z) - CLEAR.r;
  return Math.max(0, Math.min(dValley, dClear));
}
export function ground(x, z) {
  const d = outside(x, z);
  const crest = 8.5 + 7 * noise(x * 0.035 + 7, z * 0.035);                    // hill height varies along the valley
  let y = crest * sstep(0, 26, d) + 2.5 * sstep(20, 70, d);
  y += (noise(x * 0.25, z * 0.25) - 0.5) * 0.9 * sstep(0, 6, d);              // lumps on the slopes
  y += (noise(x * 0.11 + 40, z * 0.11) - 0.5) * 0.35;                         // gentle undulation everywhere
  const dp = Math.hypot(x - POND.x, z - POND.z);
  y -= 1.6 * (1 - sstep(POND.r - 4.5, POND.r + 1.5, dp));                     // the pond bowl
  const dk = Math.hypot(x - ARCHER.x, z - ARCHER.z);
  y += 1.1 * (1 - sstep(0, 6, dk));                                           // the archer's knoll
  return y;
}

// The camera's route: straight up the valley, then a slow circuit of the clearing.
const loopPts = [[pathX(VALLEY_END), VALLEY_END], [1, 186], [-6, 200], [-12, 214], [-3, 222], [1, 232], [9, 238.5], [19, 237.5], [26.5, 230], [28, 220], [23, 210], [12, 203], [0, 204], [-6, 212]];
const loop = new THREE.CatmullRomCurve3(loopPts.map(p => new THREE.Vector3(p[0], 0, p[1])), false, 'centripetal');
const loopSamples = loop.getSpacedPoints(90);
const T_VALLEY = 128;                         // seconds spent in the valley
function routeAt(t) {
  if (t <= T_VALLEY) { const z = -6 + (VALLEY_END + 6) * (t / T_VALLEY); return new THREE.Vector3(pathX(z), 0, z); }
  const u = (t - T_VALLEY) / (DURATION - T_VALLEY);
  return loop.getPointAt(Math.min(1, u * (1 - 0.06 * u)));                    // eases off towards the end
}
// Things the camera turns to look at, by time: [from, to, x, y, z]
const LOOKS = [
  [134, 150, POND.x, 0.6, POND.z],
  [156, 176, ARCHER.x, 2.0, ARCHER.z],
  [182, 232, POND.x, 0.5, POND.z],
  [240, 262, CLEAR.x - 8, 1.0, CLEAR.z - 14],          // the rams' meadow
  [270, 300, LANDMARK.x, 11, LANDMARK.z],
];
function cameraAt(t, pos, tgt) {
  const p = routeAt(t), a = routeAt(Math.min(DURATION, t + 4.5));
  if (a.distanceTo(p) < 0.5) a.copy(p).add(new THREE.Vector3(0, 0, 4));
  pos.set(p.x, ground(p.x, p.z) + EYE + 0.035 * Math.sin(t * 5.6), p.z);     // a slight walking bob
  tgt.set(a.x + 1.6 * Math.sin(t * 0.21), ground(a.x, a.z) + EYE - 0.1 + 0.5 * Math.sin(t * 0.13), a.z);
  if (t < T_VALLEY) tgt.y += 1.2 * sstep(20, 50, t) * (1 - sstep(100, 124, t)) * (0.5 + 0.5 * Math.sin(t * 0.09));   // glance up the slopes
  for (const [t0, t1, x, y, z] of LOOKS) {
    const w = sstep(t0, t0 + 5, t) * (1 - sstep(t1 - 5, t1, t));
    if (w > 0) tgt.lerp(new THREE.Vector3(x, ground(x, z) + y, z), w * 0.92);
  }
}

// ------------------------------------------------------------------ renderer and scene
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, PHONE ? 1.5 : 2));
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
const HORIZON = new THREE.Color(0xe8dcb6), ZENITH = new THREE.Color(0x7fa6bd);
scene.fog = new THREE.FogExp2(HORIZON, 0.0048);
const camera = new THREE.PerspectiveCamera(58, 16 / 9, 0.1, 700);

const sky = new THREE.Mesh(new THREE.SphereGeometry(600, 24, 12), new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false, fog: false, uniforms: { a: { value: HORIZON }, b: { value: ZENITH } },
  vertexShader: 'varying float h; void main(){ h = normalize(position).y; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
  fragmentShader: 'uniform vec3 a, b; varying float h; void main(){ gl_FragColor = vec4(mix(a, b, smoothstep(0.0, 0.55, h)), 1.0);\n#include <colorspace_fragment>\n}' }));
scene.add(sky);

scene.add(new THREE.HemisphereLight(0xfff6e2, 0xd9c08a, 1.75));
const sun = new THREE.DirectionalLight(0xfff0d6, 1.35);
const SUN_DIR = new THREE.Vector3(-0.55, 0.78, -0.3).normalize();
sun.castShadow = true; sun.shadow.mapSize.set(PHONE ? 2048 : 4096, PHONE ? 2048 : 4096);
Object.assign(sun.shadow.camera, { left: -55, right: 55, top: 55, bottom: -55, near: 1, far: 260 });
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.05; sun.shadow.radius = 3;
scene.add(sun, sun.target);

// ------------------------------------------------------------------ the land
{
  const X0 = -95, X1 = 95, Z0 = -40, Z1 = 320, STEP = 1.0;
  const nx = Math.round((X1 - X0) / STEP) + 1, nz = Math.round((Z1 - Z0) / STEP) + 1;
  const pos = new Float32Array(nx * nz * 3), col = new Float32Array(nx * nz * 3), uv = new Float32Array(nx * nz * 2), idx = [];
  const lo = new THREE.Color(0xe9dcc0), hi = new THREE.Color(0xffffff), path = new THREE.Color(0xfff6e4), c = new THREE.Color();
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const x = X0 + i * STEP, z = Z0 + j * STEP, k = (j * nx + i) * 3;
    pos[k] = x; pos[k + 1] = ground(x, z); pos[k + 2] = z; uv[k / 3 * 2] = x / 7; uv[k / 3 * 2 + 1] = z / 7;
    const n = 0.55 * noise(x * 0.06, z * 0.06) + 0.3 * noise(x * 0.31, z * 0.31) + 0.15 * noise(x * 1.7, z * 1.7);
    c.copy(lo).lerp(hi, n); if (outside(x, z) === 0) c.lerp(path, 0.35);
    col[k] = c.r; col[k + 1] = c.g; col[k + 2] = c.b;
  }
  for (let j = 0; j < nz - 1; j++) for (let i = 0; i < nx - 1; i++) { const a = j * nx + i; idx.push(a, a + nx, a + 1, a + 1, a + nx, a + nx + 1); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3)); g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(idx); g.computeVertexNormals();
  // The ground is real vellum: a blank patch of f73v (Yale scan), mirrored so it tiles, drawn at two sizes so the repeat does not show.
  const vellum = new THREE.TextureLoader().load('./tex/vellum_f73v.webp');
  vellum.wrapS = vellum.wrapT = THREE.RepeatWrapping; vellum.colorSpace = THREE.SRGBColorSpace; vellum.anisotropy = 8;
  const landMat = new THREE.MeshLambertMaterial({ vertexColors: true, map: vellum });
  landMat.onBeforeCompile = s => { s.fragmentShader = s.fragmentShader.replace('#include <map_fragment>',
    'vec4 sampledDiffuseColor = mix(texture2D(map, vMapUv), texture2D(map, vMapUv.yx * 0.19 + 0.37), 0.45); diffuseColor.rgb *= sampledDiffuseColor.rgb * 1.2;'); };
  const land = new THREE.Mesh(g, landMat);
  land.receiveShadow = true; scene.add(land);
}

// ------------------------------------------------------------------ the pond (mirror reflection, tinted, rippled)
const waterU = { time: { value: 0 } };
const water = new Reflector(new THREE.CircleGeometry(POND.r + 1.2, 96), {
  textureWidth: PHONE ? 512 : 1024, textureHeight: PHONE ? 512 : 1024, clipBias: 0.003,
  shader: {
    uniforms: { color: { value: null }, tDiffuse: { value: null }, textureMatrix: { value: null }, time: waterU.time, tint: { value: new THREE.Color(0x5b9aa6) } },
    vertexShader: 'uniform mat4 textureMatrix; varying vec4 vUv; varying vec2 p; void main(){ vUv = textureMatrix * vec4(position, 1.0); p = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: `uniform sampler2D tDiffuse; uniform float time; uniform vec3 tint; varying vec4 vUv; varying vec2 p;
      void main(){
        vec2 r = vec2(sin(p.y * 0.9 + time * 1.1) + sin(p.x * 1.3 - time * 0.8), sin(p.x * 1.1 + time * 0.9) + sin(p.y * 1.6 + time * 0.6)) * 0.0025;
        vec4 uv = vUv; uv.xy += r * uv.w;
        vec3 refl = texture2DProj(tDiffuse, uv).rgb;
        gl_FragColor = vec4(mix(refl, tint, 0.55), 1.0);
        #include <colorspace_fragment>
      }` } });
water.rotation.x = -Math.PI / 2; water.position.set(POND.x, WATER_Y, POND.z); scene.add(water);

// ------------------------------------------------------------------ drawings as cards
const loader = new THREE.TextureLoader(), timeU = { value: 0 };
const cardGeo = new THREE.PlaneGeometry(1, 1); cardGeo.translate(0, 0.5, 0);          // base at y = 0
const fishGeo = new THREE.PlaneGeometry(1, 1);
function texture(kind, name) {
  return new Promise(res => loader.load(`./tex/${TEX}/${name}.webp`, t => { t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; res(t); }, undefined, () => res(null)));
}
function materials(map, sway) {
  const m = new THREE.MeshLambertMaterial({ map, alphaTest: 0.35, alphaToCoverage: true, side: THREE.DoubleSide, emissive: 0xffffff, emissiveMap: map, emissiveIntensity: 0.42, color: 0xd8d8d8 });
  const d = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map, alphaTest: 0.5, side: THREE.DoubleSide });
  if (sway) for (const mat of [m, d]) mat.onBeforeCompile = s => {
    s.uniforms.uTime = timeU;
    s.vertexShader = 'uniform float uTime;\n' + s.vertexShader.replace('#include <begin_vertex>',
      `#include <begin_vertex>
       float ph = instanceMatrix[3][0] * 0.7 + instanceMatrix[3][2] * 1.3;
       transformed.x += (sin(uTime * 1.1 + ph) * 0.022 + sin(uTime * 2.7 + ph * 2.0) * 0.008) * uv.y * uv.y;`);
  };
  return [m, d];
}

const actors = [];                             // things that move, each with update(t, camera)
const tmpV = new THREE.Vector3(), right = new THREE.Vector3();
function actor(tex, aspect, height, geo = cardGeo) {
  const [m, d] = materials(tex, false);
  const mesh = new THREE.Mesh(geo, m); mesh.customDepthMaterial = d; mesh.castShadow = true;
  mesh.scale.set(height * aspect, height, 1); mesh.userData = { w: height * aspect, h: height };
  scene.add(mesh); return mesh;
}
// Turn a card to the camera (about the upright axis only). facing: +1 keep as drawn, -1 mirror.
function faceCamera(mesh, cam, facing = 1, roll = 0) {
  mesh.rotation.set(0, Math.atan2(cam.position.x - mesh.position.x, cam.position.z - mesh.position.z), roll, 'YXZ');
  mesh.scale.x = mesh.userData.w * facing;
}
// For a drawing that faces LEFT on the page: mirror it when it is travelling to the camera's right.
function facingFor(mesh, cam, dx, dz) {
  right.set(cam.position.z - mesh.position.z, 0, -(cam.position.x - mesh.position.x));   // camera's right, seen from the card
  return dx * right.x + dz * right.z > 0 ? -1 : 1;
}

async function build() {
  const A = await (await fetch('./assets.json')).json();
  const R = rng(408);
  // ---- plants: one instanced mesh per folio
  const spots = A.plants.map(() => []);
  const taken = [];
  const free = (x, z, r) => { for (const s of taken) if ((x - s[0]) ** 2 + (z - s[1]) ** 2 < (r + s[2]) ** 2) return false; return true; };
  taken.push([ARCHER.x, ARCHER.z, 3], [LANDMARK.x, LANDMARK.z, 7]);
  const order = A.plants.map((_, i) => i).filter(i => A.plants[i].name !== 'f93r_plant'); for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  let k = 0, tries = 0;
  const WANT = PHONE ? 1300 : 2600;
  while (k < WANT && tries++ < 200000) {
    let x, z;
    if (R() < 0.52) { z = -30 + R() * (VALLEY_END + 40); x = pathX(z) + (R() < 0.5 ? -1 : 1) * (2.4 + Math.abs(R() + R() + R() - 1.5) * 30); }
    else { const a = R() * Math.PI * 2, r = R() < 0.22 ? 8 + R() * (CLEAR.r - 8) : CLEAR.r - 3 + Math.abs(R() + R() - 1) * 55; x = CLEAR.x + Math.cos(a) * r; z = CLEAR.z + Math.sin(a) * r; }
    if (Math.abs(x) > 90 || z < -36 || z > 315) continue;
    const d = outside(x, z);
    if (d === 0 && z < VALLEY_END + 4) continue;                                           // keep the valley floor clear
    if (Math.hypot(x - POND.x, z - POND.z) < POND.r + 1.5) continue;
    if (d === 0 && loopSamples.some(p => Math.hypot(x - p.x, z - p.z) < 3.0)) continue;       // and the camera's route
    const h = d === 0 ? 0.8 + R() * 1.3 : d < 4 ? 1.2 + R() * 2.0 : d < 14 ? 2.2 + R() * 3.4 : 4 + R() * 7;
    const f = order[k % order.length], w = h * A.plants[f].aspect;
    if (!free(x, z, w * 0.3)) continue;
    taken.push([x, z, w * 0.3]);
    // face the open ground, with a little variety
    // in the valley: face back down the path (towards the walker), turned a little inwards; round the clearing: face its middle
    const inValley = z < VALLEY_END - 6 && Math.hypot(x - CLEAR.x, z - CLEAR.z) > CLEAR.r + 4;
    const yaw = inValley ? Math.atan2(-Math.sign(x - pathX(z)) * 0.5, -1) : Math.atan2(CLEAR.x - x, CLEAR.z - z);
    spots[f].push([x, ground(x, z) - 0.04 * h, z, h, w, yaw + (R() - 0.5) * 0.7]);
    k++;
  }
  const giant = A.plants.findIndex(p => p.name === 'f93r_plant');
  if (giant >= 0) spots[giant].push([LANDMARK.x, ground(LANDMARK.x, LANDMARK.z) - 0.6, LANDMARK.z, 27, 27 * A.plants[giant].aspect, Math.atan2(CLEAR.x - LANDMARK.x, CLEAR.z - LANDMARK.z)]);

  const M = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), s = new THREE.Vector3(), p = new THREE.Vector3();
  const plantTex = await Promise.all(A.plants.map(pl => texture('plants', pl.name)));
  A.plants.forEach((pl, i) => {
    if (!plantTex[i] || !spots[i].length) return;
    const [m, d] = materials(plantTex[i], true);
    const im = new THREE.InstancedMesh(cardGeo, m, spots[i].length); im.customDepthMaterial = d; im.castShadow = true; im.frustumCulled = false;
    spots[i].forEach((sp, n) => { e.set(0, sp[5], 0); q.setFromEuler(e); p.set(sp[0], sp[1], sp[2]); s.set(sp[4], sp[3], 1); M.compose(p, q, s); im.setMatrixAt(n, M); });
    scene.add(im);
  });

  const asp = (kind, name) => (A[kind].find(a => a.name === name) || { aspect: 1 }).aspect;
  // ---- nymphs in the pond
  const nymphNames = A.nymphs.map(n => n.name).slice(0, PHONE ? 10 : 14);
  const nymphTex = await Promise.all(nymphNames.map(n => texture('nymphs', n)));
  nymphNames.forEach((name, i) => {
    if (!nymphTex[i]) return;
    const a = i * 2.39996 + 0.4, r = POND.r * (0.25 + 0.5 * ((i * 0.618) % 1)), x0 = POND.x + Math.cos(a) * r, z0 = POND.z + Math.sin(a) * r, ph = R() * 6.28;
    const m = actor(nymphTex[i], asp('nymphs', name), 1.5);
    actors.push((t, cam) => {
      let x = x0, z = z0, dy = 0.05 * Math.sin(t * 0.9 + ph), dx = 0, dz = 0;
      if (i % 3 === 0) { const w = 0.16 * (i % 2 ? 1 : -1), c = t * w + ph; x += Math.cos(c) * 1.5; z += Math.sin(c) * 1.5; dx = -Math.sin(c) * w; dz = Math.cos(c) * w; }   // wading round
      if (i % 3 === 1) dy -= 0.45 * Math.max(0, Math.sin(t * 0.35 + ph)) ** 6;                                                                                           // ducking under
      m.position.set(x, WATER_Y - 0.5 + dy, z);
      faceCamera(m, cam, dx || dz ? facingFor(m, cam, dx, dz) : 1, 0.035 * Math.sin(t * 0.7 + ph));
    });
  });
  // ---- fish leaping (f70v2 Pisces)
  const fishT = await texture('animals', 'fish_pisces_70v2');
  if (fishT) for (let i = 0; i < 6; i++) {
    const m = actor(fishT, asp('animals', 'fish_pisces_70v2'), 0.5, fishGeo), period = 5.5 + i * 0.9, off = i * 1.7;
    actors.push((t, cam) => {
      const n = Math.floor((t + off) / period), u = ((t + off) / period - n) / 0.24;      // in the air for a quarter of each cycle
      if (u > 1) { m.visible = false; return; }
      const g = rng(n * 31 + i * 7 + 1), a0 = g() * 6.28, r0 = POND.r * (0.2 + 0.5 * g()), a1 = a0 + 0.9 + g();
      const x0 = POND.x + Math.cos(a0) * r0, z0 = POND.z + Math.sin(a0) * r0, x1 = x0 + Math.cos(a1) * 3.2, z1 = z0 + Math.sin(a1) * 3.2;
      if (Math.hypot(x1 - POND.x, z1 - POND.z) > POND.r - 1) { m.visible = false; return; }
      m.visible = true; m.position.set(x0 + (x1 - x0) * u, WATER_Y - 0.15 + 1.5 * 4 * u * (1 - u), z0 + (z1 - z0) * u);
      const f = facingFor(m, cam, x1 - x0, z1 - z0), climb = Math.atan2(1.5 * 4 * (1 - 2 * u), 3.2);
      faceCamera(m, cam, f, f * -climb);
    });
  }
  // ---- Aries rams bounding across the meadow (f70v1 and f71r)
  const ramT = [await texture('animals', 'ram_aries_70v1'), await texture('animals', 'ram_aries_71r')], ramN = ['ram_aries_70v1', 'ram_aries_71r'];
  for (let i = 0; i < 6; i++) {
    const w = i % 2; if (!ramT[w]) continue;
    const m = actor(ramT[w], asp('animals', ramN[w]), 1.3), zr = CLEAR.z - 22 + i * 3.1, lag = i * 0.85, SPAN = 44, V = 5.2, HOP = 3.4;
    actors.push((t, cam) => {
      const leg = SPAN / V + 3.5, n = Math.floor((t - lag) / leg), u = (t - lag) - n * leg, dir = n % 2 ? -1 : 1;
      const s = Math.min(SPAN, Math.max(0, u * V)), x = CLEAR.x - 8 + dir * (s - SPAN / 2), z = zr + 2.5 * Math.sin(s * 0.11 + i);
      const moving = u * V < SPAN, hop = moving ? Math.abs(Math.sin(Math.PI * s / HOP)) : 0;
      m.position.set(x, ground(x, z) + 0.75 * hop, z);
      const f = facingFor(m, cam, dir, 0), tilt = moving ? 0.3 * Math.cos(Math.PI * ((s / HOP) % 1)) : 0;
      faceCamera(m, cam, f, f * -tilt);
    });
  }
  // ---- the Sagittarius archer (f73v): leans into the aim and recoils. No bolt is drawn, because the page shows none in flight.
  const archT = await texture('animals', 'archer_sagittarius_73v');
  if (archT) {
    const m = actor(archT, asp('animals', 'archer_sagittarius_73v'), 2.0); m.position.set(ARCHER.x, ground(ARCHER.x, ARCHER.z) - 0.05, ARCHER.z);
    actors.push((t, cam) => { const u = (t % 7) / 7, aim = 0.07 * sstep(0, 0.8, u), kick = u > 0.82 ? -0.16 * Math.exp(-(u - 0.82) * 38) * Math.sin((u - 0.82) * 90) : 0; faceCamera(m, cam, 1, aim * (1 - sstep(0.8, 0.84, u)) + kick); });
  }
  // ---- the f76v creature crossing the valley floor now and then
  const creT = await texture('animals', 'creature_76v');
  if (creT) [40, 96, 150].forEach((z0, i) => {
    const m = actor(creT, asp('animals', 'creature_76v'), 0.8);
    actors.push((t, cam) => {
      const u = ((t + i * 9) % 26) / 9; if (u > 1) { m.visible = false; return; }
      const dir = i % 2 ? -1 : 1, x = pathX(z0) + dir * (u * 14 - 7), z = z0 + 6 + u * 2; m.visible = true;
      m.position.set(x, ground(x, z) + 0.28 * Math.abs(Math.sin(u * 14 * Math.PI / 1.6)), z); faceCamera(m, cam, facingFor(m, cam, dir, 0));
    });
  });
}

// ------------------------------------------------------------------ running it
const pos = new THREE.Vector3(), tgt = new THREE.Vector3();
function renderAt(t) {
  t = ((t % DURATION) + DURATION) % DURATION;
  cameraAt(t, pos, tgt); camera.position.copy(pos); camera.lookAt(tgt);
  timeU.value = t; water.material.uniforms.time.value = t;
  sun.target.position.set(pos.x, 0, pos.z).addScaledVector(tmpV.subVectors(tgt, pos).setY(0).normalize(), 28);
  sun.position.copy(sun.target.position).addScaledVector(SUN_DIR, 130); sun.target.updateMatrixWorld();
  sky.position.copy(pos);
  for (const a of actors) a(t, camera);
  renderer.render(scene, camera);
}
function resize() { const w = innerWidth, h = innerHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.fov = w < h ? 78 : 58; camera.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();

const ui = { play: document.getElementById('play'), scrub: document.getElementById('scrub'), clock: document.getElementById('clock'), speed: document.getElementById('speed') };
let T = +Q.get('t') || 0, playing = !Q.get('still'), rate = 1, last = 0;
const mmss = s => Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
ui.play.onclick = () => { playing = !playing; ui.play.textContent = playing ? 'Pause' : 'Play'; };
ui.speed.onclick = () => { rate = rate === 1 ? 2 : rate === 2 ? 0.5 : 1; ui.speed.textContent = rate + '×'; };
ui.scrub.oninput = () => { T = +ui.scrub.value; };
addEventListener('keydown', e => { if (e.key === ' ') ui.play.onclick(); if (e.key === 'h') document.body.classList.toggle('clean'); });

window.journey = { renderAt, DURATION, ready: false };
build().then(() => {
  document.getElementById('load').remove(); window.journey.ready = true;
  renderer.setAnimationLoop(now => {
    const dt = Math.min(0.1, (now - last) / 1000 || 0); last = now;
    if (window.journey.manual) return;                      // frame-by-frame export drives renderAt itself
    if (playing) T = (T + dt * rate) % DURATION;
    renderAt(T); ui.scrub.value = T; ui.clock.textContent = mmss(T) + ' / ' + mmss(DURATION);
  });
}).catch(err => { document.getElementById('load').textContent = 'Could not load: ' + err.message; });

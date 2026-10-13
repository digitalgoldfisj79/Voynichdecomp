/* VOYNICH WALK (SCROLL) — the main file.
   Loads the pictures and videos, builds the world, moves the camera when you
   scroll, and handles clicks. You normally edit these instead:
     walk.config.js  -> the stations (videos, titles, notes)
     settings.js     -> every number
     js/world.js     -> how the world is laid out and painted
     js/ui.js        -> the info card, the video viewer, the panel, recording */

import * as THREE from '../lib/three.module.min.js';
import { SETTINGS as S } from '../settings.js';
import { WALK } from '../walk.config.js';
import { ASSETS } from '../assets.js';
import { makeRng, makeNoise } from './random.js';
import { buildWorld } from './world.js';
import { makeUI } from './ui.js';
import { makeGuide } from './guide.js';
import { YOUTUBE } from '../youtube.js';
import { makeFramePlayer } from './youtube.js';

const params = new URLSearchParams(location.search);
if (params.has('seed')) S.seed = parseInt(params.get('seed')) || S.seed;
for (const [k, v] of params) if (k.includes('.')) {          // any setting from the address bar, e.g. ?plants.size=1.4
  const path = k.split('.'), last = path.pop(); let o = S;
  for (const p of path) o = o?.[p];
  if (o && last in o) o[last] = typeof o[last] === 'number' ? parseFloat(v) : typeof o[last] === 'boolean' ? v === 'true' : v;
}
const PACK = window.PACKED_ASSETS || {};                    // a single-file copy carries its pictures and videos inside
const resolve = f => PACK[f] || f;
// videos packed inside a single-file copy are turned back into a playable file in memory
const resolveMedia = f => {
  const d = PACK[f]; if (!d || !d.startsWith('data:')) return d || f;          // already unpacked, or a normal file
  const [head, b64] = d.split(','), bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return (PACK[f] = URL.createObjectURL(new Blob([bytes], { type: head.slice(5).split(';')[0] })));
};

// ---- the stations: fill in title, note and picture from the YouTube list ------------------
const youtubeId = s => { s = String(s || '').trim(); if (/^[\w-]{11}$/.test(s)) return s; const m = s.match(/(?:v=|youtu\.be\/|embed\/|shorts\/|live\/)([\w-]{11})/); return m ? m[1] : null; };
function firstSentence(desc) {                               // the description's opening paragraph, without its links
  const para = String(desc || '').trim().split(/\n\s*\n/)[0].replace(/\n/g, ' ').replace(/(https?:\/\/|www\.)\S+/g, '\u00a7');
  if (/^[*#]/.test(para) || /\(\d{4}\)/.test(para) || /^\d+:\d\d/.test(para)) return '';   // a footnote, references or chapter times, not a summary
  let parts = para.match(/[^.!?]+[.!?]+(\s|$)|[^.!?]+$/g) || [];
  parts = parts.length === 1 ? [parts[0].replace(/\u00a7/g, '')] : parts.filter(s => !s.includes('\u00a7'));
  let out = ''; for (const s of parts) { if (out && (out + s).length > 200) break; out += s; }
  out = out.replace(/\s+/g, ' ').trim();
  return out.length < 12 ? '' : out.length > 230 ? out.slice(0, 220).replace(/\s+\S*$/, '') + '…' : out;
}
for (const st of WALK.stations) {
  st.ytId = youtubeId(st.youtube);
  const y = st.ytId && (YOUTUBE.videos || []).find(v => v.id === st.ytId);
  st.title = st.title || (y && y.title) || (st.ytId ? 'Voynich Talk' : 'Untitled');
  if (st.note == null) st.note = y ? firstSentence(y.description) : '';
  st.published = y ? y.published : '';
  st.thumb = st.thumb || (y && y.thumb) || (st.ytId ? `https://i.ytimg.com/vi/${st.ytId}/hqdefault.jpg` : null);
  st.clip = st.clip || st.video || null;                      // 'video' was the old name
  st.ytUrl = st.ytId ? `https://www.youtube.com/watch?v=${st.ytId}` : null;
}
// play YouTube itself inside the frames? Not in a page opened from the folder (YouTube refuses), and not in the preview copy
const LIVE_YT = S.videos.youtubeInFrame && !window.PREVIEW && /^https?:$/.test(location.protocol) && WALK.stations.some(st => st.ytId && !st.clip);

// ---- renderer, scene, camera ------------------------------------------------------
const canvas = document.getElementById('world');
// with YouTube playing in the frames the picture needs see-through "windows", so it has an alpha channel
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance', alpha: LIVE_YT, preserveDrawingBuffer: false });
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(S.view.horizon, S.view.fogNear, S.view.fogFar);
scene.background = skyGradient();
scene.add(new THREE.HemisphereLight(0xffffff, 0xb8a888, 1.55));
const sunLight = new THREE.DirectionalLight(0xfff1d8, 0.7); sunLight.position.set(10, 20, 15); scene.add(sunLight);
const camera = new THREE.PerspectiveCamera(S.camera.fov, 16 / 9, 0.1, 300);
const framePlayer = makeFramePlayer({ S, WALK, enabled: LIVE_YT, renderer, camera, canvas });

function skyGradient() {
  const c = document.createElement('canvas'); c.width = 2; c.height = 256; const g = c.getContext('2d');
  const grd = g.createLinearGradient(0, 0, 0, 256); grd.addColorStop(0, S.view.sky); grd.addColorStop(0.62, S.view.horizon); grd.addColorStop(1, S.view.horizon);
  g.fillStyle = grd; g.fillRect(0, 0, 2, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// ---- load the pictures, fonts and videos -------------------------------------------------
const textures = new Map();
async function loadTextures() {
  const files = [...ASSETS.plants.map(p => p.file), ...ASSETS.creatures.map(c => c.file), ASSETS.sky.sun, ...WALK.stations.map(st => st.thumb)].filter(Boolean);
  const loader = new THREE.TextureLoader(), bar = document.getElementById('bar');
  let done = 0;
  await Promise.all(files.map(f => new Promise(res => {
    loader.load(resolve(f), t => { t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = renderer.capabilities.getMaxAnisotropy(); textures.set(f, t); bar.style.width = (++done / files.length * 100) + '%'; res(); },
      undefined, () => { console.warn('Could not load', f); bar.style.width = (++done / files.length * 100) + '%'; res(); });
  })));
}
function makeVideo(src) {
  const v = document.createElement('video');
  v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'auto'; v.crossOrigin = 'anonymous';
  // MP4 (H.264) plays in Chrome, Edge, Safari and Firefox; if a browser can't, use a .webm of the same name
  const webm = src.replace(/\.mp4$/i, '.webm');
  v.src = resolveMedia(!v.canPlayType('video/mp4; codecs="avc1.42E01E"') && webm !== src && !PACK[src] ? webm : src);
  const texture = new THREE.VideoTexture(v); texture.colorSpace = THREE.SRGBColorSpace;
  let active = false;
  return { el: v, src, texture, setActive(on) { if (on === active) return; active = on; on ? v.play().catch(() => {}) : v.pause(); } };
}
const videos = WALK.stations.map(st => st.clip ? makeVideo(st.clip) : null);

// ---- build ----------------------------------------------------------------------------------
let world = null, guide = { mesh: null, update() {}, hit() { return false; } };
function build() {
  if (world) { scene.remove(world.group); world.dispose(); }
  world = buildWorld({ S, WALK, ASSETS, textures, rng: makeRng(S.seed), noise: makeNoise(S.seed), videos });
  scene.add(world.group);
}

// ---- moving along: scroll, keys, touch, drag, glide and tour ---------------------------------------
const walk = { pos: 0, target: 0, glide: false, tour: null, keys: new Set() };
const wrap = x => ((x % world.SPAN) + world.SPAN) % world.SPAN;
const clampT = () => { if (!S.walk.loop) walk.target = THREE.MathUtils.clamp(walk.target, 0, world.SPAN - 14); };
const takeOver = () => { walk.glide = false; walk.tour = null; };
addEventListener('wheel', e => {
  if (!world || ui.modalOpen()) return; takeOver();
  let d = e.deltaY + e.deltaX; if (e.deltaMode === 1) d *= 16; else if (e.deltaMode === 2) d *= 400;   // lines or pages -> pixels
  walk.target += THREE.MathUtils.clamp(d, -160, 160) * 0.012 * S.walk.scrollSpeed; clampT();            // a hard flick can't fling you miles
}, { passive: true });
addEventListener('keydown', e => {
  if (!world || ui.modalOpen() || (e.target.closest && e.target.closest('.lil-gui'))) return;
  if (['ArrowRight', 'ArrowLeft', 'KeyD', 'KeyA'].includes(e.code)) { takeOver(); walk.keys.add(e.code); }
  if (e.code === 'Space' || e.code === 'PageDown') { e.preventDefault(); takeOver(); jumpStation(1); }
  if (e.code === 'PageUp') { takeOver(); jumpStation(-1); }
  if (e.code === 'KeyP') { walk.tour = null; walk.glide = !walk.glide; }
  if (e.code === 'KeyT') { walk.glide = false; walk.tour ? (walk.tour = null) : startTour(); }
});
addEventListener('keyup', e => walk.keys.delete(e.code));
addEventListener('blur', () => walk.keys.clear());

// dragging the picture sideways also walks; a click without a drag picks things
let drag = null;
canvas.addEventListener('pointerdown', e => { if (!world) return; drag = { x: e.clientX, y: e.clientY, moved: false }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e => {
  mouse.x = e.clientX / innerWidth * 2 - 1; mouse.y = e.clientY / innerHeight * 2 - 1;
  if (!world) return;
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) > 6) { drag.moved = true; takeOver(); }
    if (drag.moved) { walk.target -= (dx + dy * 0.5) * 0.03 * S.walk.scrollSpeed; clampT(); drag.x = e.clientX; drag.y = e.clientY; }
  } else hover(e);
});
canvas.addEventListener('pointerup', e => { if (drag && !drag.moved) pick(e); drag = null; });

function stationStop(i) { return world.stationX[i] - 1.5; }               // where the camera stands to face station i
function forwardTo(x) { let d = wrap(x) - wrap(walk.pos); if (S.walk.loop && d < 0.5) d += world.SPAN; return d; }
function jumpStation(dir) {
  const here = wrap(walk.pos), n = world.stationX.length;
  let best = null;
  for (let i = 0; i < n; i++) {
    let d = stationStop(i) - here; if (S.walk.loop) { if (dir > 0 && d < 0.5) d += world.SPAN; if (dir < 0 && d > -0.5) d -= world.SPAN; }
    if ((dir > 0 && d > 0.5) || (dir < 0 && d < -0.5)) if (!best || Math.abs(d) < Math.abs(best)) best = d;
  }
  if (best !== null) { walk.target = walk.pos + best; clampT(); }
}
function startTour() {
  const here = wrap(walk.pos); let i = world.stationX.findIndex((_, k) => stationStop(k) > here - 0.5); if (i < 0) i = 0;
  walk.glide = false; walk.tour = { i, phase: 'move', t: 0, hold: S.glide.tourHold };
}
const titleStop = () => S.walk.lead * 0.42 - 1.5;
const endStop = () => world.SPAN - 14 - 1.5;
// what is the camera standing in front of? (the guide talks about it)
function nearWhat(cx) {
  for (let i = 0; i < world.stationX.length; i++) if (Math.abs(cx - stationStop(i)) < 3) return { kind: 'station', index: i };
  if (Math.abs(cx - titleStop()) < 4.5) return { kind: 'intro' };
  if (Math.abs(cx - endStop()) < 4) return { kind: 'end' };
  return null;
}

// ---- picking: what did you click on? ----------------------------------------------------------------
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(), ndc = new THREE.Vector2();
const alphaCache = new Map();
function solidAt(hit) {                                     // ignore clicks on the transparent parts of a cut-out
  const map = hit.object.material.map;
  if (!map || map.isVideoTexture || !hit.uv) return true;
  let a = alphaCache.get(map.uuid);
  if (!a) {
    const img = map.image, N = 128, c = document.createElement('canvas'); c.width = N; c.height = N;
    const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(img, 0, 0, N, N);
    a = g.getImageData(0, 0, N, N).data; alphaCache.set(map.uuid, a);
  }
  const u = ((hit.uv.x * map.repeat.x + map.offset.x) % 1 + 1) % 1, v = 1 - hit.uv.y;
  return a[(Math.min(127, Math.floor(v * 128)) * 128 + Math.min(127, Math.floor(u * 128))) * 4 + 3] > 100;
}
function hitAt(e) {
  ndc.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  for (const h of ray.intersectObjects(world.clickables, false)) if (h.object.visible && solidAt(h)) return h.object.userData.info;
  return null;
}
let hoverT = 0;
function hover(e) { if (performance.now() - hoverT < 60) return; hoverT = performance.now(); const g = guide.mesh && (ray.setFromCamera(ndc.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1), camera), guide.hit(ray)); canvas.classList.toggle('pointing', !!(g || hitAt(e))); }
function pick(e) {
  ndc.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1); ray.setFromCamera(ndc, camera);
  if (guide.mesh && guide.hit(ray)) { guide.poke(); return; }                 // clicked the guide
  const info = hitAt(e); if (info) { walk.glide = false; walk.tour = null; ui.showInfo(info); }
}

// ---- size ---------------------------------------------------------------------------------------------
function resize() {
  let w = innerWidth, h = innerHeight;
  if (S.record.fixedSize) {
    renderer.setPixelRatio(1); renderer.setSize(S.record.width, S.record.height, false);
    const s = Math.min(innerWidth / S.record.width, innerHeight / S.record.height);
    canvas.style.width = S.record.width * s + 'px'; canvas.style.height = S.record.height * s + 'px';
    w = S.record.width; h = S.record.height;
  } else { renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5)); renderer.setSize(w, h, false); canvas.style.width = '100%'; canvas.style.height = '100%'; }
  camera.aspect = w / h;
  // narrow screens: widen the view so a whole video frame still fits
  camera.fov = w / h >= 1.2 ? S.camera.fov : 2 * Math.atan(Math.tan(S.camera.fov * Math.PI / 360) * 1.2 / (w / h)) * 180 / Math.PI;
  camera.updateProjectionMatrix();
  framePlayer.resize();
}
addEventListener('resize', resize);

// ---- the loop ----------------------------------------------------------------------------------------
let last = performance.now(), time = 0, frames = 0;
const look = new THREE.Vector3(), par = new THREE.Vector2();
function tick(dt) {
  time += dt;
  if (walk.keys.has('ArrowRight') || walk.keys.has('KeyD')) { walk.target += 7 * dt * S.walk.scrollSpeed; clampT(); }
  if (walk.keys.has('ArrowLeft') || walk.keys.has('KeyA')) { walk.target -= 7 * dt * S.walk.scrollSpeed; clampT(); }
  if (walk.glide) { walk.target += S.glide.speed * dt; clampT(); }
  if (walk.tour) {
    const T = walk.tour, d = forwardTo(stationStop(T.i));
    if (T.phase === 'move') {
      const step = Math.min(d, S.glide.tourSpeed * dt * Math.min(1, 0.35 + d / 6));      // ease in to each stop
      walk.target += step;
      if (d < 0.05 || d > world.SPAN - 0.5) { T.phase = 'hold'; T.t = 0; }
    } else if ((T.t += dt) > (T.hold = S.glide.tourHold)) {
      T.i = (T.i + 1) % world.stationX.length; T.phase = 'move';
      if (T.i === 0 && !S.walk.loop) walk.tour = null;
    }
  }
  walk.pos += (walk.target - walk.pos) * (1 - Math.exp(-S.walk.smoothing * dt));
  const cx = S.walk.loop ? wrap(walk.pos) : walk.pos;
  par.lerp(mouse, 1 - Math.exp(-3 * dt));
  // the camera drifts gently up and down and in and out as you go, like being carried
  const k = 2 * Math.PI / world.SPAN, sway = Math.sin(cx * k * 7) * 0.18 + Math.sin(cx * k * 3 + 1) * 0.12;
  camera.position.set(cx + par.x * S.camera.mouseParallax, S.camera.height + sway - par.y * S.camera.mouseParallax * 0.5, S.camera.distance + Math.sin(cx * k * 5 + 2) * 0.6);
  look.set(cx + 1.5, 1.95 + sway * 0.5, -2.5); camera.lookAt(look);
  scene.fog.near = S.view.fogNear; scene.fog.far = S.view.fogFar;
  world.update(dt, time, cx);
  framePlayer.update(dt, cx, world);
  guide.update(dt, time, walk.pos, cx, walk, nearWhat(cx));
  ui.update(cx, walk);
}
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  tick(dt); renderer.render(scene, camera); framePlayer.render(); frames++;
  requestAnimationFrame(frame);
}

const ui = makeUI({ S, WALK, renderer, canvas, camera, scene, textures, resolve, resolveMedia, videos, rebuild: () => build(), resize, getWorld: () => world, walk, startTour, framePlayer });
await Promise.all([loadTextures(), document.fonts.load('64px "IM Fell English"').catch(() => {}), document.fonts.load('italic 64px "IM Fell English"').catch(() => {})]);
build();
resize();
walk.pos = walk.target = titleStop();                        // start facing the title card
guide = makeGuide({ S, WALK, textures, scene, camera, actions: {
  tour: () => startTour(), pauseTour: () => { walk.tour = null; }, next: () => { takeOver(); jumpStation(1); },
  watch: i => ui.openVideo(i), restart: () => { takeOver(); walk.target = walk.pos + forwardTo(titleStop()); },
  hello: () => guide.hello(), pointerX: () => par.x,
  blocked: () => ui.modalOpen() || document.body.classList.contains('clean'),
} });
// draw every stop once behind the loading screen, so pictures are on the graphics card before you scroll
for (const x of [titleStop(), ...world.stationX.map((_, i) => stationStop(i)), endStop()]) {
  camera.position.set(x, S.camera.height, S.camera.distance); camera.lookAt(x + 1.5, 1.95, -2.5); renderer.render(scene, camera);
}
document.getElementById('loading').classList.add('done');
requestAnimationFrame(t => { last = t; frame(t); });

// for testing from the browser console
window.__walk = { S, scene, camera, walk, renderer, get world() { return world; }, get frames() { return frames; },
  advance(seconds) { for (let t = 0; t < seconds; t += 1 / 30) tick(1 / 30); renderer.render(scene, camera); }, ui,
  pickAt(x, y) { return hitAt({ clientX: x, clientY: y }); }, get guide() { return guide; }, framePlayer, WALK };

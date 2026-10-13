/* THE WORLD: a paper theatre seen from the side.

   Think of a toy theatre: flat painted layers stand one behind the other, and you
   slide along in front of them. From back to front:
     sky  ->  far hills (pale)  ->  near hills  ->  a bank  ->  tall plants
     ->  the video stations and smaller plants  ->  the path  ->  a few big
     plants right at the front edge.
   Because the layers are at different distances, they slide past at different
   speeds as you move (parallax), which is what makes it feel deep.

   The walk is a loop: whatever is near the end is also copied just before the
   start, so going round from the last station to the first has no visible join. */

import * as THREE from '../lib/three.module.min.js';

const TILE = 12;          // the ground picture repeats every 12 m
const VIEW = 26;          // how far either side of the camera can be seen (for the loop copies)
const PANEL_Z = -2.6;     // where the video frames stand

export function buildWorld({ S, WALK, ASSETS, textures, rng, noise, videos }) {
  const group = new THREE.Group();
  const clickables = [];
  const n = WALK.stations.length;
  const raw = S.walk.lead + n * S.walk.stationSpacing + 18;
  const SPAN = Math.ceil(raw / TILE) * TILE;                 // a whole number of ground tiles, so the loop is seamless
  const stationX = WALK.stations.map((_, i) => S.walk.lead + (i + 0.5) * S.walk.stationSpacing);
  const wind = { uTime: { value: 0 }, uSway: { value: S.plants.sway }, uSwaySpeed: { value: S.plants.swaySpeed } };
  const disposables = [];
  const keep = x => { disposables.push(x); return x; };

  // ---- ground ---------------------------------------------------------------------
  const groundTex = keep(paintGround(S, rng));
  groundTex.repeat.set((SPAN + 2 * VIEW) / TILE, 1); groundTex.offset.x = -VIEW / TILE;
  const ground = new THREE.Mesh(keep(new THREE.PlaneGeometry(SPAN + 2 * VIEW, 36).rotateX(-Math.PI / 2)), keep(new THREE.MeshLambertMaterial({ map: groundTex })));
  ground.position.set(SPAN / 2, 0, 2);                     // runs from 16 m behind the path to 20 m in front
  group.add(ground);

  // ---- painted hill flats ------------------------------------------------------------
  const flats = [
    { z: -34, h: 24, y: -3, top: 0.30, rough: 0.20, col: ['#b9c7ad', '#a7b99d'], ink: 0.25, seed: 1 },
    { z: -21, h: 13, y: -1, top: 0.35, rough: 0.30, col: ['#93ad83', '#7f9c74'], ink: 0.45, seed: 2 },
    { z: -12.5, h: 5, y: -0.4, top: 0.25, rough: 0.35, col: ['#7d9a6c', '#6b8a5f'], ink: 0.6, seed: 3 },
  ];
  for (const f of flats) {
    const tex = keep(paintFlat(SPAN, f, rng, noise));
    tex.repeat.set((SPAN + 2 * VIEW) / SPAN, 1); tex.offset.x = -VIEW / SPAN;
    const m = new THREE.Mesh(keep(new THREE.PlaneGeometry(SPAN + 2 * VIEW, f.h)), keep(new THREE.MeshLambertMaterial({ map: tex, alphaTest: 0.5 })));
    m.position.set(SPAN / 2, f.y + f.h / 2, f.z);
    group.add(m);
  }

  // ---- the sun (or moon), hung on strings like a theatre prop ----------------------------
  const sunTex = textures.get(ASSETS.sky.sun) || keep(paintSun());
  const sun = new THREE.Mesh(keep(new THREE.PlaneGeometry(6, 6)), keep(new THREE.MeshBasicMaterial({ map: sunTex, transparent: true, fog: false, depthWrite: false })));
  const sunStrings = new THREE.LineSegments(keep(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-1.2, 2.6, 0), new THREE.Vector3(-1.2, 30, 0), new THREE.Vector3(1.2, 2.6, 0), new THREE.Vector3(1.2, 30, 0)])),
    keep(new THREE.LineBasicMaterial({ color: 0x6b5a44, transparent: true, opacity: 0.5, fog: false })));
  sun.add(sunStrings); sun.renderOrder = -1;
  group.add(sun);

  // ---- plants -----------------------------------------------------------------------------
  const byFolio = new Map(ASSETS.plants.map(p => [p.folio, p]));
  const wanted = new Set(WALK.stations.flatMap(s => s.plants || []));
  const pool = rng.shuffle(ASSETS.plants.filter(p => textures.get(p.file) && !wanted.has(p.folio))).slice(0, S.plants.variety);
  const kindCache = new Map();
  const kindFor = p => {
    if (!kindCache.has(p.file)) {
      const tex = textures.get(p.file), aspect = tex.image.width / tex.image.height;
      const geo = keep(new THREE.PlaneGeometry(aspect, 1)); geo.translate(0, 0.5, 0);
      kindCache.set(p.file, { ...p, aspect, geo, mat: keep(plantMaterial(tex, wind)) });
    }
    return kindCache.get(p.file);
  };
  const plants = [];
  const shadowTex = keep(blobTexture());
  const shadowMat = keep(new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false, opacity: 0.5, color: 0x4a3a28 }));
  const shadowGeo = keep(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2));
  const pond = WALK.stations.findIndex(s => s.pond) >= 0 ? { x: stationX[WALK.stations.findIndex(s => s.pond)] + 10, z: -1.1, rx: 5, rz: 2.2 } : null;
  const inPond = (x, z, pad = 0) => pond && ((x - pond.x) / (pond.rx + pad)) ** 2 + ((z - pond.z) / (pond.rz + pad)) ** 2 < 1;
  // keep the space in front of each video frame and each title card clear of plants
  const clear = [...stationX.map(x => [x, S.videos.width / 2 + 1.4]), [S.walk.lead * 0.42, 4.2], [SPAN - 14, 3.8]];
  const nearPanel = (x, z) => z > PANEL_Z - 0.6 && z < 0.8 && clear.some(([cx, hw]) => Math.abs(x - cx) < hw);

  // Plants are placed in two steps: first decide every spot, then draw all plants of the
  // same picture in one go (an "instanced" mesh). Far fewer draw calls = smoother scrolling.
  const spots = [];
  function addPlant(p, x, z, h, opts = {}) {
    if (!opts.force && (inPond(x, z, 0.8) || nearPanel(x, z))) return;
    spots.push({ p, x, z: z + spots.length * 0.0007, h, ry: 0 });      // flat to the viewer, like stage scenery; never two on the same plane
  }
  const D = S.plants.density, Z = S.plants.size;
  // clusters: a few plants of one or two kinds close together, then a gap
  const clusters = (gap, zr, hr, per) => {
    if (!pool.length) return;
    for (let x = rng.range(0, gap); x < SPAN; x += gap / D * rng.range(0.7, 1.4)) {
      const a = rng.pick(pool), b = rng.pick(pool), n = rng.int(per[0], per[1]), zc = rng.range(...zr);
      for (let k = 0; k < n; k++) addPlant(rng() < 0.7 ? a : b, x + rng.range(-1.3, 1.3), THREE.MathUtils.clamp(zc + rng.range(-0.9, 0.9), zr[0], zr[1]), rng.range(...hr) * Z * (k === 0 ? 1.15 : 1));
    }
  };
  clusters(3.2, [-10.5, -6.5], [3.0, 7.0], [2, 4]);     // tall plants at the back
  clusters(4.2, [-5.4, -0.4], [1.0, 2.7], [2, 4]);      // the row beside the path
  clusters(19, [6.6, 8.2], [1.4, 2.4], [1, 2]);         // now and then a big one right at the front edge (in front of the guide)
  // the station's own plants flank its frame
  WALK.stations.forEach((st, i) => (st.plants || []).forEach((fo, k) => {
    const p = byFolio.get(fo); if (!p || !textures.get(p.file)) return;
    const side = k % 2 ? 1 : -1, x = stationX[i] + side * (S.videos.width / 2 + 1.1 + Math.floor(k / 2) * 1.6);
    addPlant(p, x, rng.range(-2.3, -1.2), rng.range(2.4, 3.3) * Z, { force: true });
  }));
  // loop copies: plants near either end also stand just past the other end
  for (const s of spots.slice()) {
    if (s.x < VIEW) spots.push({ ...s, x: s.x + SPAN });
    if (s.x > SPAN - VIEW) spots.push({ ...s, x: s.x - SPAN });
  }
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), v3 = new THREE.Vector3(), sc = new THREE.Vector3();
  const byKind = new Map();
  for (const s of spots) { const k = kindFor(s.p); (byKind.get(k) || byKind.set(k, []).get(k)).push(s); }
  for (const [k, list] of byKind) {
    const im = new THREE.InstancedMesh(k.geo, k.mat, list.length);
    list.forEach((s, i) => im.setMatrixAt(i, m4.compose(v3.set(s.x, -0.05 * s.h, s.z), q.setFromEuler(e.set(0, s.ry, 0)), sc.setScalar(s.h))));
    im.userData.info = { kind: 'plant', folio: k.folio, iiif: k.iiif, section: k.section, file: k.file };
    im.userData.noCopy = true; im.frustumCulled = false;
    group.add(im); clickables.push(im); plants.push(im);
  }
  const shadows = new THREE.InstancedMesh(shadowGeo, shadowMat, spots.length);
  spots.forEach((s, i) => shadows.setMatrixAt(i, m4.compose(v3.set(s.x, 0.02, s.z + 0.1), q.identity(), sc.set(s.h * kindFor(s.p).aspect * 0.8, 1, s.h * 0.25))));
  shadows.renderOrder = 1; shadows.userData.noCopy = true; shadows.frustumCulled = false;
  group.add(shadows);

  // ---- video stations --------------------------------------------------------------------------
  const stations = WALK.stations.map((st, i) => {
    const g = new THREE.Group(), w = S.videos.width, h = w * 9 / 16;
    g.position.set(stationX[i], 0, PANEL_Z);
    const frame = new THREE.Mesh(keep(new THREE.PlaneGeometry(w + 0.7, h + 0.7)), keep(new THREE.MeshLambertMaterial({ map: keep(paintFrame(i + 1)) })));
    frame.position.y = 1.15 + h / 2;
    const vid = videos[i], still = st.thumb && textures.get(st.thumb);
    // a clip plays in the frame; otherwise the YouTube picture (js/youtube.js plays the video itself over it when you're near)
    const map = vid ? vid.texture : still ? keep(paintStill(still.image)) : keep(paintPoster(st.title));
    const screen = new THREE.Mesh(keep(new THREE.PlaneGeometry(w, h)), keep(new THREE.MeshBasicMaterial({ map, fog: false })));
    screen.position.set(0, frame.position.y, 0.02);
    for (const sx of [-w / 2 + 0.4, w / 2 - 0.4]) {                    // two posts
      const post = new THREE.Mesh(keep(new THREE.BoxGeometry(0.12, 1.4, 0.08)), keep(new THREE.MeshLambertMaterial({ color: 0x6b5236 })));
      post.position.set(sx, 0.7, -0.05); g.add(post);
    }
    const label = new THREE.Mesh(keep(new THREE.PlaneGeometry(w * 0.86, w * 0.86 / 6)), keep(new THREE.MeshBasicMaterial({ map: keep(paintLabel(st.label || st.title)), transparent: true })));
    label.position.set(0, 0.62, 0.3);
    g.add(frame, screen, label);
    const info = { kind: 'station', index: i };
    frame.userData.info = screen.userData.info = label.userData.info = info;
    clickables.push(frame, screen, label);
    group.add(g);
    return { x: stationX[i], group: g, screen, video: vid };
  });

  // ---- title and closing cards ------------------------------------------------------------------
  const card = (lines, x, w, h, big) => {
    const m = new THREE.Mesh(keep(new THREE.PlaneGeometry(w, h)), keep(new THREE.MeshLambertMaterial({ map: keep(paintCard(lines, w / h, big)), transparent: true, alphaTest: 0.3 })));
    m.position.set(x, 1.0 + h / 2, PANEL_Z + 0.2);
    const strings = new THREE.LineSegments(keep(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-w * 0.35, h / 2 - 0.1, 0), new THREE.Vector3(-w * 0.35, 30, 0), new THREE.Vector3(w * 0.35, h / 2 - 0.1, 0), new THREE.Vector3(w * 0.35, 30, 0)])),
      keep(new THREE.LineBasicMaterial({ color: 0x6b5a44, transparent: true, opacity: 0.55 })));
    m.add(strings); group.add(m); return m;
  };
  const titleCard = card([WALK.title, WALK.subtitle, WALK.byline], S.walk.lead * 0.42, 7.2, 3.4, true);
  const closingCard = card(WALK.closing, SPAN - 14, 6.4, 2.6, false);

  // ---- the pond ----------------------------------------------------------------------------------
  const pondBits = { nymphs: [], fish: null, state: 'present', timer: 0, fishNext: 4, fishT: -1, fishA: 0, rings: [] };
  if (pond) {
    const water = new THREE.Mesh(keep(new THREE.CircleGeometry(1, 48).rotateX(-Math.PI / 2)), keep(new THREE.MeshLambertMaterial({ map: keep(waterTexture(rng)), transparent: true, opacity: 0.95 })));
    water.scale.set(pond.rx, 1, pond.rz); water.position.set(pond.x, 0.025, pond.z); water.renderOrder = 2;
    const rim = new THREE.Mesh(keep(new THREE.RingGeometry(0.98, 1.06, 48).rotateX(-Math.PI / 2)), keep(new THREE.MeshBasicMaterial({ color: 0x2f4a55, transparent: true, opacity: 0.7 })));
    rim.scale.copy(water.scale); rim.position.set(pond.x, 0.03, pond.z); rim.renderOrder = 3;
    group.add(water, rim);
    const pics = rng.shuffle(ASSETS.creatures.filter(c => c.role === 'pond' && textures.get(c.file)).slice()).slice(0, 5);
    pics.forEach((c, i) => {
      const m = cutout(textures.get(c.file), c.height * rng.range(0.85, 1.05));
      const x = pond.x + (i - (pics.length - 1) / 2) * (pond.rx * 1.6 / pics.length) + rng.range(-0.3, 0.3);
      const zz = pond.z + rng.range(-0.8, 0.8);
      m.userData.base = { x, z: zz, h: m.scale.y, phase: rng() * 6, delay: rng.range(0, 0.7) };
      m.userData.info = { kind: 'figure', folio: c.folio, iiif: c.iiif, section: c.section, file: c.file, what: 'Bathing figure' };
      group.add(m); clickables.push(m); pondBits.nymphs.push(m);
    });
    const fish = ASSETS.creatures.find(c => c.role === 'fish' && textures.get(c.file));
    if (fish) { pondBits.fish = cutout(textures.get(fish.file), fish.height); pondBits.fish.visible = false; pondBits.fish.userData.info = { kind: 'figure', folio: fish.folio, iiif: fish.iiif, section: fish.section, file: fish.file, what: 'Fish' }; group.add(pondBits.fish); clickables.push(pondBits.fish); }
    pondBits.water = water;
  }
  function cutout(tex, h) {
    const a = tex.image.width / tex.image.height, geo = keep(new THREE.PlaneGeometry(a, 1)); geo.translate(0, 0.5, 0);
    const mat = keep(new THREE.MeshBasicMaterial({ map: tex, alphaTest: 0.5, side: THREE.DoubleSide })); mat.alphaToCoverage = true;
    const m = new THREE.Mesh(geo, mat); m.scale.setScalar(h); return m;
  }
  const ringTex = keep(ringTexture());
  function splash(x, z, s) {
    const m = new THREE.Mesh(keep(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2)), new THREE.MeshBasicMaterial({ map: ringTex, transparent: true, depthWrite: false, color: 0x2c4a5c }));
    m.position.set(x, 0.04, z); m.userData = { t: 0, s }; m.renderOrder = 4; group.add(m); pondBits.rings.push(m);
  }

  // ---- loop copies: things near either end also appear just past the other end ----------------------
  const originals = group.children.slice();
  for (const o of originals) {
    if (o === ground || o === sun || o.userData.noCopy) continue;
    if (o.geometry && o.geometry.parameters && o.geometry.parameters.width > SPAN) continue;   // ground and flats tile by themselves
    if (pondBits.nymphs.includes(o) || o === pondBits.fish) continue;
    const x = o.position.x;
    for (const shift of [x < VIEW ? SPAN : null, x > SPAN - VIEW ? -SPAN : null]) {
      if (shift === null) continue;
      const c = o.clone(); c.position.x += shift; group.add(c);
      c.traverse(q => { if (q.userData.info) clickables.push(q); });
    }
  }

  // ---- creatures crossing the path -----------------------------------------------------------------
  const crossers = ASSETS.creatures.filter(c => (c.role === 'hop' || c.role === 'fly') && textures.get(c.file));
  let crossing = null, crossNext = S.events.creatureEvery * 0.4;

  // ---- every frame ----------------------------------------------------------------------------------
  function update(dt, time, camX, light = 1) {
    wind.uTime.value = time; wind.uSway.value = S.plants.sway; wind.uSwaySpeed.value = S.plants.swaySpeed;
    sun.position.set(camX + 16, 12.5, -42);                          // so far away it seems to travel with you
    const cx = ((camX % SPAN) + SPAN) % SPAN;

    // videos play when you are near them, and pause otherwise (keeps the computer calm)
    for (const st of stations) {
      const d = Math.min(Math.abs(cx - st.x), Math.abs(cx - st.x - SPAN), Math.abs(cx - st.x + SPAN));
      if (st.video) st.video.setActive(d < S.videos.playWithin);
    }

    // pond: the nymphs dive away as you approach, and come back once you have gone
    if (pond) {
      const B = pondBits, d = Math.min(Math.abs(cx - pond.x), SPAN - Math.abs(cx - pond.x));
      if (B.state === 'present' && d < 7.5) { B.state = 'diving'; B.timer = 0; }
      if (B.state !== 'present') B.timer += dt;
      if (B.state === 'diving' && B.timer > 3) B.state = 'gone';
      if (B.state === 'gone' && d > 26 && B.timer > 12) { B.state = 'present'; B.nymphs.forEach(m => m.userData.splashed = false); }
      for (const m of B.nymphs) {
        const u = m.userData.base; let sink = 0.32, tilt = 0;
        if (B.state !== 'present') {
          const t = THREE.MathUtils.clamp((B.timer - u.delay) / 1.0, 0, 1); sink += t * t * 0.8; tilt = t * 0.4;
          if (t > 0 && !m.userData.splashed) { splash(u.x, u.z, 1.6); m.userData.splashed = true; }
        }
        m.visible = sink < 1.1;
        m.position.set(u.x, -u.h * sink + Math.sin(time * 1.2 + u.phase) * 0.03, u.z); m.rotation.z = tilt;
      }
      if (B.fish) {
        B.fishA += dt * 0.5; B.fishNext -= dt;
        const fx = pond.x + Math.cos(B.fishA) * pond.rx * 0.5, fz = pond.z + Math.sin(B.fishA) * pond.rz * 0.4;
        if (B.fishNext < 0 && B.fishT < 0) { B.fishT = 0; splash(fx, fz, 1.1); }
        if (B.fishT >= 0) {
          B.fishT += dt / 1.2; const t = B.fishT;
          B.fish.visible = true; B.fish.position.set(fx, -0.15 + 4 * t * (1 - t) * 1.4, fz); B.fish.rotation.z = (0.5 - t) * 1.4;
          if (t >= 1) { B.fishT = -1; B.fish.visible = false; B.fishNext = S.events.fishJumpEvery * rng.range(0.5, 1.5); splash(fx, fz, 1.1); }
        }
      }
      B.water.material.map.offset.x = Math.sin(time * 0.1) * 0.02;
    }
    for (let i = pondBits.rings.length - 1; i >= 0; i--) {
      const r = pondBits.rings[i]; r.userData.t += dt / 1.5;
      r.scale.setScalar(r.userData.s * (0.4 + r.userData.t * 2)); r.material.opacity = Math.max(0, 0.8 * (1 - r.userData.t));
      if (r.userData.t >= 1) { group.remove(r); r.material.dispose(); pondBits.rings.splice(i, 1); }
    }

    // a creature hops across the path ahead of you now and then
    crossNext -= dt;
    if (!crossing && crossNext < 0 && crossers.length) {
      const c = rng.pick(crossers), dir = rng() < 0.5 ? 1 : -1;
      const m = cutout(textures.get(c.file), c.height); m.scale.x *= dir > 0 ? 1 : -1;   // faces the way it travels
      m.userData.info = { kind: 'figure', folio: c.folio, iiif: c.iiif, section: c.section, file: c.file, what: 'Creature' };
      group.add(m); clickables.push(m);
      crossing = { m, c, t: 0, x0: camX - dir * 16, dir, fly: c.role === 'fly' };
    }
    if (crossing) {
      const k = crossing; k.t += dt / (k.fly ? 8 : 7);
      const t = k.t, x = k.x0 + k.dir * 32 * t;                          // along the path, across the whole view
      k.m.position.set(x, k.fly ? 7 + Math.sin(t * 9) * 0.4 : Math.abs(Math.sin(t * Math.PI * 9)) * 0.45, k.fly ? -4 : 2.2);
      if (t >= 1) {
        group.remove(k.m); clickables.splice(clickables.indexOf(k.m), 1); k.m.geometry.dispose(); k.m.material.dispose();
        crossing = null; crossNext = S.events.creatureEvery * rng.range(0.6, 1.4);
      }
    }
  }

  return {
    group, clickables, SPAN, stations, stationX, pond, update,
    stopX: i => stationX[i],
    state: () => ({ SPAN, plants: spots.length, drawCalls: plants.length, stations: stations.length, pond: pondBits.state, crossing: !!crossing }),
    dispose() { disposables.forEach(d => d.dispose && d.dispose()); },
  };
}

/* ---------------------------------------------------------------------------------------
   Materials and painted pictures
   --------------------------------------------------------------------------------------- */

// plant cut-outs: picture shown exactly as it is, bent gently by the wind (more at the top)
export function plantMaterial(tex, wind) {
  const mat = new THREE.MeshBasicMaterial({ map: tex, alphaTest: 0.5, side: THREE.DoubleSide });
  mat.alphaToCoverage = true;
  mat.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, wind);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float uTime; uniform float uSway; uniform float uSwaySpeed;')
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        #ifdef USE_INSTANCING
          vec4 wpos = modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
        #else
          vec4 wpos = modelMatrix * vec4(0.0, 0.0, 0.0, 1.0);
        #endif
        float ph = wpos.x * 0.37 + wpos.z * 0.21;
        float gust = 0.55 + 0.45 * sin(uTime * 0.23 * uSwaySpeed + wpos.x * 0.05);
        transformed.x += sin(uTime * 1.25 * uSwaySpeed + ph) * uv.y * uv.y * uSway * gust * 0.045;
        transformed.y += sin(uTime * 2.3 * uSwaySpeed + ph * 1.7) * uSway * 0.01 * smoothstep(0.8, 1.0, uv.y);`);
  };
  mat.customProgramCacheKey = () => 'plant';
  return mat;
}

const FONT = '"IM Fell English", Georgia, serif';
const tex = cv => { const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t; };
const canvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return [c, c.getContext('2d')]; };

// the ground: parchment, a path with inked edges, a green wash at the back. Repeats every 12 m.
function paintGround(S, rng) {
  const W = 1024, H = 3072, [cv, g] = canvas(W, H), zToY = z => (z + 16) / 36 * H;
  g.fillStyle = S.view.paper; g.fillRect(0, 0, W, H);
  const grd = g.createLinearGradient(0, zToY(-16), 0, zToY(0.4));
  grd.addColorStop(0, 'rgba(120,150,100,0.85)'); grd.addColorStop(0.75, 'rgba(160,175,125,0.45)'); grd.addColorStop(1, 'rgba(160,175,125,0)');
  g.fillStyle = grd; g.fillRect(0, 0, W, zToY(0.4));
  const g2 = g.createLinearGradient(0, zToY(5.2), 0, H);
  g2.addColorStop(0, 'rgba(160,175,125,0)'); g2.addColorStop(1, 'rgba(130,155,105,0.6)');
  g.fillStyle = g2; g.fillRect(0, zToY(5.2), W, H - zToY(5.2));
  for (let k = 0; k < 900; k++) {                                         // watercolour blotches
    g.fillStyle = `rgba(${rng() < 0.5 ? '110,90,60' : '255,250,235'},${rng.range(0.02, 0.05)})`;
    const x = rng() * W, y = rng() * H, r = rng.range(10, 70);
    for (const dx of [-W, 0, W]) { g.beginPath(); g.arc(x + dx, y, r, 0, 7); g.fill(); }
  }
  // path edges: wavy inked lines whose waves repeat exactly every tile (so the loop is seamless)
  const edge = (z0, amp, ph) => x => zToY(z0 + amp * Math.sin(2 * Math.PI * x / W + ph) + amp * 0.5 * Math.sin(4 * Math.PI * x / W + ph * 2));
  for (const [z0, ph] of [[1.0, 0.7], [4.7, 2.1]]) {
    const f = edge(z0, 0.25, ph);
    g.strokeStyle = 'rgba(70,50,32,0.65)'; g.lineWidth = 3;
    g.beginPath(); for (let x = 0; x <= W; x += 4) { const y = f(x) + rng.range(-1, 1); x ? g.lineTo(x, y) : g.moveTo(x, y); } g.stroke();
  }
  for (let k = 0; k < 3600; k++) {                                        // grass strokes at the back and front
    const x = rng() * W, back = rng() < 0.6, z = back ? rng.range(-16, 0.6) : rng.range(5.3, 20), y = zToY(z), L = rng.range(4, 10);
    g.strokeStyle = `rgba(60,70,40,${rng.range(0.12, 0.3)})`; g.lineWidth = 1.2;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + rng.range(-2, 2), y - L); g.stroke();
  }
  for (let k = 0; k < 60000; k++) { g.fillStyle = rng() < 0.5 ? 'rgba(80,60,30,0.08)' : 'rgba(255,250,235,0.1)'; g.fillRect(rng() * W, rng() * H, 1.5, 1.5); }
  const t = tex(cv); t.wrapS = THREE.RepeatWrapping; return t;
}

// a painted hill flat: watercolour green below a wavy inked skyline, transparent above.
// The skyline is built from waves that fit the walk's length exactly, so it loops without a seam.
function paintFlat(SPAN, f, rng) {
  const W = 4096, H = 512, [cv, g] = canvas(W, H);
  const waves = [1, 2, 3, 5, 8, 13, 21].map(k => ({ k: Math.round(k * SPAN / 120) || k, a: rng.range(0.4, 1) / k ** 0.6, p: rng() * 7 }));
  const sum = waves.reduce((s, w) => s + w.a, 0);
  const top = x => H * (f.top + f.rough * (0.5 + 0.5 * waves.reduce((s, w) => s + w.a * Math.sin(2 * Math.PI * w.k * x / W + w.p), 0) / sum));
  g.beginPath(); g.moveTo(0, H);
  for (let x = 0; x <= W; x += 8) g.lineTo(x, top(x));
  g.lineTo(W, H); g.closePath();
  const grd = g.createLinearGradient(0, 0, 0, H); grd.addColorStop(0, f.col[0]); grd.addColorStop(1, f.col[1]);
  g.fillStyle = grd; g.fill();
  g.save(); g.clip();
  for (let k = 0; k < 500; k++) {                                         // blotchy wash
    g.fillStyle = `rgba(${rng() < 0.5 ? '60,90,60' : '230,225,190'},${rng.range(0.03, 0.08)})`;
    g.beginPath(); g.arc(rng() * W, rng() * H, rng.range(20, 90), 0, 7); g.fill();
  }
  for (let k = 0; k < 900 * f.ink; k++) {                                 // pen hatching just under the skyline
    const x = rng() * W, y = top(x) + rng.range(6, 90), L = rng.range(8, 22);
    g.strokeStyle = `rgba(50,55,35,${0.25 * f.ink})`; g.lineWidth = 1.4;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + L, y + rng.range(-3, 3)); g.stroke();
  }
  g.restore();
  g.strokeStyle = `rgba(55,45,30,${0.4 + f.ink * 0.5})`; g.lineWidth = 3;
  g.beginPath(); for (let x = 0; x <= W; x += 6) { const y = top(x) + rng.range(-0.8, 0.8); x ? g.lineTo(x, y) : g.moveTo(x, y); } g.stroke();
  const t = tex(cv); t.wrapS = THREE.RepeatWrapping; return t;
}

// the frame round each video: parchment card, double ink rule, station number in the corner
function paintFrame(num) {
  const [cv, g] = canvas(1024, 600);
  g.fillStyle = '#e9dcbf'; g.fillRect(0, 0, 1024, 600);
  g.strokeStyle = '#4a3825'; g.lineWidth = 6; g.strokeRect(10, 10, 1004, 580);
  g.lineWidth = 2; g.strokeRect(26, 26, 972, 548);
  g.fillStyle = '#9a4b32'; g.font = `44px ${FONT}`; g.textAlign = 'right'; g.fillText(String(num), 1000, 58);
  return tex(cv);
}
function paintPoster(title) {
  const [cv, g] = canvas(640, 360);
  g.fillStyle = '#3b3328'; g.fillRect(0, 0, 640, 360);
  g.fillStyle = '#efe5cf'; g.font = `30px ${FONT}`; g.textAlign = 'center'; g.fillText(title, 320, 190);
  return tex(cv);
}
// the YouTube picture in the frame: cropped to 16:9 (the smaller YouTube pictures are 4:3 with black bars), with a small play mark
function paintStill(img) {
  const [cv, g] = canvas(1280, 720), iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
  const sh = Math.min(ih, iw * 9 / 16), sy = (ih - sh) / 2;
  g.drawImage(img, 0, sy, iw, sh, 0, 0, 1280, 720);
  g.fillStyle = 'rgba(59,47,33,0.62)'; g.beginPath(); g.arc(640, 360, 62, 0, Math.PI * 2); g.fill();
  g.strokeStyle = 'rgba(239,229,207,0.9)'; g.lineWidth = 4; g.stroke();
  g.fillStyle = '#efe5cf'; g.beginPath(); g.moveTo(618, 326); g.lineTo(618, 394); g.lineTo(676, 360); g.closePath(); g.fill();
  return tex(cv);
}
function paintLabel(title) {
  const [cv, g] = canvas(1536, 256);
  g.fillStyle = 'rgba(239,229,207,0.92)';
  g.beginPath(); g.moveTo(30, 40); g.lineTo(1506, 30); g.lineTo(1490, 220); g.lineTo(40, 226); g.closePath(); g.fill();
  g.strokeStyle = 'rgba(74,56,37,0.7)'; g.lineWidth = 3; g.stroke();
  g.fillStyle = '#3b2f21'; g.textAlign = 'center'; g.textBaseline = 'middle';
  const fits = (s, size) => { g.font = `${size}px ${FONT}`; return g.measureText(s).width <= 1380; };
  let size = 104; while (!fits(title, size) && size > 64) size -= 4;
  if (fits(title, size)) { g.fillText(title, 768, 130); return tex(cv); }
  // long titles go on two lines, split near the middle at a space (and shortened if still too long)
  const words = title.split(' '); let best = null;
  for (let k = 1; k < words.length; k++) { const a = words.slice(0, k).join(' '), b = words.slice(k).join(' '), m = Math.max(a.length, b.length); if (!best || m < best[2]) best = [a, b, m]; }
  let [a, b] = best || [title, '']; size = 76;
  while ((!fits(a, size) || !fits(b, size)) && size > 50) size -= 2;
  while (!fits(b, size) && b.length > 4) b = b.replace(/\s*\S+…?$/, '…');
  while (!fits(a, size) && a.length > 4) a = a.replace(/\s*\S+…?$/, '…');
  g.fillText(a, 768, 92); g.fillText(b, 768, 170);
  return tex(cv);
}
function paintCard(lines, aspect, big) {
  const W = 1600, H = Math.round(W / aspect), [cv, g] = canvas(W, H);
  g.fillStyle = '#efe5cf'; g.fillRect(8, 8, W - 16, H - 16);
  g.strokeStyle = '#4a3825'; g.lineWidth = 5; g.strokeRect(14, 14, W - 28, H - 28); g.lineWidth = 2; g.strokeRect(30, 30, W - 60, H - 60);
  g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillStyle = '#3b2f21';
  const sizes = big ? [210, 84, 64] : [110, 64, 60];
  const ys = big ? [0.4, 0.66, 0.84] : [0.4, 0.68, 0.84];
  lines.forEach((l, i) => { g.font = `${i === 0 ? '' : 'italic '}${sizes[i] || 60}px ${FONT}`; g.fillStyle = i === 2 ? '#9a4b32' : '#3b2f21'; g.fillText(l, W / 2, H * (ys[i] || 0.9)); });
  return tex(cv);
}
function paintSun() {
  const N = 256, [cv, c] = canvas(N, N); c.translate(N / 2, N / 2);
  c.strokeStyle = 'rgba(120,80,30,0.9)'; c.lineWidth = 3;
  for (let k = 0; k < 16; k++) { c.save(); c.rotate(k / 16 * Math.PI * 2); c.beginPath(); c.moveTo(52, 0); c.quadraticCurveTo(80, k % 2 ? 10 : -10, 108, 0); c.stroke(); c.restore(); }
  c.fillStyle = '#e9c45c'; c.beginPath(); c.arc(0, 0, 48, 0, 7); c.fill();
  c.fillStyle = 'rgba(230,150,60,0.35)'; c.beginPath(); c.arc(-8, 6, 34, 0, 7); c.fill();
  c.beginPath(); c.arc(0, 0, 48, 0, 7); c.stroke();
  return tex(cv);
}
function waterTexture(rng) {
  const N = 512, [cv, c] = canvas(N, N);
  c.fillStyle = '#6f9ba0'; c.fillRect(0, 0, N, N);
  for (let k = 0; k < 500; k++) { c.fillStyle = `rgba(${rng() < 0.5 ? '40,80,110' : '180,210,205'},${rng.range(0.03, 0.08)})`; c.beginPath(); c.arc(rng() * N, rng() * N, rng.range(6, 40), 0, 7); c.fill(); }
  c.strokeStyle = 'rgba(30,55,80,0.5)'; c.lineWidth = 2;
  for (let k = 0; k < 120; k++) { const x = rng() * N, y = rng() * N, w = rng.range(10, 26); c.beginPath(); c.moveTo(x, y); c.quadraticCurveTo(x + w / 2, y - 5, x + w, y); c.stroke(); }
  const t = tex(cv); t.wrapS = t.wrapT = THREE.RepeatWrapping; return t;
}
function blobTexture() {
  const N = 128, [cv, c] = canvas(N, N), g = c.createRadialGradient(N / 2, N / 2, 2, N / 2, N / 2, N / 2);
  g.addColorStop(0, 'rgba(255,255,255,0.75)'); g.addColorStop(0.6, 'rgba(255,255,255,0.25)'); g.addColorStop(1, 'rgba(255,255,255,0)');
  c.fillStyle = g; c.fillRect(0, 0, N, N); return new THREE.CanvasTexture(cv);
}
function ringTexture() {
  const N = 128, [cv, c] = canvas(N, N); c.strokeStyle = 'rgba(255,255,255,0.9)';
  for (const [r, w] of [[54, 3], [42, 2], [30, 1.5]]) { c.lineWidth = w; c.beginPath(); c.arc(N / 2, N / 2, r, 0, 7); c.stroke(); }
  return new THREE.CanvasTexture(cv);
}

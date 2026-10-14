// ============================================================================
// plants.js  -  decides WHERE plants grow and draws them as upright cut-outs.
//
// Two steps:
//   1. makePlan():   pick clump centres (beside paths, on the hills, in the
//                    clearings), then place plants round each centre.
//                    Big plants are placed FIRST and kept apart so they never
//                    clip into each other; small shrubs fill in afterwards.
//   2. build():      draw them. All plants of one picture are drawn together
//                    ("instancing"), which is why thousands are cheap.
// ============================================================================

// A plant is a flat card with its origin at the bottom-centre (the root).
// 'crossed' = two cards in an X. segY = how finely the card is cut up
// (the wind bends the card, so it needs a few slices).
function makeCardGeometry(crossed, segY) {
  const quads = crossed ? 2 : 1, pos = [], uv = [], idx = [], quad = [];
  for (let q = 0; q < quads; q++) {
    const a = q * Math.PI / 2, dx = Math.cos(a) * 0.5, dz = -Math.sin(a) * 0.5, base = pos.length / 3;
    for (let j = 0; j <= segY; j++) {
      const v = j / segY;
      pos.push(-dx, v, -dz,  dx, v, dz);
      uv.push(0, v, 1, v);
      quad.push(q, q);                              // [quality] which of the two cards (for lighting)
    }
    for (let j = 0; j < segY; j++) {
      const a0 = base + j * 2;
      idx.push(a0, a0 + 1, a0 + 3, a0, a0 + 3, a0 + 2);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute('aQuad', new THREE.Float32BufferAttribute(quad, 1));
  g.setIndex(idx);
  return g;
}

// Wind: runs on the graphics card, so it costs almost nothing.
// The base of each card stays put; the higher up, the more it leans (height squared).
// Every plant gets a slightly different timing, and the wind comes in slow gusts.
function addSway(material, wind) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = wind.uTime; shader.uniforms.uAmp = wind.uAmp;
    shader.vertexShader = 'uniform float uTime;\nuniform float uAmp;\n' +
      shader.vertexShader.replace('#include <begin_vertex>', `
        #include <begin_vertex>
        #ifdef USE_INSTANCING
          mat4 swayM = instanceMatrix;
        #else
          mat4 swayM = modelMatrix;
        #endif
        {
          vec3 ip = swayM[3].xyz;
          float sx = max(length(swayM[0].xyz), 0.001);
          float sz = max(length(swayM[2].xyz), 0.001);
          float sy = length(swayM[1].xyz);
          float seed = fract(sin(dot(ip.xz, vec2(12.9898, 78.233))) * 43758.5453);
          float ph = dot(ip.xz, vec2(0.045, 0.03)) - uTime * 1.25 + seed * 1.6;
          float gust = 0.7 + 0.3 * sin(uTime * 0.31 + ip.x * 0.021);
          float y = clamp(position.y, 0.0, 1.0);
          float off = uAmp * gust * sy * y * y * (sin(ph) + 0.4 * sin(ph * 2.3 + 1.7));
          transformed.x += off / sx;
          transformed.z += off * 0.5 / sz;
        }`);
  };
}

class Plants {
  constructor(parent, terrain, S, seed, kinds) {
    this.parent = parent; this.terrain = terrain; this.S = S;
    this.kindMap = {};
    kinds.forEach((k) => (this.kindMap[k.id] = k));
    this.group = new THREE.Group();
    parent.add(this.group);
    this.layers = []; this.landmark = null; this.geoms = [];
    this.wind = { uTime: { value: 0 }, uAmp: { value: 0 } };
    this.windLandmark = { uTime: this.wind.uTime, uAmp: { value: 0 } };
    this.plan = this.makePlan(makeRng(seed, 'plants'));
    this.build();
  }

  // ---- Step 1: where do plants go? -------------------------------------------------
  makePlan(rng) {
    const S = this.S, P = S.plants, W = S.world, T = this.terrain, sp = P.spacing;
    const clusters = [];
    const half = W.size / 2 - 10;

    // Clumps beside the paths, and now and then a clump of giants on the hill behind.
    for (const path of T.paths) {
      let d = rng.range(3, 8);
      while (d < path.length - 3) {
        const u = d / path.length;
        const p = path.curve.getPointAt(u), t = path.curve.getTangentAt(u);
        const nx = -t.z, nz = t.x;
        let side = rng.chance(0.5) ? 1 : -1, off = W.pathHalfWidth + rng.range(1.2, 6);
        clusters.push({ zone: 'path', x: p.x + nx * off * side, z: p.z + nz * off * side, radius: rng.range(1.8, 4), weight: rng.range(0.6, 1.4) });
        if (rng.chance(0.8)) {
          side = rng.chance(0.5) ? 1 : -1; off = W.pathHalfWidth + rng.range(10, 22);
          clusters.push({ zone: 'edge', x: p.x + nx * off * side, z: p.z + nz * off * side, radius: rng.range(4, 8), weight: rng.range(0.3, 0.7) });
        }
        d += rng.range(5, 12);
      }
    }
    // Themed clearings. w = how thickly planted; lo/hi = where in the clearing (fraction of radius).
    const zoneInfo = {
      grove:  { w: 2.2, r: [2.5, 5], lo: 0.05, hi: 0.95 },
      pond:   { w: 1.4, r: [2, 3.5], lo: 0.70, hi: 1.05 },   // a ring round the water
      meadow: { w: 0.6, r: [3, 5],   lo: 0.35, hi: 0.95 },   // centre kept empty for animals
      glade:  { w: 1.2, r: [2.5, 4], lo: 0.05, hi: 0.90 },
    };
    for (const c of S.layout.clearings) {
      const zi = zoneInfo[c.type] || zoneInfo.glade;
      const n = Math.max(2, Math.round(Math.PI * c.radius * c.radius / 110));
      for (let i = 0; i < n; i++) {
        const a = rng.range(0, Math.PI * 2), r = c.radius * Math.sqrt(rng.range(zi.lo * zi.lo, zi.hi * zi.hi));
        clusters.push({ zone: zoneInfo[c.type] ? c.type : 'glade', x: c.x + Math.cos(a) * r, z: c.z + Math.sin(a) * r,
                        radius: rng.range(zi.r[0], zi.r[1]), weight: zi.w * rng.range(0.7, 1.3) });
      }
    }

    // Decide, for every clump, what plants it will have (kind + height) ...
    const totalW = clusters.reduce((s, c) => s + c.weight, 0);
    const allIds = Object.keys(this.kindMap);
    const jobs = [];
    for (const c of clusters) {
      const n = Math.round(P.count * c.weight / totalW);
      let set = expandPlantSet(P.sets[c.zone] || []).filter((id) => this.kindMap[id]);   // group names like 'tall' are allowed (js/images.js)
      if (!set.length) set = allIds;
      const favs = [rng.pick(set), rng.pick(set), rng.pick(set)];       // a clump has 2-3 favourite pictures
      c.bigKind = favs[0];                                              // ...and its big plants are all ONE kind
      c.rowYaw = rng.range(0, Math.PI);                                 // ...standing in a row, in the same plane
      c.bigCount = 0;
      const [h0, h1] = P.heights[c.zone];
      for (let k = 0; k < n; k++) {
        let h = rng.range(h0, h1) * P.heightScale, big = h >= sp.bigHeight;
        if (big && c.bigCount >= sp.maxBigPerCluster) { h = sp.bigHeight * rng.range(0.5, 0.95); big = false; }   // too many: make it a shrub
        if (big) c.bigCount++;
        jobs.push({ c, h, big, kind: big ? c.bigKind : (rng.chance(0.8) ? rng.pick(favs) : rng.pick(set)) });
      }
    }
    jobs.sort((a, b) => (b.big - a.big) || (b.h - a.h));                // ... then place the BIG ones first

    // ---- placing with spacing -------------------------------------------------------
    const CELL = 6, grid = new Map(); let maxR = 0.5;
    const cellKey = (i, j) => i * 100003 + j;
    const fits = (x, z, r, kind, big) => {
      const reach = Math.max(1, sp.bigSame, sp.bigDifferent, sp.bigToSmall) * (r + maxR);
      const span = Math.ceil(reach / CELL), ci = Math.floor(x / CELL), cj = Math.floor(z / CELL);
      for (let i = ci - span; i <= ci + span; i++) for (let j = cj - span; j <= cj + span; j++) {
        const list = grid.get(cellKey(i, j));
        if (!list) continue;
        for (const o of list) {
          const f = (!big && !o.big) ? sp.smallOverlap : (big && o.big) ? (kind === o.kind ? sp.bigSame : sp.bigDifferent) : sp.bigToSmall;
          if (Math.hypot(x - o.x, z - o.z) < f * (r + o.r)) return false;
        }
      }
      return true;
    };
    const plan = [];
    for (const j of jobs) {
      const c = j.c, kind = this.kindMap[j.kind];
      const r = j.h * kind.aspect * 0.5 * sp.footprint;                 // footprint radius of this plant
      for (let attempt = 0; attempt < 14; attempt++) {
        const grow = 1 + attempt * 0.18;                                // widen the search if it is crowded
        let x, z;
        if (j.big) {                                                    // big plants line up side by side along the row
          const a = c.rowYaw, along = rng.gauss() * c.radius * 1.5 * grow / P.density, across = rng.gauss() * c.radius * 0.4 * grow / P.density;
          x = c.x + Math.cos(a) * along + Math.sin(a) * across; z = c.z - Math.sin(a) * along + Math.cos(a) * across;
        } else {
          x = c.x + rng.gauss() * c.radius * grow / P.density; z = c.z + rng.gauss() * c.radius * grow / P.density;
        }
        if (Math.abs(x) > half || Math.abs(z) > half) continue;
        if (T.pathDistance(x, z) < W.pathHalfWidth * 0.8 + (j.big ? r * 0.5 : 0)) continue;   // keep the path clear
        const y = T.height(x, z);
        if (y < W.waterLevel + 0.25) continue;                          // not in the water
        if (!fits(x, z, r, j.kind, j.big)) continue;
        const yaw = j.big ? c.rowYaw + clamp(rng.gauss() * 0.15, -0.35, 0.35) : rng.range(0, Math.PI);
        const item = { kind: j.kind, x, y, z, h: j.h, r, big: j.big, yaw, flip: rng.chance(0.5) ? -1 : 1 };
        plan.push(item);
        const key = cellKey(Math.floor(x / CELL), Math.floor(z / CELL));
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key).push(item);
        maxR = Math.max(maxR, r);
        break;
      }
    }
    return plan;
  }

  // ---- Step 2: draw them ---------------------------------------------------------------
  build() {
    const P = this.S.plants;
    const crossed = P.mode === 'crossed';
    const byKind = {};
    for (const p of this.plan) (byKind[p.kind] = byKind[p.kind] || []).push(p);

    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), pos = new THREE.Vector3(), sc = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    this.geometry = makeCardGeometry(crossed, 8);
    const QS = this.S.quality || {};
    if (QS.shadows) this.shadowGeometry = makeCardGeometry(false, 6);
    for (const id of Object.keys(byKind)) {
      const kind = this.kindMap[id], items = byKind[id];
      // MeshBasicMaterial = flat colour, no shading: right for paper cut-outs.
      // alphaTest = pixels more transparent than this are not drawn at all (clean cut edges, no sorting glitches).
      const material = new THREE.MeshBasicMaterial({ map: kind.texture, alphaTest: 0.5, side: THREE.DoubleSide, alphaToCoverage: P.smoothEdges });
      addSway(material, this.wind);
      if (QS.plantShading) addPlantShading(material);       // [quality] light from the sun, darker near the roots
      const mesh = new THREE.InstancedMesh(this.geometry, material, items.length);
      mesh.frustumCulled = false;                     // instanced meshes would otherwise vanish at odd moments
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      const mats = new Float32Array(items.length * 16);   // [quality] every plant's placement, kept so only the visible ones are sent to be drawn
      items.forEach((p, i) => {
        q.setFromAxisAngle(up, p.yaw);
        pos.set(p.x, p.y - p.h * 0.03, p.z);          // sink the root a hair into the ground
        sc.set(p.h * kind.aspect * p.flip, p.h, 1);
        m.compose(pos, q, sc);
        mesh.setMatrixAt(i, m);
        m.toArray(mats, i * 16);
      });
      this.group.add(mesh);
      let shadow = null;
      if (QS.shadows) {                               // [quality] each plant's shadow, laid on the ground away from the sun
        const sm = new THREE.MeshBasicMaterial({ map: kind.texture, color: 0x000000, transparent: true, depthWrite: false, fog: false, side: THREE.DoubleSide,
          polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
        addSway(sm, this.wind); addPlantShadow(sm, this.S);
        shadow = new THREE.InstancedMesh(this.shadowGeometry, sm, items.length);
        shadow.instanceMatrix = mesh.instanceMatrix;  // same placements as the plants (one list, sent once)
        shadow.frustumCulled = false; shadow.renderOrder = 1;
        this.group.add(shadow);
      }
      this.layers.push({ mesh, items, material, aspect: kind.aspect, mats, shadow });
    }

    // The landmark: one huge plant you can spot from far away.
    const L = this.S.layout.landmark, kind = this.kindMap[P.landmarkKind] || this.kindMap[Object.keys(this.kindMap)[0]];
    if (L && kind) {
      this.landGeom = makeCardGeometry(true, 8);
      const mat = new THREE.MeshBasicMaterial({ map: kind.texture, alphaTest: 0.5, side: THREE.DoubleSide, fog: !P.landmarkIgnoresFog });
      addSway(mat, this.windLandmark);
      if ((this.S.quality || {}).plantShading) addPlantShading(mat);   // [quality]
      const mesh = new THREE.Mesh(this.landGeom, mat);
      mesh.position.set(L.x, this.terrain.height(L.x, L.z) - 0.5, L.z);
      mesh.scale.set(P.landmarkHeight * kind.aspect, P.landmarkHeight, 1);
      mesh.frustumCulled = false;
      this.group.add(mesh);
      this.landmark = { mesh, mat };
    }
    this._lastCam = new THREE.Vector3(1e9, 0, 0);
  }

  // Every frame. level = overall brightness 0..1 (darker at night).
  update(camera, level, dt) {
    const W = this.S.plants.wind;
    this.wind.uTime.value += dt * W.speed;
    this.wind.uAmp.value = W.strength;
    this.windLandmark.uAmp.value = W.strength * 0.35;            // the giant landmark leans less
    for (const l of this.layers) l.material.color.setScalar(level);
    if (this.landmark) this.landmark.mat.color.setScalar(level * 0.85);
    const QS = this.S.quality || {};
    const shadowA = (QS.shadowStrength ?? 0.38) * Q.sun.power.value;
    for (const l of this.layers) if (l.shadow) { l.shadow.material.opacity = shadowA; l.shadow.visible = shadowA > 0.01; }

    // [quality] Only the plants in view (and not lost in the fog) are sent to be drawn.
    // Re-checked when you move or turn; in billboard mode the cards are also turned to face you.
    camera.updateMatrixWorld();
    const moved = this._lastCam.distanceToSquared(camera.position) > 0.04, turned = !this._lastQ || this._lastQ.angleTo(camera.quaternion) > 0.02;
    const billboard = this.S.plants.mode === 'billboard';
    if (!(moved || turned || this._dirty !== false)) return;
    this._dirty = false; this._lastCam.copy(camera.position); this._lastQ = (this._lastQ || new THREE.Quaternion()).copy(camera.quaternion);
    const fr = this._frustum || (this._frustum = new THREE.Frustum()), pm = this._pm || (this._pm = new THREE.Matrix4());
    fr.setFromProjectionMatrix(pm.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
    const far = QS.cullPlants === false ? 1e9 : this.S.fog.far + 6, far2 = far * far;
    const sph = this._sph || (this._sph = new THREE.Sphere()), m = new THREE.Matrix4(), q = new THREE.Quaternion(), pos = new THREE.Vector3(), sc = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0), cx = camera.position.x, cz = camera.position.z;
    const shadowReach = QS.shadows ? 1 / Math.max(Q.sun.dir.value.y, 0.35) : 0;
    for (const l of this.layers) {
      const arr = l.mesh.instanceMatrix.array; let n = 0;
      l.items.forEach((p, i) => {
        const dx = p.x - cx, dz = p.z - cz;
        if (QS.cullPlants !== false) {
          if (dx * dx + dz * dz > far2) return;
          sph.center.set(p.x, p.y + p.h * 0.5, p.z); sph.radius = p.h * (0.6 + 0.5 * shadowReach);   // a bit bigger so its shadow is kept too
          if (!fr.intersectsSphere(sph)) return;
        }
        if (billboard) {
          q.setFromAxisAngle(up, Math.atan2(camera.position.x - p.x, camera.position.z - p.z));
          pos.set(p.x, p.y - p.h * 0.03, p.z); sc.set(p.h * l.aspect * p.flip, p.h, 1);
          m.compose(pos, q, sc); m.toArray(arr, n * 16);
        } else arr.set(l.mats.subarray(i * 16, i * 16 + 16), n * 16);
        n++;
      });
      l.mesh.count = n; l.mesh.visible = n > 0; l.mesh.instanceMatrix.needsUpdate = true;
      if (l.shadow) { l.shadow.count = n; if (n === 0) l.shadow.visible = false; }
    }
  }

  dispose() {
    for (const l of this.layers) { l.mesh.dispose(); l.material.dispose(); }
    if (this.landmark) this.landmark.mat.dispose();
    this.geometry.dispose();
    if (this.shadowGeometry) this.shadowGeometry.dispose();
    for (const l of this.layers) if (l.shadow) l.shadow.material.dispose();
    if (this.landGeom) this.landGeom.dispose();
    this.parent.remove(this.group);
  }
}

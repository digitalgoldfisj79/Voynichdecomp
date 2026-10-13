// ============================================================================
// terrain.js  -  builds the hills, the carved paths, the clearings and the water.
//
// How the ground is made (plain terms):
//  1. For every point on a grid we measure "how far is the nearest path or
//     clearing?"  (that number is called dOpen below; negative = inside one).
//  2. Close to a path the ground stays low and flat. Further away it rises into
//     hills. So the hills become walls that hug the paths.
//  3. We turn that grid of heights into a mesh of small triangles and colour it.
// ============================================================================

class Terrain {
  constructor(S, seed) {
    const W = S.world, L = S.layout;
    this.S = S;
    this.size = W.size;
    this.N = W.gridCells;
    this.cell = W.size / W.gridCells;
    const N = this.N, n1 = N + 1, half = this.size / 2;
    const rng = makeRng(seed, 'terrain');
    const noise = makeNoise2D(rng);

    // ---- 1. Turn each path into a smooth curve and a list of sample points ----
    this.paths = L.paths.map((p) => {
      const curve = new THREE.CatmullRomCurve3(
        p.points.map((q) => new THREE.Vector3(q[0], 0, q[1])), false, 'centripetal');
      const length = curve.getLength();
      const samples = curve.getSpacedPoints(Math.max(4, Math.ceil(length / 1.5)));
      return { name: p.name, curve, length, samples };
    });
    let total = 0;
    this.paths.forEach((p) => (total += p.samples.length));
    const sx = new Float32Array(total), sz = new Float32Array(total);
    let k = 0;
    this.paths.forEach((p) => p.samples.forEach((v) => { sx[k] = v.x; sz[k] = v.z; k++; }));

    // ---- 2. Heights, plus two distance maps we keep for later (plants, walking) ----
    this.H = new Float32Array(n1 * n1);       // ground height
    this.dPath = new Float32Array(n1 * n1);   // distance to nearest path centre line
    this.dOpen = new Float32Array(n1 * n1);   // distance to nearest open space (<0 = inside)
    const ponds = L.clearings.filter((c) => c.type === 'pond');

    for (let j = 0; j < n1; j++) {
      for (let i = 0; i < n1; i++) {
        const x = -half + i * this.cell, z = -half + j * this.cell;
        let best = 1e12;
        for (let s = 0; s < total; s++) {
          const dx = x - sx[s], dz = z - sz[s];
          const d2 = dx * dx + dz * dz;
          if (d2 < best) best = d2;
        }
        const dPath = Math.sqrt(best);
        let dOpen = dPath - W.pathHalfWidth;
        for (const c of L.clearings) {
          dOpen = Math.min(dOpen, Math.hypot(x - c.x, z - c.z) - c.radius);
        }

        // hills: bumpy noise, remapped so there is always a decent wall
        const hn = noise.fbm(x * W.hillScale + 10, z * W.hillScale + 10, 3);
        const hills = W.hillBase + W.hillHeight * clamp((hn - 0.25) / 0.5, 0, 1);
        const wall = smoothstep(0, W.wallRamp, dOpen);           // 0 on path, 1 in hills
        const roll = (noise(x * 0.045 + 200, z * 0.045 + 200) - 0.5) * 2 * W.groundRoll;
        let h = roll * (1 - 0.6 * wall) + wall * hills;
        h += smoothstep(half - 30, half - 5, Math.max(Math.abs(x), Math.abs(z))) * W.edgeWall;
        for (const c of ponds) {                                  // dig the pond basins
          const d = Math.hypot(x - c.x, z - c.z);
          h -= W.pondDepth * (1 - smoothstep(c.radius * 0.2, c.radius * 0.85, d));
        }
        for (const c of ponds) {                                  // a shallow ford where a path crosses a pond
          const d = Math.hypot(x - c.x, z - c.z);
          const inPond = 1 - smoothstep(c.radius * 0.8, c.radius * 1.0, d);
          const ford = (1 - smoothstep(W.pathHalfWidth * 0.8, W.pathHalfWidth * 1.8, dPath)) * inPond;
          if (ford > 0) h = lerp(h, Math.max(h, W.waterLevel - W.fordDepth), ford);
        }
        const idx = j * n1 + i;
        this.H[idx] = h; this.dPath[idx] = dPath; this.dOpen[idx] = dOpen;
      }
    }

    // ---- 3. Colours per grid point -------------------------------------------------
    const cPath = new THREE.Color(W.colors.path), cMid = new THREE.Color(W.colors.mid);
    const cHigh = new THREE.Color(W.colors.high), cBed = new THREE.Color(W.colors.pondBed);
    const vc = new Float32Array(n1 * n1 * 3);
    const tmp = new THREE.Color();
    for (let j = 0; j < n1; j++) {
      for (let i = 0; i < n1; i++) {
        const idx = j * n1 + i, x = -half + i * this.cell, z = -half + j * this.cell;
        const h = this.H[idx], dOpen = this.dOpen[idx];
        const away = smoothstep(1.0, 9.0, dOpen);
        tmp.copy(cPath).lerp(cMid, away);
        const tall = clamp(h / (W.hillBase + W.hillHeight), 0, 1);
        tmp.lerp(cHigh, smoothstep(0.35, 0.95, tall) * away);
        if (h < W.waterLevel + 0.4) tmp.lerp(cBed, 0.7);
        tmp.multiplyScalar(1 + (noise(x * 0.08 + 50, z * 0.08 + 50) - 0.5) * 0.25);   // wash variation
        vc[idx * 3] = tmp.r; vc[idx * 3 + 1] = tmp.g; vc[idx * 3 + 2] = tmp.b;
      }
    }

    // ---- 4. Build the triangle mesh (each triangle has its own vertices so it can be flat-coloured) ----
    const quads = N * N;
    const pos = new Float32Array(quads * 6 * 3), col = new Float32Array(quads * 6 * 3);
    const nor = new Float32Array(quads * 6 * 3);
    const jit = makeRng(seed, 'facets');
    let p = 0;
    const self = this;
    function pushTri(a, b, c) {
      const f = 1 + (jit.next() - 0.5) * 2 * W.faceJitter;       // little brightness change per facet
      for (const v of [a, b, c]) {
        const i = v % n1, j = (v - i) / n1;
        pos[p * 3] = -half + i * self.cell; pos[p * 3 + 1] = self.H[v]; pos[p * 3 + 2] = -half + j * self.cell;
        if (W.flatShading) {                                       // one colour per facet
          col[p * 3]     = (vc[a * 3]     + vc[b * 3]     + vc[c * 3])     / 3 * f;
          col[p * 3 + 1] = (vc[a * 3 + 1] + vc[b * 3 + 1] + vc[c * 3 + 1]) / 3 * f;
          col[p * 3 + 2] = (vc[a * 3 + 2] + vc[b * 3 + 2] + vc[c * 3 + 2]) / 3 * f;
        } else {                                                   // smooth: colour blends between points
          col[p * 3] = vc[v * 3]; col[p * 3 + 1] = vc[v * 3 + 1]; col[p * 3 + 2] = vc[v * 3 + 2];
        }
        if (!W.flatShading) {                                      // smooth normals from the height slope
          const i0 = Math.max(i - 1, 0), i1 = Math.min(i + 1, N), j0 = Math.max(j - 1, 0), j1 = Math.min(j + 1, N);
          const gx = (self.H[j * n1 + i1] - self.H[j * n1 + i0]) / ((i1 - i0) * self.cell);
          const gz = (self.H[j1 * n1 + i] - self.H[j0 * n1 + i]) / ((j1 - j0) * self.cell);
          const l = Math.hypot(gx, 1, gz);
          nor[p * 3] = -gx / l; nor[p * 3 + 1] = 1 / l; nor[p * 3 + 2] = -gz / l;
        }
        p++;
      }
    }
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const v00 = j * n1 + i, v10 = v00 + 1, v01 = v00 + n1, v11 = v01 + 1;
        pushTri(v00, v01, v10);      // triangle 1
        pushTri(v10, v01, v11);      // triangle 2
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    if (W.flatShading) geo.computeVertexNormals();
    else geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    const ground = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true }));

    this.group = new THREE.Group();
    this.group.add(ground);

    // ---- 5. Water for the ponds -----------------------------------------------------
    this.waterColor = new THREE.Color(W.colors.water);
    this.waterMat = new THREE.MeshBasicMaterial({ color: this.waterColor.clone(), transparent: true, opacity: 0.85, depthWrite: false });
    for (const c of ponds) {
      const g = new THREE.CircleGeometry(c.radius, 48);
      g.rotateX(-Math.PI / 2);
      const m = new THREE.Mesh(g, this.waterMat);
      m.position.set(c.x, W.waterLevel, c.z);
      m.renderOrder = 1;
      this.group.add(m);
    }

    this.ponds = ponds;

    // Where the player starts: beginning of the main path, looking along it.
    const main = this.paths[0];
    const a = main.curve.getPointAt(0), t = main.curve.getTangentAt(0);
    this.start = { x: a.x, z: a.z, yaw: Math.atan2(-t.x, -t.z) };
  }

  // Called every frame: darken the water at night.
  update(level) {
    this.waterMat.color.copy(this.waterColor).multiplyScalar(level);
  }

  pathByName(name) {
    return this.paths.find((p) => p.name === name) || this.paths[0];
  }

  // ---- Reading the maps at any x,z --------------------------------------------------
  _grid(x, z) {
    const half = this.size / 2, n = this.N;
    const gx = clamp((x + half) / this.cell, 0, n - 0.0001);
    const gz = clamp((z + half) / this.cell, 0, n - 0.0001);
    const i = Math.floor(gx), j = Math.floor(gz);
    return [i, j, gx - i, gz - j];
  }
  // Exact ground height (matches the triangles you see).
  height(x, z) {
    const [i, j, fx, fz] = this._grid(x, z), n1 = this.N + 1, H = this.H;
    const h00 = H[j * n1 + i], h10 = H[j * n1 + i + 1], h01 = H[(j + 1) * n1 + i], h11 = H[(j + 1) * n1 + i + 1];
    if (fx + fz <= 1) return h00 + (h10 - h00) * fx + (h01 - h00) * fz;
    return h11 + (h01 - h11) * (1 - fx) + (h10 - h11) * (1 - fz);
  }
  _sample(arr, x, z) {
    const [i, j, fx, fz] = this._grid(x, z), n1 = this.N + 1;
    const a = arr[j * n1 + i], b = arr[j * n1 + i + 1], c = arr[(j + 1) * n1 + i], d = arr[(j + 1) * n1 + i + 1];
    return a + (b - a) * fx + (c - a) * fz + (a - b - c + d) * fx * fz;
  }
  pathDistance(x, z) { return this._sample(this.dPath, x, z); }
  openDistance(x, z) { return this._sample(this.dOpen, x, z); }

  // Can the player step from (fx,fz) to (x,z)? Steep hills and the world edge say no.
  inWater(x, z) { return this.height(x, z) < this.S.world.waterLevel - 0.05; }
  isBlocked(x, z, fx, fz) {
    const limit = this.size / 2 - 4;
    if (Math.abs(x) > limit || Math.abs(z) > limit) return true;
    const h = this.height(x, z);                      // (water does not block: you can wade)
    const d = Math.hypot(x - fx, z - fz);
    if (d > 1e-6 && (h - this.height(fx, fz)) / d > this.S.player.maxSlope) return true;
    return false;
  }

  dispose() {
    this.group.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
    if (this.group.parent) this.group.parent.remove(this.group);
  }
}

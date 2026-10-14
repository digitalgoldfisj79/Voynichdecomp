// ============================================================================
// groundfog.js  -  mist that hugs the ground and slowly swirls.
//
// How it stays cheap (no lag):
//   * It is only THREE thin sheets of mesh draped over the terrain just around
//     you (settings.groundFog.radius). The sheets follow you as you walk.
//   * The swirling is not simulated. A tiny shader (graphics-card program)
//     computes a moving noise pattern per pixel: the pattern is bent by a second
//     slowly drifting noise, which makes it curl and churn.
//   * Mist is thicker over ponds and down in the path valleys, thinner on hilltops.
// ============================================================================

class GroundFog {
  constructor(scene, terrain, S) {
    this.scene = scene; this.terrain = terrain; this.S = S;
    const F = S.groundFog, N = F.gridCells, n1 = N + 1;
    this.N = N; this.cell = (F.radius * 2) / N; this.time = 0;
    this.center = new THREE.Vector2(1e9, 1e9);

    // One grid shared by all three sheets. Its x/z never change (it is moved as a whole);
    // only the ground height under it is refreshed when you have walked a cell or two.
    this.pos = new Float32Array(n1 * n1 * 3);
    this.boost = new Float32Array(n1 * n1);
    for (let j = 0; j < n1; j++) for (let i = 0; i < n1; i++) {
      const k = (j * n1 + i) * 3;
      this.pos[k] = -F.radius + i * this.cell; this.pos[k + 2] = -F.radius + j * this.cell;
    }
    const idx = [];
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const a = j * n1 + i;
      idx.push(a, a + n1, a + 1, a + 1, a + n1, a + n1 + 1);
    }
    this.geo = new THREE.BufferGeometry();
    this.posAttr = new THREE.BufferAttribute(this.pos, 3); this.posAttr.setUsage(THREE.DynamicDrawUsage);
    this.boostAttr = new THREE.BufferAttribute(this.boost, 1); this.boostAttr.setUsage(THREE.DynamicDrawUsage);
    this.geo.setAttribute('position', this.posAttr);
    this.geo.setAttribute('aBoost', this.boostAttr);
    this.geo.setIndex(idx);

    this.group = new THREE.Group();
    scene.add(this.group);
    this.sheets = F.layers.map((l, k) => {
      const material = new THREE.ShaderMaterial({
        transparent: true, depthWrite: false, side: THREE.DoubleSide,
        uniforms: {
          uTime: { value: 0 }, uOffset: { value: 0 }, uDensity: { value: 0 }, uScale: { value: l.scale },
          uSeed: { value: l.seed }, uSwirl: { value: 0 }, uRadius: { value: F.radius },
          uCam: { value: new THREE.Vector3() }, uColor: { value: new THREE.Color() },
        },
        vertexShader: `
          uniform float uOffset, uTime, uScale, uSeed, uSwirl; attribute float aBoost;
          varying vec3 vWorld; varying float vBoost; varying vec2 vWarp;
          float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
          float vnoise(vec2 p) {
            vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
            return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
          }
          void main() {
            vec3 p = position; p.y += uOffset;
            vec4 wp = modelMatrix * vec4(p, 1.0);
            vWorld = wp.xyz; vBoost = aBoost;
            // [quality] the slow swirl is worked out here, once per mesh point, instead of for every pixel
            // (it changes over tens of metres, so the result looks the same and costs half as much)
            vec2 q = wp.xz * uScale + uSeed;
            vWarp = (vec2(vnoise(q * 0.8 + vec2(uTime * 0.05, -uTime * 0.04)),
                          vnoise(q * 0.8 + vec2(5.2 - uTime * 0.045, 1.3 + uTime * 0.05))) - 0.5) * uSwirl * 5.0;
            gl_Position = projectionMatrix * viewMatrix * wp;
          }`,
        fragmentShader: `
          uniform float uTime, uDensity, uScale, uSeed, uSwirl, uRadius;
          uniform vec3 uCam, uColor;
          varying vec3 vWorld; varying float vBoost; varying vec2 vWarp;
          float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
          float vnoise(vec2 p) {
            vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
            return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
          }
          void main() {
            vec2 p = vWorld.xz * uScale + uSeed;
            p += vWarp;                                           // swirl (from the vertex shader)
            p += vec2(0.035, 0.02) * uTime;                       // overall drift with the wind
            float n = vnoise(p) * 0.65 + vnoise(p * 2.1 + 3.7) * 0.35;
            float cloud = smoothstep(0.30, 0.78, n);              // patches with gaps between them
            float d = length(vWorld.xz - uCam.xz);
            float fade = 1.0 - smoothstep(uRadius * 0.45, uRadius, d);   // melts away with distance
            float near = smoothstep(2.0, 11.0, d);                        // never blinds the camera
            float a = cloud * uDensity * vBoost * fade * near;
            gl_FragColor = vec4(uColor, clamp(a, 0.0, 0.85));
          }`,
      });
      const mesh = new THREE.Mesh(this.geo, material);
      mesh.frustumCulled = false; mesh.renderOrder = 2 + k;
      this.group.add(mesh);
      return { mesh, material, layer: l };
    });
  }

  // Refresh heights + thickness under the sheets (done only when you have moved a cell or so).
  _rebuild(cx, cz) {
    const T = this.terrain, n1 = this.N + 1, W = this.S.world;
    for (let j = 0; j < n1; j++) for (let i = 0; i < n1; i++) {
      const k = j * n1 + i, wx = cx + this.pos[k * 3], wz = cz + this.pos[k * 3 + 2];
      const h = T.height(wx, wz);
      this.pos[k * 3 + 1] = h;
      let pondMask = 0;
      for (const c of T.ponds) pondMask = Math.max(pondMask, 1 - smoothstep(c.radius * 0.8, c.radius * 1.7, Math.hypot(wx - c.x, wz - c.z)));
      const valley = 1 - smoothstep(0, 7, T.openDistance(wx, wz));           // low ground between the hills
      const high = smoothstep(4, 16, h);                                      // thinner up on hilltops
      const slope = Math.hypot(T.height(wx + 1.5, wz) - T.height(wx - 1.5, wz), T.height(wx, wz + 1.5) - T.height(wx, wz - 1.5)) / 3;
      const steep = smoothstep(0.25, 0.8, slope);                             // no mist clinging to steep walls
      this.boost[k] = (0.45 + 0.5 * valley + 0.65 * pondMask) * (1 - 0.7 * high) * (1 - 0.85 * steep);
    }
    this.posAttr.needsUpdate = true; this.boostAttr.needsUpdate = true;
    this.group.position.set(cx, 0, cz);
    this.center.set(cx, cz);
  }

  update(camera, dt, horizonColor, level) {
    const F = this.S.groundFog;
    this.group.visible = F.enabled;
    if (!F.enabled) return;
    this.time += dt * F.speed;
    const cx = Math.round(camera.position.x / this.cell) * this.cell, cz = Math.round(camera.position.z / this.cell) * this.cell;
    if (Math.abs(cx - this.center.x) > this.cell * 1.5 || Math.abs(cz - this.center.y) > this.cell * 1.5 || this.center.x > 1e8) this._rebuild(cx, cz);
    const col = horizonColor.clone().lerp(new THREE.Color(1, 1, 1), 0.35).multiplyScalar(level);
    for (const s of this.sheets) {
      const u = s.material.uniforms;
      u.uTime.value = this.time; u.uOffset.value = s.layer.height * F.height;
      u.uDensity.value = s.layer.density * F.density; u.uSwirl.value = F.swirl;
      u.uCam.value.copy(camera.position); u.uColor.value.copy(col);
    }
  }

  dispose() {
    this.geo.dispose(); this.sheets.forEach((s) => s.material.dispose());
    this.scene.remove(this.group);
  }
}

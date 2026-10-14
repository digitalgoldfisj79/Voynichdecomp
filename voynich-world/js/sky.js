// ============================================================================
// sky.js  -  sky colours, sun, moon, stars, fog colour and lights.
//
// Time of day (settings.sky.timeOfDay) picks colours from the palette list in
// settings.js. The HORIZON colour is also used for fog, so the hills fade
// into the sky instead of ending in a visible edge.
// The sun/moon/stars are simple placeholders; later you swap in the
// manuscript's astronomical drawings from the sky/ folder.
// ============================================================================

class Sky {
  constructor(scene, renderer, S) {
    this.scene = scene; this.renderer = renderer; this.S = S;
    this.hour = S.sky.timeOfDay;
    this.level = 1;                                   // overall brightness, read by plants/water
    this.horizon = new THREE.Color();

    // Sky dome: a big sphere with a vertical colour gradient, drawn behind everything.
    this.uniforms = { top: { value: new THREE.Color() }, horizon: { value: new THREE.Color() } };
    const dome = new THREE.Mesh(
      new THREE.SphereGeometry(1000, 32, 16),
      new THREE.ShaderMaterial({
        uniforms: this.uniforms, side: THREE.BackSide, depthWrite: false, fog: false,
        vertexShader: 'varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
        fragmentShader: 'uniform vec3 top; uniform vec3 horizon; varying vec3 vDir; void main(){ float t = pow(clamp(vDir.y,0.0,1.0), 0.55); gl_FragColor = vec4(mix(horizon, top, t), 1.0); }',
      }));
    dome.renderOrder = -10; dome.frustumCulled = false;
    scene.add(dome); this.dome = dome;

    // Stars: scattered dots over the upper sky (they fade in at night).
    const rng = makeRng(S.seed, 'stars'), n = S.sky.starCount, sp = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const a = rng.range(0, Math.PI * 2), e = Math.asin(rng.range(0.05, 1)), r = 900;
      sp[i * 3] = Math.cos(a) * Math.cos(e) * r; sp[i * 3 + 1] = Math.sin(e) * r; sp[i * 3 + 2] = Math.sin(a) * Math.cos(e) * r;
    }
    const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.BufferAttribute(sp, 3));
    this.starMat = new THREE.PointsMaterial({ color: '#f3e8c4', size: 2.6, sizeAttenuation: false, transparent: true, opacity: 0, fog: false, depthWrite: false });
    this.stars = new THREE.Points(sg, this.starMat); this.stars.frustumCulled = false;
    scene.add(this.stars);

    // Sun and moon: flat sprites that always face you.
    this.sun = this._sprite(this._sunTexture(), 120);
    this.moon = this._sprite(this._moonTexture(), 80);

    // Lights: soft all-round light plus one direct light from the sun (or moon).
    this.hemi = new THREE.HemisphereLight('#ffffff', '#7a6648', 1.0);
    this.dir = new THREE.DirectionalLight('#ffffff', 0.8);
    scene.add(this.hemi, this.dir, this.dir.target);

    this._cols = S.sky.palette.map((k) => ({ hour: k.hour, top: new THREE.Color(k.top), horizon: new THREE.Color(k.horizon),
      ambient: new THREE.Color(k.ambient), sun: new THREE.Color(k.sun), sunPower: k.sunPower, level: k.level }));
    this.update(null, 0);
  }

  _sprite(texture, size) {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, fog: false, transparent: true, depthWrite: false }));
    s.scale.set(size, size, 1); s.renderOrder = -5; s.frustumCulled = false;
    this.scene.add(s);
    return s;
  }
  _sunTexture() {
    const c = document.createElement('canvas'); c.width = c.height = 256;
    const x = c.getContext('2d'); x.translate(128, 128);
    x.fillStyle = '#e6b84a'; x.strokeStyle = INK; x.lineWidth = 4; x.lineJoin = 'round';
    for (let i = 0; i < 16; i++) {                      // pointed rays
      x.save(); x.rotate(i * Math.PI / 8); x.beginPath(); x.moveTo(-12, 66); x.lineTo(0, 122); x.lineTo(12, 66); x.closePath(); x.fill(); x.stroke(); x.restore();
    }
    x.beginPath(); x.arc(0, 0, 70, 0, Math.PI * 2); x.fillStyle = '#f0cc62'; x.fill(); x.stroke();
    x.beginPath(); x.arc(0, 0, 50, 0, Math.PI * 2); x.lineWidth = 2; x.stroke();
    return new THREE.CanvasTexture(c);
  }
  _moonTexture() {
    const c = document.createElement('canvas'); c.width = c.height = 256;
    const x = c.getContext('2d'), R = 90, cx = 128, cy = 128;
    x.beginPath(); x.arc(cx, cy, R, Math.PI * 0.5, Math.PI * 1.5, false);       // outer curve, left side
    x.quadraticCurveTo(cx - 1.1 * R, cy, cx, cy + R);                           // inner curve back down
    x.fillStyle = '#efe6c6'; x.fill(); x.lineWidth = 4; x.strokeStyle = INK; x.lineJoin = 'round'; x.stroke();
    return new THREE.CanvasTexture(c);
  }

  // Blend the palette to the current hour.
  _blend() {
    const k = this._cols, h = ((this.hour % 24) + 24) % 24;
    let i = 0;
    while (i < k.length - 2 && h >= k[i + 1].hour) i++;
    const a = k[i], b = k[i + 1], t = smoothstep(a.hour, b.hour, h);
    return {
      top: a.top.clone().lerp(b.top, t), horizon: a.horizon.clone().lerp(b.horizon, t),
      ambient: a.ambient.clone().lerp(b.ambient, t), sun: a.sun.clone().lerp(b.sun, t),
      sunPower: lerp(a.sunPower, b.sunPower, t), level: lerp(a.level, b.level, t),
    };
  }

  update(camera, dt) {
    const S = this.S;
    if (S.sky.timeSpeed) this.hour = (this.hour + S.sky.timeSpeed * dt / 60 + 24) % 24;
    else this.hour = S.sky.timeOfDay;
    // rebuild palette cache if the panel edited colours
    this._cols.forEach((c, i) => { const p = S.sky.palette[i]; c.top.set(p.top); c.horizon.set(p.horizon); c.ambient.set(p.ambient); c.sun.set(p.sun); c.sunPower = p.sunPower; c.level = p.level; });
    const c = this._blend();
    this.uniforms.top.value.copy(c.top); this.uniforms.horizon.value.copy(c.horizon);
    this.horizon.copy(c.horizon);
    this.scene.fog.color.copy(c.horizon); this.scene.fog.near = S.fog.near; this.scene.fog.far = S.fog.far;
    this.renderer.setClearColor(c.horizon);
    this.hemi.color.copy(c.ambient); this.hemi.groundColor.set('#7a6648').multiplyScalar(c.level); this.hemi.intensity = 0.62 * (0.4 + 0.6 * c.level);
    this.level = c.level;

    // Sun travels east -> west: rises 6:00, highest 12:00, sets 18:00. Moon is opposite.
    const ang = (this.hour - 6) / 12 * Math.PI;
    const sunDir = new THREE.Vector3(Math.cos(ang) * 0.92, Math.sin(ang), -0.4).normalize();
    const moonDir = sunDir.clone().multiplyScalar(-1);
    const sunUp = sunDir.y > -0.05;
    this.dir.color.copy(c.sun); this.dir.intensity = c.sunPower * 0.55;
    this.dir.position.copy(sunUp ? sunDir : moonDir).multiplyScalar(100);
    if (typeof Q !== 'undefined') {                    // [quality] the plants' light and shadows follow the sun (or, faintly, the moon)
      Q.sun.dir.value.copy(sunUp ? sunDir : moonDir); Q.sun.color.value.copy(c.sun);
      Q.sun.power.value = sunUp ? clamp(c.sunPower, 0, 1) * smoothstep(-0.05, 0.25, sunDir.y) : 0.12 * (1 - c.level);
    }

    const night = clamp(1 - (c.level - 0.5) * 3, 0, 1);               // 1 at deep night, 0 by day
    this.starMat.opacity = night;
    this.sun.material.color.setScalar(1); this.moon.material.color.setScalar(1);
    this.moon.material.opacity = clamp(0.35 + night, 0, 1);

    if (camera) {
      this.dome.position.copy(camera.position); this.stars.position.copy(camera.position);
      this.sun.position.copy(camera.position).addScaledVector(sunDir, 800);
      this.moon.position.copy(camera.position).addScaledVector(moonDir, 800);
    }
  }
}

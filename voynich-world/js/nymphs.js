// ============================================================================
// nymphs.js  -  placeholder bathing nymphs in the ponds.
//
// Each pond has a few "places" (settings.nymphs.perPond). A place can be:
//   bathing  - standing in the water, bobbing gently, facing you
//   diving   - noticed you: sinks under the surface, leaving a ripple
//   fleeing  - noticed you: wades out of the pond and runs away, then fades
//   rising   - coming back: rises out of the water
//   absent   - gone. After a random delay, and only while you are far from the
//              pond, it may come back (so they do not pop up in front of you).
//
// The water hides their lower body with a "clipping plane" at the water level:
// anything below the surface is simply not drawn. Diving is just sinking.
// The pictures are the bathing-page cut-outs (js/images.js, settings.nymphs.pictures);
// if those are missing, the simple stand-ins below are drawn instead.
// ============================================================================

// Placeholder nymph picture: a plain paper figure with long hair (no detail).
function makeNymphTexture(variant) {
  const w = 256, h = 512, c = document.createElement('canvas'); c.width = w; c.height = h;
  const x = c.getContext('2d'); x.lineJoin = 'round'; x.lineCap = 'round';
  const skin = '#ecd3ae', hair = ['#8a4a2c', '#4a3022', '#c9a05a'][variant % 3], ink = '#3a2a1c', cx = w / 2;
  const shape = (fn, fill) => { x.beginPath(); fn(); x.closePath(); x.fillStyle = fill; x.fill(); x.lineWidth = 5; x.strokeStyle = ink; x.stroke(); };
  // hair behind
  shape(() => { x.moveTo(cx - 40, 70); x.quadraticCurveTo(cx - 78, 190, cx - 50, 300); x.lineTo(cx + 50, 300); x.quadraticCurveTo(cx + 78, 190, cx + 40, 70); }, hair);
  // body (simple column) and legs
  shape(() => { x.moveTo(cx - 34, 120); x.quadraticCurveTo(cx - 44, 230, cx - 36, 330); x.lineTo(cx - 28, 500); x.lineTo(cx - 4, 500); x.lineTo(cx, 360); x.lineTo(cx + 4, 500); x.lineTo(cx + 28, 500); x.lineTo(cx + 36, 330); x.quadraticCurveTo(cx + 44, 230, cx + 34, 120); }, skin);
  // arms: raised to the hair (variant 0), folded (1), one lifted (2)
  const arm = (side, up) => { const s = side;
    shape(() => { x.moveTo(cx + s * 30, 130); x.quadraticCurveTo(cx + s * 72, up ? 110 : 190, cx + s * (up ? 36 : 22), up ? 62 : 250);
                  x.lineTo(cx + s * (up ? 28 : 36), up ? 70 : 250); x.quadraticCurveTo(cx + s * 52, up ? 125 : 190, cx + s * 22, 150); }, skin); };
  if (variant === 0) { arm(-1, true); arm(1, true); } else if (variant === 1) { arm(-1, false); arm(1, false); } else { arm(-1, true); arm(1, false); }
  // head and hair front
  shape(() => x.ellipse(cx, 62, 25, 31, 0, 0, Math.PI * 2), skin);
  shape(() => { x.moveTo(cx - 28, 60); x.quadraticCurveTo(cx - 20, 22, cx + 2, 30); x.quadraticCurveTo(cx + 26, 24, cx + 28, 62); x.quadraticCurveTo(cx + 10, 44, cx - 28, 60); }, hair);
  const t = new THREE.CanvasTexture(c); t.anisotropy = 4; return t;
}

class Nymphs {
  constructor(scene, terrain, S, seed) {
    this.scene = scene; this.terrain = terrain; this.S = S;
    this.rng = makeRng(seed, 'nymphs');
    this.time = 0; this.slots = []; this.group = new THREE.Group(); scene.add(this.group);
    const N = S.nymphs;
    if (!N.enabled) return;
    this.clip = new THREE.Plane(new THREE.Vector3(0, 1, 0), -S.world.waterLevel);   // keeps only what is above the water
    this.kinds = makeNymphKinds(N.pictures) || [0, 1, 2].map((v) => ({ id: 'placeholder' + v, aspect: 0.5, texture: makeNymphTexture(v) }));
    this.textures = this.kinds.map((k) => k.texture);
    const cardGeo = new THREE.PlaneGeometry(1, 1); cardGeo.translate(0, 0.5, 0);    // origin at the feet
    const ringGeo = new THREE.RingGeometry(0.55, 0.7, 40); ringGeo.rotateX(-Math.PI / 2);
    this.geos = [cardGeo, ringGeo];
    for (const pond of terrain.ponds) {
      for (let i = 0; i < N.perPond; i++) {
        const kind = this.rng.pick(this.kinds);
        const mat = new THREE.MeshBasicMaterial({ map: kind.texture, transparent: true, alphaTest: 0.05, side: THREE.DoubleSide, clippingPlanes: [this.clip] });
        const mesh = new THREE.Mesh(cardGeo, mat); mesh.visible = false; mesh.frustumCulled = false;
        mesh.scale.set(N.height * kind.aspect, N.height, 1);
        const ringMat = new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0, depthWrite: false });
        const ring = new THREE.Mesh(ringGeo, ringMat); ring.visible = false; ring.renderOrder = 3;
        this.group.add(mesh, ring);
        const slot = { pond, mesh, mat, ring, ringMat, state: 'absent', t: 0, x: 0, z: 0, y: 0, dirX: 0, dirZ: 0,
                       react: null, scare: this.rng.range(0.7, 1.0), readyAt: 0, ringT: 1, bobPhase: this.rng.range(0, 6.28) };
        this.slots.push(slot);
        if (this.rng.chance(N.presence)) this._appear(slot, true);
        else slot.readyAt = this.rng.range(5, N.returnDelay[1]);
      }
    }
  }

  _place(slot) {                       // a free spot in the deeper part of the pond
    const p = slot.pond, r = this.S.nymphs;
    for (let k = 0; k < 20; k++) {
      const a = this.rng.range(0, Math.PI * 2), d = p.radius * Math.sqrt(this.rng.range(0.02, 0.4));
      const x = p.x + Math.cos(a) * d, z = p.z + Math.sin(a) * d;
      if (this.terrain.height(x, z) > this.S.world.waterLevel - 0.35) continue;
      if (this.slots.some((o) => o !== slot && o.state !== 'absent' && o.pond === p && Math.hypot(o.x - x, o.z - z) < 2.4)) continue;
      return [x, z];
    }
    return [p.x, p.z];
  }
  _appear(slot, instant) {
    const [x, z] = this._place(slot), h = this.S.nymphs.height;
    slot.x = x; slot.z = z; slot.react = null; slot.mat.opacity = 1;
    slot.mesh.visible = true;
    if (instant) { slot.state = 'bathing'; slot.y = this.S.world.waterLevel - h * 0.42; }
    else { slot.state = 'rising'; slot.y = this.S.world.waterLevel - h * 1.1; this._ripple(slot); }
  }
  _ripple(slot) { slot.ring.position.set(slot.x, this.S.world.waterLevel + 0.03, slot.z); slot.ring.visible = true; slot.ringT = 0; }
  _gone(slot) {
    const N = this.S.nymphs;
    slot.state = 'absent'; slot.mesh.visible = false;
    slot.readyAt = this.time + this.rng.range(N.returnDelay[0], N.returnDelay[1]);
  }

  update(dt, playerPos, camera, level) {
    const N = this.S.nymphs, W = this.S.world, T = this.terrain;
    if (!N.enabled) return;
    this.time += dt;
    for (const s of this.slots) {
      const dPlayer = Math.hypot(playerPos.x - s.x, playerPos.z - s.z);
      const dPond = Math.hypot(playerPos.x - s.pond.x, playerPos.z - s.pond.z);

      if (s.state === 'bathing') {
        s.bobPhase += dt * 1.3;
        s.y = W.waterLevel - N.height * 0.42 + Math.sin(s.bobPhase) * 0.03;
        if (dPlayer < N.scareDistance * s.scare && s.react === null) s.react = this.rng.range(0, 1.2);   // notice, with a little delay
        if (s.react !== null) {
          s.react -= dt;
          if (s.react <= 0) {
            if (this.rng.chance(N.diveChance)) { s.state = 'diving'; this._ripple(s); }
            else {                                                       // run: away from the player, out through the nearest shore
              let dx = s.x - playerPos.x, dz = s.z - playerPos.z; const l = Math.hypot(dx, dz) || 1; dx /= l; dz /= l;
              const ox = s.x - s.pond.x, oz = s.z - s.pond.z, ol = Math.hypot(ox, oz) || 1;
              dx = dx * 0.6 + ox / ol * 0.4; dz = dz * 0.6 + oz / ol * 0.4; const m = Math.hypot(dx, dz) || 1;
              s.dirX = dx / m; s.dirZ = dz / m; s.state = 'fleeing';
            }
          }
        }
      } else if (s.state === 'diving') {
        s.y -= N.diveSpeed * dt;
        if (s.y < W.waterLevel - N.height * 1.1) { this._ripple(s); this._gone(s); }
      } else if (s.state === 'rising') {
        s.y += N.diveSpeed * 0.7 * dt;
        if (s.y >= W.waterLevel - N.height * 0.42) { s.state = 'bathing'; s.react = null; }
      } else if (s.state === 'fleeing') {
        s.x += s.dirX * N.runSpeed * dt; s.z += s.dirZ * N.runSpeed * dt;
        s.y = Math.max(T.height(s.x, s.z), W.waterLevel - N.height * 0.42);          // wade, then stand on the ground
        const away = Math.hypot(s.x - s.pond.x, s.z - s.pond.z);
        if (away > s.pond.radius + 9 || T.openDistance(s.x, s.z) > 1.5) {
          s.mat.opacity -= dt / 1.2;                                                  // fade out of sight
          if (s.mat.opacity <= 0) this._gone(s);
        }
      } else if (s.state === 'absent') {
        if (this.time >= s.readyAt) {
          if (dPond > s.pond.radius + N.returnDistance && this.rng.chance(N.returnChance)) this._appear(s, false);
          else if (dPond > s.pond.radius + N.returnDistance) s.readyAt = this.time + this.rng.range(15, 40);   // not this time
        }
      }

      if (s.state !== 'absent') {
        s.mesh.position.set(s.x, s.y, s.z);
        s.mesh.rotation.y = Math.atan2(camera.position.x - s.x, camera.position.z - s.z);   // always face you
        s.mat.color.setScalar(level);
      }
      if (s.ring.visible) {                                                          // expanding ripple
        s.ringT += dt / 1.6;
        if (s.ringT >= 1) s.ring.visible = false;
        else { const k = 1 + s.ringT * 3.5; s.ring.scale.set(k, 1, k); s.ringMat.opacity = 0.55 * (1 - s.ringT); s.ringMat.color.setScalar(Math.min(1, level + 0.2)); }
      }
    }
  }

  dispose() {
    this.slots.forEach((s) => { s.mat.dispose(); s.ringMat.dispose(); });
    (this.textures || []).forEach((t) => t.dispose()); (this.geos || []).forEach((g) => g.dispose());
    this.scene.remove(this.group);
  }
}

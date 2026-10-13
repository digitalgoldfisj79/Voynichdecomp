// ============================================================================
// player.js  -  the camera: walking, flying and the autopilot glide.
//
// Walking feels smooth because nothing jumps straight to its target:
//   * mouse look chases where you pointed (mouseSmoothness)
//   * speed ramps up (acceleration) and eases off (braking)
//   * height follows the ground softly, with a small head-bob (bobAmount)
// ============================================================================

class Player {
  constructor(camera, S) {
    this.camera = camera; this.S = S; this.terrain = null;
    this.mode = 'walk';                         // 'walk' | 'fly' | 'auto'
    this.keys = {};
    this.stick = { x: 0, y: 0 };                // on-screen joystick on phones/tablets (js/touch.js): -1..1
    this.pos = new THREE.Vector3();             // ground position (x,z used)
    this.vx = 0; this.vz = 0;                   // current walking velocity
    this.yaw = 0; this.pitch = 0;               // where the camera really looks
    this.targetYaw = 0; this.targetPitch = 0;   // where the mouse says to look
    this.groundY = 0; this.flyY = 10; this.bobPhase = 0;
    this.autoU = 0; this.autoDir = 1; this.time = 0;

    window.addEventListener('keydown', (e) => {
      if (this._typing(e)) return;
      this.keys[e.code] = true;
      if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
    });
    window.addEventListener('keyup', (e) => { this.keys[e.code] = false; });
    window.addEventListener('blur', () => { this.keys = {}; });
  }
  _typing(e) { const t = e.target; return t && t.tagName === 'INPUT' && (t.type === 'number' || t.type === 'text'); }

  setTerrain(t) { this.terrain = t; this.resetToStart(); }

  resetToStart() {
    const s = this.terrain.start;
    this.pos.set(s.x, 0, s.z);
    this.yaw = this.targetYaw = s.yaw; this.pitch = this.targetPitch = -0.05;
    this.groundY = this.terrain.height(s.x, s.z);
    this.vx = this.vz = 0; this.autoU = 0; this.autoDir = 1;
  }

  look(dx, dy) {                                // called with raw mouse movement
    const sens = this.S.player.mouseSensitivity;
    this.targetYaw -= dx * sens;
    this.targetPitch = clamp(this.targetPitch - dy * sens, -1.45, 1.45);
  }

  setMode(mode) {
    if (mode === this.mode) return;
    if (mode === 'fly') this.flyY = this.camera.position.y;
    if (mode === 'auto') {                      // start gliding from the nearest point of the chosen path
      const path = this.terrain.pathByName(this.S.autopilot.path);
      let best = 1e9, bu = 0;
      for (let i = 0; i <= 200; i++) {
        const p = path.curve.getPointAt(i / 200), d = Math.hypot(p.x - this.pos.x, p.z - this.pos.z);
        if (d < best) { best = d; bu = i / 200; }
      }
      this.autoU = bu;
    }
    if (mode === 'walk') { this.groundY = this.terrain.height(this.pos.x, this.pos.z); this.vx = this.vz = 0; }
    this.targetYaw = this.yaw; this.targetPitch = this.pitch;
    this.mode = mode;
  }

  _key(a, b) { return (this.keys[a] || this.keys[b]) ? 1 : 0; }

  update(dt) {
    const P = this.S.player, T = this.terrain;
    this.time += dt;
    const f = clamp(this._key('KeyW', 'ArrowUp') - this._key('KeyS', 'ArrowDown') + this.stick.y, -1, 1);
    const r = clamp(this._key('KeyD', 'ArrowRight') - this._key('KeyA', 'ArrowLeft') + this.stick.x, -1, 1);
    const anyMove = f !== 0 || r !== 0;
    if (this.mode === 'auto' && anyMove) this.setMode('walk');           // moving yourself cancels the autopilot

    // ---- looking (walk and fly) -------------------------------------------------
    if (this.mode !== 'auto') {
      const smooth = clamp(P.mouseSmoothness, 0, 1);
      const a = smooth <= 0 ? 1 : 1 - Math.exp(-lerp(60, 2.5, smooth) * dt);
      this.yaw += (this.targetYaw - this.yaw) * a;
      this.pitch += (this.targetPitch - this.pitch) * a;
    }

    let eyeY, lateral = 0;
    if (this.mode === 'walk') {
      // direction you want to go, from the keys and where you face
      let dx = -Math.sin(this.yaw) * f + Math.cos(this.yaw) * r;
      let dz = -Math.cos(this.yaw) * f - Math.sin(this.yaw) * r;
      const len = Math.hypot(dx, dz), mag = Math.min(1, len);    // mag < 1 = joystick pushed only part-way: walk slower
      if (len > 0) { dx /= len; dz /= len; }
      const speed = P.walkSpeed * (this.keys.ShiftLeft || this.keys.ShiftRight ? P.runMultiplier : 1)
                    * (T.inWater(this.pos.x, this.pos.z) ? P.wadeSpeed : 1);   // slower in water
      // speed ramps toward the wanted speed (acceleration) or toward zero (braking)
      const tvx = dx * speed * mag, tvz = dz * speed * mag;
      const ex = tvx - this.vx, ez = tvz - this.vz, el = Math.hypot(ex, ez), step = (len > 0 ? P.acceleration : P.braking) * dt;
      if (el <= step) { this.vx = tvx; this.vz = tvz; } else { this.vx += ex / el * step; this.vz += ez / el * step; }
      this._tryMove(this.vx * dt, 0); this._tryMove(0, this.vz * dt);

      // ground following and head-bob
      const g = T.height(this.pos.x, this.pos.z);
      this.groundY += (g - this.groundY) * (1 - Math.exp(-14 * dt));
      const sp = Math.hypot(this.vx, this.vz);
      this.bobPhase += sp * dt / P.stepLength * Math.PI * 2;
      const amt = P.bobAmount * Math.min(1, sp / P.walkSpeed);
      eyeY = this.groundY + P.eyeHeight + Math.sin(this.bobPhase) * amt;
      lateral = Math.cos(this.bobPhase * 0.5) * amt * 0.6;

    } else if (this.mode === 'fly') {
      const up = (this.keys.Space ? 1 : 0) - (this.keys.KeyC ? 1 : 0) + (this.keys.KeyE ? 1 : 0) - (this.keys.KeyQ ? 1 : 0) + (this.stickUp || 0);
      const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      const fwd = [-Math.sin(this.yaw) * cp, sp, -Math.cos(this.yaw) * cp];
      const speed = P.flySpeed * (this.keys.ShiftLeft || this.keys.ShiftRight ? 2.5 : 1) * dt;
      this.pos.x += (fwd[0] * f + Math.cos(this.yaw) * r) * speed;
      this.pos.z += (fwd[2] * f - Math.sin(this.yaw) * r) * speed;
      this.flyY += (fwd[1] * f + up) * speed;
      const lim = T.size / 2 - 2;
      this.pos.x = clamp(this.pos.x, -lim, lim); this.pos.z = clamp(this.pos.z, -lim, lim);
      this.flyY = Math.max(this.flyY, T.height(this.pos.x, this.pos.z) + 0.8);
      eyeY = this.flyY;

    } else {                                    // 'auto': glide along a path
      const A = this.S.autopilot, path = T.pathByName(A.path);
      this.autoU += this.autoDir * A.speed * dt / path.length;
      if (this.autoU > 1) { this.autoU = 1; this.autoDir = -1; }          // turn round at the ends
      if (this.autoU < 0) { this.autoU = 0; this.autoDir = 1; }
      const p = path.curve.getPointAt(this.autoU);
      const ahead = clamp(this.autoU + this.autoDir * A.lookAhead / path.length, 0, 1);
      const q = path.curve.getPointAt(ahead);
      const want = Math.atan2(-(q.x - p.x), -(q.z - p.z)) + Math.sin(this.time * 0.37) * A.drift;
      this.yaw += wrapPi(want - this.yaw) * (1 - Math.exp(-A.turnRate * dt));
      this.pitch += (-0.04 + Math.sin(this.time * 0.23) * 0.03 - this.pitch) * (1 - Math.exp(-2 * dt));
      this.targetYaw = this.yaw; this.targetPitch = this.pitch;
      this.pos.x = p.x; this.pos.z = p.z;
      const g = T.height(p.x, p.z);
      this.groundY += (g - this.groundY) * (1 - Math.exp(-6 * dt));
      this.bobPhase += A.speed * dt / P.stepLength * Math.PI * 2;
      eyeY = this.groundY + P.eyeHeight + Math.sin(this.bobPhase) * P.bobAmount * 0.8;
    }

    const cam = this.camera;
    cam.position.set(this.pos.x + Math.cos(this.yaw) * lateral, eyeY, this.pos.z - Math.sin(this.yaw) * lateral);
    cam.rotation.set(this.pitch, this.yaw, 0);
  }

  // Move if the new spot is allowed; if a hill/water/edge is in the way, stop that part of the motion.
  _tryMove(dx, dz) {
    if (dx === 0 && dz === 0) return;
    const nx = this.pos.x + dx, nz = this.pos.z + dz;
    if (this.terrain.isBlocked(nx, nz, this.pos.x, this.pos.z)) {
      if (dx !== 0) this.vx = 0;
      if (dz !== 0) this.vz = 0;
    } else { this.pos.x = nx; this.pos.z = nz; }
  }
}

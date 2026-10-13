/* THE GUIDE: a figure from the manuscript who goes along with you.

   By default it's one of the bathers in barrels from the Pisces page (f70v2), turned
   upright. A barrel can't walk, so it hops: squashing as it lands and stretching as it
   leaps. A nymph from the bathing pages (f76v) is the alternative; she walks with a sway.
   Choose in walk.config.js -> guide.

   The guide stays a little to the left of the middle of the picture. When you scroll it
   hurries after you and turns the way you're going. When you stop at a video it hops
   once and says something in a speech bubble, with buttons for the tour, the next stop
   and the video. Click the guide for a remark. */

import * as THREE from '../lib/three.module.min.js';

export function makeGuide({ S, WALK, textures, scene, camera, actions }) {
  const G = WALK.guide || {};
  const tex = textures.get(G.file);
  const group = new THREE.Group();
  if (!tex) return { group, update() {}, hit() { return false; }, poke() {}, mesh: null };

  const aspect = tex.image.width / tex.image.height, H = G.height || 1.5;
  const geo = new THREE.PlaneGeometry(aspect, 1); geo.translate(0, 0.5, 0);
  const mat = new THREE.MeshBasicMaterial({ map: tex, alphaTest: 0.5, side: THREE.DoubleSide }); mat.alphaToCoverage = true;
  const body = new THREE.Mesh(geo, mat);   // normal depth: it stands in front of the scenery, and only the odd plant right at the front can pass in front of it
  const shadowC = document.createElement('canvas'); shadowC.width = shadowC.height = 64;
  const sg = shadowC.getContext('2d'), grd = sg.createRadialGradient(32, 32, 2, 32, 32, 32);
  grd.addColorStop(0, 'rgba(60,45,30,0.55)'); grd.addColorStop(1, 'rgba(60,45,30,0)'); sg.fillStyle = grd; sg.fillRect(0, 0, 64, 64);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(shadowC), transparent: true, depthWrite: false }));
  shadow.renderOrder = 2;
  group.add(body, shadow);
  scene.add(group);

  const hop = (G.gait || 'hop') === 'hop';
  const st = { x: null, v: 0, face: 1, phase: 0, air: 0, jump: 0, jumpV: 0, idleT: 3, lean: 0, squash: 1 };
  const bubble = makeBubble(G, WALK, actions);

  function update(dt, t, posU, cx, walk, near) {
    // where the guide wants to be: a little left of the middle of the view (in walk distance)
    const want = posU + (G.offset ?? -2.6);
    if (st.x === null) st.x = want;
    const prev = st.x;
    st.x += (want - st.x) * (1 - Math.exp(-(hop ? 4.5 : 5.5) * dt));
    st.x = THREE.MathUtils.clamp(st.x, want - 1.6, want + 1.6);              // never falls out of the picture, however fast you scroll
    st.v = (st.x - prev) / Math.max(dt, 1e-4);
    const speed = Math.abs(st.v), moving = speed > 0.25;
    if (st.v > 0.4) st.face = 1; else if (st.v < -0.4) st.face = -1;     // turn the way it's going (with a little hysteresis)

    // gait
    let y = 0, rz = 0, sx = 1, sy = 1;
    if (moving) {
      st.phase += dt * (hop ? Math.min(3.2, 1.4 + speed * 0.5) : speed * 1.5 + 1);
      if (hop) {
        const p = st.phase % 1, arc = Math.sin(p * Math.PI);                    // one hop per phase
        y = arc * Math.min(0.42, 0.18 + speed * 0.06);
        sy = p < 0.12 || p > 0.88 ? 0.9 : 1 + arc * 0.05; sx = 2 - sy;       // squash on landing, stretch in the air
        rz = -st.face * Math.min(0.12, speed * 0.03);                          // leans into the hop
      } else {
        y = Math.abs(Math.sin(st.phase * Math.PI)) * 0.05; rz = Math.sin(st.phase * Math.PI) * 0.05;
      }
      st.idleT = 3 + Math.random() * 4;
    } else {
      sy = 1 + Math.sin(t * 2.1) * 0.012; sx = 2 - sy;                       // breathing
      st.idleT -= dt;
      if (st.idleT < 0 && st.jump === 0) { st.jumpV = 2.2; st.idleT = 4 + Math.random() * 5; }   // the odd little hop on the spot
      rz = THREE.MathUtils.clamp(-actions.pointerX() * 0.06, -0.06, 0.06) * -st.face;      // leans a little towards the pointer
    }
    // poke / arrival jump (simple gravity)
    st.jumpV -= 9.8 * dt; st.jump = Math.max(0, st.jump + st.jumpV * dt); if (st.jump === 0 && st.jumpV < 0) st.jumpV = 0;
    st.lean += (rz - st.lean) * (1 - Math.exp(-8 * dt));

    const gx = cx + (st.x - posU), z = G.z ?? 4.2;
    body.position.set(gx, y + st.jump, z);
    body.scale.set(H * st.face * sx, H * sy, H);
    body.rotation.z = st.lean;
    shadow.position.set(gx, 0.03, z + 0.05);
    const ss = 1 - Math.min(0.5, (y + st.jump) * 0.8); shadow.scale.set(H * aspect * 0.9 * ss, 1, H * 0.25 * ss);

    bubble.update(dt, { moving: moving || speed > 0.1, near, walk, head: body.position.clone().setY(body.position.y + H * sy + 0.1), camera });
  }

  return {
    group, update, mesh: body,
    hit(ray) { const h = ray.intersectObject(body, false)[0]; return !!h; },
    poke() { st.jumpV = 3; bubble.quip(); },
    hello() { st.jumpV = 2.4; },
    bubble,
  };
}

/* The speech bubble: an HTML box pinned above the guide's head. It only speaks about
   where you are (navigation), the stop's own title and note, and a few light remarks. */
function makeBubble(G, WALK, actions) {
  const el = document.getElementById('guide-bubble');
  const kick = el.querySelector('.gb-k'), title = el.querySelector('.gb-t'), say = el.querySelector('.gb-s'), acts = el.querySelector('.gb-a'), bar = el.querySelector('.gb-bar i');
  const name = G.name || 'Your guide';
  const L = Object.assign({
    intro: "I'll take you along the walk. Scroll, swipe or use the arrow keys, or let me lead the way.",
    stop: 'Click the frame to watch it with sound.',
    end: "That's the whole walk. Shall we go round again?",
    back: 'Back again. Click me any time.',
  }, G.lines || {});
  const quips = G.quips || [];
  const S = { key: null, still: 0, typed: 0, full: '', quipUntil: 0, t: 0, muted: false, dismissed: null };

  function show(c) {
    kick.textContent = c.kicker || name; title.textContent = c.title || ''; title.hidden = !c.title;
    S.full = c.speech || ''; S.typed = matchMedia('(prefers-reduced-motion: reduce)').matches ? S.full.length : 0; say.textContent = '';
    acts.replaceChildren(...(c.actions || []).map(a => { const b = document.createElement('button'); b.type = 'button'; b.textContent = a.label; if (a.sec) b.className = 'sec'; b.onclick = e => { e.stopPropagation(); a.fn(); }; return b; }));
    el.querySelector('.gb-bar').hidden = !c.bar; bar.style.width = '0%';
    el.hidden = false;
  }
  const hide = () => { el.hidden = true; };
  el.querySelector('.gb-x').onclick = () => { S.muted = true; hide(); };

  function content(near, walk) {
    if (near.kind === 'intro') return { kicker: name, title: `${WALK.title}`, speech: L.intro,
      actions: [{ label: 'Take the tour →', fn: () => actions.tour() }, { label: "I'll explore", sec: true, fn: () => { S.dismissed = 'intro'; hide(); } }] };
    if (near.kind === 'end') return { kicker: name, speech: L.end, actions: [{ label: 'Start again ↺', fn: () => actions.restart() }] };
    const i = near.index, s = WALK.stations[i], n = WALK.stations.length;
    const a = [{ label: '▶ Watch', fn: () => actions.watch(i) }];
    if (walk.tour) a.push({ label: 'Pause tour', sec: true, fn: () => actions.pauseTour() });
    else a.push({ label: i === n - 1 ? 'To the end →' : 'Next stop →', sec: true, fn: () => actions.next() });
    return { kicker: `${name} · stop ${i + 1} of ${n}`, title: s.title, speech: s.note || L.stop, actions: a, bar: !!walk.tour };
  }

  return {
    quip() {
      if (S.muted) { S.muted = false; show({ speech: L.back }); }
      else if (quips.length) show({ speech: quips[Math.floor(Math.random() * quips.length)] });
      S.quipUntil = S.t + 5; S.key = 'quip';
    },
    update(dt, { moving, near, walk, head, camera }) {
      S.t += dt;
      if (S.quipUntil && (S.t > S.quipUntil)) { S.quipUntil = 0; S.key = null; hide(); }
      if (actions.blocked()) { hide(); S.key = null; return; }
      if (!S.quipUntil) {
        if (moving) { S.still = 0; if (S.key) { hide(); S.key = null; } }
        else {
          S.still += dt;
          const key = near ? (near.kind === 'station' ? 's' + near.index : near.kind) : null;
          if (S.still > 0.4 && key && key !== S.key && !S.muted && S.dismissed !== key) { S.key = key; show(content(near, walk)); actions.hello(); }
          if (key !== S.key && S.key && !key) { hide(); S.key = null; }
        }
      }
      if (!el.hidden) {
        if (S.typed < S.full.length) { S.typed = Math.min(S.full.length, S.typed + dt * 60); say.textContent = S.full.slice(0, Math.floor(S.typed)); }
        if (walk.tour && walk.tour.phase === 'hold') bar.style.width = Math.min(100, walk.tour.t / walk.tour.hold * 100) + '%';
        // pin above the guide's head, kept on screen
        const v = head.clone().project(camera);
        if (v.z > 1) { hide(); return; }
        const sx = (v.x + 1) / 2 * innerWidth, sy = (1 - v.y) / 2 * innerHeight, bw = el.offsetWidth, bh = el.offsetHeight;
        const x = Math.min(Math.max(16, sx - bw * 0.3), innerWidth - bw - 16), y = Math.max(56, sy - bh - 18);
        el.style.transform = `translate3d(${Math.round(x)}px, ${Math.round(y)}px, 0)`;
        el.style.setProperty('--tail', Math.min(bw - 24, Math.max(24, sx - x)) + 'px');
      }
    },
    hide,
  };
}

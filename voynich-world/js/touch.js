// ============================================================================
// touch.js  -  walking on phones and tablets (does nothing with a mouse).
//
//   left thumb   drag anywhere on the lower-left of the screen: a joystick
//                appears under your thumb; push it the way you want to walk
//                (a little = slow, all the way = full speed)
//   right thumb  drag anywhere else to look around
//   buttons      Autopilot, Fly/Walk, and up/down while flying
// Sizes and speeds: settings.js -> touch
// ============================================================================

(function () {
  const coarse = window.matchMedia && matchMedia('(pointer: coarse)').matches;
  if (!(coarse || 'ontouchstart' in window) || !window.VW) return;
  const { player, app, S } = window.VW;
  const T = S.touch || {};
  const size = T.stickSize || 120, R = size / 2;
  document.body.classList.add('touch');
  const panelEl = document.getElementById('panel');
  if (panelEl) panelEl.classList.add('collapsed');               // the settings panel starts folded up (tap its title to open)

  const style = document.createElement('style');
  style.textContent = `
    body.touch #keys, body.touch #click-hint { display: none; }
    body.touch #panel { width: min(290px, 62vw); }
    #view { touch-action: none; }
    .t-ring { position: fixed; width: ${size}px; height: ${size}px; margin: -${R}px 0 0 -${R}px; border-radius: 50%;
              border: 2px solid rgba(255,246,220,.55); background: rgba(40,28,16,.18); pointer-events: none; z-index: 5; }
    .t-knob { position: absolute; left: 50%; top: 50%; width: ${size * 0.42}px; height: ${size * 0.42}px; margin: -${size * 0.21}px 0 0 -${size * 0.21}px;
              border-radius: 50%; background: rgba(255,246,220,.6); }
    .t-ring.idle { opacity: .5; }
    #t-btns { position: fixed; right: calc(12px + env(safe-area-inset-right, 0px)); bottom: calc(14px + env(safe-area-inset-bottom, 0px));
              display: flex; flex-direction: column; gap: 10px; z-index: 6; }
    #t-btns button { min-width: 64px; min-height: 48px; padding: 6px 12px; font: 15px Georgia, serif; color: #3a2a1c;
              background: rgba(238,226,192,.85); border: 1px solid #8a7550; border-radius: 24px; touch-action: none; user-select: none; -webkit-user-select: none; }
    #t-btns button.on { background: #f0cc62; }
    #t-btns .fly-only { display: none; }
    body.flying #t-btns .fly-only { display: block; }
    body.ui-hidden #t-btns, body.ui-hidden .t-ring { opacity: .15; }
  `;
  document.head.append(style);

  // ---- the joystick ----------------------------------------------------------------
  const ring = document.createElement('div'); ring.className = 't-ring idle';
  const knob = document.createElement('div'); knob.className = 't-knob'; ring.append(knob);
  document.body.append(ring);
  const home = () => ({ x: 24 + R, y: innerHeight - 24 - R });      // where it waits when you're not touching it
  function placeRing(x, y, idle) { ring.style.left = x + 'px'; ring.style.top = y + 'px'; ring.classList.toggle('idle', !!idle); knob.style.transform = ''; }
  placeRing(home().x, home().y, true);
  addEventListener('resize', () => { if (ring.classList.contains('idle')) placeRing(home().x, home().y, true); });

  // ---- buttons ----------------------------------------------------------------------
  const box = document.createElement('div'); box.id = 't-btns';
  const btn = (label, cls) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = label; if (cls) b.className = cls; box.append(b); return b; };
  const bUp = btn('▲ up', 'fly-only'), bDown = btn('▼ down', 'fly-only'), bFly = btn('Fly'), bAuto = btn('Autopilot');
  document.body.append(box);
  bAuto.addEventListener('click', () => { app.toggleAuto(); sync(); });
  bFly.addEventListener('click', () => { app.toggleFly(); sync(); });
  for (const [b, v] of [[bUp, 1], [bDown, -1]]) {
    b.addEventListener('pointerdown', (e) => { e.preventDefault(); player.stickUp = v; });
    for (const ev of ['pointerup', 'pointercancel', 'pointerleave']) b.addEventListener(ev, () => { player.stickUp = 0; });
  }
  function sync() {
    bAuto.classList.toggle('on', player.mode === 'auto');
    bFly.textContent = player.mode === 'fly' ? 'Walk' : 'Fly';
    document.body.classList.toggle('flying', player.mode === 'fly');
  }
  setInterval(sync, 300);                       // the autopilot also stops when you start walking

  // ---- touches on the picture ---------------------------------------------------------------
  const canvas = document.getElementById('view');
  const touches = new Map();
  canvas.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse') return;
    e.preventDefault();
    const stickFree = ![...touches.values()].some((t) => t.type === 'stick');
    const type = stickFree && e.clientX < innerWidth * 0.5 && e.clientY > innerHeight * 0.4 ? 'stick' : 'look';
    touches.set(e.pointerId, { type, x: e.clientX, y: e.clientY, cx: e.clientX, cy: e.clientY });
    try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
    if (type === 'stick') placeRing(e.clientX, e.clientY, false);   // the joystick appears under your thumb
  }, { passive: false });
  canvas.addEventListener('pointermove', (e) => {
    const t = touches.get(e.pointerId); if (!t) return;
    e.preventDefault();
    if (t.type === 'stick') {
      let dx = e.clientX - t.cx, dy = e.clientY - t.cy; const l = Math.hypot(dx, dy);
      if (l > R) { dx *= R / l; dy *= R / l; }
      knob.style.transform = `translate(${dx}px, ${dy}px)`;
      const m = Math.min(1, l / R), dead = 0.12, k = m < dead ? 0 : (m - dead) / (1 - dead) / (m || 1);   // small dead zone in the middle
      player.stick.x = dx / R * k; player.stick.y = -dy / R * k;
    } else {
      player.look((e.clientX - t.x) * (T.lookSpeed || 2.2), (e.clientY - t.y) * (T.lookSpeed || 2.2));
      t.x = e.clientX; t.y = e.clientY;
    }
  }, { passive: false });
  const end = (e) => {
    const t = touches.get(e.pointerId); if (!t) return;
    touches.delete(e.pointerId);
    if (t.type === 'stick') { player.stick.x = player.stick.y = 0; placeRing(home().x, home().y, true); }
  };
  canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
  sync();
})();

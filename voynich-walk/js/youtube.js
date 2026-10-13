/* YOUTUBE IN THE FRAMES
   When you stand in front of a station, its YouTube video starts playing inside the
   frame, silently, and the still picture fades away to reveal it. Walk on and it
   pauses and the picture comes back. Clicking the frame still opens the big player
   with sound (ui.js).

   How: there is one YouTube player for the whole walk. It sits in a layer *behind*
   the 3-D picture, placed and tilted so it lines up exactly with the frame
   (Three.js CSS3DRenderer). The 3-D picture then leaves a see-through window where
   the frame's screen is. Anything standing in front of the frame (the guide, a plant)
   still covers the video, because the window is only cut where the screen itself
   is visible.

   The still picture only fades once YouTube reports that the video is really playing,
   so if YouTube refuses (no internet, embedding turned off, a page opened straight
   from the folder) you simply keep seeing the picture. Turn the whole thing off with
   settings.js -> videos.youtubeInFrame. */

import * as THREE from '../lib/three.module.min.js';
import { CSS3DRenderer, CSS3DObject } from '../lib/CSS3DRenderer.js';

const PX = 1280, PY = 720;                                     // the player's size in CSS pixels before it is scaled into the frame
const ORIGIN = 'https://www.youtube-nocookie.com';

export function makeFramePlayer({ S, WALK, enabled, renderer, camera, canvas }) {
  if (!enabled) return { update() {}, render() {}, resize() {}, suspend() {}, solidify() {}, state: () => ({ enabled: false }) };

  // ---- the layer behind the picture ------------------------------------------------------------
  const css = new CSS3DRenderer();
  Object.assign(css.domElement.style, { position: 'fixed', left: '0', top: '0', zIndex: '0', pointerEvents: 'none' });
  css.domElement.setAttribute('aria-hidden', 'true');
  canvas.style.position = 'relative'; canvas.style.zIndex = '1';
  document.body.prepend(css.domElement);
  const cssScene = new THREE.Scene();
  const holder = document.createElement('div');
  Object.assign(holder.style, { width: PX + 'px', height: PY + 'px', background: '#000', pointerEvents: 'none' });
  const iframe = document.createElement('iframe');
  iframe.title = 'YouTube video'; iframe.tabIndex = -1;
  iframe.allow = 'autoplay; encrypted-media; picture-in-picture';
  iframe.referrerPolicy = 'strict-origin-when-cross-origin';   // YouTube needs to know which site it is on
  Object.assign(iframe.style, { width: '100%', height: '100%', border: '0', pointerEvents: 'none' });
  holder.append(iframe);
  const obj = new CSS3DObject(holder); obj.visible = false; cssScene.add(obj);

  // ---- the see-through window -------------------------------------------------------------------
  const holeScene = new THREE.Scene();
  // first make the whole picture fully solid (the soft edges of the cut-outs leave it a little see-through)
  const solid = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
    vertexShader: 'void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: 'void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }',
    depthTest: false, depthWrite: false,
    blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
    blendSrc: THREE.ZeroFactor, blendDst: THREE.OneFactor,              // colour unchanged
    blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.ZeroFactor,    // alpha = 1
  }));
  solid.frustumCulled = false; solid.renderOrder = 0;
  // then fade the screen out where it can be seen (depth-tested, so whatever stands in front stays)
  const holeMat = new THREE.MeshBasicMaterial({ color: 0x000000, fog: false, depthWrite: false,
    blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
    blendSrc: THREE.ZeroFactor, blendDst: THREE.SrcAlphaFactor,          // colour and alpha both multiplied by (1 - fade)
    blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.SrcAlphaFactor,
    polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -4 });
  const hole = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), holeMat);
  hole.renderOrder = 1; hole.frustumCulled = false; hole.visible = false;
  holeScene.add(solid, hole);

  // ---- talking to the YouTube player ------------------------------------------------------------
  const P = { loaded: null, idx: -1, shift: 0, playing: false, ready: false, paused: false, f: 0, listenT: 0, failed: new Set(), why: new Set(), errors: [] };
  const post = o => { if (!P.frameLoaded) return; try { iframe.contentWindow.postMessage(JSON.stringify(o), ORIGIN); } catch (_) {} };
  const command = (func, args = []) => post({ event: 'command', func, args, id: 1, channel: 'widget' });
  const listen = () => post({ event: 'listening', id: 1, channel: 'widget' });
  iframe.addEventListener('load', () => { P.frameLoaded = !!P.loaded; listen(); });
  addEventListener('message', e => {
    if (e.source !== iframe.contentWindow || e.origin !== ORIGIN) return;
    let d; try { d = typeof e.data === 'string' ? JSON.parse(e.data) : e.data; } catch (_) { return; }
    if (!d || !d.event) return;
    P.ready = true;
    if (d.event === 'onError') { P.failed.add(P.loaded); P.playing = false; P.errors.push([P.loaded, d.info]); }
    const s = d.info && d.info.playerState;
    if (s === 1) P.playing = true;
    else if (s === 0 || s === 2 || s === -1 || s === 5) P.playing = false;   // 3 = buffering: keep showing what we have
  });
  function load(st) {
    const q = new URLSearchParams({ autoplay: 1, mute: 1, controls: 0, playsinline: 1, rel: 0, iv_load_policy: 3, disablekb: 1, fs: 0,
      loop: 1, playlist: st.ytId, enablejsapi: 1, origin: location.origin, start: Math.max(0, Math.floor(st.start || 0)) });
    iframe.src = `${ORIGIN}/embed/${st.ytId}?${q}`;
    P.loaded = st.ytId; P.frameLoaded = false; P.playing = false; P.ready = false; P.paused = false; P.listenT = 0;
  }

  const v = new THREE.Vector3();
  function update(dt, cx, world) {
    if (!world) return;
    const SPAN = world.SPAN;
    // the nearest station with a YouTube video (the walk loops, so look one lap either way)
    let best = -1, bd = Infinity, bshift = 0;
    world.stations.forEach((s, i) => {
      const st = WALK.stations[i];
      if (!st.ytId || st.clip || P.failed.has(st.ytId)) return;
      for (const k of [-1, 0, 1]) { const d = Math.abs(cx + 1.5 - (s.x + k * SPAN)); if (d < bd) { bd = d; best = i; bshift = k * SPAN; } }
    });
    const want = best >= 0 && bd < S.videos.playWithin && P.why.size === 0 && !document.hidden ? best : -1;

    if (want >= 0) {
      const st = WALK.stations[want];
      if (P.loaded !== st.ytId) { if (P.f === 0) { load(st); P.idx = want; P.shift = bshift; } }   // swap only once the last one has faded out
      else { if (P.paused) { command('playVideo'); P.paused = false; } P.idx = want; P.shift = bshift; }
    } else if (P.loaded && !P.paused && P.f === 0) { command('pauseVideo'); P.paused = true; }

    if (P.loaded && !P.ready && (P.listenT += dt) > 0.5) { P.listenT = 0; listen(); }   // keep knocking until the player answers

    const show = want >= 0 && P.idx === want && P.loaded === WALK.stations[want].ytId && P.playing;
    P.f += ((show ? 1 : 0) - P.f) * (1 - Math.exp(-(show ? 2.5 : 7) * dt));
    if (!show && P.f < 0.01) P.f = 0;

    const s = world.stations[P.idx];
    if (s) {
      s.screen.updateWorldMatrix(true, false); s.screen.getWorldPosition(v); v.x += P.shift;
      hole.geometry = s.screen.geometry; hole.position.copy(v);
      const w = s.screen.geometry.parameters.width;
      obj.position.copy(v); obj.scale.setScalar(w * 1.01 / PX);
    }
    holeMat.opacity = 1 - P.f;
    hole.visible = P.f > 0;
    obj.visible = !!s && !!P.loaded && (P.f > 0 || !P.paused);       // kept "on" (behind the solid picture) while it loads
  }

  function render() {
    const ac = renderer.autoClear; renderer.autoClear = false;
    renderer.render(holeScene, camera);
    renderer.autoClear = ac;
    css.render(cssScene, camera);
  }

  function resize() {
    css.setSize(innerWidth, innerHeight);
    suspend('fixed size', S.record.fixedSize);                      // the fixed-size recording view is scaled, the layer behind can't follow it
  }
  function suspend(reason, on) { on ? P.why.add(reason) : P.why.delete(reason); if (on) P.f = 0, hole.visible = false; }

  return {
    update, render, resize, suspend,
    // for a screenshot: make the picture solid without cutting the window
    solidify() { hole.visible = false; render(); },
    state: () => ({ enabled: true, loaded: P.loaded, playing: P.playing, ready: P.ready, fade: +P.f.toFixed(2), station: P.idx, failed: [...P.failed], errors: P.errors, suspended: [...P.why] }),
  };
}

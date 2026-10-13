/* ON-SCREEN PARTS: the info card, the video viewer, the "where am I" label,
   the settings panel, and recording.

   Keys:  ← → (or A D) walk    Space next video    T tour    P glide
          H hide everything    G settings panel    R record video    K screenshot
          N new plant arrangement    Esc close a card or the video viewer */

import GUI from '../lib/lil-gui.esm.min.js';

const YALE_SCAN = id => `https://collections.library.yale.edu/iiif/2/${id}/full/2000,/0/default.jpg`;
const BEINECKE = 'https://beinecke.library.yale.edu/beinecke/collections/beinecke-cipher-voynich-manuscript';

export function makeUI({ S, WALK, renderer, canvas, camera, scene, textures, resolve, resolveMedia, videos, rebuild, resize, getWorld, walk, startTour, framePlayer }) {
  const $ = s => document.querySelector(s);
  const info = $('#info'), box = $('#lightbox'), where = $('#where'), bar = $('#progress i');
  let clean = false, recorder = null, chunks = [], flashT = 0;

  // ---- the info card (click a plant, a figure or a video) ----------------------------------------
  function showInfo(it) {
    const body = info.querySelector('.body');
    body.replaceChildren();
    const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text) e.textContent = text; body.append(e); return e; };
    const link = (parent, href, text) => { const a = document.createElement('a'); a.href = href; a.target = '_blank'; a.rel = 'noopener'; a.textContent = text + ' ↗'; parent.append(a); return a; };
    if (it.kind === 'station') {
      const st = WALK.stations[it.index];
      if (st.thumb) { const img = el('img', 'thumb wide'); img.alt = ''; img.src = resolve(st.thumb); }
      el('p', 'eyebrow', [`Stop ${it.index + 1} of ${WALK.stations.length}`, st.ytId && (YOUTUBE_NAME || 'YouTube'), dateText(st.published)].filter(Boolean).join(' · '));
      el('h2', '', st.title);
      if (st.note) el('p', 'note', st.note);
      const acts = el('div', 'actions');
      const play = document.createElement('button'); play.type = 'button'; play.textContent = '▶ Play with sound'; play.onclick = () => openVideo(it.index); acts.append(play);
      if (st.ytUrl) link(acts, st.ytUrl, 'Watch on YouTube');
    } else {
      const img = el('img', 'thumb'); img.alt = `Cut-out from folio ${it.folio}`; img.src = resolve(it.file);
      el('p', 'eyebrow', it.kind === 'plant' ? 'Plant' : (it.what || 'Figure'));
      el('h2', '', it.folio ? `Folio ${it.folio}` : 'Beinecke MS 408');
      el('p', 'meta', [it.section, 'Beinecke MS 408'].filter(Boolean).join(' · '));
      const note = WALK.notes && WALK.notes[it.folio];
      if (note) el('p', 'note', note);
      const acts = el('div', 'actions');
      if (it.iiif) link(acts, YALE_SCAN(it.iiif), 'Full page scan (Yale)');
      link(acts, BEINECKE, 'About the manuscript');
    }
    info.hidden = false; info.querySelector('.close').focus({ preventScroll: true });
  }
  info.querySelector('.close').onclick = () => { info.hidden = true; };

  // ---- the video viewer ---------------------------------------------------------------------------
  // YouTube plays here with sound. The preview copy can't show YouTube inside itself, and nor can a page
  // opened straight from the folder, so those show the picture with a button that opens YouTube instead.
  function openVideo(i) {
    const st = WALK.stations[i], stage = box.querySelector('.stage'), foot = box.querySelector('.foot');
    stage.replaceChildren(); foot.replaceChildren();
    const canEmbed = st.ytId && !window.PREVIEW && /^https?:$/.test(location.protocol);
    if (canEmbed) {
      const f = document.createElement('iframe');
      f.src = `https://www.youtube-nocookie.com/embed/${st.ytId}?autoplay=1&rel=0&playsinline=1`;
      f.allow = 'autoplay; encrypted-media; fullscreen; picture-in-picture'; f.allowFullscreen = true; f.referrerPolicy = 'strict-origin-when-cross-origin'; f.title = st.title;
      stage.append(f);
    } else if (st.clip) {
      const v = document.createElement('video'); const webm = st.clip.replace(/\.mp4$/i, '.webm');
      v.src = resolveMedia(!v.canPlayType('video/mp4; codecs="avc1.42E01E"') && webm !== st.clip && !(window.PACKED_ASSETS || {})[st.clip] ? webm : st.clip); v.controls = true; v.autoplay = true; v.playsInline = true;
      stage.append(v);
    } else if (st.ytId) {
      const a = document.createElement('a'); a.className = 'yt-out'; a.href = st.ytUrl; a.target = '_blank'; a.rel = 'noopener';
      if (st.thumb) { const img = document.createElement('img'); img.src = resolve(st.thumb); img.alt = ''; a.append(img); }
      const b = document.createElement('span'); b.textContent = '▶ Watch on YouTube ↗'; a.append(b);
      stage.append(a);
    }
    if (st.ytUrl && (canEmbed || st.clip)) {
      const a = document.createElement('a'); a.href = st.ytUrl; a.target = '_blank'; a.rel = 'noopener'; a.textContent = 'Open on YouTube ↗'; foot.append(a);
    }
    box.querySelector('h2').textContent = st.title;
    box.hidden = false; info.hidden = true;
    framePlayer.suspend('viewer', true);                                     // the silent one in the frame pauses meanwhile
  }
  function closeVideo() { box.querySelector('.stage').replaceChildren(); box.hidden = true; framePlayer.suspend('viewer', false); }
  const YOUTUBE_NAME = WALK.byline;
  const dateText = d => { if (!d) return ''; const x = new Date(d + 'T12:00:00Z'); return isNaN(x) ? '' : x.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }); };
  box.querySelector('.close').onclick = closeVideo;
  box.addEventListener('click', e => { if (e.target === box) closeVideo(); });

  // ---- the settings panel (G) -------------------------------------------------------------------------
  const gui = new GUI({ title: 'Voynich Walk settings' }); gui.hide();
  const act = {
    rebuild, newSeed() { S.seed = Math.floor(Math.random() * 100000); rebuild(); gui.controllersRecursive().forEach(c => c.updateDisplay()); },
    tour: () => { walk.glide = false; walk.tour ? (walk.tour = null) : startTour(); }, glide: () => { walk.tour = null; walk.glide = !walk.glide; },
    record: toggleRecord, screenshot,
    copy() {
      const text = 'export const SETTINGS = ' + JSON.stringify(S, null, 2) + ';\n';
      Promise.resolve(navigator.clipboard?.writeText(text)).then(() => flash('Settings copied: paste them over settings.js to keep them'))
        .catch(() => { console.log(text); flash('Could not copy here; the settings are printed in the browser console (F12)'); });
    },
  };
  const fW = gui.addFolder('Plants (press Rebuild after changing)');
  fW.add(S, 'seed', 1, 100000, 1); fW.add(S.plants, 'density', 0.2, 3, 0.05); fW.add(S.plants, 'size', 0.4, 2.5, 0.05); fW.add(S.plants, 'variety', 1, 120, 1);
  fW.add(act, 'rebuild').name('⟳ Rebuild'); fW.add(act, 'newSeed').name('⚄ New arrangement (N)');
  const fL = gui.addFolder('Live');
  fL.add(S.plants, 'sway', 0, 4, 0.05).name('wind strength'); fL.add(S.plants, 'swaySpeed', 0, 4, 0.05).name('wind speed');
  fL.add(S.walk, 'scrollSpeed', 0.2, 4, 0.05).name('scroll speed'); fL.add(S.walk, 'smoothing', 0.5, 10, 0.1).name('camera catch-up');
  fL.add(S.camera, 'height', 0.8, 8, 0.05).name('camera height'); fL.add(S.camera, 'distance', 6, 30, 0.5).name('camera distance');
  fL.add(S.camera, 'fov', 20, 70, 1).name('field of view').onChange(resize); fL.add(S.camera, 'mouseParallax', 0, 2, 0.05).name('mouse parallax');
  fL.add(S.view, 'fogNear', 0, 80, 1).name('fog starts'); fL.add(S.view, 'fogFar', 20, 300, 1).name('fog ends');
  fL.add(S.videos, 'playWithin', 2, 40, 0.5).name('videos play within (m)');
  fL.add(S.glide, 'speed', 0.3, 10, 0.1).name('glide speed (P)'); fL.add(S.glide, 'tourHold', 1, 60, 0.5).name('tour stop (s)'); fL.add(S.glide, 'tourSpeed', 0.5, 15, 0.5).name('tour speed');
  const fR = gui.addFolder('Recording');
  fR.add(S.record, 'fixedSize').name('fixed size').onChange(resize); fR.add(S.record, 'width', 640, 3840, 2).onChange(resize); fR.add(S.record, 'height', 360, 2160, 2).onChange(resize);
  fR.add(act, 'tour').name('▶ Tour (T)'); fR.add(act, 'glide').name('→ Glide (P)'); fR.add(act, 'record').name('● Record (R)'); fR.add(act, 'screenshot').name('📷 Screenshot (K)');
  gui.add(act, 'copy').name('Copy settings');
  [fW, fR].forEach(f => f.close());

  // ---- keys ---------------------------------------------------------------------------------------------
  addEventListener('keydown', e => {
    if (e.code === 'Escape') { if (!box.hidden) closeVideo(); else info.hidden = true; return; }
    if (modalOpen() || (e.target.closest && (e.target.closest('.lil-gui') || e.target.tagName === 'INPUT'))) return;
    if (e.code === 'KeyH') { clean = !clean; document.body.classList.toggle('clean', clean); if (clean) { gui.hide(); info.hidden = true; } }
    if (e.code === 'KeyG') gui._hidden ? gui.show() : gui.hide();
    if (e.code === 'KeyR') toggleRecord();
    if (e.code === 'KeyK') screenshot();
    if (e.code === 'KeyN') act.newSeed();
  });
  const modalOpen = () => !box.hidden;

  // ---- recording and screenshots ---------------------------------------------------------------------
  function toggleRecord() {
    if (window.PREVIEW) { flash('Recording works in the downloaded version'); return; }
    if (recorder) { recorder.stop(); return; }
    const type = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find(t => MediaRecorder.isTypeSupported(t));
    recorder = new MediaRecorder(canvas.captureStream(S.record.fps), { mimeType: type, videoBitsPerSecond: S.record.bitrateMbps * 1e6 });
    chunks = []; recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
    recorder.onstop = () => { save(new Blob(chunks, { type }), `voynich-walk-${stamp()}.${type.includes('mp4') ? 'mp4' : 'webm'}`); recorder = null; $('#rec').hidden = true; framePlayer.suspend('recording', false); };
    framePlayer.suspend('recording', true);                                  // a recording can't capture YouTube; the frames show their pictures instead
    recorder.start(1000); $('#rec').hidden = false;
  }
  function screenshot() {
    if (window.PREVIEW) { flash('Screenshots work in the downloaded version'); return; }
    renderer.render(scene, camera); framePlayer.solidify(); canvas.toBlob(b => save(b, `voynich-walk-${stamp()}.png`));
  }
  function save(blob, name) { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000); }
  const stamp = () => new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  function flash(msg) { where.textContent = msg; flashT = 3; }

  // ---- every frame: where am I? -------------------------------------------------------------------------
  let lastLabel = '';
  function update(cx, w) {
    const world = getWorld(); if (!world) return;
    bar.style.width = (cx / world.SPAN * 100).toFixed(2) + '%';
    if (flashT > 0) { flashT -= 1 / 60; return; }
    let label = WALK.title;
    world.stationX.forEach((x, i) => { if (Math.abs(cx + 1.5 - x) < S.walk.stationSpacing / 2) label = `${i + 1} / ${WALK.stations.length} · ${WALK.stations[i].title}`; });
    if (w.tour) label = '▶ tour · ' + label; else if (w.glide) label = '→ gliding · ' + label;
    if (label !== lastLabel) { where.textContent = label; lastLabel = label; }
  }

  return { showInfo, openVideo, update, modalOpen };
}

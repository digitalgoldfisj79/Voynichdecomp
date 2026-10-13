// ============================================================================
// main.js  -  starts everything, runs the frame loop, handles keys + recording.
// ============================================================================

(function () {
  const S = SETTINGS;
  const qs = new URLSearchParams(location.search);
  if (qs.has('seed')) S.seed = parseInt(qs.get('seed'), 10) || S.seed;     // index.html?seed=777

  // ---- Renderer, scene, camera ------------------------------------------------
  const canvas = document.getElementById('view');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.localClippingEnabled = true;             // lets the water hide the lower body of nymphs
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xe8d9a8, S.fog.near, S.fog.far);
  const camera = new THREE.PerspectiveCamera(S.camera.fov, 1, 0.1, 3000);
  camera.rotation.order = 'YXZ';                    // turn left/right first, then up/down

  const sky = new Sky(scene, renderer, S);
  const kinds = makeImageKinds() || makePlaceholderKinds();   // the Voynich cut-outs (js/images.js), or the placeholders if they are missing
  const player = new Player(camera, S);
  let terrain = null, plants = null, groundFog = null, nymphs = null;

  // ---- Building the world ---------------------------------------------------------
  function buildPlants() {
    if (plants) plants.dispose();
    plants = new Plants(scene, terrain, S, S.seed, kinds);
  }
  function buildNymphs() {
    if (nymphs) nymphs.dispose();
    nymphs = new Nymphs(scene, terrain, S, S.seed);
  }
  function buildWorld() {
    if (plants) { plants.dispose(); plants = null; }
    if (groundFog) groundFog.dispose();
    if (terrain) terrain.dispose();
    terrain = new Terrain(S, S.seed);
    scene.add(terrain.group);
    player.setTerrain(terrain);
    groundFog = new GroundFog(scene, terrain, S);
    buildPlants();
    buildNymphs();
  }

  // ---- Picture size (normal window, or fixed recording size) ------------------------
  function applySize() {
    const R = S.recording;
    let w, h;
    if (R.enabled) { w = R.width; h = R.height; renderer.setPixelRatio(1); }
    else { w = window.innerWidth; h = window.innerHeight; renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, (typeof IS_TOUCH !== 'undefined' && IS_TOUCH) ? 1.5 : 2)); }   // phones: a little less sharp, much faster
    renderer.setSize(w, h, false);                  // the CSS scales it to fit the window, keeping proportions
    camera.aspect = w / h;
    camera.fov = S.camera.fov;
    camera.updateProjectionMatrix();
    document.body.classList.toggle('recording', !!R.enabled);
  }
  window.addEventListener('resize', applySize);

  // ---- Things the panel and keys can trigger ---------------------------------------------
  const hud = document.getElementById('hud');
  let uiHidden = false;
  const app = {
    apply(kind) {
      if (kind === 'world') buildWorld();
      else if (kind === 'plants') buildPlants();
      else if (kind === 'nymphs') buildNymphs();
      else if (kind === 'resize' || kind === 'fov') applySize();
    },
    newSeed() { S.seed = Math.floor(Math.random() * 1000000); buildWorld(); panel.refresh(); },
    toggleUI() {
      uiHidden = !uiHidden;
      document.body.classList.toggle('ui-hidden', uiHidden);
    },
    toggleAuto() { player.setMode(player.mode === 'auto' ? 'walk' : 'auto'); },
    toggleFly() { player.setMode(player.mode === 'fly' ? 'walk' : 'fly'); },
    toggleRecording() { S.recording.enabled = !S.recording.enabled; applySize(); panel.refresh(); },
    copySettings() {
      const text = 'const SETTINGS = ' + JSON.stringify(S, null, 2) + ';\n';
      const done = () => { hud.dataset.flash = 'Settings copied - paste them over the block in js/settings.js'; setTimeout(() => (hud.dataset.flash = ''), 4000); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, () => window.prompt('Copy this:', text));
      else window.prompt('Copy this:', text);
    },
  };
  const panel = new Panel(document.getElementById('panel'), S, app);
  panel.refresh();

  window.addEventListener('keydown', (e) => {
    if (e.repeat) return;
    const t = e.target;
    if (t && t.tagName === 'INPUT' && (t.type === 'number' || t.type === 'text')) return;
    switch (e.code) {
      case 'KeyH': app.toggleUI(); break;
      case 'KeyF': app.toggleFly(); break;
      case 'KeyP': app.toggleAuto(); break;
      case 'KeyR': app.toggleRecording(); break;
      case 'KeyN': app.newSeed(); break;
      case 'KeyB': S.player.bobAmount = S.player.bobAmount > 0 ? 0 : 0.045; panel.refresh(); break;
    }
  });

  // ---- Mouse look: pointer lock (best), or click-and-drag if the browser refuses ----------
  canvas.addEventListener('click', () => {
    if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
    if (canvas.requestPointerLock) canvas.requestPointerLock();
  });
  document.addEventListener('pointerlockchange', () => {
    document.body.classList.toggle('locked', document.pointerLockElement === canvas);
  });
  let dragging = false;
  canvas.addEventListener('mousedown', () => { dragging = true; });
  window.addEventListener('mouseup', () => { dragging = false; });
  document.addEventListener('mousemove', (e) => {
    if (document.pointerLockElement === canvas || dragging) player.look(e.movementX, e.movementY);
  });

  // ---- The frame loop ------------------------------------------------------------------------
  const clock = new THREE.Clock();
  let hudTimer = 0;
  function frame() {
    requestAnimationFrame(frame);
    const dt = Math.min(clock.getDelta(), 0.05);
    player.update(dt);
    sky.update(camera, dt);
    terrain.update(sky.level);
    plants.update(camera, sky.level, dt);
    groundFog.update(camera, dt, sky.horizon, sky.level);
    nymphs.update(dt, player.pos, camera, sky.level);
    hudTimer -= dt;
    if (hudTimer <= 0) {
      hudTimer = 0.25;
      const m = { walk: 'walking', fly: 'flying', auto: 'autopilot' }[player.mode];
      hud.textContent = hud.dataset.flash || ('seed ' + S.seed + '  ·  ' + m + '  ·  ' + Math.floor(sky.hour) + ':' + String(Math.floor((sky.hour % 1) * 60)).padStart(2, '0'));
    }
    renderer.render(scene, camera);
  }

  applySize();
  buildWorld();
  frame();

  // Handy for testing and for you in the browser console (F12): VW.player, VW.S ...
  window.VW = { S, scene, camera, renderer, player, sky, app, get terrain() { return terrain; }, get plants() { return plants; }, get nymphs() { return nymphs; }, get fog() { return groundFog; } };
})();

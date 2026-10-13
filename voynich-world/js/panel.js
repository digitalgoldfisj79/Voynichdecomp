// ============================================================================
// panel.js  -  the on-screen settings panel (top right).
//
// It is built from the list below, so adding a slider is one line:
//   { path: 'fog.far', label: 'Fog far', min: 30, max: 300, step: 5 }
// 'path' is the address of the value in SETTINGS.
// apply: 'plants' = rebuild plants when you let go of the slider,
//        'world'  = rebuild hills + plants,   'resize' = refit the picture,
//        (nothing) = changes immediately.
// ============================================================================

const PANEL_SCHEMA = [
  { title: 'World', open: true, items: [
    { path: 'seed', label: 'Seed', type: 'number', apply: 'world' },
    { type: 'button', label: 'New random seed  (N)', action: 'newSeed' },
    { path: 'plants.count', label: 'Plant count', min: 0, max: 6000, step: 50, apply: 'plants' },
    { path: 'plants.heightScale', label: 'Plant height', min: 0.3, max: 3, step: 0.05, apply: 'plants' },
    { path: 'plants.density', label: 'Clump tightness', min: 0.4, max: 3, step: 0.05, apply: 'plants' },
    { path: 'plants.wind.strength', label: 'Wind lean (0 = off)', min: 0, max: 0.15, step: 0.005 },
    { path: 'plants.wind.speed', label: 'Wind speed', min: 0.1, max: 3, step: 0.05 },
    { path: 'plants.mode', label: 'Plant style', type: 'select', options: ['crossed', 'billboard'], apply: 'plants' },
    { path: 'plants.spacing.bigDifferent', label: 'Gap between big plants', min: 1, max: 3, step: 0.05, apply: 'plants' },
    { path: 'plants.spacing.maxBigPerCluster', label: 'Big plants per clump', min: 1, max: 8, step: 1, apply: 'plants' },
    { path: 'world.hillHeight', label: 'Hill height', min: 4, max: 40, step: 1, apply: 'world' },
    { path: 'world.flatShading', label: 'Faceted paper ground', type: 'checkbox', apply: 'world' },
    { path: 'world.wallRamp', label: 'Hill steepness (low = steep)', min: 4, max: 30, step: 1, apply: 'world' },
  ] },
  { title: 'Walking & camera', items: [
    { path: 'player.walkSpeed', label: 'Walking speed', min: 1, max: 12, step: 0.1 },
    { path: 'player.mouseSmoothness', label: 'Mouse smoothness', min: 0, max: 1, step: 0.01 },
    { path: 'player.bobAmount', label: 'Head-bob (0 = off)', min: 0, max: 0.2, step: 0.005 },
    { path: 'player.acceleration', label: 'Acceleration', min: 2, max: 40, step: 1 },
    { path: 'player.braking', label: 'Braking', min: 2, max: 40, step: 1 },
    { path: 'camera.fov', label: 'Field of view', min: 40, max: 100, step: 1, apply: 'fov' },
    { path: 'autopilot.speed', label: 'Autopilot speed', min: 0.5, max: 12, step: 0.1 },
  ] },
  { title: 'Fog & sky', items: [
    { path: 'fog.near', label: 'Fog starts', min: 0, max: 200, step: 1 },
    { path: 'fog.far', label: 'Fog full at', min: 20, max: 400, step: 1 },
    { path: 'groundFog.enabled', label: 'Ground mist', type: 'checkbox' },
    { path: 'groundFog.density', label: 'Mist thickness', min: 0, max: 2, step: 0.05 },
    { path: 'groundFog.height', label: 'Mist height', min: 0.4, max: 3, step: 0.05 },
    { path: 'groundFog.swirl', label: 'Mist swirl', min: 0, max: 1.5, step: 0.05 },
    { path: 'groundFog.speed', label: 'Mist speed', min: 0, max: 3, step: 0.05 },
    { path: 'sky.timeOfDay', label: 'Time of day (hour)', min: 0, max: 24, step: 0.1 },
    { path: 'sky.timeSpeed', label: 'Time passing (hours/min)', min: 0, max: 12, step: 0.1 },
    { path: 'sky.palette.3.top', label: 'Day: sky top', type: 'color' },
    { path: 'sky.palette.3.horizon', label: 'Day: horizon + fog', type: 'color' },
    { path: 'sky.palette.0.top', label: 'Night: sky top', type: 'color' },
    { path: 'sky.palette.0.horizon', label: 'Night: horizon + fog', type: 'color' },
    { path: 'world.colors.path', label: 'Path colour', type: 'color', apply: 'world' },
    { path: 'world.colors.mid', label: 'Ground colour', type: 'color', apply: 'world' },
    { path: 'world.colors.high', label: 'Hill colour', type: 'color', apply: 'world' },
  ] },
  { title: 'Nymphs', items: [
    { path: 'nymphs.enabled', label: 'Nymphs', type: 'checkbox', apply: 'nymphs' },
    { path: 'nymphs.perPond', label: 'Places per pond', min: 0, max: 10, step: 1, apply: 'nymphs' },
    { path: 'nymphs.scareDistance', label: 'Notice you at', min: 6, max: 50, step: 1 },
    { path: 'nymphs.diveChance', label: 'Dive (vs run away)', min: 0, max: 1, step: 0.05 },
    { path: 'nymphs.returnChance', label: 'Chance to return', min: 0, max: 1, step: 0.05 },
  ] },
  { title: 'Recording', items: [
    { path: 'recording.enabled', label: 'Fixed recording size  (R)', type: 'checkbox', apply: 'resize' },
    { path: 'recording.width', label: 'Width', type: 'number', apply: 'resize' },
    { path: 'recording.height', label: 'Height', type: 'number', apply: 'resize' },
    { type: 'button', label: 'Hide all UI  (H)', action: 'toggleUI' },
    { type: 'button', label: 'Autopilot on/off  (P)', action: 'toggleAuto' },
    { type: 'button', label: 'Fly mode on/off  (F)', action: 'toggleFly' },
  ] },
  { title: 'Keep my settings', items: [
    { type: 'button', label: 'Copy settings as text', action: 'copySettings' },
  ] },
];

class Panel {
  constructor(root, S, app) {
    this.root = root; this.S = S; this.app = app; this.fields = [];
    const head = document.createElement('div');
    head.className = 'panel-head';
    head.innerHTML = '<span>Voynich world &middot; v2</span><span class="chev">&#9662;</span>';
    const body = document.createElement('div'); body.className = 'panel-body';
    head.addEventListener('click', () => root.classList.toggle('collapsed'));
    root.append(head, body);

    for (const sec of PANEL_SCHEMA) {
      const det = document.createElement('details'); det.open = !!sec.open;
      const sum = document.createElement('summary'); sum.textContent = sec.title; det.append(sum);
      for (const item of sec.items) det.append(this._row(item));
      body.append(det);
    }
    const tip = document.createElement('p'); tip.className = 'tip';
    tip.textContent = 'Press Esc to free the mouse, then use the panel. Click the picture to go back to walking.';
    body.append(tip);
  }

  _row(item) {
    const row = document.createElement('label'); row.className = 'row';
    if (item.type === 'button') {
      const b = document.createElement('button'); b.textContent = item.label;
      b.addEventListener('click', () => { this.app[item.action](); b.blur(); });
      row.className = 'row btn'; row.append(b); return row;
    }
    const name = document.createElement('span'); name.className = 'name'; name.textContent = item.label;
    const input = document.createElement(item.type === 'select' ? 'select' : 'input');
    const out = document.createElement('span'); out.className = 'val';
    if (item.type === 'select') {
      item.options.forEach((o) => { const op = document.createElement('option'); op.value = o; op.textContent = o; input.append(op); });
    } else if (item.type === 'color') input.type = 'color';
    else if (item.type === 'checkbox') input.type = 'checkbox';
    else if (item.type === 'number') input.type = 'number';
    else { input.type = 'range'; input.min = item.min; input.max = item.max; input.step = item.step; }

    const read = () => {
      if (item.type === 'checkbox') return input.checked;
      if (item.type === 'select' || item.type === 'color') return input.value;
      return parseFloat(input.value);
    };
    input.addEventListener('input', () => {
      const v = read();
      if (typeof v === 'number' && isNaN(v)) return;
      setPath(this.S, item.path, v);
      out.textContent = this._fmt(v);
      if (!item.apply || item.apply === 'fov') this.app.apply(item.apply);       // live changes
    });
    input.addEventListener('change', () => {                                       // on release: heavy rebuilds
      if (item.apply && item.apply !== 'fov') this.app.apply(item.apply);
      if (item.type !== 'number') input.blur();
    });
    this.fields.push({ item, input, out });
    row.append(name, input, out);
    return row;
  }

  _fmt(v) { return typeof v === 'number' ? (Math.abs(v) >= 100 ? String(Math.round(v)) : String(Math.round(v * 1000) / 1000)) : ''; }

  // Copy the current values from SETTINGS into the controls (after a new seed etc.).
  refresh() {
    for (const f of this.fields) {
      const v = getPath(this.S, f.item.path);
      if (f.item.type === 'checkbox') f.input.checked = !!v; else f.input.value = v;
      f.out.textContent = this._fmt(v);
    }
  }
}

// ============================================================================
// images.js  -  the real Voynich cut-outs (Stage 2).
//
// The pictures live inside images/plants.js, images/nymphs.js and
// images/animals.js (made by tools/pack_images.py from the PNG folders), so the
// world still runs by double-clicking index.html: Chrome blocks loose PNG files
// on pages opened from disk, but allows pictures stored inside a .js file.
//
// If those files are missing, everything falls back to the placeholders.
//
// Plant names are the file names without .png, e.g. 'f93r_plant'.
// In settings.js -> plants.sets you can also use these group names:
//   'tall'  narrow, upright plants (width < half the height)
//   'mid'   in between
//   'wide'  broad, bushy plants (width > 0.8 x height)
//   'all'   every plant
// ============================================================================

function textureFromPacked(p) {
  const img = new Image();
  const texture = new THREE.Texture(img);
  texture.anisotropy = 4;
  img.onload = () => { texture.needsUpdate = true; };     // drawn as soon as the picture is decoded (a moment after start)
  img.src = p.src;
  return texture;
}

// Same shape of list as makePlaceholderKinds(): [{id, aspect, texture}]
function makeImageKinds() {
  const P = window.VOYNICH_PLANTS;
  if (!P || !Object.keys(P).length) return null;
  return Object.keys(P).map((id) => ({ id, aspect: P[id].w / P[id].h, texture: textureFromPacked(P[id]) }));
}

// Nymph pictures: [{id, aspect, texture}]. 'names' = which ones to use (settings.nymphs.pictures); empty = all.
function makeNymphKinds(names) {
  const N = window.VOYNICH_NYMPHS;
  if (!N || !Object.keys(N).length) return null;
  let ids = (names && names.length ? names : Object.keys(N)).filter((id) => N[id]);
  if (!ids.length) ids = Object.keys(N);
  return ids.map((id) => ({ id, aspect: N[id].w / N[id].h, texture: textureFromPacked(N[id]) }));
}

// Turns a set like ['tall', 'f93r_plant'] into a list of plant names.
function expandPlantSet(list) {
  const P = window.VOYNICH_PLANTS || {}, out = [];
  const group = (g) => Object.keys(P).filter((id) => {
    const a = P[id].w / P[id].h;
    return g === 'all' || (g === 'tall' && a < 0.5) || (g === 'mid' && a >= 0.5 && a < 0.8) || (g === 'wide' && a >= 0.8);
  });
  for (const name of list || []) {
    if (['all', 'tall', 'mid', 'wide'].includes(name)) out.push(...group(name));
    else out.push(name);
  }
  return [...new Set(out)];
}

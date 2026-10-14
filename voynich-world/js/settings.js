// ============================================================================
// settings.js  -  THE FILE YOU EDIT.
// Everything you will want to tweak is here, in plain numbers.
// The on-screen panel changes these same values live; its "Copy settings"
// button gives you text you can paste back over this block to keep a look.
//
// Units: 1 unit is roughly 1 metre. The camera "eyes" are 1.7 units high.
// ============================================================================

const SETTINGS = {

  // The seed decides the hills and where plants grow. Same seed = same world.
  // You can also open index.html?seed=777 to try a seed without editing.
  seed: 1234,

  camera: {
    fov: 70,                 // field of view in degrees. Lower = more zoomed in.
  },

  // ---- Terrain ------------------------------------------------------------
  world: {
    size: 360,               // the world is a square this many units wide
    gridCells: 220,          // terrain detail. More = smoother but slower to build
    pathHalfWidth: 3.2,      // half the width of the walking paths
    hillBase: 8,             // minimum height of the "walls" beside the paths
    hillHeight: 16,          // how much taller the tallest hills are than that
    hillScale: 0.018,        // size of hills. Smaller number = broader hills
    wallRamp: 11,            // distance over which the ground rises from path to hill
                             //   (smaller = steeper, harder walls)
    groundRoll: 0.7,         // gentle bumps on the path floor itself
    edgeWall: 28,            // extra tall hills around the world edge
    pondDepth: 1.7,          // how deep the pond basins are dug
    waterLevel: -0.9,        // height of the water surface
    fordDepth: 0.3,          // where a path crosses a pond the ground is only this far below the water (wading)
    faceJitter: 0.0,         // random brightness per facet (only used with flatShading)
    flatShading: false,      // false = smooth ground, true = faceted paper look

    // Ground colours (parchment, olive wash, darker hill wash)
    colors: {
      path:    '#e8d8aa',    // worn parchment on the walking path
      mid:     '#b0b57c',    // olive wash beside the path
      high:    '#77987a',    // darker wash on the hills
      pondBed: '#9a9a78',
      water:   '#7aa6a6',
    },
  },

  // ---- Layout: paths and clearings ------------------------------------------
  // Points are [x, z] on the ground. x is left/right, z is toward/away.
  // Paths are smooth curves through the points. A branch must START at a
  // point that lies on another path (copy the numbers) - dead ends are just
  // branches that stop.
  layout: {
    paths: [
      { name: 'main', points: [
        [-120, 135], [-95, 112], [-98, 82], [-78, 56], [-50, 48], [-22, 58],
        [6, 48], [28, 22], [48, 4], [62, -18], [58, -48], [30, -68], [0, -62],
        [-30, -72], [-58, -96], [-52, -126], [-24, -146], [6, -150] ] },
      { name: 'branch-north',  points: [[-22, 58], [-30, 90], [-12, 118], [14, 126], [38, 112]] },
      { name: 'branch-east',   points: [[28, 22], [60, 40], [92, 58], [118, 40]] },
      { name: 'branch-south',  points: [[-30, -72], [-8, -92], [20, -108], [44, -100]] },
      { name: 'branch-far',    points: [[58, -48], [90, -62], [116, -90], [132, -120]] },
    ],
    // Open areas. type decides which plants grow there (see plants.sets below).
    clearings: [
      { name: 'plant grove',    type: 'grove',  x: -74, z: 52,   radius: 20 },
      { name: 'pond',           type: 'pond',   x: 52,  z: -5,   radius: 20 },
      { name: 'animal meadow',  type: 'meadow', x: -55, z: -110, radius: 26 },
      { name: 'landmark glade', type: 'glade',  x: 6,   z: -150, radius: 16 },
      { name: 'north glade',    type: 'glade',  x: 40,  z: 112,  radius: 12 },
      { name: 'east glade',     type: 'glade',  x: 120, z: 38,   radius: 10 },
      { name: 'south glade',    type: 'glade',  x: 46,  z: -100, radius: 9 },
      { name: 'far glade',      type: 'glade',  x: 134, z: -122, radius: 10 },
    ],
    // The giant plant you can see from far away.
    landmark: { x: 10, z: -158 },
  },

  // ---- Plants -----------------------------------------------------------------
  plants: {
    count: 2500,             // total number of plants (the budget is shared out among clusters)
    heightScale: 1.0,        // multiplies every plant height
    density: 1.0,            // clumpiness: higher = tighter clumps, lower = looser spread
    mode: 'crossed',         // 'crossed' = two planes in an X (good from all sides)
                             // 'billboard' = one plane that always turns to face you
    smoothEdges: true,       // softer cut-out edges (alpha-to-coverage)

    // Wind: tops of plants lean and sway, bases stay put.
    wind: {
      strength: 0.05,        // lean as a fraction of plant height. 0 = no wind
      speed: 1.0,            // how fast the wind moves
    },

    // Keeping big plants from clipping into each other.
    // A plant's "footprint" is a circle as wide as the plant. Values below say how
    // far apart two footprints must be (1.0 = just touching, 1.7 = a clear gap).
    spacing: {
      footprint: 0.8,        // footprint size relative to the picture width
      bigHeight: 4,          // plants at least this tall count as "big"
      maxBigPerCluster: 3,   // at most this many big plants in one clump
      bigSame: 1.05,         // two big plants of the SAME kind (they line up side by side)
      bigDifferent: 1.7,     // big plants of DIFFERENT kinds (bigger gap)
      bigToSmall: 0.7,       // a small plant next to a big one
      smallOverlap: 0.3,     // small shrubs may overlap each other (0 = on top, 1 = touching)
    },

    // Height range [min, max] in units for each kind of place
    heights: {
      path:   [1.2, 3.2],    // small plants beside the path
      edge:   [7, 16],       // giants on the hills at the edges
      grove:  [2.5, 6],
      pond:   [1.5, 3.8],
      meadow: [0.8, 2],
      glade:  [1.5, 4],
    },

    // WHICH IMAGES APPEAR WHERE. Names are plant picture names (the file name in
    // plants/ without .png, e.g. 'f93r_plant'), or a whole group:
    //   'tall' (narrow, upright)  'mid'  'wide' (broad, bushy)  'all'
    // Mix freely: ['tall', 'f93r_plant']. Placeholder names still work if you
    // remove the images/ files.
    sets: {
      path:   ['mid', 'wide'],
      edge:   ['tall', 'mid'],
      grove:  ['all'],
      pond:   ['tall'],
      meadow: ['wide', 'mid'],
      glade:  ['all'],
    },

    landmarkKind: 'f93r_plant', // which image is the giant landmark (f93r: O'Neill's "sunflower")
    emptyZonesBare: true,       // a place whose list is empty [] gets no plants (false = it gets any plant)
    landmarkHeight: 70,
    landmarkIgnoresFog: true,   // true = it stays visible over the hills from far away
  },

  // ---- Ground fog: thin mist sheets that follow the terrain and slowly swirl ------------
  groundFog: {
    enabled: true,
    density: 1.0,            // overall thickness (0 = none, 2 = heavy)
    height: 1.0,             // multiplies how high the mist reaches
    swirl: 0.6,              // how much the mist curls and churns
    speed: 1.0,              // how fast it moves
    radius: 75,              // mist only exists this far around you (keeps it cheap)
    gridCells: 72,           // detail of the mist sheets
    layers: [                // three thin sheets give depth. height = metres above ground
      { height: 0.35, density: 0.55, scale: 0.045, seed: 0 },
      { height: 1.00, density: 0.42, scale: 0.035, seed: 11.3 },
      { height: 1.90, density: 0.22, scale: 0.028, seed: 27.1 },
    ],
  },

  // ---- Picture quality and speed (js/quality.js) ------------------------------------
  quality: {
    targetFps: 60,           // the frame rate to hold
    adaptiveResolution: true,// draw a little less sharp when the computer can't keep up (never in recording mode)
    minResolution: 0.6,      // ...but never below this fraction of full sharpness
    maxPixelRatio: 2,        // on very sharp screens, at most this many pixels per screen pixel
    plantShading: true,      // light from the sun on the plants, darker near the roots
    shadows: true,           // plant shadows on the ground
    shadowStrength: 0.38,    // how dark the shadows are (0-1)
    shadowDistance: 60,      // shadows fade out by this distance
    shadowMinLength: 0.7,    // shadows are at least this long (x plant height), even at noon - it looks better
    cullPlants: true,        // only draw the plants in view (faster; no visible difference)
    groundShading: true,     // hollows and the foot of the hills a little darker (needs a new world: N)
    paperGrain: 0.55,        // paper grain and fibres on the ground up close (0 = off)
    waterShading: true,      // ponds reflect the sky colour and ripple
  },

  // ---- Phones and tablets (js/touch.js) -------------------------------------------
  touch: {
    lookSpeed: 2.2,          // how fast dragging turns the view (x the mouse sensitivity)
    stickSize: 120,          // size of the on-screen joystick, in pixels
    maxPicture: 384,         // pictures are shrunk to this many pixels on phones (saves memory; 0 = keep)
  },

  // ---- Nymphs bathing in the ponds ----------------------------------------------------------
  nymphs: {
    enabled: true,
    perPond: 5,              // how many places there are for nymphs at each pond
    height: 1.9,             // size of the figures
    // which bathing-figure pictures to use (files in nymphs/ without .png); [] = all of them.
    // Some files are groups of two or three figures - left out here, add them if you like.
    pictures: ['nymph_75r_1', 'nymph_75r_3', 'nymph_75r_4', 'nymph_76v_0', 'nymph_76v_1', 'nymph_76v_2', 'nymph_76v_3', 'nymph_77r_2'],
    presence: 0.75,          // chance each place is occupied when the world starts
    scareDistance: 24,       // they notice you from about this far
    diveChance: 0.5,         // when scared: 0 = always run away, 1 = always dive under
    runSpeed: 3.2,
    diveSpeed: 1.3,
    returnDistance: 45,      // you must be this far from the pond (beyond the pond edge) for them to come back
    returnDelay: [25, 70],   // seconds before a gone nymph may come back
    returnChance: 0.6,       // each time it may come back, the chance that it does
  },

  // ---- Fog ----------------------------------------------------------------------
  fog: {
    near: 14,                // fog starts to appear this far away
    far: 95,                 // everything is fully fogged out at this distance
  },

  // ---- Sky and time of day --------------------------------------------------------
  sky: {
    timeOfDay: 11,           // hour of the day, 0-24 (12 = noon, 0 = midnight)
    timeSpeed: 0,            // hours that pass per real minute. 0 = frozen
    starCount: 700,
    // Colours at certain hours; between them the sky blends smoothly.
    //   top      = colour overhead          horizon = colour at the horizon
    //   (the horizon colour is also the FOG colour, so hills melt into the sky)
    //   ambient  = general light colour     sun     = direct light colour
    //   sunPower = strength of direct light level   = overall brightness (0-1)
    palette: [
      { hour: 0,    top: '#161c33', horizon: '#2b3352', ambient: '#7480ad', sun: '#8fa0d0', sunPower: 0.25, level: 0.50 },
      { hour: 5,    top: '#2a3050', horizon: '#6a5a6a', ambient: '#8a86a6', sun: '#b09aa0', sunPower: 0.25, level: 0.55 },
      { hour: 6.5,  top: '#7d8fae', horizon: '#eab68a', ambient: '#c9b296', sun: '#f2b27c', sunPower: 0.60, level: 0.85 },
      { hour: 9,    top: '#a6c0b8', horizon: '#efe0b4', ambient: '#f0e2bc', sun: '#fff0c8', sunPower: 0.80, level: 1.00 },
      { hour: 14,   top: '#9fbfbf', horizon: '#f0e1b2', ambient: '#f0e2bc', sun: '#fff0c8', sunPower: 0.80, level: 1.00 },
      { hour: 17.5, top: '#9d86a6', horizon: '#eca87a', ambient: '#d8b49a', sun: '#f4a470', sunPower: 0.60, level: 0.88 },
      { hour: 20,   top: '#252b4a', horizon: '#4a4560', ambient: '#8084ac', sun: '#8fa0d0', sunPower: 0.25, level: 0.58 },
      { hour: 24,   top: '#161c33', horizon: '#2b3352', ambient: '#7480ad', sun: '#8fa0d0', sunPower: 0.25, level: 0.50 },
    ],
  },

  // ---- Walking ---------------------------------------------------------------------
  player: {
    eyeHeight: 1.7,
    walkSpeed: 4.0,          // units per second
    runMultiplier: 1.9,      // hold Shift
    flySpeed: 14,            // speed in fly mode
    acceleration: 14,        // how quickly you reach full speed (bigger = snappier)
    braking: 18,             // how quickly you stop when you let go
    mouseSensitivity: 0.0022,
    mouseSmoothness: 0.45,   // 0 = raw mouse, 1 = very floaty. 0.3-0.6 feels nice
    bobAmount: 0.045,        // walking head-bob in units. 0 = off
    stepLength: 1.7,         // distance of one bob cycle
    wadeSpeed: 0.65,         // speed multiplier when walking in water
    maxSlope: 0.8,           // steepest slope you can climb (1 = 45 degrees). Makes hills walls
  },

  // ---- Autopilot camera --------------------------------------------------------------
  autopilot: {
    path: 'main',            // which path to glide along (a name from layout.paths)
    speed: 3.2,              // units per second
    lookAhead: 7,            // how far ahead along the path the camera looks
    turnRate: 2.2,           // how quickly it turns into bends (smaller = lazier)
    drift: 0.06,             // gentle side-to-side looking around
  },

  // ---- Recording -----------------------------------------------------------------------
  recording: {
    enabled: false,          // true = render at the fixed size below (press R)
    width: 1920,
    height: 1080,
  },
};

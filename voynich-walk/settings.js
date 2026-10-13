/* ==========================================================================
   VOYNICH WALK (SCROLL) — SETTINGS
   Every number worth changing. Save, then reload the page.
   Most of these can also be changed live in the panel (press G).
   Distances are in metres; the walk runs from left to right along x.
   ========================================================================== */

export const SETTINGS = {
  seed: 1408,              // decides where the plants stand. Same seed = same walk. Also ?seed=123 in the address.

  walk: {
    stationSpacing: 30,    // metres between videos
    lead: 18,              // empty stretch before the first video (the title card stands here)
    scrollSpeed: 1.0,      // how far one turn of the mouse wheel takes you
    smoothing: 3.2,        // how gently the camera catches up (lower = floatier)
    loop: true,            // after the last station, carry on round to the start
  },

  camera: {
    height: 2.6,           // eye height
    distance: 13,          // how far in front of the path the camera stands
    fov: 38,               // field of view in degrees
    mouseParallax: 0.35,   // how much the view shifts with the mouse (0 = none)
  },

  plants: {
    density: 1.0,          // more or fewer plants overall
    size: 1.0,             // multiplies every plant's height
    variety: 50,           // how many different plant pictures to load
    sway: 1.0,             // wind strength
    swaySpeed: 1.0,        // wind speed
  },

  videos: {
    playWithin: 11,        // a video starts playing when you are this close (metres)
    width: 5.6,            // width of the video frames
    youtubeInFrame: true,  // play the YouTube video itself (silently) in the frame you stand at; false = just its picture
  },

  glide: {
    speed: 2.2,            // P: steady glide speed (metres per second)
    tourHold: 9,           // T: seconds the tour stops at each video
    tourSpeed: 4,          // T: travel speed between stops
  },

  view: {
    fogNear: 22,
    fogFar: 85,
    sky: '#9cb6c6',        // top of the sky
    horizon: '#ece2ca',    // bottom of the sky and the fog colour
    paper: '#efe5cf',      // parchment of the ground
  },

  events: {
    creatureEvery: 25,     // roughly every N seconds a creature crosses the path
    fishJumpEvery: 9,
  },

  record: {
    fixedSize: false, width: 1920, height: 1080, fps: 60, bitrateMbps: 24,
  },

  showHelp: true,
};

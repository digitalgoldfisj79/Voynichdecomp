VOYNICH WORLD - v2 (placeholders only)
======================================

  ** This copy has the Voynich cut-outs in place of the placeholders - see CHANGES-from-Ed.txt **
  ** To choose which plants grow where, open picker.html **

RUN IT (no server needed - it is just files)
  Mac:      open voynich-world-v2/index.html
  Windows:  start voynich-world-v2\index.html
  Linux:    xdg-open voynich-world-v2/index.html
  (or double-click index.html). Chrome, Edge or Firefox. Click the picture to look around; Esc frees the mouse.

KEYS
  W A S D / arrows   walk            Shift  run
  F  fly (Space up, C down)          P      autopilot along the main path
  H  hide ALL interface              R      fixed recording size
  N  new random seed                 B      head-bob on/off
  Any walking key cancels autopilot.   index.html?seed=777 opens a specific seed.

WHAT'S NEW IN v2 (all numbers live in js/settings.js and in the panel)
  * Smooth terrain (world.flatShading: true brings the paper facets back)
  * Big plants never clip into each other: plants.spacing
      bigDifferent = gap between big plants of different kinds, maxBigPerCluster = size of a clump of big plants
  * Wind lean: plants.wind.strength (0 = off) and plants.wind.speed
  * Ground mist: groundFog.* (thickness, height, swirl, speed)
  * You can wade through ponds (slower: player.wadeSpeed); the path crosses a shallow ford
  * Placeholder nymphs bathe in the pond: nymphs.*  They notice you, dive or run away, and may return later when you are far away.

FILES
  js/settings.js   <-- YOUR FILE: every number you may want to change
  js/terrain.js    hills, paths, clearings, water       js/plants.js     where plants grow + wind
  js/groundfog.js  the swirling mist                    js/nymphs.js     bathing nymphs (placeholders)
  js/sky.js        sky, sun, moon, stars, fog colour    js/player.js     walking, flying, autopilot
  js/placeholders.js  stand-in plants                   js/panel.js      the on-screen panel
  js/main.js       starts everything                    js/utils.js      seeded random + noise
  vendor/three.min.js  Three.js r149

KEEPING A LOOK YOU LIKE: tweak in the panel, press "Copy settings as text", paste over the SETTINGS block in js/settings.js.

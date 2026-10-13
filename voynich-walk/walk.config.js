/* ==========================================================================
   WHAT IS ON THE WALK
   --------------------------------------------------------------------------
   Each "station" is a stop along the walk: a framed video standing among the
   plants, with a handwritten title. Add, remove or reorder stations freely.

     youtube  the video's id or link, e.g. 'Jf5lsVx-qkA' or
              'https://www.youtube.com/watch?v=Jf5lsVx-qkA'.
              The frame shows the video's picture; when you stand in front of it
              the video itself starts playing in the frame, silently. Clicking it
              opens it big, with sound.
     title    optional: the handwritten title. Left out = the YouTube title.
     note     optional: a sentence shown on the video's card and by the guide.
              Left out = the first sentence of the YouTube description.
     start    optional: where the silent in-frame playing starts, in seconds.
     clip     optional: a short video file in videos/ to play in the frame instead
              of YouTube (MP4, H.264). Clicking still opens YouTube if set.
     plants   optional: folios whose plants stand next to this station, e.g. ['1v', '2r']
     pond     true for the stop by the pond (one only)

   Titles, descriptions and pictures come from youtube.js. To bring in new
   videos, run:   python tools/fetch_youtube.py
   It lists every video with its id, ready to copy in here.
   ========================================================================== */

export const WALK = {
  title: 'Voynich Walk',
  subtitle: 'a stroll through Beinecke MS 408',
  byline: 'Voynich Talk',
  channel: 'UCvUpIY3QGcCWdKbcm1do5zw',     // the YouTube channel (@voynichtalk)

  stations: [
    { youtube: 'Jf5lsVx-qkA', plants: ['93r', '33v'] },            // O'Neill's "sunflower" is f93r
    { youtube: '7tDr9zbtgKY', plants: ['1v', '2r'] },
    { youtube: 'JnGBl5usAMM', plants: ['3r', '9r'] },
    { youtube: '26btWwKw508', plants: ['25v', '44r'] },            // f25v has a creature at its foot
    { youtube: '9ZDzpM9ARE8', pond: true },
    { youtube: 'cihoF5BJ0KE', plants: ['56r', '42r'] },
    { youtube: 'SnRrE9PLU6Y', plants: ['21r', '24v'] },
    { youtube: 'V_2sOPwAQtw', plants: ['46v', '47r'] },
    { youtube: 'enJg0mp8qqo', plants: ['17v', '35v'] },
  ],

  closing: ['Thank you for walking.', 'The plants are cut from Beinecke MS 408, Yale.'],

  // The figure who goes along with you. Swap 'file' for another picture to change it:
  //   creatures/guide_barrel_70v2_21.png  bather in a barrel, arm raised (f70v2)   gait: 'hop'
  //   creatures/guide_barrel_70v2_28.png  bather in a barrel, blue cap (f70v2)     gait: 'hop'
  //   creatures/guide_barrel_70v2_12.png  bather in a barrel (f70v2)               gait: 'hop'
  //   creatures/nymph_76v_1.png           a nymph from the bathing pages (f76v)    gait: 'walk', height: 1.8
  // Delete the whole guide block to walk alone.
  guide: {
    file: 'creatures/guide_barrel_70v2_21.png',
    gait: 'hop',
    height: 1.7,
    name: 'Your guide',
    offset: -2.6,        // where it stands, in metres left (-) or right (+) of the middle of the view
    lines: {
      intro: "I'll take you along the walk. Scroll, swipe or use the arrow keys, or let me lead the way.",
      stop: 'Click the frame to watch it with sound.',
      end: "That's the whole walk. Shall we go round again?",
    },
    quips: [
      'I travel light. Just me and the barrel.',
      'Every plant here is cut straight from the manuscript. Nobody has touched them up.',
      'Click any plant to see which folio it comes from.',
      'Mind the nymphs by the pond. They are shy.',
      "Can I read the writing? No. Nobody can. I just do the walking.",
      'From the Pisces page, folio 70v. Lovely view of the fish.',
    ],
  },

  // Optional notes shown when someone clicks a plant or figure, by folio.
  // Leave empty to show just the folio and section.
  notes: {
    // '1v': 'The first plant of the herbal section.',
  },
};

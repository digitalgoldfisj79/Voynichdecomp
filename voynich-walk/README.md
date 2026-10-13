# Voynich Walk: side-scrolling edition

A paper-theatre walk you scroll through sideways. Painted hills stand behind rows of
real plant cut-outs from the Voynich manuscript. Along the way there are framed videos that
play as you pass, and a pond with bathing figures who dive away when you come close.
Click anything (a plant, a figure, a video) to see what it is.

It's built like a toy theatre: flat painted layers stand one behind the other, and you
slide along in front of them. The layers are at different depths, so they move past at
different speeds, and that's what gives it depth.

## Starting it

Browsers won't load pictures and videos for a page opened by double-clicking it, so it needs
a tiny local web server:

- **Windows:** double-click `start-windows.bat`. Keep the black window open. If Python
  isn't found, install it from python.org and tick "Add python.exe to PATH".
- **Mac:** double-click `start-mac.command`. If macOS refuses, right-click it, choose Open,
  then Open again.
- **On a website:** upload the whole folder as it is. Everything here is plain files.

## The YouTube videos

The stops are videos from the Voynich Talk channel. Each frame shows the video's picture,
and when you stop in front of it **the video itself starts playing inside the frame**,
silently. Walk on and it pauses; the picture comes back. Click the frame (or the guide's
**▶ Watch**) to watch it big, with sound, still inside the walk.

- **Adding a new video:** run `python tools/fetch_youtube.py`. It reads the channel, saves
  the titles, descriptions and pictures (in `youtube.js` and `videos/thumbs/`), and prints
  every video with its id. Put the id in `walk.config.js`, e.g.
  `{ youtube: 'enJg0mp8qqo', plants: ['17v'] }`. The title and note are taken from YouTube
  unless you write your own (`title:`, `note:`). No API key is needed.
- The channel's feed only lists the newest 15 videos. Older ones still work: put their id in
  `walk.config.js` and run the tool; it looks them up one by one (title and picture only).
- Playing in the frame needs the page to come from a web server (the start scripts, or
  your website). Opened straight from the folder, YouTube refuses, and the frames just
  show their pictures. If a video can't be played (embedding turned off, no internet) the
  frame keeps its picture; nothing breaks.
- Turn the in-frame playing off with `videos.youtubeInFrame: false` in `settings.js`.
  `start: 90` on a station starts the silent in-frame playing 90 seconds in.
- You can still use a short clip of your own in a frame (`clip: 'videos/my_clip.mp4'`);
  clicking it then opens YouTube if the station also has a `youtube` id.

## The guide

A bather in a barrel from the Pisces page (f70v2), turned upright, goes along with you. It
hops after you as you scroll, faces the way you're going, and hops on the spot now and then.
When you stop at a video it says the stop's title and note in a speech bubble, with buttons
to watch, go to the next stop, or take the tour. Click it for a remark; the × hides its
messages until you click it again.

To change it, edit `guide` in `walk.config.js`. Two other barrel bathers and a walking nymph
(f76v) are ready to swap in, and you can change the name, the lines, the remarks, the height
and where it stands. Delete the `guide` block to walk alone.

## Moving around

| | |
|---|---|
| scroll wheel, ← →, A D, or drag the picture | walk |
| **Space** / PageUp | next / previous video |
| **click** a plant, figure or video | info card; a video's card has "Play with sound" |
| **T** | tour: glides to each video, waits while it plays, then moves on (loops) |
| **P** | glide: a steady drift to the right, for b-roll |
| **H** | hide everything for a clean picture |
| **G** | settings panel |
| **R** / **K** | record a video / save a screenshot |
| **N** | rearrange the plants (new seed) |
| Esc | close the card or the video |

The walk loops: after the last video it carries on to the start without a join.

## Making it yours

- **`walk.config.js`** is the content: the title card, and for each stop a YouTube id and
  the folios of the plants that stand beside it (see above). Notes for individual plants
  go in `notes`, by folio.
- **`videos/`**: the YouTube pictures (`thumbs/`) and, if you want them, your own short clips.
  The four clips there are unused examples (slow pans over manuscript pages).
- **`settings.js`** holds every number: spacing between videos, camera height and distance,
  plant density and size, wind, scroll feel, glide and tour speeds, fog, colours, and the
  recording size.
- **New pictures:** drop PNGs into `plants/` (named after the folio, e.g. `f12v.png`) or
  `creatures/` (`nymph_…`, `fish_…`, `hop_…`, `fly_…`), then run
  `python tools/make_asset_list.py`. Check new cut-outs with
  `python tools/check_and_fix_pngs.py plants`.

## Files

| File | What it does |
|---|---|
| `index.html` | the page: info card, video viewer, labels, styles |
| `walk.config.js` | **the stops: YouTube ids, plants, optional titles and notes** |
| `youtube.js` | the channel's videos (made by `tools/fetch_youtube.py`) |
| `settings.js` | **every number** |
| `assets.js` | list of pictures (made by the tool) |
| `js/main.js` | loading, scrolling, the tour, clicking |
| `js/world.js` | the painted layers, plant placement, video frames, pond, creatures |
| `js/ui.js` | info card, video viewer, panel, recording |
| `js/guide.js` | the guide: hopping, speech bubble, tour buttons |
| `js/youtube.js` | plays YouTube inside the frame you're standing at |
| `js/random.js` | seeded randomness |
| `lib/` | Three.js 0.160 (with its CSS3DRenderer) and lil-gui; don't edit |

## Recording b-roll

Press **T** (tour) or **P** (glide), then **H** to hide the text, and record with OBS. OBS
records the YouTube videos playing in the frames too. **R** records straight from the 3-D
picture, in real time, but it can't capture YouTube: while it records, the frames show their
pictures. For an exact size, tick **fixed size** under Recording in the settings panel (G)
(this also shows pictures in the frames).

## Pictures

120 plant cut-outs, one per herbal folio, cut from Yale's scans of Beinecke MS 408 using
hand-checked masks. There are also 17 bathing figures (f75r–f77r), the Pisces fish
(f70v2) and the f76v creature, which hops across the path. Clicking a picture links to
Yale's full-page scan of its folio.

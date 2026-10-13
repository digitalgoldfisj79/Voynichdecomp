"""Fetch the channel's videos from YouTube and save them for the walk.

    python tools/fetch_youtube.py

What it does
  1. Reads the channel's public feed (the newest 15 videos; YouTube's feed holds no more).
  2. Also looks up any video named in walk.config.js that is older than that, one by one.
  3. Saves a still picture of each video in videos/thumbs/ (used in the frames).
  4. Writes youtube.js: id, title, date, description and picture for every video.
  5. Prints the list, so you can copy ids into walk.config.js.

Run it again whenever you publish something new. It needs internet, nothing else
(only Python's own modules). No API key is needed.

Options
  --channel UC...   another channel id (default: the one in walk.config.js -> channel)
"""
import json, os, re, sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THUMBS = os.path.join(ROOT, 'videos', 'thumbs')
NS = {'a': 'http://www.w3.org/2005/Atom', 'yt': 'http://www.youtube.com/xml/schemas/2015', 'media': 'http://search.yahoo.com/mrss/'}
UA = {'User-Agent': 'Mozilla/5.0 (VoynichWalk fetch_youtube.py)'}


def get(url, binary=False):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        data = r.read()
    return data if binary else data.decode('utf-8')


def config():
    txt = open(os.path.join(ROOT, 'walk.config.js'), encoding='utf-8').read()
    m = re.search(r"channel:\s*'([\w-]+)'", txt)
    ids = re.findall(r"youtube:\s*'([^']+)'", txt)
    return (m.group(1) if m else None), [video_id(i) for i in ids if video_id(i)]


def video_id(s):
    s = s.strip()
    if re.fullmatch(r'[\w-]{11}', s): return s
    m = re.search(r'(?:v=|youtu\.be/|embed/|shorts/)([\w-]{11})', s)
    return m.group(1) if m else None


def from_feed(channel):
    root = ET.fromstring(get(f'https://www.youtube.com/feeds/videos.xml?channel_id={channel}'))
    out = []
    for e in root.findall('a:entry', NS):
        g = e.find('media:group', NS)
        out.append({
            'id': e.find('yt:videoId', NS).text,
            'title': e.find('a:title', NS).text,
            'published': e.find('a:published', NS).text[:10],
            'description': (g.find('media:description', NS).text or '') if g is not None else '',
        })
    return root.find('a:title', NS).text, out


def from_oembed(vid):
    d = json.loads(get('https://www.youtube.com/oembed?format=json&url=' + urllib.parse.quote(f'https://www.youtube.com/watch?v={vid}')))
    return {'id': vid, 'title': d.get('title', ''), 'published': '', 'description': ''}


def save_thumb(vid):
    """The largest still YouTube has. maxresdefault is 16:9; the smaller ones are 4:3 with black bars (the walk crops those)."""
    path = os.path.join(THUMBS, vid + '.jpg')
    if os.path.exists(path) and os.path.getsize(path) > 2000: return path
    for name in ('maxresdefault', 'sddefault', 'hqdefault'):
        try:
            data = get(f'https://i.ytimg.com/vi/{vid}/{name}.jpg', binary=True)
            if len(data) > 2000:                       # YouTube returns a tiny grey picture when a size is missing
                open(path, 'wb').write(data); return path
        except Exception:
            pass
    return None


def main():
    channel, wanted = config()
    if '--channel' in sys.argv: channel = sys.argv[sys.argv.index('--channel') + 1]
    if not channel: sys.exit("No channel id. Put channel: 'UC...' in walk.config.js or use --channel UC...")
    os.makedirs(THUMBS, exist_ok=True)
    name, videos = from_feed(channel)
    have = {v['id'] for v in videos}
    for vid in wanted:                                 # older videos named in walk.config.js
        if vid not in have:
            try: videos.append(from_oembed(vid)); have.add(vid)
            except Exception as e: print(f'  ! could not look up {vid}: {e}')
    for v in videos:
        p = save_thumb(v['id'])
        v['thumb'] = f"videos/thumbs/{v['id']}.jpg" if p else f"https://i.ytimg.com/vi/{v['id']}/hqdefault.jpg"
    data = {'channel': channel, 'name': name, 'fetched': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), 'videos': videos}
    with open(os.path.join(ROOT, 'youtube.js'), 'w', encoding='utf-8') as f:
        f.write('// Made by tools/fetch_youtube.py. Run it again to update; edits here are overwritten.\n')
        f.write('export const YOUTUBE = ' + json.dumps(data, indent=1, ensure_ascii=False) + ';\n')
    print(f'{name}: {len(videos)} videos saved to youtube.js\n')
    for v in videos:
        print(f"  {v['id']}  {v['published'] or '          '}  {v['title']}")
    print("\nPut an id in walk.config.js, e.g.  { youtube: '%s', plants: ['1v'] }" % videos[0]['id'])


if __name__ == '__main__':
    main()

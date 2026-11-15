"""Frame-by-frame export of the journey to a video file (no screen recording needed).
   python tools/export.py START END out.mp4 [WIDTH] [FPS]        e.g.  python tools/export.py 0 300 journey.mp4 1920 30
Needs: pip install playwright, Google Chrome, ffmpeg. Serve this folder first:  python -m http.server 8765"""
import sys, asyncio, subprocess, time
from playwright.async_api import async_playwright
t0, t1, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
W = int(sys.argv[4]) if len(sys.argv) > 4 else 1920; FPS = int(sys.argv[5]) if len(sys.argv) > 5 else 30; H = W * 9 // 16
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel='chrome', args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox'])
        pg = await b.new_page(viewport={'width': W, 'height': H})
        await pg.goto('http://localhost:8765/index.html?q=hi&clean=1&still=1')
        await pg.wait_for_function('window.journey && window.journey.ready', timeout=300000)
        await pg.evaluate('window.journey.manual = true')
        ff = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'image2pipe', '-framerate', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '16', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
        n = int(round((t1 - t0) * FPS)); start = time.time()
        for i in range(n):
            t = t0 + i / FPS
            await pg.evaluate('t => new Promise(r => { window.journey.renderAt(t); requestAnimationFrame(() => { window.journey.renderAt(t); r(); }); })', t)
            ff.stdin.write(await pg.screenshot(type='jpeg', quality=95))
            if i % 30 == 0: print('frame', i, 'of', n, '%.1f s/frame' % ((time.time() - start) / (i + 1)), flush=True)
        ff.stdin.close(); ff.wait(); await b.close()
asyncio.run(main())

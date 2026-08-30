from pathlib import Path

p = Path(__file__).with_name('taccola_calibration.py')
src = p.read_text(encoding='utf-8')
old = '",\nraw='
if old not in src:
    raise RuntimeError('expected loader tuple marker not found; refusing non-frozen repair')
src = src.replace(old, '"\nraw=', 1)
exec(compile(src, str(p), 'exec'))

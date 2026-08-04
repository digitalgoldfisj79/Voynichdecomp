#!/usr/bin/env python3
"""Execute pilot.py after applying the frozen syntax correction."""
from pathlib import Path

path = Path(__file__).with_name("pilot.py")
source = path.read_text(encoding="utf-8")
broken = '''        src = cv2.resize(raw_rgb[key], (int(raw_rgb[key].shape[1] * min(1.0, 2200.0 / max(raw_rgb[key].shape[:2])),
                                       int(raw_rgb[key].shape[0] * min(1.0, 2200.0 / max(raw_rgb[key].shape[:2])))), interpolation=cv2.INTER_AREA)'''
fixed = '''        src_scale = min(1.0, 2200.0 / max(raw_rgb[key].shape[:2]))
        src_size = (int(raw_rgb[key].shape[1] * src_scale), int(raw_rgb[key].shape[0] * src_scale))
        src = cv2.resize(raw_rgb[key], src_size, interpolation=cv2.INTER_AREA)'''
if broken not in source:
    raise RuntimeError("Expected frozen syntax defect not found; refusing silent rewrite")
source = source.replace(broken, fixed, 1)
exec(compile(source, str(path), "exec"), globals(), globals())

#!/usr/bin/env python3
"""Load and execute the hashable compressed stage-2 implementation."""
from pathlib import Path
import base64
import gzip

payload_path = Path(__file__).with_name("run_stage2.py.gz.b64")
source = gzip.decompress(base64.b64decode(payload_path.read_text(encoding="utf-8").strip())).decode("utf-8")
# OpenCV 5 rejects a float64 Sobel input when a CV_32F destination is requested.
# Freeze the one-line compatibility correction without changing scientific logic.
source = source.replace(
    "return np.clip((x - lo) / (hi - lo), 0, 1)\n",
    "return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)\n",
)
exec(compile(source, str(Path(__file__).with_name("run_stage2_impl.py")), "exec"), globals(), globals())

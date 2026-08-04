#!/usr/bin/env python3
"""Load and execute the hashable compressed stage-2 implementation."""
from pathlib import Path
import base64
import gzip

payload_path = Path(__file__).with_name("run_stage2.py.gz.b64")
source = gzip.decompress(base64.b64decode(payload_path.read_text(encoding="utf-8").strip())).decode("utf-8")
exec(compile(source, str(Path(__file__).with_name("run_stage2_impl.py")), "exec"), globals(), globals())

#!/usr/bin/env python3
"""Download only the f116v acquisitions required by the preregistered preflight.

The public Drive root contains several folios and many derived products. Recursive
acquisition is therefore both wasteful and vulnerable to Drive rate limiting.
This manifest pins the exact f116v source captures used in the first pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import gdown

# One raw reflectance acquisition per primary spectral band, eight transmitted-
# light acquisitions, and two UV-excited white-balance/fluorescence captures.
FILES = {
    "Voynich_116v+MB365UV_007_F.tif": "1fFFH6lVG7UgwSj49CdI_JqsBHdhplnQX",
    "Voynich_116v+MB450RB_001_F.tif": "1BM3_Cr392BPkrHKZ1FQLXUGtp6EkQLPI",
    "Voynich_116v+MB470LB_002_F.tif": "14u86TZmNe9y2hF5EecO6Q1qZkx9hmCjo",
    "Voynich_116v+MB505CN_003_F.tif": "1-Lu1UZYXpj1megBz_PMtz7P5n_DYqYoE",
    "Voynich_116v+MB535GN_004_F.tif": "1g42uO4dKjR4upi-JH3lKRShlBG0iia5D",
    "Voynich_116v+MB570AM_005_F.tif": "1EEY3_zWZOnb6yNQsCh_aF0-gREbzifQx",
    "Voynich_116v+MB625RD_006_F.tif": "1NK4cOby79jgWBomBzLHFStT4Bcx7xr3Z",
    "Voynich_116v+MB700IR_008_F.tif": "1E0Tcgm7B8nZErRlvCAATO0bDlSRPz2Xz",
    "Voynich_116v+MB735IR_009_F.tif": "1nW88TTTtClgwMzJ2mbLAHTfN_DWL2G6t",
    "Voynich_116v+MB780IR_010_F.tif": "1jcGlP8aUmHxtE6_E4o_qXfNSUZZvtF9L",
    "Voynich_116v+MB870IR_011_F.tif": "1bLj_i2DWej_n7efvD66KTlLhEiW9A4Pb",
    "Voynich_116v+MB940IR_012_F.tif": "1duzrV74dw0E3TjmPYZOfG6c3QXzoDuzp",
    "Voynich_116v+TX450RB_039_F.tif": "1D3NzXyalp0eREkkObOhnxED5VHhit6Hu",
    "Voynich_116v+TX535GN_040_F.tif": "1YMyjHSfmW_ztVkKNRCKuPROloR-ta8n1",
    "Voynich_116v+TX570AM_041_F.tif": "1KiQTQxNmwcpTe3cTHhp57Fkrg-nzYRmN",
    "Voynich_116v+TX625RD_042_F.tif": "1ReRMPHVteHqaLi9ovdiQ8tbuvCBNs-XX",
    "Voynich_116v+TX700IR_043_F.tif": "1YHlfgAIaZyERsN1EfkfoEr4vnWR27y0f",
    "Voynich_116v+TX780IR_044_F.tif": "1yEXm-Kmm8yIyutGUokcltCjpVLi-kj-9",
    "Voynich_116v+TX870IR_045_F.tif": "1XFBnhb0R-juSFds9T27MLOrMh5Hvw8kw",
    "Voynich_116v+TX940IR_046_F.tif": "1i8m_PULtBHNpfknm1oKda4l3Qf26KyLL",
    "Voynich_116v+WBUVUVP_019_F.tif": "15AUFF6yeUbzFHkI-AN-hXvUGozcZAQtb",
    "Voynich_116v+WBUVUVB_022_F.tif": "111WyehzhwPuJ2vlsaPzoqOyxZtNKDVIL",
}
README_ID = "1nzKNlV2BqCEz3VvheAFZ_4IULFwLExng"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(file_id: str, target: Path, retries: int) -> tuple[bool, str]:
    if target.exists() and target.stat().st_size > 50_000_000:
        return True, "existing"
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            result = gdown.download(id=file_id, output=str(target), quiet=False)
            if result and target.exists() and target.stat().st_size > 50_000_000:
                return True, f"downloaded_attempt_{attempt}"
            last_error = f"download incomplete: {result!r}"
        except Exception as exc:  # network/provider errors are recorded, not hidden
            last_error = f"{type(exc).__name__}: {exc}"
        target.unlink(missing_ok=True)
        time.sleep(3 * attempt)
    return False, last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--minimum", type=int, default=14)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    for name, file_id in FILES.items():
        target = args.output / name
        ok, note = fetch(file_id, target, args.retries)
        record = {"name": name, "file_id": file_id, "ok": ok, "note": note}
        if ok:
            record.update({"bytes": target.stat().st_size, "sha256": sha256(target)})
        records.append(record)

    # Documentation is useful but not a scientific input, so failure is non-fatal.
    try:
        gdown.download(id=README_ID, output=str(args.output / "READ_ME.pdf"), quiet=True)
    except Exception:
        pass

    manifest = {
        "requested": len(FILES),
        "downloaded": sum(int(r["ok"]) for r in records),
        "minimum_required": args.minimum,
        "records": records,
    }
    (args.output / "DOWNLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("requested", "downloaded", "minimum_required")}, indent=2))
    if manifest["downloaded"] < args.minimum:
        raise SystemExit("Insufficient f116v raw captures acquired")


if __name__ == "__main__":
    main()

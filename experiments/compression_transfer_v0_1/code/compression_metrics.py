#!/usr/bin/env python3
"""Deterministic compression-transfer and normalized compression metrics.

The module deliberately separates:
  * internal compressibility C(x)/|x|;
  * directional conditional cost C(A||b)-C(A);
  * self-normalized excess cost relative to a reference from b's own source;
  * symmetric normalized compression distance (NCD).

No metric is interpreted as semantic evidence by this module.
"""
from __future__ import annotations

import bz2
import gzip
import hashlib
import importlib.util
import json
import lzma
import zlib
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, Mapping

BOUNDARY = b"\n\x00CTD_BOUNDARY\x00\n"


class CompressorUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CompressorSpec:
    name: str
    implementation: str
    version: str
    deterministic_parameters: Mapping[str, object]


@dataclass(frozen=True)
class CompressionObservation:
    compressor: str
    reference_bytes: int
    probe_bytes: int
    c_reference: int
    c_probe: int
    c_reference_probe: int
    c_probe_reference: int
    incremental_bytes: int
    incremental_bits_per_probe_byte: float
    ncd_forward: float
    ncd_reverse: float
    ncd_symmetric: float


def _zlib(data: bytes) -> bytes:
    return zlib.compress(data, level=9)


def _gzip(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9, mtime=0)


def _bz2(data: bytes) -> bytes:
    return bz2.compress(data, compresslevel=9)


def _lzma(data: bytes) -> bytes:
    return lzma.compress(data, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)


def _zstd(data: bytes) -> bytes:
    try:
        import zstandard as zstd  # type: ignore
    except ImportError as exc:
        raise CompressorUnavailable("zstandard is not installed") from exc
    return zstd.ZstdCompressor(level=19, threads=0, write_checksum=True).compress(data)


def _ppmd(data: bytes) -> bytes:
    try:
        import pyppmd  # type: ignore
    except ImportError as exc:
        raise CompressorUnavailable("pyppmd is not installed") from exc
    return pyppmd.compress(data, max_order=6, mem_size=64 << 20)


_COMPRESSORS: Dict[str, Callable[[bytes], bytes]] = {
    "zlib9": _zlib,
    "gzip9": _gzip,
    "bz2_9": _bz2,
    "lzma9e": _lzma,
    "zstd19": _zstd,
    "ppmd6": _ppmd,
}


def available_compressors() -> Dict[str, bool]:
    return {
        "zlib9": True,
        "gzip9": True,
        "bz2_9": True,
        "lzma9e": True,
        "zstd19": importlib.util.find_spec("zstandard") is not None,
        "ppmd6": importlib.util.find_spec("pyppmd") is not None,
    }


def compressor_spec(name: str) -> CompressorSpec:
    if name == "zlib9":
        return CompressorSpec(name, "python-zlib", zlib.ZLIB_VERSION, {"level": 9})
    if name == "gzip9":
        return CompressorSpec(name, "python-gzip/zlib", zlib.ZLIB_VERSION, {"level": 9, "mtime": 0})
    if name == "bz2_9":
        return CompressorSpec(name, "python-bz2/libbz2", "stdlib", {"level": 9})
    if name == "lzma9e":
        return CompressorSpec(name, "python-lzma/xz", "stdlib", {"preset": "9|EXTREME"})
    if name == "zstd19":
        try:
            import zstandard as zstd  # type: ignore
        except ImportError as exc:
            raise CompressorUnavailable("zstandard is not installed") from exc
        return CompressorSpec(name, "python-zstandard", zstd.__version__, {"level": 19, "threads": 0, "checksum": True})
    if name == "ppmd6":
        try:
            import pyppmd  # type: ignore
        except ImportError as exc:
            raise CompressorUnavailable("pyppmd is not installed") from exc
        return CompressorSpec(name, "pyppmd", getattr(pyppmd, "__version__", "unknown"), {"max_order": 6, "mem_size": 64 << 20})
    raise KeyError(f"unknown compressor: {name}")


def compressed_size(data: bytes, compressor: str) -> int:
    try:
        fn = _COMPRESSORS[compressor]
    except KeyError as exc:
        raise KeyError(f"unknown compressor {compressor!r}; choices={sorted(_COMPRESSORS)}") from exc
    return len(fn(data))


def _join(left: bytes, right: bytes, boundary: bytes = BOUNDARY) -> bytes:
    return left + boundary + right


def incremental_cost(reference: bytes, probe: bytes, compressor: str, boundary: bytes = BOUNDARY) -> int:
    base = reference + boundary
    return compressed_size(base + probe, compressor) - compressed_size(base, compressor)


def conditional_bits_per_byte(reference: bytes, probe: bytes, compressor: str, boundary: bytes = BOUNDARY) -> float:
    if not probe:
        raise ValueError("probe must be non-empty")
    return 8.0 * incremental_cost(reference, probe, compressor, boundary) / len(probe)


def directional_excess_bits_per_byte(candidate_reference: bytes, own_reference: bytes, probe: bytes, compressor: str, boundary: bytes = BOUNDARY) -> float:
    """Excess conditional cost relative to the probe's own-source reference."""
    if not probe:
        raise ValueError("probe must be non-empty")
    candidate = incremental_cost(candidate_reference, probe, compressor, boundary)
    own = incremental_cost(own_reference, probe, compressor, boundary)
    return 8.0 * (candidate - own) / len(probe)


def normalized_compression_distance(x: bytes, y: bytes, compressor: str, boundary: bytes = BOUNDARY) -> tuple[float, float, float]:
    """Return forward, reverse, and averaged NCD."""
    cx = compressed_size(x, compressor)
    cy = compressed_size(y, compressor)
    denom = max(cx, cy)
    if denom == 0:
        raise ValueError("empty compressed denominator")
    cxy = compressed_size(_join(x, y, boundary), compressor)
    cyx = compressed_size(_join(y, x, boundary), compressor)
    ncd_xy = (cxy - min(cx, cy)) / denom
    ncd_yx = (cyx - min(cx, cy)) / denom
    return ncd_xy, ncd_yx, (ncd_xy + ncd_yx) / 2.0


def observe(reference: bytes, probe: bytes, compressor: str, boundary: bytes = BOUNDARY) -> CompressionObservation:
    if not reference or not probe:
        raise ValueError("reference and probe must be non-empty")
    cr = compressed_size(reference, compressor)
    cp = compressed_size(probe, compressor)
    crp = compressed_size(_join(reference, probe, boundary), compressor)
    cpr = compressed_size(_join(probe, reference, boundary), compressor)
    inc = incremental_cost(reference, probe, compressor, boundary)
    ncd_f, ncd_r, ncd_s = normalized_compression_distance(reference, probe, compressor, boundary)
    return CompressionObservation(compressor, len(reference), len(probe), cr, cp, crp, cpr, inc, 8.0 * inc / len(probe), ncd_f, ncd_r, ncd_s)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def specs_json(names: Iterable[str]) -> str:
    payload = {
        "available": available_compressors(),
        "selected": [asdict(compressor_spec(name)) for name in names],
        "boundary_hex": BOUNDARY.hex(),
    }
    return json.dumps(payload, indent=2, sort_keys=True)

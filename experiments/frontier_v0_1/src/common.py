from __future__ import annotations
import hashlib, json, os, pickle, pickletools, tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

FORBIDDEN_PICKLE_OPS = {
    "GLOBAL","STACK_GLOBAL","REDUCE","BUILD","OBJ","NEWOBJ","NEWOBJ_EX",
    "INST","PERSID","BINPERSID","EXT1","EXT2","EXT4"
}

class GateFailure(RuntimeError):
    pass

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def safe_pickle_load(path: Path) -> Any:
    data = path.read_bytes()
    bad = sorted({op.name for op, arg, pos in pickletools.genops(data)
                  if op.name in FORBIDDEN_PICKLE_OPS})
    if bad:
        raise GateFailure(f"unsafe pickle opcodes in {path}: {bad}")
    return pickle.loads(data)

def find_first(root: Path, candidates: Iterable[str]) -> Path:
    candidates = list(candidates)
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    raise GateFailure(f"none of the required paths exist: {candidates}")

def load_config(path: Path) -> Dict[str, Any]:
    cfg = load_json(path)
    if cfg.get("schema") != "voynich-frontier-programme-v0.1":
        raise GateFailure("unexpected config schema")
    return cfg

def load_records(repo_root: Path, cfg: Dict[str, Any]) -> List[dict]:
    p = find_first(repo_root, cfg["canonical"]["record_paths"])
    recs = load_json(p) if p.suffix == ".json" else safe_pickle_load(p)
    if not isinstance(recs, list):
        raise GateFailure("canonical records are not a list")
    return recs

def normalize_folio(s: str) -> str:
    s = str(s).strip()
    return s if s.startswith("f") else "f" + s

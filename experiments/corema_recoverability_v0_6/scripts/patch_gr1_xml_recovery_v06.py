#!/usr/bin/env python3
"""Apply the frozen-gate-neutral XML recovery erratum to the v0.6 runner.

The original reconstructed runner is verified byte-for-byte before patching.
Only XML parsing/audit behavior changes: recoverable TEI validity errors such as
CoReMA Gr1's duplicate xml:id declarations are retained and logged. Models,
splits, features, labels, thresholds and random seed are untouched.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ORIGINAL_SHA256 = "213603fd38d99725cc99cb640bccb428eea0480de48298630dd3834426a49616"
MARKER = '"xml_recovery_issues": []'


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    here = Path(__file__).resolve().parent
    target = here / "run_corema_recoverability_v06.py"
    sidecar = here / "PATCHED_RUNNER_SHA256_v0_6.txt"
    raw = target.read_bytes()
    before = sha256(raw)
    text = raw.decode("utf-8")

    if MARKER in text:
        after = before
    else:
        if before != ORIGINAL_SHA256:
            raise RuntimeError(f"refusing to patch unexpected runner sha256={before}")

        text = replace_once(
            text,
            'audit = {"files": [], "parse_failures": [], "types": Counter(), "roles": Counter()}',
            'audit = {"files": [], "parse_failures": [], "xml_recovery_issues": [], "types": Counter(), "roles": Counter()}',
            "parse audit schema",
        )
        text = replace_once(
            text,
            '''        try:\n            root = etree.fromstring(path.read_bytes())\n        except Exception as exc:\n            audit["parse_failures"].append({"file": path.name, "error": str(exc)})\n            continue''',
            '''        try:\n            parser = etree.XMLParser(recover=True, huge_tree=True)\n            root = etree.fromstring(path.read_bytes(), parser)\n            if root is None:\n                raise ValueError("lxml recovery returned no root")\n            if parser.error_log:\n                audit["xml_recovery_issues"].append({\n                    "file": path.name,\n                    "issues": [str(item) for item in parser.error_log],\n                })\n        except Exception as exc:\n            audit["parse_failures"].append({"file": path.name, "error": str(exc)})\n            continue''',
            "TEI file parse",
        )
        text = replace_once(
            text,
            '''                        try:\n                            etree.fromstring(r.content)\n                        except Exception as exc:\n                            errors.append(f"parse:{exc}")\n                            break''',
            '''                        try:\n                            parser = etree.XMLParser(recover=True, huge_tree=True)\n                            root = etree.fromstring(r.content, parser)\n                            if root is None:\n                                raise ValueError("lxml recovery returned no root")\n                        except Exception as exc:\n                            errors.append(f"parse:{exc}")\n                            break''',
            "direct acquisition validation",
        )
        target.write_text(text, encoding="utf-8")
        after = sha256(target.read_bytes())

    sidecar.write_text(
        "CoReMA v0.6 runner technical erratum\n"
        f"original_sha256 {ORIGINAL_SHA256}\n"
        f"patched_sha256  {after}\n"
        "scope XML recovery and audit only; frozen scientific gates unchanged\n",
        encoding="utf-8",
    )
    print(f"original_sha256={ORIGINAL_SHA256}")
    print(f"patched_sha256={after}")


if __name__ == "__main__":
    main()

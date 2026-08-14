from __future__ import annotations
import argparse, subprocess, sys

def call(mod, args):
    cmd = [sys.executable, "-m", mod] + args
    print("+", " ".join(map(str, cmd)))
    raise SystemExit(subprocess.call(cmd))

def main():
    ap = argparse.ArgumentParser(description="Voynich Frontier Programme v0.1 gate-safe runner")
    sub = ap.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("gate0")
    g.add_argument("--repo-root", required=True)
    g.add_argument("--fold-manifest", required=True)
    g.add_argument("--out", default="results/gate0")
    u1 = sub.add_parser("u1-build")
    u1.add_argument("--slim", required=True)
    u1.add_argument("--out", default="results/u1")
    u2 = sub.add_parser("u2")
    u2.add_argument("--panel", required=True)
    u2.add_argument("--out", default="results/u2")
    u3 = sub.add_parser("u3")
    u3.add_argument("--features", required=True)
    u3.add_argument("--out", default="results/u3")
    a = ap.parse_args()
    if a.stage == "gate0":
        call("src.gate0_freeze", ["--repo-root", a.repo_root, "--fold-manifest", a.fold_manifest, "--out", a.out])
    if a.stage == "u1-build":
        call("src.transliteration_uncertainty", ["build", "--slim", a.slim, "--out", a.out])
    if a.stage == "u2":
        call("src.dimperio_replication", ["--panel", a.panel, "--out", a.out])
    if a.stage == "u3":
        call("src.latent_regime", ["--features", a.features, "--out", a.out])

if __name__ == "__main__":
    main()

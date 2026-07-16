#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    phase = sys.argv[1]
    repo = Path(sys.argv[2])
    if phase == "two":
        import v060_family_s_lattice_mapping_phase as lattice

        sys.argv = [sys.argv[0], str(repo)]
        lattice.main()
        return
    if phase == "three":
        import v060_family_s_final_scoring_phase as scoring

        scoring.run(repo)
        return

    import v060_family_s_neural_final_evaluate as implementation

    args = type("Args", (), {
        "repo": repo,
        "signer_url": implementation.DEFAULT_SIGNER_URL,
    })()
    if phase == "one":
        implementation.phase1(args)
    else:
        raise SystemExit(f"unknown phase: {phase}")


if __name__ == "__main__":
    main()

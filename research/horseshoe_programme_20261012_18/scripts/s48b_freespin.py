#!/usr/bin/env python3
"""Post-hoc bound for PREREG 33 (flagged): does a device whose dials are spun far between words also give D2 ~ 0 (i.e. look like VMS)?"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "s48_turn_vs_slip.py")).read().split("out = {}")[0])
r = np.random.default_rng(SEED + 201)
for t, s in ((.8, 9), (1.0, 9), (1.0, 18)):
    x = measure(to_lines(gen_device(10000, r, t, s, 0), LL, r), "STOLFI", r, boot=False)
    print("FREESPIN t=%.1f s=%d  D1 %.3f D2 %.3f  E1 by lag %s  SW by lag %s" % (t, s, x["D1"], x["D2"], np.round(x["E1"], 3), np.round(x["SW"], 3)), flush=True)

#!/usr/bin/env python3
from correction_ammerbach_v04 import make_loader
from correction_cv_v04 import make_external_cv, make_family_cv
from correction_target_v04 import make_fit_and_predict


def install(h):
    h.load_ammerbach=make_loader(h)
    h.external_cv=make_external_cv(h)
    h.family_cv=make_family_cv(h)
    h.fit_and_predict=make_fit_and_predict(h)
    return h

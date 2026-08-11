# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import amadi_driver_final_v1 as d
m=d.m

# Exact transport headers inherited from the successful Cipher Coverage v1 RF fetch.
# This changes no parser, representation, split, scorer, solver or threshold.
m.HEADERS={
    "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language":"en-GB,en;q=0.9",
    "Referer":"https://www.voynich.nu/transcr.html",
}

if __name__=="__main__": m.main()

# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import sys
sys.path.insert(0,'experiments/amadi_residuals_v1');sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
import vbm_typed_fast_v1 as fast
b=fast.b
b.NS='VBMV1TYPEDQ2'
b.PROPS=160000
b.MAX_RESTARTS=24
b.BATCH=6
if __name__=='__main__':b.main()

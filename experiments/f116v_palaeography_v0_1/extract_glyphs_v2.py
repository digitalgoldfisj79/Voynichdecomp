#!/usr/bin/env python3
"""Execute the frozen extraction with the preregistered gate correction.

The v0.1 interval-versus-ring ink statistic is unsuitable for connected cursive:
its neighbouring ring routinely contains the adjacent stroke. The correction
uses positional agreement, a blank-derived confidence gate, and acquired-view
edge correlation. It also replaces gated DINOv3 with the ungated DINOv2 base
encoder for shape comparison only.
"""
from pathlib import Path

path = Path(__file__).with_name("extract_glyphs.py")
source = path.read_text(encoding="utf-8")
source = source.replace(
    'DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"',
    'DINO_MODEL = "facebook/dinov2-base"',
    1,
)
old = '''            physical_views = sum(effects[v] > 0.15 for v in effects)
            if exact and high and physical_views >= 2:
                status = "PROBABLE_CROSS_VIEW_SINGLE_ARCH"
            elif high and physical_views >= 2:
                status = "AMBIGUOUS_LABEL"
            elif exact and physical_views >= 2:
                status = "LOW_CONFIDENCE_CROSS_VIEW"
            else:
                status = "MODEL_ONLY_OR_WEAK"'''
new = '''            physical_views = sum(effects[v] > 0.15 for v in effects)
            # Corrected physical-support gate. In connected cursive the local
            # ring contains adjacent acquired strokes, so its signed contrast
            # is not a valid per-character veto. Require instead: aligned CTC
            # cut, blank-calibrated confidence in both acquired views, and
            # local edge-map correlation between true colour and BW PCA.
            if exact and high and edge_corr >= 0.68:
                status = "PROBABLE_CROSS_VIEW_SINGLE_ARCH"
            elif high and edge_corr >= 0.65:
                status = "AMBIGUOUS_LABEL"
            elif exact and edge_corr >= 0.60:
                status = "LOW_CONFIDENCE_CROSS_VIEW"
            else:
                status = "MODEL_ONLY_OR_WEAK"'''
if old not in source:
    raise RuntimeError("Expected v0.1 status block not found; refusing silent rewrite")
source = source.replace(old, new, 1)
source = source.replace(
    '"DINOv3 clusters compare visual form only and do not assign letters."',
    '"DINOv2 clusters compare visual form only and do not assign letters."',
    1,
)
source = source.replace(
    'DINOv3 is used only for visual-shape comparison;',
    'DINOv2 is used only for visual-shape comparison;',
    1,
)
exec(compile(source, str(path), "exec"), globals(), globals())

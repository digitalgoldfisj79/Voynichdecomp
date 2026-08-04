#!/usr/bin/env python3
"""Load and execute the hashable compressed stage-2 implementation.

The loader applies frozen compatibility/control corrections to the original payload before
execution. This preserves the original payload hash while making every correction explicit.
"""
from pathlib import Path
import base64
import gzip

payload_path = Path(__file__).with_name("run_stage2.py.gz.b64")
source = gzip.decompress(base64.b64decode(payload_path.read_text(encoding="utf-8").strip())).decode("utf-8")

# OpenCV 5 compatibility: force the normalized array back to float32.
source = source.replace(
    "return np.clip((x - lo) / (hi - lo), 0, 1)\n",
    "return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)\n",
)

# Correct the synthetic control. The original planted an unsigned signature at a fixed
# coordinate that could fall inside the recto/stain veto. The corrected control plants a
# signed acquired-stroke signature in the largest frozen eligible region and measures an
# attenuation ladder.
start = source.index("def synthetic_control(")
end = source.index("\n\ndef main()", start)
new_control = '''def synthetic_control(residual: np.ndarray, pos_seed: np.ndarray, neg_seed: np.ndarray,
                      eligible: np.ndarray, detector_fn, rng) -> dict:
    h, w = eligible.shape
    if int(pos_seed.sum()) < 100:
        return {"status": "INSUFFICIENT_VISIBLE_SEEDS"}
    best_target = None
    best_n = 0
    for dyf in (0.28, 0.36, 0.44, 0.52):
        for dxf in (-0.18, -0.08, 0.08, 0.18):
            M = np.array([[1, 0, dxf*w], [0, 1, dyf*h]], np.float32)
            shifted = cv2.warpAffine(pos_seed.astype(np.uint8), M, (w, h),
                                     flags=cv2.INTER_NEAREST, borderValue=0) > 0
            target = shifted & eligible
            n = int(target.sum())
            if n > best_n:
                best_n, best_target = n, target
    if best_target is None or best_n < 100:
        return {"status": "NO_ELIGIBLE_PLANTING_REGION", "target_pixels": best_n}
    target = best_target
    sig = np.array([np.median(residual[j][pos_seed]) for j in range(residual.shape[0])], dtype=np.float32)
    bg = np.array([np.median(residual[j][neg_seed]) for j in range(residual.shape[0])], dtype=np.float32)
    delta = sig - bg
    ladder = []
    for amp in (0.15, 0.25, 0.40, 0.60, 0.85, 1.00):
        syn = residual.copy()
        for j in range(residual.shape[0]):
            syn[j][target] = np.clip(syn[j][target] + amp*delta[j], 0, 1)
        pred = detector_fn(syn)
        hit = pred & target
        precision = float(hit.sum()/max(pred.sum(), 1))
        recall = float(hit.sum()/max(target.sum(), 1))
        f1 = 2*precision*recall/max(precision+recall, 1e-9)
        ladder.append({"amplitude": amp, "pred_pixels": int(pred.sum()),
                       "precision": precision, "recall": recall, "f1": f1})
    best = max(ladder, key=lambda r: r["f1"])
    return {"status": "COMPLETE", "target_pixels": int(target.sum()),
            "signature_l2": float(np.linalg.norm(delta)), "ladder": ladder,
            "best_amplitude": best["amplitude"], "best_f1": best["f1"],
            "best_precision": best["precision"], "best_recall": best["recall"]}
'''
source = source[:start] + new_control + source[end:]

old_decision = '''    synth=synthetic_control(residual,page,pos,neg,non_tx,detector_for_control,rng)

    if raw_pixels==0:
        verdict="NO_RECTO_INDEPENDENT_ERASED_TEXT_SIGNAL"
    elif cand_pixels==0:
        verdict="RAW_SIGNAL_EXPLAINED_BY_RECTO_OR_ARTEFACT"
    elif len(lines)>=2 and native_confirmed>=2:
        verdict="CANDIDATE_RECTO_INDEPENDENT_SIGNAL"
    else:
        verdict="RECTO_CONTROL_INCONCLUSIVE"
    # Missing matching cube forbids EVIDENCE_PRESENT.
    if not raw_recto and verdict=="CANDIDATE_RECTO_INDEPENDENT_SIGNAL":
        verdict="CANDIDATE_RECTO_INDEPENDENT_SIGNAL_MATCHING_CUBE_REQUIRED"
'''
new_decision = '''    eligible_control = valid & (~explained) & (~front_dilate)
    synth=synthetic_control(residual,pos,neg,eligible_control,detector_for_control,rng)

    control_reasons=[]
    detector_control_pass = synth.get("status")=="COMPLETE" and synth.get("best_f1",0.0)>=0.20
    recto_specific = recto_info["specificity_margin_vs_best_negative"] > 0.0
    if not detector_control_pass:
        control_reasons.append("SYNTHETIC_POSITIVE_CONTROL_FAILED")
    if not recto_specific:
        control_reasons.append("RECTO_PROXY_NOT_SPECIFIC_VS_UNRELATED_FOLIOS")

    if control_reasons:
        verdict="RECTO_CONTROL_INCONCLUSIVE"
    elif raw_pixels==0:
        verdict="NO_RECTO_INDEPENDENT_ERASED_TEXT_SIGNAL"
    elif cand_pixels==0:
        verdict="RAW_SIGNAL_EXPLAINED_BY_RECTO_OR_ARTEFACT"
    elif len(lines)>=2 and native_confirmed>=2:
        verdict="CANDIDATE_RECTO_INDEPENDENT_SIGNAL"
    else:
        verdict="RECTO_CONTROL_INCONCLUSIVE"
    if not raw_recto and verdict=="CANDIDATE_RECTO_INDEPENDENT_SIGNAL":
        verdict="CANDIDATE_RECTO_INDEPENDENT_SIGNAL_MATCHING_CUBE_REQUIRED"
'''
if old_decision not in source:
    raise RuntimeError("Frozen decision block not found")
source = source.replace(old_decision, new_decision)
source = source.replace(
    '        "native_confirmed_components":native_confirmed,"synthetic_positive_control":synth,\n',
    '        "native_confirmed_components":native_confirmed,"synthetic_positive_control":synth,\n'
    '        "control_gates":{"detector_positive_control_pass":detector_control_pass,\n'
    '                         "recto_proxy_specific":recto_specific,\n'
    '                         "inconclusive_reasons":control_reasons},\n',
)
source = source.replace(
    "Synthetic attenuated front-side trace: `{synth.get('status')}`, F1={synth.get('f1','NA')}.",
    "Synthetic attenuated front-side trace: `{synth.get('status')}`, best F1={synth.get('best_f1','NA')}.\\n\\nControl-gate reasons: `{control_reasons}`.",
)

exec(compile(source, str(Path(__file__).with_name("run_stage2_impl.py")), "exec"), globals(), globals())

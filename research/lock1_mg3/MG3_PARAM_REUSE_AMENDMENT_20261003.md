# MG3 parameter-checkpoint reuse amendment — 2026-10-03

Timing: written before any MG3 fit completion, free-generation output, or scientific statistic was observed.

The original preregistration specified refitting the unchanged MG2 base parameters from scratch in each MG3 job. After launch, the persistent HF Storage Bucket Digitalgoldfish79/vms-lock1 was recovered. It contains the exact converged MG2 parameter checkpoints from the audited MG2 ZLZI and TTLI jobs.

This amendment permits reuse of those checkpoints instead of recomputing the same convex MG2 fit, conditional on ALL provenance checks below passing before generation:
1. frozen MG2 code hash remains 9a89e05be1616243a3b5aabb8a89e9a5b91de3af630b836d3833cb98b046343d;
2. source/dump hashes remain the preregistered ZLZI/TTLI hashes;
3. each saved parameter pickle SHA-256 equals the frozen list below;
4. every saved fit has converged=True and its nit, nll_per_event and novelty fit rate are logged;
5. the stored MG2 result JSON reproduces the already-audited MG2 max-|z|/argmax/null95/runaway result.

Frozen parameter SHA-256:
ZLZI:
f0 3e75671f69535a40d0e77c0c7e71f18f9739d579e4937f98b4ce928de7966128
f1 37b6698b47a8275f74015380e78a59fc6e0a6ae6757e1be24f441d254b99b9f6
f2 6d9e7a728c74469bcfef9c2dc2815b4d0c2ca546262aa510dc0491482a8fa2f8
f3 2fb8662cbb1ef2f8de15daf8b1f7ae51440226ae08155cc9e61d5cf5ee442b2b
f4 d9e7208f47959514546ce0bd531c82b9b88822db4c52fdeca402d2dbc731de8f
MG2 result e6d849a89498f33097e822c2105353c37dbe4c1205e12fac51082bd06a055071.

TTLI:
f0 9478d1dcd39297442d1d3c5f8394de06e19af661fea676cc043ba5122eab9e2a
f1 5e75d245f21d666a38eee4d4366afe0921da97a30348afe992ff87a9c785f6b0
f2 06203f9ad64f50958a9e264e6b1d73cd2f5ccb887ae104d7a98fa0d0dd1b364a
f3 cecb421d6c259207c988963a357b27edb42399bc45c25fb02e6ec36cfd372e49
f4 b1ff6eb423ba6b7e69fa2741a00d94b8fb161829815a81084ded06c4e6842271
MG2 result 77437f7a4435d3673c6e30b84a7894ba2fe4b40c18a00e1c4b789757717ce97e.

Scientific model, MG3 palette module, generation seeds, diagnostics, closure/falsification thresholds and scoring amendment are unchanged. This is equivalent to resuming from a verified deterministic base-fit checkpoint. Any mismatch aborts the run.

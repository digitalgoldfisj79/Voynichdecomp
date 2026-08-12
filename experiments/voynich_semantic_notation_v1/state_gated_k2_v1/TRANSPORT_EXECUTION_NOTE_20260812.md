# VSN-B3-v1 execution transport note — 2026-08-12

Before launching the frozen state-gated target scorer, Hugging Face Jobs `ps` was attempted three times through the authenticated connector. Each call failed upstream with HTTP 502 **before any job launch**.

No HF state-gated job was created by those calls.

Frozen scientific anchor remains commit `924ef5edd25f884227410dbfad0b59998b33f62f` containing:
- `STATE_GATED_K2_PROTOCOL_V1.md`
- `state_gated_targets_v1.json`
- `state_gated_k2_v1.py`

If local fallback execution is required, the committed Python algorithm must not be edited. A wrapper may replace network transport only by supplying the exact hash-pinned lexical bytes and committed target JSON locally. Such a run must record wrapper code, input hashes and output hashes, and must be labelled local transport fallback rather than an HF execution.

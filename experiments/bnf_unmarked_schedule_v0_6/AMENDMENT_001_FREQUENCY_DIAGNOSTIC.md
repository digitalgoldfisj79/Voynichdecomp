# Amendment 001 — frequency diagnostic implementation

Date: 2026-08-09

No v0.6 schedule-feasibility result has been observed.

The frequency-plausibility diagnostic in the protocol is explicitly **non-binding**. Exact minimization of its quadratic aggregate-frequency objective would require a separate integer quadratic optimizer and is unnecessary for the structural gate.

Prospective implementation clarification:

- Structural assignment feasibility remains exact and unchanged.
- If a schedule+rotation is feasible, the runner reports the SSE of the first exact feasible assignment plus the best SSE found by a deterministic seeded local-search diagnostic over legal assignments.
- This diagnostic may not be the global minimum and is labelled `approx_frequency_sse`.
- It cannot change PASS/REJECT status or select a rotation for later language testing.

All structural criteria remain exactly as frozen.
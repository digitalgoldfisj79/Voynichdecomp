# Tranchedino × STA v2.0 — T20-only optimizer diagnosis

Date: 2026-08-09
Diagnostic runner: `a927e37ca1939171d12ae9c234291603dd48a0db`
No H20/C20 or replication stream was scored by these diagnostic jobs.

Jobs:
- C: `6a787d6eda2af92a634f0286`
- D: `6a787d8a3e1f34a7e32c0645`
- E: `6a787da2da2af92a634f028a`
- F: `6a787dbada2af92a634f028c`

Each independent ensemble received 72 restarts with unchanged f069v multiplicities, Paduan model, RF K=36 stream and exact pair-block polish.

Final T20 objectives:

- C: **-2.472681680372687**
- D: -2.479377938651999
- E: **-2.472681680372687**
- F: -2.475699360505827

Ensembles C and E independently recovered the **identical best 36-state numerical map**, not merely the same objective:

`[4,0,4,18,16,17,8,17,15,3,10,8,18,10,16,11,15,9,9,3,11,12,13,7,13,2,12,1,14,6,6,5,14,7,5,1]`

C first reached this basin at restart 72; E first reached it at restart 54. The best 36-restart v2.0 binding objective had been -2.4856662191641186, so the common C/E basin improves the fit by 0.01298453879 nats/retained T20 symbol.

## Diagnosis

The v2.0 36-restart failure was an **insufficient-search / local-basin problem**. T20 does support a reproducible fixed f069v-geometry optimum under the stronger 72-restart search.

This diagnosis uses T20 only. It does not rehabilitate the accidentally exposed H20 panel, which remains permanently inadmissible.

A continuation may prospectively freeze the stronger search and create a new held-out hierarchy solely from the never-scored v2.0 C20 folios.

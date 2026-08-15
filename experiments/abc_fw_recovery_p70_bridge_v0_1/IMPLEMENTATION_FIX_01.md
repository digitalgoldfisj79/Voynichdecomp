# Implementation fix 01 — 2026-08-15

Run 1 failed after completing the permutation loop but before writing any result, at the carrier-profile reporting line. The error was a Python unpacking typo:

`dominant,n=parse=parses.most_common(1)[0][0],parses.most_common(1)[0][1],None`

It is replaced at execution with:

`dominant,n=parses.most_common(1)[0]`

This changes no corpus selection, permutation, seed, statistic, threshold, or hypothesis. No recovery or bridge target value was written or inspected before this fix. The workflow applies this one-line hotfix immediately before execution so the original frozen runner remains auditable.
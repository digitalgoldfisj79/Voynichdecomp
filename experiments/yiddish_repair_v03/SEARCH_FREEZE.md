# Search-budget repair after coverage diagnostic

The balanced-coverage trigram run remains DEVELOPMENT_NOT_RESOLVED: F166 has 26/32 correct-model recovery passes; every other development family has 32/32. All other inherited gates pass. Failed F166 returned keys score strictly below the known correct key under the revised fit objective; this is a directly tested search deficit.

Freeze before new outcomes: increase restarts from 8 to 32 for both anonymous language models and both primary and shuffled recovery arms. Keep 5,000 proposals/restart, 60 greedy passes, training data/model bytes, extraction, random-map nulls, call rule, and all recovery/qualification gates fixed. Rerun all 16 development families and all 32 keys from scratch; do not select failed keys for preferential compute. The N1 mapping control remains unchanged and has no search restart parameter.

This can only produce a DEVELOPMENT pass. Previously exposed keys and source families remain exposed. A pass cannot authorize Voynich without fresh confirmation and representation qualification. Existing failed results are retained.

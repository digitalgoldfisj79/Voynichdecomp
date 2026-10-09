# LANG2 CI 24-pair CLI-only extension, 2026-10-09

The preregistered new seed manifest contains twenty-four model/scramble seed pairs. The original 2026-10-09 wrapper at commit 10e8d498c745f833223b58ac2fdbfb66c7602371 hardcodes the original four seed choices. Initial jobs 6ac9445d... and 6ac9445f... exited before training on argparse invalid choices, not as statistical failures.

For confirmatory jobs, download the immutable original wrapper, then perform exactly these engineering edits prior to execution:
1. Extend SEEDS list by appending the 24 model_seed integers in research/data/LANG2_CI_24PAIR_SEEDS_20261009.json.
2. Extend SCRAMBLE_SEEDS by the 24 corresponding scramble_seed integers in precisely the same index order. The original argparse validation and scramble mapping assertions then operate unchanged.
3. Change the cache output file to a per-mode, per-seed path in /tmp/lang2_ger2_frozen, rather than a single shared path, to prevent concurrent packed replicas overwriting the same cache transport file.

The solver, source corpus, fold hashes, MCF, model architecture, 10x150 training, and checkpoint selection are all unchanged. Retry failed jobs only with the same predefined paired seeds. Original four-seed GER2 prereg remains FAIL.

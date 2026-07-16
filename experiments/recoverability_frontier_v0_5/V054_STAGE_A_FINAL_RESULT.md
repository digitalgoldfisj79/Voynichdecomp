# Recoverability frontier v0.5.4 — final nomenclator Stage A result

Date: 2026-07-16

Verdict: **BOTH COMPONENT GATES FAIL. STAGE B JOINT RECOVERY IS PROHIBITED. CLOSE THE GENERIC NOMENCLATOR LINE.**

No Voynich text was scored.

## Frozen family

Each trial combined:

- a fresh monoalphabetic character key;
- a fresh whole-word nomenclator codebook sampled from a train-only candidate pool;
- opaque one-symbol replacements for all occurrences of selected words;
- a joint random relabelling of character and code symbols;
- first-occurrence canonicalisation;
- no raw range, padding or symbol-type shortcut.

Stage A1 supplied the true character key and symbol-type partition but hid the codeword mapping. Stage A2 supplied the true codeword mapping and observed plaintext-character inventory but hid the residual character key.

## A1 — unknown codeword identities

### Train-only word n-gram frontier

Eight English development chunks were evaluated in each cell.

| Plaintext characters | Candidate pool | Codebook size | Mean observed codes | Mapping accuracy | Occurrence accuracy | Expanded recovery | Scientific SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 384 | 32 | 16 | 5.75 | 44.2361% | 58.9822% | 96.7387% | `b56c2be5bc0a8d67b93f4586923037688771af05040a2c6f594cfb0ee5418d9e` |
| 384 | 64 | 24 | 7.125 | 37.9613% | 45.6727% | 94.8009% | `aaa5f34da71eb4a8928bfb82b1bcda5f00b81bbf07ba6291ceb533ebb96af5b6` |
| 384 | 96 | 24 | 4.50 | 29.8512% | 44.3624% | 97.1964% | `18d7ab0bc473c3278f1c14bba2e2648bf68925c4ece2464e6c261b49e99f29ca` |
| 768 | 32 | 16 | 9.25 | 44.3998% | 61.0475% | 96.9196% | `00cef355cd4792639412bf925e24bbcf3297f558b7da73dff5989c4e583daa12` |
| 768 | 64 | 24 | 10.25 | 34.7671% | 56.8646% | 96.1223% | `2f9ad8fe0fa2946eeae85f58dee804f961d303deffaec4b0e7555846999acd33` |
| 768 | 96 | 24 | 8.75 | 29.6001% | 51.9045% | 96.7872% | `59e4f51b8eae8cb525cf868e194a9bf9c5516f8ef78171ac4203c7e4412522f3` |
| 1536 | 32 | 16 | 11.125 | **55.6133%** | **74.7558%** | 97.7417% | `95287187151f39bae6b8dfa3e864913c19931f35b0841ebd40169ca1101f0bd7` |
| 1536 | 64 | 24 | 14.375 | 39.2410% | 59.0044% | 96.0026% | `bfea763e35987bf9c6fda4ffd7b4a68d77c91bcc466b785dfb8033ad79c66f4f` |
| 1536 | 96 | 24 | 12.125 | 35.0562% | 58.4265% | 96.6392% | `f698b954e43dbdef2210e4c65187de6f94c393a9d31001b7a36d56f351744e7c` |

More text increased support but never approached the frozen 80% mapping gate.

### Train-only masked-word Transformers

Medium model:

- job: `Digitalgoldfish79/6a586630b1669a49bf076c87`;
- model: d_model 256, four layers, 10,000 updates;
- scientific SHA-256: `cc849ee55e96919bff6b27216c08e56508f6fff11d32bbbf776c6346d8564bbf`;
- best mapping accuracy: 39.3750%, at 384 characters and candidate pool 32;
- no frontier cell passed.

Large model:

- job: `Digitalgoldfish79/6a58663fb1669a49bf076c89`;
- model: d_model 384, six layers, 15,000 updates;
- final masked-word training loss: 0.7002;
- scientific SHA-256: `cebc2b5c314102ae9193f2033dc977727029484bae47b4aabd31dcedd9364615`;
- best mapping accuracy: **44.9869%**, at 1536 characters and candidate pool 32;
- corresponding occurrence accuracy: 71.4003%;
- no frontier cell passed.

The low neural training loss did not translate into reliable code identity. Context often supports several plausible frequent words, while each opaque symbol has too few independent occurrences to resolve those alternatives.

**A1 verdict: fail.**

## A2 — true code words, unknown residual character key

### Random-pair annealing

Initial schedule, `300,000 × 35`:

- mean expanded recovery: 65.9118%;
- median: 99.4838%;
- five near-complete basin hits.

`700,000 × 50`:

- job: `Digitalgoldfish79/6a5864a985d9643ce16d5d37`;
- mean: **88.4126%**;
- median: 100%;
- seven of eight at least 90%;
- one failure at 8.33%;
- SHA-256: `fd472f5e2f0f34ae1207fe4675b8c487e34af6f2d94650e2a202413ca3e9ce61`.

`1,200,000 × 70`:

- job: `Digitalgoldfish79/6a5864b2b1669a49bf076c3d`;
- mean: 73.0074%;
- median: 99.4838%;
- five of eight at least 90%;
- SHA-256: `b71af27df4b35c7244a3ac498d1bb64731e8e2a27acccae260c234119de72c91`.

The search was non-monotonic under the language-model objective.

### Exhaustive-pair CrypTool-style search

Job: `Digitalgoldfish79/6a5866cbb1669a49bf076c9b`

Scientific SHA-256: `a32f68bb82b089b664929b0cebe1bbc1a3527ecce49a3435265bcef9ef466e98`

| Restart prefix | Mean recovery | Median | Minimum | Trials ≥90% |
|---:|---:|---:|---:|---:|
| 12 | 68.2048% | 99.4838% | 8.07% | 5/8 |
| 24 | 68.2048% | 99.4838% | 8.07% | 5/8 |
| 48 | 68.1723% | 99.4838% | 7.81% | 5/8 |
| 96 | **77.4751%** | **99.7442%** | **8.33%** | **6/8** |

The correct basin is recoverable for most texts, but neither random nor exhaustive search meets the all-eight reliability gate. Higher objective values can also select lower-accuracy plaintexts, demonstrating residual source-model misspecification in some cells.

**A2 verdict: fail.**

## Scientific conclusion

Generic fresh-codebook nomenclator identity is underdetermined in the tested regime:

- 384–1536 plaintext characters;
- 32–96 plausible train-derived candidate words;
- 16–24 fresh code entries;
- train-only n-gram and masked-Transformer source models.

High expanded character recovery is misleading because code tokens occupy a minority of the plaintext. The programme therefore retains exact code mapping as the primary A1 criterion.

The residual monoalphabetic component is frequently almost perfectly recoverable, but not with the frozen reliability requirement. Since both component gates are required, no joint Stage B solver or locked test is permitted.

This does not show that historical nomenclators are generally insoluble. Real archives may provide repeated keys, multiple messages, known names, candidate plaintext lists, parallel correspondence or external historical anchors. Those belong to a separate anchored-decipherment programme, not generic fresh-key recovery.

## Programme decision

- stop generic nomenclator development;
- do not condition the synthetic codebook on words known to occur in the test message;
- do not relax exact code mapping in favour of edit-distance recovery;
- do not score the Voynich Manuscript;
- proceed to substitution plus fixed block transposition under v0.5.5.
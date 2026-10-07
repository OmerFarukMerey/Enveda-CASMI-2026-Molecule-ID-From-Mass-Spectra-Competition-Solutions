<!-- index: fusion-v4n 2026-10-07 local replay of v40a (v33b + FUSE_SKIP_LIB 0.97 + POP_LIB_OFF + FRAG_LIB_OFF, submitted on outside evidence, score pending) against v33b with real ICEBERG / GLACIER scores on the 380 mixed regime and the Enveda panel (cross-fitted engine 2 on 226, in-sample on 250): per type MRR@25, top-1, paired differences, exposure; the three flags alone and in pairs, the skip threshold at 0.90 / 0.95 / 0.97 / 0.99, how many lists each flag can touch and the effect inside them, split by the answer's start rank and by lib_max band, headroom of protection; fpnet_full1 is not on this machine so v41a (FP_BANK 'ens') cannot be replayed; result: no local support, a small loss (composite -0.002, 1.3 to 1.6 SE) that sits in lists whose answer starts at rank 2 to 25, nothing to protect locally -->
# What the local replay says about v40a (library-hit protection), 2026-10-07

Offline only. Nothing was submitted, pushed, committed, uploaded or downloaded; no `kaggle` command was run; no network was
trained; the visible `test.parquet` was not read; no NIST data. **No GPU was used**: the three flags do not change which
candidates `nb/07` scores, so every ICEBERG / GLACIER / MetFrag score was already cached (candidates to score: 0 of 31,047
and 0 of 26,070; coverage gap 0 in every stored variant). CPU: two processes of mine at below-normal priority, about four
minutes each. Scripts: `scratch/libprotect/` (git-ignored). Outputs: `D:\casmi-data\libprotect\` (new; the score caches there
are copies of `D:\casmi-data\uregime\`; every other cache was read, not written). No file of the solution was changed except
this review. `INDEX.md` and `EXPERIMENTS.md` not edited. v40a's score was pending when this was written; nothing here changes
the submission.

**Every number below was computed in this study** unless a file is named. MRR = MRR@25 on the metric key; "±" = one standard
error of a paired per-molecule (per-list where said) difference; (wins / losses). **No leaderboard estimate is given.**

## 0. Bottom line

1. **The local replay does not support v40a, and does not condemn it.** Composite (mean of the four type means) against
   v33b: mixed regime **−0.0021 ± 0.0013**, panel with cross-fitted engine 2 **−0.0025 ± 0.0019**, panel in-sample
   −0.0022 ± 0.0017, fully unseen base chain −0.0017 ± 0.0013 and −0.0024 ± 0.0014. One cell is beyond two standard
   errors: mixed-regime `dA` −0.0063 ± 0.0022 (2 / 18). One cell is positive: mixed-regime `dB` +0.0027 ± 0.0014 (5 / 1).
   Exposure is unchanged in every set (the flags act at `lib_max` ≥ 0.9, the join below 0.7).
2. **The flags touch few lists and move fewer.** `lib_max` ≥ 0.9: 192 of 1,180 lists in the mixed regime (16%; `c1` 95 of
   150), 84 of 904 on the panel (9%; `c1` 57 of 226). `lib_max` ≥ 0.97: 113 lists in the mixed regime (`c1` 76), **none on
   the panel**, so `FUSE_SKIP_LIB=0.97` is tested on one set only. Reciprocal rank changes in 37 lists (mixed regime) and 9
   (panel). Inside the touched lists: −0.013 ± 0.008 (9 / 28) and −0.027 ± 0.017 (1 / 8).
3. **Which flag does what.** `FUSE_SKIP_LIB` 0.97: `dA` −0.161 ± 0.038 inside its 12 lists (0 / 10: engine 2 held the removed
   answer), `dB` +0.073 ± 0.032 (5 / 1), pool answers 0 (3 of 89 lists move). `POP_LIB_OFF`: the whole panel effect (`c1`
   −0.005, `c2` −0.005; 1 / 5 to 1 / 8), and `c1` −0.002 on the mixed regime. `FRAG_LIB_OFF`: nothing on the panel (0 lists
   change), `c2` −0.003 ± 0.002 (0 / 3) on the mixed regime; it can only act on the 31 touched lists ICEBERG does not cover.
   A lower threshold is not better: 0.90 costs `dA` −0.010 ± 0.003 (0 / 23) and gains `dB` +0.005 ± 0.003 (7 / 1).
4. **Where the effect sits: in lists whose answer starts below first.** Rank 1: +0.001 ± 0.001 (one list moves in each set).
   Rank 2 to 10: −0.007 ± 0.004 (7 / 13) in the mixed regime, −0.019 ± 0.009 (0 / 7) on the panel. Absent from the base
   list: −0.002 ± 0.001 (1 / 12). **This is the opposite of the pattern that preceded the v34a loss** (a local gain carried
   by lists whose answer already starts first). Here there is no local gain to mistrust; there is a small local loss, in the
   stratum the U-regime review found closest to the leaderboard, on 7 to 23 moving lists.
5. **Nothing to protect locally.** In pool-answer lists with `lib_max` ≥ 0.9 the answer starts first in 106 of 128 (mixed
   regime) and 59 of 66 (panel); v33b's later stages take the first place away in 2 and 3 of those, and lift 3 and 4 answers
   to first from below. v40a keeps 1 and 1 of the former and loses 2 and 4 of the latter. The harm protection is meant to
   prevent does not occur in the local sets, so a local replay cannot show its benefit.
6. **`fpnet_full1` is not on this machine** (`find D:/casmi-data/input -iname "*full1*"`: no hit; `ahmedberatozer/` holds
   `casmi26-glacier`, `casmi26-iceberg`, `casmi26-v2-pool`, `casmi26-v3-models`, `casmi26-v4b-models` and two others).
   **v41a (`FP_BANK='ens'`) cannot be replayed locally without that 198.8 MB download**; it was not downloaded.
7. **Weight: low.** Sections 5 and 6.

## 1. What was run

`scratch/libprotect/lp_replay.py` executes `scratch/uregime/u_replay.py` unchanged (path shim: output root and variant
table), i.e. `fusion_core.build_submission` with real forward scores and the submitted fusion config
(`ICE_LAM = GL_LAM = 0.75`, `POP_MU` 0.25, `FRAG_LAM` 0.5 `gap`, `ICE_LAM_LIB = GL_LAM_LIB = 0`, join N 100 below 0.7).
Variants: `lp_vars.py`.

| Chain | Base list | Engine 2 | Sets |
|---|---|---|---|
| **P** (v33b's emulation; primary) | public FPNet rows + public ranker, rank-averaged with our out-of-fold ranker below `lib_max` 0.7 | public | mixed regime (380 molecules: `c1` 150, `c2` 380, `dA` / `dB` 325); panel, all 250 (**in-sample for engine 2's rankers**) |
| **C** (panel, secondary) | as P | cross-fitted rankers (`regime\eng2\p2_*_pubcf`) | panel, the 226 molecules with cross-fitted lists |
| U (sensitivity) | fold-safe FPNet rows + our out-of-fold ranker alone, no gate | cross-fitted on the panel, public on the mixed regime | both |

Checks. Reference P on the mixed regime: 0.8466 / 0.7600 / 0.5536 / 0.3553, the fsranker2 review's v33b row to four
decimals. On the panel 0.9452 / 0.8636 / 0.6743 / 0.3986: `c2` is 0.0026 below that review's 0.8662 and equal to the
U-regime study's value (its fuller MetFrag cache, used here). Outside `lib_max` ≥ 0.9 the largest difference of any variant
to the reference is exactly 0 in all five chain / set pairs, as the code says it must be.

What v40a changes in a list, given v33b's config: at `lib_max` ≥ 0.9 the forward weights are already 0 and the PubChem
merge is already off; v33b still applies the popularity prior to the base list, the MetFrag term when ICEBERG does not cover
the molecule, and the rank fusion with engine 2. v40a removes the first two at ≥ 0.9 and the third at ≥ 0.97: **at ≥ 0.97
the submitted list is the base list in the ranker's order.**

## 2. v40a against v33b

Exposure = pool-answer lists with a joined PubChem row first / answer lost the first place to a joined row, v33b → v40a.

**Mixed regime (chain P; primary)**

| type | n | MRR v33b | MRR v40a | top-1 v33b | top-1 v40a | difference (wins / losses) | exposure | `lib_max` ≥ 0.9 | ≥ 0.97 | lists that move |
|---|---|---|---|---|---|---|---|---|---|---|
| `c1` | 150 | 0.8466 | 0.8444 | 0.807 | 0.807 | −0.0021 ± 0.0014 (0 / 3) | 5 / 0 → 5 / 0 | 95 | 76 | 3 |
| `c2` | 380 | 0.7600 | 0.7572 | 0.682 | 0.679 | −0.0028 ± 0.0036 (2 / 6) | 23 / 5 → 23 / 5 | 33 | 13 | 8 |
| `dA` | 325 | 0.5536 | 0.5473 | 0.422 | 0.418 | **−0.0063 ± 0.0022 (2 / 18)** | – | 32 | 12 | 20 |
| `dB` | 325 | 0.3553 | 0.3579 | 0.240 | 0.240 | +0.0027 ± 0.0014 (5 / 1) | – | 32 | 12 | 6 |
| composite | | | | | | −0.0021 ± 0.0013 | | 192 of 1,180 | 113 | 37 |

**Enveda panel, cross-fitted engine 2 (chain C, 226 molecules; secondary)**

| type | n | MRR v33b | MRR v40a | top-1 v33b | top-1 v40a | difference (wins / losses) | exposure | `lib_max` ≥ 0.9 | ≥ 0.97 | lists that move |
|---|---|---|---|---|---|---|---|---|---|---|
| `c1` | 226 | 0.9329 | 0.9278 | 0.898 | 0.889 | −0.0052 ± 0.0037 (0 / 2) | 0 / 0 → 0 / 0 | 57 | 0 | 2 |
| `c2` | 226 | 0.8323 | 0.8277 | 0.752 | 0.748 | −0.0046 ± 0.0050 (1 / 3) | 4 / 1 → 4 / 1 | 9 | 0 | 4 |
| `dA` | 226 | 0.6556 | 0.6555 | 0.540 | 0.540 | −0.0001 ± 0.0001 (0 / 1) | – | 9 | 0 | 1 |
| `dB` | 226 | 0.3972 | 0.3970 | 0.217 | 0.217 | −0.0003 ± 0.0002 (0 / 2) | – | 9 | 0 | 2 |
| composite | | | | | | −0.0025 ± 0.0019 | | 84 of 904 | 0 | 9 |

**Same panel in-sample (chain P, 250 molecules, public engine 2):** `c1` 0.9452 → 0.9405, −0.0047 ± 0.0033 (0 / 2); `c2`
0.8636 → 0.8593, −0.0043 ± 0.0045 (1 / 3); `dA`, `dB` 0 (no list moves); exposure `c1` 1 / 1, `c2` 4 / 3, unchanged; 91
lists at ≥ 0.9, none at ≥ 0.97; composite −0.0022 ± 0.0017.

**Unseen base chain U:** mixed regime `c1` +0.0012 ± 0.0036 (1 / 3), `c2` −0.0048 ± 0.0032 (3 / 10), `dA` −0.0040 ± 0.0014
(4 / 15), `dB` +0.0009 ± 0.0005 (5 / 3), composite −0.0017 ± 0.0013; panel 226 −0.0041 ± 0.0032, −0.0054 ± 0.0038, −0.0002,
−0.0001, composite −0.0024 ± 0.0014. Same picture with a base list that has seen nothing.

Share of lists at `lib_max` ≥ 0.9 / ≥ 0.95 / ≥ 0.97 / ≥ 0.99: mixed regime `c1` 95 / 84 / 76 / 71 of 150, `c2` 33 / 20 / 13
/ 7 of 380, `dA` = `dB` 32 / 19 / 12 / 6 of 325; panel (226) `c1` 57 / 5 / 0 / 0, the other types 9 / 0 / 0 / 0. The
panel's highest library similarities end below 0.97.

## 3. The parts

Difference to v33b over all lists of the type; in brackets the mean inside the lists the flag can touch and their number.

**Mixed regime (chain P)**

| variant | `c1` | `c2` | `dA` | `dB` | all lists | inside touched | lists that move |
|---|---|---|---|---|---|---|---|
| `FUSE_SKIP_LIB` 0.97 | −0.0006 ± 0.0006 (−0.001; 76) | 0.0000 ± 0.0006 (0.000; 13) | **−0.0059 ± 0.0022, 0 / 10 (−0.161 ± 0.038; 12)** | +0.0027 ± 0.0014, 5 / 1 (+0.073 ± 0.032; 12) | −0.0010 ± 0.0007 | −0.010 ± 0.008 (113) | 19 |
| `POP_LIB_OFF` | −0.0017 ± 0.0012, 0 / 3 (−0.003; 95) | −0.0003 ± 0.0024, 1 / 5 (−0.003; 33) | +0.0017 ± 0.0021, 4 / 8 (+0.018; 32) | +0.0026 ± 0.0027, 3 / 1 (+0.027; 32) | +0.0009 ± 0.0012 | +0.006 ± 0.008 (192) | 25 |
| `FRAG_LIB_OFF` | −0.0006 ± 0.0006, 0 / 1 (−0.001; 95) | −0.0033 ± 0.0022, 0 / 3 (−0.038; 33) | −0.0005 ± 0.0004 (−0.005; 32) | −0.0002 ± 0.0002 (−0.002; 32) | −0.0013 ± 0.0007 | −0.008 ± 0.004 (192; 31 not covered by ICEBERG) | 8 |
| skip 0.97 + pop | −0.0016 ± 0.0011 | −0.0016 ± 0.0029 | −0.0049 ± 0.0021 (3 / 16) | +0.0039 ± 0.0018 (6 / 0) | −0.0010 ± 0.0012 | −0.006 ± 0.008 | 36 |
| skip 0.97 + frag | −0.0011 ± 0.0008 | −0.0036 ± 0.0026 | −0.0061 ± 0.0022 (1 / 11) | +0.0027 ± 0.0014 | −0.0022 ± 0.0011 | −0.014 ± 0.007 | 25 |
| pop + frag | −0.0022 ± 0.0015 | −0.0026 ± 0.0033 | −0.0010 ± 0.0006 (3 / 10) | −0.0003 ± 0.0003 | −0.0015 ± 0.0011 | −0.009 ± 0.007 | 27 |
| **v40a** (all three) | −0.0021 ± 0.0014 | −0.0028 ± 0.0036 | −0.0063 ± 0.0022 | +0.0027 ± 0.0014 | −0.0022 ± 0.0014 (9 / 28) | −0.013 ± 0.008 | 37 |
| skip 0.90 alone | +0.0002 ± 0.0008 | +0.0012 ± 0.0016 (4 / 5) | **−0.0104 ± 0.0025 (0 / 23)** | +0.0048 ± 0.0025 (7 / 1) | −0.0011 ± 0.0011 | −0.007 ± 0.007 (192) | 43 |
| skip 0.95 alone | +0.0002 ± 0.0008 | +0.0016 ± 0.0015 (3 / 3) | −0.0074 ± 0.0023 (0 / 15) | +0.0027 ± 0.0014 | −0.0008 ± 0.0009 | −0.006 ± 0.007 (142) | 30 |
| skip 0.99 alone | −0.0006 ± 0.0006 | 0.0000 ± 0.0006 | −0.0017 ± 0.0009 (0 / 4) | +0.0015 ± 0.0010 (3 / 0) | −0.0001 ± 0.0004 | −0.002 ± 0.006 (90) | 10 |
| v40a with skip 0.90 / 0.95 / 0.99 | −0.0054 / −0.0021 / −0.0021 | −0.0020 / −0.0017 / −0.0026 | −0.0104 / −0.0073 / −0.0027 | +0.0048 / +0.0027 / +0.0013 | −0.0029 / −0.0021 / −0.0015 | | 47 / 38 / 31 |

**Panel, cross-fitted engine 2 (chain C, 226)**

| variant | `c1` | `c2` | `dA` | `dB` | all lists | inside touched | lists that move |
|---|---|---|---|---|---|---|---|
| `FUSE_SKIP_LIB` 0.97, 0.95, 0.99 | 0 | 0 | 0 | 0 | 0 | 0 lists at ≥ 0.97; 5 at ≥ 0.95, none moves | 0 |
| `POP_LIB_OFF` | −0.0052 ± 0.0037, 0 / 2 (−0.021; 57) | −0.0046 ± 0.0050, 1 / 3 (−0.117; 9) | −0.0001 (9) | −0.0003 (9) | −0.0025 ± 0.0016 (1 / 8) | −0.027 ± 0.017 (84) | 9 |
| `FRAG_LIB_OFF` | 0 | 0 | 0 | 0 | 0 | 0 (2 touched lists not covered by ICEBERG) | 0 |
| **v40a** | = `POP_LIB_OFF` | | | | −0.0025 ± 0.0016 | −0.027 ± 0.017 | 9 |
| skip 0.90 alone | −0.0015 ± 0.0023 (1 / 1) | −0.0015 ± 0.0012 (0 / 2) | −0.0024 ± 0.0012 (0 / 5) | **+0.0101 ± 0.0051 (5 / 0; +0.253 inside 9)** | +0.0012 ± 0.0015 | +0.013 ± 0.016 (84) | 14 |
| v40a with skip 0.90 | −0.0022 ± 0.0044 | −0.0041 ± 0.0060 | −0.0024 ± 0.0012 | +0.0101 ± 0.0051 | +0.0003 ± 0.0023 | +0.004 ± 0.025 | 20 |

In-sample panel (chain P, 250): the same rows within 0.001 (`POP_LIB_OFF` `c1` −0.0047, `c2` −0.0043; skip 0.90 `dA`
−0.0026 ± 0.0012, `dB` +0.0084 ± 0.0043). Chain U: `D:\casmi-data\libprotect\report.md`.

Reading of the parts.
- **`FUSE_SKIP_LIB` is a trade between `dA` and `dB`, and nothing for pool answers.** Without the fusion a removed answer
  that only engine 2 lists is gone (`dA`), and engine 2's rows no longer stand above an answer the base list holds lower
  down (`dB`). In pool-answer lists at ≥ 0.97 three of 89 move. The trade's size per touched list is large (−0.16 against
  +0.07 on 12 lists each) and its sign on the hidden test depends on a mix nobody knows: how often a list with a
  near-perfect library match has its answer only in engine 2.
- **`POP_LIB_OFF` is the pool-answer cost.** In library-hit lists whose answer is not first, the popularity prior lifts it;
  without it, it stays where the ranker put it. Local answers are library compounds with high popularity, so this is the
  term most likely flattered locally (`reviews/uregime-2026-10-07.md`, table of section 1).
- **`FRAG_LIB_OFF` is almost inert**: 31 lists can change in the mixed regime, 8 do, all on the losing side by a small
  amount; on the panel none.

## 4. By where the answer starts, and by `lib_max` band

v40a − v33b per list (n: mean ± SE (wins / losses)); rank = the answer's place in the base list before the popularity prior.

**By start rank**

| set (chain) | lists | first | 2 to 10 | 11 to 25 | lower or absent |
|---|---|---|---|---|---|
| mixed regime (P) | all | 595: +0.0013 ± 0.0013 (1 / 0) | 231: −0.0066 ± 0.0044 (7 / 13) | 22: −0.044 ± 0.042 (0 / 3) | 332: −0.0024 ± 0.0010 (1 / 12) |
| | `c1` + `c2` | 381: +0.0020 ± 0.0020 (1 / 0) | 93: −0.0124 ± 0.0072 (1 / 6) | 14: −0.069 ± 0.065 (0 / 3) | 42: 0 |
| | `dA` | 107: 0 | 69: −0.0180 ± 0.0088 (1 / 6) | 4: 0 | 145: −0.0056 ± 0.0024 (1 / 12) |
| | `dB` | 107: 0 | 69: +0.0126 ± 0.0064 (5 / 1) | 4: 0 | 145: 0 |
| panel 226 (C) | all | 579: +0.0009 ± 0.0009 (1 / 0) | 146: −0.0190 ± 0.0089 (0 / 7) | 24: 0 | 155: −0.0001 ± 0.0001 (0 / 1) |
| | `c1` + `c2` | 359: +0.0014 ± 0.0014 (1 / 0) | 74: −0.0367 ± 0.0173 (0 / 5) | 10: 0 | 9: 0 |
| mixed regime (U) | all | 560: +0.0015 ± 0.0013 (2 / 1) | 254: −0.0093 ± 0.0045 (7 / 16) | 13: −0.020 ± 0.014 (0 / 3) | 353: −0.0025 ± 0.0010 (4 / 11) |
| panel 226 (U) | all | 497: 0 | 192: −0.0107 ± 0.0059 (2 / 7) | 41: 0.000 | 174: −0.0009 ± 0.0004 (2 / 7) |

Sum of the reciprocal-rank differences, mixed regime (P): total −2.55 = first +0.75 (one list), rank 2 to 25 −2.49, absent
−0.81. Panel (C): −2.29 = +0.50, −2.77, −0.02.

**By `lib_max` band**

| set (chain) | lists | < 0.7 | 0.7 to 0.9 | 0.9 to 0.97 | ≥ 0.97 |
|---|---|---|---|---|---|
| mixed regime (P) | all | 767: 0 | 221: 0 | 79: −0.0034 ± 0.0119 (3 / 14) | 113: −0.0202 ± 0.0114 (6 / 14) |
| | `c1` | 25: 0 | 30: 0 | 19: −0.0096 ± 0.0084 (0 / 2) | 76: −0.0018 ± 0.0018 (0 / 1) |
| | `c2` | 274: 0 | 73: 0 | 20: +0.002 ± 0.047 (1 / 4) | 13: −0.083 ± 0.076 (1 / 2) |
| | `dA` | 234: 0 | 59: 0 | 20: −0.0058 ± 0.0046 (2 / 8) | 12: −0.161 ± 0.038 (0 / 10) |
| | `dB` | 234: 0 | 59: 0 | 20: 0 | 12: +0.073 ± 0.032 (5 / 1) |
| panel 226 (C) | all | 473: 0 | 347: 0 | 84: −0.0273 ± 0.0165 (1 / 8) | 0 lists |
| | `c1` / `c2` | 23, 150: 0 | 146, 67: 0 | 57: −0.021 ± 0.015 (0 / 2) / 9: −0.117 ± 0.127 (1 / 3) | 0 lists |

**Inside `lib_max` ≥ 0.9, band by start rank (all types, chain P / C)**

| set | band | first | 2 to 10 | 11 to 25 | lower or absent |
|---|---|---|---|---|---|
| mixed regime | 0.9 to 0.97 | 27: +0.028 ± 0.028 (1 / 0) | 18: −0.044 ± 0.030 (1 / 4) | 2: −0.025 (0 / 2) | 32: −0.005 ± 0.002 (1 / 8) |
| | ≥ 0.97 | 81: 0 (no list moves) | 23: −0.032 ± 0.038 (6 / 9) | 1: −0.917 | 8: −0.080 ± 0.033 (0 / 4) |
| panel 226 | 0.9 to 0.97 | 65: +0.008 ± 0.008 (1 / 0) | 11: −0.252 ± 0.096 (0 / 7) | 0 | 8: −0.002 (0 / 1) |

The answer starts first in 56% of the touched lists of the mixed regime (72% at ≥ 0.97; all lists 50%) and in 77% of the
panel's (all lists 64%).

**Headroom** (`lp_headroom.py`; pool-answer lists `c1` + `c2` at `lib_max` ≥ 0.9, chain P / C):

| set | lists | base MRR | MRR v33b | MRR v40a | answer starts first | … and is not first after v33b / v40a | starts lower and is first after v33b / v40a |
|---|---|---|---|---|---|---|---|
| mixed regime | 128 | 0.8747 | 0.8869 | 0.8761 | 106 | 2 / 1 | 3 / 1 |
| mixed regime, ≥ 0.97 | 89 | 0.9423 | 0.9560 | 0.9423 | 81 | 0 / 0 | 1 / 0 |
| panel 226 | 66 | 0.9278 | 0.9495 | 0.9159 | 59 | 3 / 2 | 4 / 0 |

v33b's stages above the base list are worth +0.012 to +0.022 in these lists locally; v40a returns most of that. What
protection could save (a first place the later stages destroy) is 2 and 3 lists, of which v40a saves 1 and 1; what it gives up is 2 of 3 and 4 of 4 lifted answers.

## 5. v41a (`FP_BANK='ens'`)

`fpnet_full1` is not present: no file matching `*full1*` under `D:\casmi-data\input` or four levels below `D:\casmi-data`;
the fpbank review's statement holds. `'ens'` averages `fpnet_full1` and FPNet A in the main engine's bank; without the
first file the engine cannot be run with that bank, so **v41a was not replayed, and no engine run or forward-model scoring
was started.** What exists on file: every local "public" chain uses FPNet A as the stand-in for full1
(`reviews/fpbank-2026-10-06.md`), and an earlier leaked Kaggle validation of `'ens'` gave −0.002 ± 0.003 with 5 of 250
first places changed (quoted there from `EXPERIMENTS.md`, v3a). A local replay would need the download (198.8 MB, public
dataset `ahmedberatozer/casmi26-fpnet-full1`, the user's approval), main-engine re-runs for both sets with two banks
(`full1` as the new reference, and `ens`), and forward scores for the new candidates. Even then both banks have seen every
mixed-regime molecule and the panel's spectra.

## 6. How much weight these numbers deserve

- **Who has seen the molecules.** Chain P / C: the public FPNet (A) and the public ranker have seen every molecule of both
  sets; our ranker is out of fold and enters only below `lib_max` 0.7, **so in every list v40a touches the base list is the
  leaked public chain alone**. Engine 2: in-sample rankers on the panel in chain P, cross-fitted in chain C; on the mixed
  regime its FPNets have seen the 42% of molecules with collision energy. ICEBERG / GLACIER: very probably everything, but
  their weight is 0 in the touched lists. Popularity and the library search are not models; the local answers are library
  compounds with high popularity. Chain U removes the leak from the base list and gives the same result.
- **The two local losses sit on the side the leak pushes.** Popularity flatters a local answer, and an engine 2 that knows
  the molecule lists a removed answer it would not otherwise have. A hidden-test molecule with a near-perfect library
  match and a wrong popular competitor, or an engine-2 list that displaces a correct first row, is exactly what the forks
  guard against and what the local sets do not contain (section 4, headroom).
- **Track record.** Local replays had the leaderboard's sign in 5 of 11 known contrasts (U regime) to 7 of 11 (leaked
  regime); "always worse" scores 10 of 11 (`reviews/uregime-2026-10-07.md`). A local composite of −0.002 at 1.3 to 1.6
  standard errors, carried by 9 to 37 moving lists, is inside what that record cannot distinguish from zero. One answer
  between place 1 and 2 on the public leaderboard is 0.0038.
- **Pattern check.** v34a (and, as a warning, v35a) gained locally on lists whose answer already starts first and lost or
  was flat where it starts at 2 to 25; v34a then lost 0.011 on the leaderboard. **v40a shows the opposite pattern in
  location and no gain to mistrust**: rank-1 lists are left alone (one list moves), the change is at rank 2 to 25 and it is
  a loss. The U-regime review found rank 2 to 25 the stratum whose signs matched the leaderboard most often (8 and 9 of
  11); that stratification was post hoc there and is post hoc here, and here it rests on 7 to 23 lists.
- **Coverage of the claim.** `FUSE_SKIP_LIB=0.97` is exercised on the mixed regime only (the panel has no list at ≥ 0.97).
  The share of hidden-test molecules at `lib_max` ≥ 0.9 is unknown; locally it is 16% and 9% of the lists, 63% and 25% of
  the class-1 lists.
- **What this adds up to.** The replay is a harm check and it finds a small harm of the kind a library-compound test set
  would produce; it finds no instance of the harm v40a is meant to prevent. The submission rests on outside evidence (one
  fork at 0.426 against 0.421, with `FP_BANK='ens'` changed in the same notebook), and the local numbers neither add to
  that evidence nor outweigh it.

## 7. Limits

- One seed of the out-of-fold ranker; the class-2 tie emulation of the public ranker uses one random draw. Neither matters
  for the touched lists except through the reference of chain U.
- The reference is an emulation of v33b (FPNet A for `fpnet_full1`, out-of-fold ranker), not the submitted files.
- "Lists that move" counts lists whose reciprocal rank changes; a reorder below or above the answer that leaves its rank
  alone is not counted (the replay stores ranks, not lists).
- `dA` / `dB` lists at `lib_max` ≥ 0.9 are constructed cases (the answer is removed from the pool while a near-identical
  library spectrum of another structure remains); 32 and 9 lists. Panel `dA` / `dB` are not answer-outside lists in two of
  five cases (`reviews/panel2-2026-10-05.md`).
- Composite weights are equal by decree; the hidden class mix is unknown.
- `GATE_TAU` was left at its default (`LIB_TAU` 0.9), as in v40a; no other value was replayed.
- Not done: v41a; a look at the individual moving lists; `INDEX.md`, `EXPERIMENTS.md`.
- Corrected on the way: a shell here-document broke on quoting (the report script was written with the file tool); the
  first composite counted molecules without a class-1 list as zero (replaced by the mean of the four type means).

## 8. Reproduce

```bash
export CASMI_DATA='D:\casmi-data' PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1; P=.venv/Scripts/python; S=scratch/libprotect
# once: copy D:/casmi-data/uregime/fwd_<reg>/{ice,gl}.json and frag_<reg>.json to D:/casmi-data/libprotect/ (same layout)
for r in mx p2; do $P $S/lp_replay.py --reg $r --phase cands --vars P:ref,C:ref,U:ref; done   # "to score 0", "missing 0"
for r in mx p2; do $P $S/lp_replay.py --reg $r --phase replay; done                           # replay_<reg>.parquet (appends, resumable); on mx chain C equals P
$P $S/lp_report.py; $P $S/lp_headroom.py                                                      # report.md, report.json
```

Read: `D:\casmi-data\uregime\` (score caches, copied), `rankbank3\cvmodels\base`, `rankbank3\data\mols.parquet`,
`regime\eng2\p2_*_pubcf`, `regime\cf_folds.parquet`, `pcjoin`, `panel2\join\pub`, `rankbank2\join\*_pub`, `mixed_regime`,
`panel2\regime`. Written: `D:\casmi-data\libprotect\` only. This file is new: run `python tools/gen_index.py` when it is
committed.

<!-- index: fusion-v4n 2026-10-05 variants of the join with the fine-tuned selection (ft_hn): N 25 / 50 / 150, the 5 ppm corrected PubChem window, the union of the first 50 of both selections, against N 100 with the current and with the ft_hn selection; local engine before the fusion, per type, with exposure -->
# Variants of the join with the fine-tuned selection (2026-10-05)

Offline only. Nothing was submitted, pushed or committed, no `kaggle` command and no notebook was run, the GPU was not
used, the visible `test.parquet` was not read. Continuation of `reviews/hardneg-2026-10-05.md` section 6 with the same
tools. Scripts: `scratch/hnvar/` (git-ignored). Outputs: `D:\casmi-data\hnvar\` (0.3 GB, all created here; nothing
else under `D:\` was written or deleted).

Marks: **[M]** measured in this study; **[E]** re-used from the earlier report (in every case I also recomputed it
from the earlier runs' stored rows and got the same number); **[C]** a count that does not pass through the ranker;
**[R]** local engine MRR@25, the kind of number that has not predicted the leaderboard for changes in the joined
candidates. "±" is one standard error of a paired per-molecule difference, (wins / losses) in brackets.

## 0. Bottom line

1. **No variant is better than "N 100 with the ft_hn selection" on our local metrics.** Every difference against it
   is within about 1.5 standard errors on every type, and each variant that gains on one type pays on another.
2. **N 50 with ft_hn is the only real alternative, and it is a trade, not an improvement.** It keeps 72% of the
   answer-removed gain over v13a (+0.020 ± 0.009 against +0.028 ± 0.009 [R]) with the exposure of the old N 50
   (joined rows in the first 25 of a `c2` list 10.18 → 9.42, joined row on top 9.6% → 8.4%; `c1` +0.006 ± 0.005,
   5 / 0). Its 48 candidates hold the answer more often than the 96 of v13a (74.2% against 70.5% [C]).
3. **N 25: too few.** Answer-removed −0.018 ± 0.008 against ft_hn N 100 and only +0.010 ± 0.010 against v13a.
4. **N 150: locally a little better everywhere, and that is exactly the reading that was wrong for N 200.** Answer
   removed +0.007 ± 0.006, `c2` +0.005 ± 0.004; exposure up (10.18 → 10.45 rows, 9.6% → 10.1% on top). Not proposed.
5. **5 ppm + offset on top of ft_hn: no gain on the whole regime, a gain where the answer is within 5 ppm, and more
   exposure.** Answer removed +0.004 ± 0.009 (within 5 ppm: +0.011 ± 0.008; timsTOF stratum, 39 molecules:
   +0.062 ± 0.041), `c2` −0.003 ± 0.004 (22 / 31), joined rows in the first 25 10.18 → 10.41, on top 9.6% → 10.3%.
   Most of what the 5 ppm window gave the current selection (the shortlist binding less often) ft_hn already has.
6. **Union of the two first-50 lists: dominated.** 64 candidates that hold the answer exactly as often as the 48 of
   ft_hn alone (74.2% [C]; the current selection's first 50 add no answer), with more exposure than ft_hn N 50.
7. **Recommendation**: keep `v22a` (`PC_MODELS=hn`, `PC_JOIN_N=100`) as the main candidate; `v22b` (N 50) is the one
   variant worth a second submission, as a lower-exposure hedge; do not build configs for N 25, N 150 or the union;
   do not stack `PC_PPM=5` on `hn` before `v21a` (5 ppm on the current selection) has a leaderboard score.

## 1. Headline table

Local v1 engine, public ranker, join gate `lib_max` < 0.7, rank of the answer **before the fusion**. Three types:
answer removed from the pool (dA / dB; 325 molecules with a PubChem record), answer a pool row without spectra
(`c2`, 380), class-1-like (`c1`, 150; 25 below the gate). Exposure on `c2`: joined rows among the first 25 structures
and share of lists with a joined row on top. All rows [M] except the two references, which are [E] for answer-removed
and `c1` and [M] for `c2` (section 3 explains why `c2` had to be re-scored).

| Variant | rows handed | answer among them [C] | answer removed: MRR [R] | minus ft_hn N 100 | minus cur N 100 | `c2`: MRR [R] | minus ft_hn N 100 | `c2` joined in first 25 | `c2` joined on top | `c1`: MRR [R] | minus ft_hn N 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N 0 (no join) [E] | 0 | | 0.067 | | | 0.678 | | 0 | 0 | 0.800 | |
| **cur N 100** (v13a, LB 0.431) | 96 | 70.5% | 0.307 | −0.028 ± 0.009 (16 / 44) | | 0.663 | +0.000 ± 0.004 (41 / 33) | 10.16 | 9.8% | 0.789 | +0.002 ± 0.001 (4 / 1) |
| **ft_hn N 100** (v22a) | 96 | 77.8% | 0.335 | | +0.028 ± 0.009 (44 / 16) | 0.662 | | 10.18 | 9.6% | 0.787 | |
| ft_hn N 25 | 24 | 70.5% | 0.317 | −0.018 ± 0.008 (23 / 39) | +0.010 ± 0.010 (40 / 33) | 0.660 | −0.002 ± 0.005 (51 / 48) | 8.41 | 7.1% | 0.798 | +0.011 ± 0.007 (5 / 0) |
| ft_hn N 50 (v22b) | 48 | 74.2% | 0.327 | −0.008 ± 0.005 (22 / 25) | +0.020 ± 0.009 (40 / 21) | 0.663 | +0.001 ± 0.005 (43 / 43) | 9.42 | 8.4% | 0.794 | +0.006 ± 0.005 (5 / 0) |
| ft_hn N 150 | 142 | 80.3% | 0.342 | +0.007 ± 0.006 (27 / 21) | +0.034 ± 0.009 (51 / 17) | 0.667 | +0.005 ± 0.004 (48 / 36) | 10.45 | 10.1% | 0.787 | 0.000 ± 0.002 (1 / 2) |
| ft_hn N 100, 5 ppm + offset | 94 | 78.5% | 0.339 | +0.004 ± 0.009 (21 / 20) | +0.032 ± 0.011 (51 / 24) | 0.659 | −0.003 ± 0.004 (22 / 31) | 10.41 | 10.3% | 0.791 | +0.004 ± 0.005 (1 / 2) |
| ft_hn N 50, 5 ppm + offset | 48 | 74.5% | 0.332 | −0.003 ± 0.008 (25 / 32) | +0.025 ± 0.011 (45 / 27) | 0.660 | −0.003 ± 0.005 (38 / 45) | 9.78 | 9.0% | 0.792 | +0.005 ± 0.005 (3 / 2) |
| union, first 50 of ft_hn + first 50 of cur | 64 | 74.2% | 0.326 | −0.009 ± 0.005 (22 / 29) | +0.018 ± 0.009 (37 / 19) | 0.662 | −0.001 ± 0.004 (47 / 42) | 9.82 | 8.6% | 0.793 | +0.006 ± 0.005 (4 / 0) |
| cur N 25 (existing runs) | 24 | 59.1% [E] | 0.274 | −0.061 ± 0.013 (20 / 57) | −0.033 ± 0.010 (21 / 39) | 0.666 | +0.004 ± 0.006 (51 / 47) | 8.30 | 6.4% | 0.798 | +0.011 ± 0.007 (5 / 0) |
| cur N 50 (v12a, LB 0.420; existing runs) | 48 | 65.2% | 0.292 | −0.043 ± 0.011 (17 / 42) | −0.016 ± 0.007 (20 / 24) | 0.669 | +0.006 ± 0.005 (53 / 40) | 9.42 | 7.9% | 0.798 | +0.010 ± 0.007 (5 / 0) |
| cur N 150 | 142 | not counted | 0.319 | −0.016 ± 0.009 (25 / 39) | +0.011 ± 0.005 (24 / 12) | 0.668 | +0.006 ± 0.005 (55 / 31) | 10.45 | 10.5% | 0.788 | 0.000 ± 0.001 (2 / 2) |

"Rows handed" and "answer among them" are over the 325 record molecules, answer counted as outside the pool
(`sel_counts.py`); below the gate the engine receives the same lists.

How to read it against the leaderboard. The three scored points of the current selection are N 50 = 0.420,
N 100 = 0.431, N 200 = 0.406. Locally the step 50 → 100 is answer-removed +0.016, `c2` −0.006, `c1` −0.008, and the
step 100 → 150 is +0.011, +0.006, −0.002. So the local table would have ranked N 150 above N 100 and had no opinion
between 50 and 100. **The local `c2` / `c1` columns do not price the pool-side cost**; the exposure columns are the
only local signal that moved with the leaderboard loss at depth, and they are monotone in N for both selections.

## 2. The questions

### 2.1 Does a better selection allow fewer joined rows?

Partly. The gain on answer-removed molecules does not survive halving N intact, because the ranker's gain comes from
answers at any depth of the list, not only from the first 25:

| ft_hn | answer among the candidates [C] | answer-removed gain over N 0 [R] | share of the N 100 gain | `c2` joined in first 25 | `c2` on top | `c1` joined in first 25 | `c1` on top |
|---|---|---|---|---|---|---|---|
| N 25 | 70.5% | +0.251 | 93% | 8.41 | 7.1% | 1.63 | 5.3% |
| N 50 | 74.2% | +0.261 | 97% | 9.42 | 8.4% | 1.84 | 5.3% |
| N 100 | 77.8% | +0.268 | 100% | 10.18 | 9.6% | 1.97 | 6.0% |
| N 150 | 80.3% | +0.275 | 103% | 10.45 | 10.1% | 2.02 | 6.0% |
| cur N 50 | 65.2% | +0.225 | 84% | 9.42 | 7.9% | 1.90 | 5.3% |
| cur N 100 | 70.5% | +0.241 | 90% | 10.16 | 9.8% | 2.00 | 5.3% |

All [M] except the two N 100 answer-removed values [E]. Two readings:

- **ft_hn N 50 against v13a (cur N 100)**: more answers handed (74.2% against 70.5%), answer-removed +0.020 ± 0.009
  (40 / 21), `c2` +0.000 ± 0.005, `c1` +0.004 ± 0.004 (2 / 0), and less exposure on both pool-side types (9.42 against
  10.16 rows; 8.4% against 9.8% on top). On every local column it is at or above v13a.
- **ft_hn N 50 against ft_hn N 100**: −0.008 ± 0.005 on answer-removed (22 / 25), +0.006 ± 0.005 on `c1` (5 / 0),
  nothing on `c2`, 0.76 fewer joined rows in the first 25 and 1.2 points fewer lists with a joined row on top. The
  exposure is that of cur N 50, which scored 0.011 below cur N 100: with the current selection the leaderboard
  preferred the extra answers of N 100 to the lower exposure of N 50. With ft_hn the extra answers between 50 and 100
  are fewer (3.6 points against 5.3), so the balance is closer, and nothing local can settle it.
- **At equal N the fine-tune's rows are not easier decoys.** ft_hn N 50 against cur N 50: answer-removed
  +0.035 ± 0.010 (41 / 19), `c2` −0.006 ± 0.004 (31 / 39), `c1` −0.004 ± 0.005 (1 / 2); joined rows in the first 25
  equal (9.42), joined row on top 8.4% against 7.9%. At N 150: +0.023 ± 0.009, −0.001 ± 0.004, 0.000. The pool-side
  sign at N 50 is 1.6 standard errors; it is the cost one would expect from candidates that look more like measured
  structures, and it is the reason N 50 with ft_hn is not simply "v12a's exposure with more answers".
- The exposure saving of a smaller N is modest: the ranker fills the first 25 with joined rows almost as readily
  from 48 as from 96 (9.4 against 10.2). Halving N does not halve the exposure.
- On the timsTOF stratum of `c2` (40 molecules, 28 below the gate) the sign is the other way: N 50 has a joined row
  on top for 9.7% of the lists against 3.7% at N 100 (MRR −0.030 ± 0.017, 0 / 3); the current selection shows the same sign more weakly (5.9% against 3.7%). Small sample;
  noted because it is the test's instrument and agrees in sign with N 100 beating N 50 on the leaderboard.

### 2.2 The 5 ppm corrected window with ft_hn

`sel_w5.py`: `PC_PPM=5.0`, `MASS_OFFSET_BY_ADDUCT='enveda'` (table `mass_window.ENVEDA_PPM`), both passes of the
channel redone with ft_hn's logits. As in the mass-calibration study the offset is applied to the queries of the
Enveda libraries only (40 of the 380 molecules); the other 340 get a plain 5 ppm window, and 7.1% of the record
molecules then lose an answer to instrument error that does not exist on the test's instrument. The rows "answer
within 5 ppm" are the closer reading.

**The cached windows were sufficient for every molecule.** The cache holds rows [a, a + n) of the mass-sorted tier
for target ± 10 ppm. The corrected target differs from the plain one by at most 2.22 ppm, so
[t' − 5 ppm, t' + 5 ppm] ⊂ [t − 7.3 ppm, t + 5 ppm] always lies inside it. I did not rely on the argument: the
script recomputes the row range of the new window on the tier's mass array for each of the 380 molecules, asserts
that the cached target equals the recomputed plain target, and checks that the new range is inside the cached one:
**0 of 380 needed a recomputed window**; the 5 ppm window is an exact slice of cached rows. Window size 7,585 → 4,962
rows, windows above the 5,000 shortlist 57.9% → 38.4%, both equal to the mass-calibration study's real run.

| 325 record molecules [C], [M] | ft_hn 10 ppm | ft_hn 5 ppm + offset | difference (gained / lost) |
|---|---|---|---|
| answer in the tier window | 91.4% | 89.2% | |
| answer in the 5,000 shortlist | 87.4% | 88.0% | |
| answer among the first 50 | 74.2% | 74.5% | +0.003 ± 0.011 (7 / 6) |
| answer among the first 100 | 77.8% | 78.5% | +0.006 ± 0.012 (8 / 6) |
| ... answer within 5 ppm of the corrected target (302) | 81.8% | 84.4% | +0.026 ± 0.009 (8 / 0) |
| ... natural-product catalogue (256) | 75.0% | 74.2% | −0.008 ± 0.011 (3 / 5) |
| ... enveda-180, timsTOF (39) | 92.3% | 100% | +0.077 ± 0.043 (3 / 0) |
| shared structures among the first 100 | | 77.9 of 100 | |

With the current selection the same change gave +4.3 points at N 100 on all 325 and +6.0 on the within-5-ppm
molecules [E, mass-calibration study]; with ft_hn it gives +0.6 and +2.6. The fine-tune already keeps the answer in
the shortlist (87.4% against 83.7%), which was the main mechanism of the window's gain.

Engine, N 100, against ft_hn N 100 at 10 ppm [R], [M]:

| Type | n | MRR 10 ppm | MRR 5 ppm | difference | joined in first 25 | joined on top |
|---|---|---|---|---|---|---|
| answer removed, record | 325 | 0.335 | 0.339 | +0.004 ± 0.009 (21 / 20) | 11.06 → 11.40 | 41.8% → 43.1% |
| ... answer within 5 ppm | 302 | 0.354 | 0.365 | +0.011 ± 0.008 (21 / 16) | 10.75 → 11.09 | 41.4% → 42.4% |
| ... natural-product catalogue | 256 | 0.277 | 0.269 | −0.008 ± 0.007 (13 / 17) | 9.73 → 10.05 | 34.4% → 35.2% |
| ... enveda-180 (timsTOF) | 39 | 0.532 | 0.594 | +0.062 ± 0.041 (5 / 1) | 15.49 → 15.90 | 71.8% → 71.8% |
| answer removed, no record | 55 | 0.228 | 0.226 | −0.002 ± 0.002 (1 / 2) | 11.84 → 11.53 | 36.4% → 36.4% |
| `c2` all | 380 | 0.662 | 0.659 | −0.003 ± 0.004 (22 / 31) | 10.18 → 10.41 | 9.6% → 10.3% |
| ... answer within 5 ppm | 350 | 0.713 | 0.710 | −0.003 ± 0.004 (21 / 29) | 9.66 → 10.01 | 6.2% → 6.3% |
| ... enveda-180 (timsTOF) | 40 | 0.948 | 0.942 | −0.006 ± 0.006 (0 / 1) | 13.72 → 14.01 | 3.7% → 5.0% |
| `c1` | 150 | 0.787 | 0.791 | +0.004 ± 0.005 (1 / 2) | 1.97 → 2.06 | 6.0% → 5.3% |

The same picture as with the current selection, smaller: a coverage gain concentrated on the test's instrument
(39 molecules), a small negative sign on `c2`, and the exposure moves the wrong way (the window swaps other-formula
joined rows for same-formula isomers). At N 50 the combination is not better than ft_hn N 50 either (answer-removed
+0.005 ± 0.008 (14 / 18) against it, `c2` −0.003 ± 0.003 (18 / 27), exposure 9.42 → 9.78).

### 2.3 Depth by rank agreement: union of the two first-50 lists

`build_dirs.py`: first 50 of ft_hn and first 50 of the current selection, interleaved (ft_hn first), one row per
metric key. The two lists share 34 structures, so the union has 66 rows on average (57 / 66 / 74 at 10 / 50 / 90%),
not 100. For `c2` / `c1` the union is built after the answer is removed from both lists.

- [C] The union holds the answer for 74.2% of the record molecules: exactly ft_hn N 50. The current selection's
  first 50 contribute **no answer** that ft_hn's first 50 lack (0 gained), and 16 extra decoys per list.
- [R] Against ft_hn N 100: answer-removed −0.009 ± 0.005 (22 / 29), `c2` −0.001 ± 0.004, `c1` +0.006 ± 0.005 (4 / 0).
  Against ft_hn N 50: answer-removed −0.002 ± 0.004 (16 / 20), `c2` −0.001 ± 0.004, exposure 9.42 → 9.82 rows and
  8.4% → 8.6% on top.
- It is ft_hn N 50 with more decoys. This agrees with the earlier report's "nothing for an ensemble". The variant in
  the other direction (keep only the rows both models put in their first 100, about 68) was not run.

## 3. The `c2` re-scoring (why the `c2` reference numbers differ slightly from the earlier report)

`e2e_eval.py` emulates a class-2 answer by moving the answer to the position of a random COCONUT-only row of its
formula, with one random stream per run. That is only paired between runs that hold the same molecules. The new runs
hold the 274 molecules below the gate (the other 106 keep their N 0 list by construction), so with that method their
`c2` differences carry draw noise: ft_hn N 50 minus ft_hn N 100 came out as −0.012 ± 0.010 [M], pure noise.

`hn_eval.py --c2 mol` draws per molecule, seeded by its id (the fix of `scratch/masscal/s3_eval.py`), and averages
8 draws; every run, including the references, is scored the same way. Consequences:

- With the original method I reproduce the earlier report exactly: answer-removed 0.3074 / 0.3349,
  +0.0276 ± 0.0085 (44 / 16); `c2` 0.6708 / 0.6719, +0.0012 ± 0.0040 (13 / 20); `c1` 0.7892 / 0.7873,
  −0.0019 ± 0.0013 (1 / 4); exposure columns identical.
- With the per-molecule draws `c2` is: N 0 0.678, cur N 100 0.663, ft_hn N 100 0.662, difference
  −0.000 ± 0.004 (33 / 41) instead of +0.001 ± 0.004. Same conclusion, different third decimal. A single draw gave
  −0.004 ± 0.004 (12 / 24): the draw itself is worth about ±0.004 on these 380 molecules.
- Wins / losses in the `c2` columns count molecules whose 8-draw mean differs, so they are larger counts than the
  earlier report's.
- Answer-removed and `c1` use no emulation; those numbers are the earlier method unchanged.

## 4. Recommendation

On our local metrics **nothing here is better than N 100 with the ft_hn selection**. Side by side, each variant
against it (answer-removed MRR / `c2` MRR / `c1` MRR; `c2` joined in first 25; `c2` joined on top):

| Variant | answer removed | `c2` | `c1` | `c2` rows in first 25 | `c2` on top | verdict |
|---|---|---|---|---|---|---|
| ft_hn N 100 | 0.335 | 0.662 | 0.787 | 10.18 | 9.6% | **keep (v22a)** |
| ft_hn N 50 | −0.008 ± 0.005 | +0.001 ± 0.005 | +0.006 ± 0.005 | 9.42 | 8.4% | trade; the only hedge worth a submission (v22b) |
| ft_hn N 25 | −0.018 ± 0.008 | −0.002 ± 0.005 | +0.011 ± 0.007 | 8.41 | 7.1% | no: loses 2 standard errors on the gain side |
| ft_hn N 150 | +0.007 ± 0.006 | +0.005 ± 0.004 | 0.000 ± 0.002 | 10.45 | 10.1% | no: the direction the leaderboard punished |
| ft_hn N 100, 5 ppm + offset | +0.004 ± 0.009 | −0.003 ± 0.004 | +0.004 ± 0.005 | 10.41 | 10.3% | not now; wait for v21a |
| union 50 + 50 | −0.009 ± 0.005 | −0.001 ± 0.004 | +0.006 ± 0.005 | 9.82 | 8.6% | no: ft_hn N 50 with more decoys |

- **Main candidate: `v22a` unchanged.**
- **`v22b` (N 50)** is defensible as a second submission only because it is at or above v13a on every local column
  with less exposure than v13a. It is not predicted to beat `v22a`; the one leaderboard comparison we have at these
  depths (N 50 against N 100 with the current selection) went to N 100.
- **5 ppm**: `v21a` isolates the window on the current selection. If it scores at or above v13a, `hn` + 5 ppm is a
  reasonable next config (the gain is concentrated on timsTOF molecules, 39 locally); if it scores below, do not
  stack it. No config was written for it.
- No leaderboard estimate is given for any row.

## 5. What was run, limits, what failed

Runs (`scratch/hnvar/queue.sh`, one worker per call, at most four at a time, below-normal priority, 2.3 to 3.4 s per
molecule): ft_hn at N 25 / 50 / 150, ft_hn with the 5 ppm window at N 100 / 50, the union, and the current selection
at N 150, each for the three versions, on the molecules below the gate only (274 + 274 + 25 per variant; 4,011
molecule-runs, 0 engine errors). Re-used: N 0, cur N 25 / 50 / 100 (`D:\casmi-data\pcjoin\rows`), ft_hn N 100
(`D:\casmi-data\hardneg\join_ft_hn\rows`), ft_hn's lists and logits, the window cache, the metric-key cache.

- **The engine's bank is the fold-safe FPNet in every run**, as in section 6 of the earlier report; only the joined
  rows differ. Production uses public A + B in the engine and the fine-tuned A / B for the selection; that
  combination cannot be evaluated on unseen molecules locally.
- **Before the fusion only.** The fusion replay (engine-2, slot merge, ICEBERG prefix) was not run for any variant.
- The regime is 40 timsTOF molecules of 380; the 5 ppm rows for the test's instrument rest on 39 to 40 molecules.
- One fine-tune, one seed; the union was tried in one construction; N 150 is the only depth above 100.
- The answer-removed type uses lists of 200; 33 of the 380 molecules have fewer than 150 candidates (15 have none).
- **Machine.** Background shells were stopped by the harness after 30 minutes (I had not set the longer limit on the
  first four). Two runs lost their worker at a memory restart and were resumed (resumable by design, no molecule
  lost: every variant was checked for all molecules below the gate before it entered a table). While re-queuing I
  started one job (`cur drop 150`) that the surviving old queue also started: **for about seven minutes, 12:22 to
  12:29, the same job had two workers, and for roughly two of those minutes five engine processes ran instead of
  four.** I killed the duplicate; the run's chunks are complete and molecules are counted once.
- Not done: `tools/gen_index.py` (INDEX.md was not regenerated for this file), `docs/experiments.md` (no submission).

```bash
export CASMI_DATA='D:\casmi-data' PYTHONHASHSEED=0 OMP_NUM_THREADS=1; P=.venv/Scripts/python; S=scratch/hnvar
$P $S/build_dirs.py                                   # data roots: ft_hn, cur, union, union_in
$P $S/sel_w5.py                                       # ft_hn selection, 5 ppm + offset, from the cached windows (3 min)
bash $S/queue.sh A "ft_hn drop 50" "ft_hn c2 50" "ft_hn c1 50" "ft_hn drop 150"      # and B..G likewise; resumable
$P $S/sel_counts.py                                   # counts [C]
$P $S/hn_eval.py --c2 run                             # answer-removed, c1; c2 with the earlier method (reproduction)
$P $S/hn_eval.py --c2 mol --draws 8 --vers c2         # c2 as reported here
```

Files: `D:\casmi-data\hnvar\eval_run.md`, `eval_mol.md`, `sel_counts.md` (full tables with the natural-product and
within-5-ppm subsets), `eval\permol_*.parquet`, `sel\regimeall_ft_hn_w5*`, `join_*\rows\`.

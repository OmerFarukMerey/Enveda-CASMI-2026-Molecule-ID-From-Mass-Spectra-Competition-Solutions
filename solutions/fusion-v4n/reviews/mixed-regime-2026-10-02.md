<!-- index: fusion-v4n 2026-10-02 mixed validation regime: engine-2 and the PubChem channel run locally on 380 hold-out molecules with the truth in the pool and removed from it, why every earlier validation run mis-valued engine-2 (its candidate files contain no training structure), mixtures of molecule types against the eight LB points, and which flagged changes the regime supports -->
# Mixed regime (2026-10-02): truth sometimes outside the pool, all channels run

Offline only. Nothing was submitted, pushed or committed, no Kaggle notebook was run, no existing file was edited.
Scripts: `scratch/mixed/` (git-ignored, log in `scratch/mixed/LOG.md`). Reusable code with a test:
`solutions/fusion-v4n/prototypes/mixed_regime.py`. Data and outputs: `D:\casmi-data\mixed_regime\` (1.6 GB).
MRR = MRR@25 on the metric key. "±" is one standard error; differences are paired per molecule; mixture errors are
clustered by molecule (a molecule's versions are one cluster). Reproduce: section 8.

**How to read the numbers.** Every molecule here is a training structure with relatives in the public libraries, and
its truth was hidden by masking, not by absence (section 7). A local gain is not a leaderboard gain. Where a number is
put on the leaderboard scale, the mixture it assumes is named next to it. Two replays of the fusion are reported:
**zero-forward** (all 380 molecules, ICEBERG / GLACIER terms at zero) and **ICEBERG prefix** (the first 169 molecules,
real ICEBERG scores, no GLACIER); section 4 says why.

## 0. Bottom line

1. **The old harness could not value engine-2 because engine-2 never had the answer there.** prvsiyan's COCONUT and
   ChEBI / LIPID MAPS files contain no training structure (40 keys of 275,810; quercetin, caffeine, rutin are absent).
   Training structures enter engine-2's candidate set only from the `train.parquet` it is given, and a class-2
   validation run deletes the hold-out structures from that file. Truth in engine-2's top 40 on the Kaggle stage
   caches: 3.0% (valA), 0.9% (valF), 8.0% (val1), against 99.6% on valc1. So "gate engine-2 off = +0.07 to +0.14"
   measured a channel that could only add wrong candidates. On the hidden test a class-2 answer is a COCONUT row of
   engine-2 like any other (section 2).
2. **Built**: engine-2 (ported to Windows, 4.7 s per molecule on one thread), the PubChem channel (fold-safe FPNet,
   2 to 15 s per molecule) and the existing v1 rows for 380 hold-out molecules in a truth-in-pool and a truth-removed
   version, 150 of them also class-1-like; ICEBERG for 139 of the 306 covered molecules (the GPU was shared with the
   user's games, section 4), GLACIER not at all; the final fusion replayed with `fusion_core.build_submission`.
3. **The regime reproduces the v7b sign, in both replays.** v7b minus v4b per type (zero-forward / ICEBERG prefix):
   class 1 −0.002 ± 0.005 / −0.011 ± 0.008; class 2 with the truth in both engines −0.010 ± 0.006 / +0.008 ± 0.012;
   truth outside the v1 pool but among engine-2's candidates −0.088 ± 0.008 / −0.179 ± 0.023; truth in neither +0.018
   ± 0.004 / −0.008 ± 0.009; and **+0.149 ± 0.013 / +0.113 ± 0.018 for the old harness's situation** (truth in the v1
   pool, absent from engine-2). Under three mixtures scaled to the leaderboard level the difference is −0.003 to
   0.000 ± 0.002 (zero-forward) and −0.011 to −0.001 ± 0.005 (ICEBERG prefix) against −0.004 on the LB; the old
   harness gives +0.116 ± 0.010 (section 5).
4. **It does not reproduce the order of all eight points.** v5a below v4b comes out with the right sign and at most a
   third of the size (−0.002 to −0.006, within one error of zero). **v3b below v2 is not reproduced**: with ICEBERG
   v3b − v2 is positive for every one of 1,771 weight vectors (+0.002 to +0.003 ± 0.004 to 0.006 under the named
   mixtures), and without it −0.002 ± 0.003 at best (LB −0.014). In the zero-forward replay 299 weight vectors pass
   all three signs, best Spearman +0.80 and 11 of 12 pairs; on the 169-molecule prefix none does, with or without
   ICEBERG. The eight points do not identify a mixture, and rank correlation is a weak judge: with ICEBERG the old
   harness also reaches +0.77 while being wrong by +0.09 on v7b.
5. **Engine-2's contribution** (section 5.3): when the v1-side list lacks the truth and engine-2 may have it, engine-2
   lists it in its top 40 for 83 to 85% of those molecules, at rank 1 to 2 in the median. The fusion then puts it in
   the top 25 for 90%, in the top 5 for 30% (52% with ICEBERG), first for 0% (6%); median rank 7 (5). Value of the
   fusion, zero-forward / ICEBERG prefix: +0.149 ± 0.009 / +0.267 ± 0.022 there, +0.017 ± 0.007 / −0.005 ± 0.015
   for class 2 in both pools, −0.002 ± 0.008 / +0.012 ± 0.014 for class 1, −0.042 ± 0.006 / +0.003 ± 0.014 when the
   truth is in neither.
6. **Flagged changes** (section 6; mixtures A / B / C of section 5.2, which are assumptions): supported by both
   replays: `RANKER='blend'` with `RANKER_LIB_GATE=0.7` (+0.005 to +0.011), `'public_free'` with the gate (+0.003 to
   +0.006), `SLOTS_GENTLE=[3,6,9,12,15]` (+0.002 to +0.004) and, where ICEBERG exists, `PC_FWD_T1=0.0` (+0.003 ±
   0.001; both together +0.005 ± 0.002). Rejected by both: `FUSE_MIN_LIB` at any value, `FUSE_ALPHA` 0.8 and 1.0
   (−0.002 to −0.013). Not supported: the soft library gate (−0.002 to −0.004 ± 0.002 on the prefix; ICEBERG is
   optimistic there and GLACIER is missing, so this is weak). A detector-gated merge is worth no more than moving the
   PubChem slots for everyone, unless more than about a tenth of the hidden molecules have their answer only among
   engine-2's candidates.

## 1. Molecules (step 1: GO)

`s1_select.py` draws from the 800 hold-out molecules of the ranker study that exist with the truth in the v1 pool
(`c2_fold0`) and with its pool row removed (`drop_fold0`); the fold-safe FPNet has never seen them.

| Stratum | n | PubChem record | median popularity | median mass | spectra per molecule | ICEBERG-covered | class-1 subset |
|---|---|---|---|---|---|---|---|
| natural-product catalogue (`pool_np > 0`) | 260 | 98.5% | 5.7 | 371 | 3.8 | 78% | 141 |
| no PubChem record, natural-product libraries | 50 | 0% | 0 | 419 | 2.2 | 86% | 3 |
| enveda-180 (synthetic, timsTOF) | 40 | 97.5% | 1.4 | 326 | 4.4 | 90% | 0 |
| other | 30 | 100% | 3.2 | 327 | 5.0 | 90% | 6 |
| **all** | **380** | **325 (85.5%)** | | | 3.7 (1,423 spectra) | 82% | **150** |

Query instruments: Orbitrap 171, QTOF 131, timsTOF 40, other 38; polarity: positive 237, both 95, negative 48; query
libraries: gnps 130, pluskal_ms2 68, enveda-180 40, riken 35, spectraverse 34, mona 30, msdial 20, massbank 17, others 6.
The class-1 subset is 150 of the 158 selected molecules that keep spectra in a second library (random choice).

v1 rows taken from the ranker study (`build_base.py`): truth-in-pool version 373 molecules with candidates (truth among
them 96.0%, median `lib_max` 0.47), class-1-like 149 (94.0%, 0.97), truth-removed 372 (truth generated 17.5%, 0.48).
Base orderings: class 1 and truth-removed lists as the notebook scores them; truth-in-pool lists with the truth placed
where a COCONUT-only row sits (`rk_c2`, the fair emulation of a hidden class-2 answer; ranker review section 2).

## 2. Engine-2 locally (step 2: GO)

**Port.** `scratch/mixed/eng2_local.py` imports `nb/embed/eng/casmi_engine.py`, `pv.py` and `pv_fp.py` unchanged and
replaces `eng_runner.py` and `casmi_engine.main`: no `/kaggle` paths, `spawn` pools with their own worker functions,
checkpoints loaded with `weights_only=True`, the `*_meta.pkl` files through the restricted unpickler, the ranker rows
without `allow_pickle`, every stage cached (training fingerprints 165 s on 3 workers, the 12 HistGradientBoosting
rankers 110 s, fragment masses, canonical keys, a checkpoint every 40 molecules). The library is `train.parquet` read
by engine-2's own loader (2,539,608 spectra, 401.8 M peaks; 5.6 GB resident after `release_unused()`), purged in
memory by metric key. Pool: 774,943 structures = COCONUT 436,389 + ChEBI / LIPID MAPS 62,744 + training 275,810.
Knobs as in `nb/00_config.py` (`ENG_*`). Downloaded: `prvsiyan/casmi26-fp-models-v2` (288 MB),
`prvsiyan/coconut-casmi26-candidates` (423 MB), `prvsiyan/chebi-lipidmaps-casmi26` (60 MB); the two rank-row files
were already local.

**What engine-2's candidate set is, and what "removed" means.**

| Check | Result |
|---|---|
| training structures (275,810; raw or metric key) in engine-2's COCONUT file / ChEBI-LIPID MAPS file | 30 / 10 keys |
| v1 pool rows that are COCONUT-only (436,515) with their key in engine-2's COCONUT file | 89.6% by key; of the rest, 87% of a sample of 400 match by raw InChIKey14 (engine-2 keys are not tautomer-canonical), so about 1.4% are really missing |
| engine-2 COCONUT rows whose key is not a v1 pool key (45,471) | 96.5% of a sample of 400 are v1 pool structures under the canonical key |
| ChEBI / LIPID MAPS rows not in the v1 pool | 62,671 of 62,744 by key; 99.3% of a sample of 400 after canonicalisation |
| truth in engine-2's top 40, Kaggle stage caches | valA 3.0%, valF 0.9%, val1 8.0% (class-2 purge); valc1 99.6% (class-1 purge) |

So engine-2's candidate set = COCONUT without the training structures + ChEBI / LIPID MAPS + whatever structures the
`train.parquet` of the run contains. Its real surplus over the v1 pool is ChEBI / LIPID MAPS (62.7k structures); the
v1 pool's surplus over it is nil. The directions review explained engine-2's 7.6% on val1 by "it cannot rank a
structure without reference spectra"; the structure was not there to be ranked. Consequences for the variants:

- **`in`**: the truth is a candidate of engine-2. Its training rows are kept in the pool although its spectra are
  purged; in the class-2 run they are moved directly behind the COCONUT / ChEBI / LIPID MAPS rows of the same mass
  (a hidden class-2 answer is a COCONUT row). Used for class 2 with the truth in the pool and for variant (a).
- **`out`**: every candidate with the truth's metric key (raw-key twins, and tautomer twins found by the canonical key)
  is removed before the features are computed. Used for variant (b), and with the truth-in-pool base list it is
  exactly what the Kaggle validation runs did (type `c2x` below).
- Variant (a) therefore stands for "the answer is a ChEBI / LIPID MAPS structure that is not in COCONUT", variant (b)
  for "PubChem-only or novel".
- In every variant the truth's spectra are absent from engine-2's library (class 2: 9,396 spectra of the 380 molecules
  and their tautomer twins; class 1: the 1,786 spectra of the query libraries) and from the v1 library (per-molecule
  mask of the ranker study, same metric key).

**Results** (`eng_diag.py`).

| Engine-2 list alone | n | truth among candidates | top 40 | top 1 | MRR | MRR without fingerprint features |
|---|---|---|---|---|---|---|
| class-2 run, `in` | 380 | 95.7% of 373 | 89.5% | 51.1% | 0.622 | 0.590 |
| ... natural-product catalogue / no PubChem record / enveda-180 / other | 257 / 46 / 40 / 30 | | 88.7 / 93.5 / 100 / 96.7% | 41 / 63 / 88 / 80% | 0.538 / 0.751 / 0.938 / 0.869 | 0.503 / 0.780 / 0.890 / 0.791 |
| class-1 run, `in` | 150 | 93.3% of 149 | 91.3% | 78.7% | 0.820 | 0.748 |
| `out` (both runs) | | 0 | 0 | 0 | 0 | 0 |

Median 54 candidates per molecule (21 COCONUT, 3 ChEBI / LIPID MAPS, 13 training rows); lists hold 30 keys on average.

**Which models have seen the hold-out molecules, and what that does.**

| Model | Seen them? | Effect |
|---|---|---|
| v1 engine bank and `model_views` (here: fold-safe FPNet) | no | honest; weaker than the three public models the notebook uses |
| v1 public ranker | fitted on simulations that include some of them (rosters unknown) | slightly optimistic |
| **engine-2's fingerprint models** (`fp_single_s2`, `fp_merged_m1`) | **yes**: f·z alone puts the truth first in its same-mass group for 89.6% (class-2 run, 327 groups of mean size 70, chance 0.10; fold-safe FPNet 0.47, FPNet A which saw them 0.76) | optimistic, but small: the ranker gives the fingerprint little weight (blend in-group top-1 0.53), the list loses 0.03 MRR (class 2) and 0.07 (class 1) when the fingerprint features are zeroed. Types `c2n` / `dAn` use those lists; no conclusion below changes (v7b − v4b −0.007 ± 0.005 and −0.079 ± 0.007) |
| engine-2's rankers | fitted at run time from the public simulation rows (rosters unknown) | unknown, probably small |
| PubChem channel (here: fold-safe FPNet instead of A + B) | no | honest, weaker than the two-model ensemble |
| ICEBERG (MassSpecGym checkpoint) | very likely | optimistic for the truth wherever it is used (ICEBERG prefix) |

## 3. PubChem channel locally (step 3: GO)

`pc_local.py` follows `pc_runner.py` + `probe_core2.py` (`PC_N1` 5000) with the fold-safe FPNet. One pass yields the
list with the truth treated as a pool member (truth-in-pool versions) and with its pool row removed (the truth stays
in the PubChem tier), in f·z order and in the popularity order of v5a / v5c (`PC_POP_LAM` 0.25, union 200).

| Truth-removed version | all 380 | natural-product catalogue 260 | no PubChem record 50 | enveda-180 40 | other 30 |
|---|---|---|---|---|---|
| truth has a PubChem record (`pool_lsid > 0`) | 325 | 98.5% | 0% | 97.5% | 100% |
| truth is a row of the PubChem tier (raw key in the ±10 ppm window) | 297 (78.2%) | 88.1% | 0% | 97.5% | 96.7% |
| survives pass 1 (top 5000) | 272 | 81.9% | 0% | 80.0% | 90.0% |
| listed among the 25 (f·z order) | 196 (51.6%) | 55.4% | 4.0% (tautomer forms) | 77.5% | 63.3% |
| first / MRR of the list alone | 15.8% / 0.244 | 16.5% / 0.252 | 0 / 0.020 | 32.5% / 0.461 | 13.3% / 0.250 |
| MRR, popularity order | 0.500 | 0.603 | 0.040 | 0.387 | 0.524 |

The popularity order finds the local truths by fame (they are library compounds). `pc_mask.py` re-scores the truth
with the popularity of an ordinary structure outside the pool (1.1, pubchem review): MRR over the 258 molecules whose
truth the popularity list holds 0.737 → 0.263. The truth-removed types use the masked list for the v5a / v5c configs;
`dAf` / `dBf` keep the real popularity (section 5.2).

## 4. Forward models (step 4: PARTIAL)

`fwd_local.py` calls the ICEBERG runner of the Kaggle package unchanged on chunks of molecules (RDKit 2026.03.3
instead of 2025.03.6, `--any-rdkit`; 1.5 to 1.7 GB of GPU memory; 29 to 57 predictions per second). Work: 306 covered
molecules, 33,412 candidates (base top 60 under each ordering, engine-2 top 40, PubChem lists; same-formula groups of
two or more), about 4.3 predictions per candidate. `gpushare` kept the job at a duty of 0.25 for almost the whole
session (the user played two games, and between them another agent's training held more than 5 GB), so every chunk
of about 2,500 jobs was followed by a pause three times its length: 40 s per molecule. After 105 minutes the job was
stopped between chunks: **139 covered molecules, 15,412 candidates scored = the first 169 molecules of the regime's
order** (natural-product catalogue 118, no PubChem record 21, enveda-180 17, other 13; 64 of the class-1 subset).
**GLACIER was not run.**

Two replays follow from that:
- **zero-forward** (all 380 molecules): forward-model terms at zero with the coverage structure kept
  (`regime_lib._Const`): an ICEBERG-covered molecule gets equal scores for all its candidates, so nothing is
  re-ordered, but `FRAG_MODE='gap'` applies to the uncovered molecules only, as on Kaggle. `ICE_LAM_LIB`,
  `GL_LAM_LIB`, `GL_MH_ONLY`, `GATE_TAU` and `PC_FWD_T1` do nothing here: v4a equals v2, v7b equals v4b + `FUSE_MIN_LIB`.
- **ICEBERG prefix** (169 molecules): real ICEBERG scores, GLACIER terms empty. `GL_MH_ONLY` still does nothing, so
  v7b is again v4b + `FUSE_MIN_LIB`, and v5a / v5c lack their GLACIER change.

## 5. The regime (step 5)

### 5.1 Types and the eight configs

| Type | v1 base list | engine-2 | PubChem list | Hidden-test situation it stands for |
|---|---|---|---|---|
| `c1` | class-1-like run, as the notebook scores it | class-1 run, `in` | truth skipped (pool member) | reference spectrum in a public library |
| `c2` | truth in the pool, placed as a COCONUT-only row | class-2 run, `in` | truth skipped | COCONUT structure without spectra |
| `c2x` | as `c2` | `out` | truth skipped | the old harness (and the ~1.4% of COCONUT answers engine-2's file lacks) |
| `dA` | truth's pool row removed | `in` | truth may be listed | answer only in ChEBI / LIPID MAPS (variant a) |
| `dB` | truth's pool row removed | `out` | truth may be listed | PubChem-only (325 of 380) or novel (55) answer (variant b) |

MRR@25 per type, zero-forward replay (`regime.py --no_fwd`; 380 molecules, `c1` 150):

| Config | LB | `c1` | `c2` | `c2x` | `dA` | `dB` |
|---|---|---|---|---|---|---|
| v1 | 0.399 | 0.818 | 0.714 | 0.430 | 0.269 | 0.083 |
| v2 = v4a | 0.404 | 0.833 | 0.742 | 0.451 | 0.275 | 0.085 |
| v3b | 0.390 | 0.834 | 0.735 | 0.445 | 0.290 | 0.088 |
| v4b | 0.409 | 0.832 | 0.743 | 0.455 | 0.277 | 0.085 |
| v5a | 0.393 | 0.839 | 0.740 | 0.448 | 0.253 | 0.075 |
| v5c | 0.405 | 0.835 | 0.743 | 0.453 | 0.255 | 0.077 |
| v7b | 0.405 | 0.831 | 0.733 | 0.604 | 0.188 | 0.103 |
| v1 without engine-2 (public lineage: 0.384) | | 0.796 | 0.680 | 0.680 | 0.127 | 0.127 |
| v4b without engine-2 | | 0.835 | 0.726 | 0.726 | 0.127 | 0.127 |

| Paired difference | LB | `c1` | `c2` | `c2x` | `dA` | `dB` |
|---|---|---|---|---|---|---|
| v7b − v4b (engine-2 gate) | −0.004 | −0.002 ± 0.005 | −0.010 ± 0.006 (10 / 24) | **+0.149 ± 0.013** (119 / 0) | −0.088 ± 0.008 (4 / 153) | +0.018 ± 0.004 (62 / 2) |
| v3b − v2 | −0.014 | +0.000 ± 0.007 | −0.007 ± 0.006 | −0.006 ± 0.007 | +0.015 ± 0.005 | +0.003 ± 0.002 |
| v5a − v4b | −0.016 | +0.007 ± 0.008 | −0.003 ± 0.007 | −0.007 ± 0.008 | −0.024 ± 0.006 | −0.011 ± 0.002 |
| v5a − v4b, truth keeps its popularity | | | | | +0.062 ± 0.010 | +0.031 ± 0.004 |
| v4b − v1 | +0.010 | +0.015 ± 0.013 | +0.029 ± 0.010 | +0.025 ± 0.010 | +0.008 ± 0.004 | +0.003 ± 0.002 |
| v1 − v1 without engine-2 | (+0.015, public) | +0.021 ± 0.009 | +0.034 ± 0.007 | −0.249 ± 0.014 | +0.142 ± 0.009 | −0.045 ± 0.006 |

With the leaked base order of the old harness (`c2s`, `c2xs`): v7b − v4b +0.004 ± 0.005 and +0.166 ± 0.014; the
Kaggle caches gave +0.118 to +0.14. The local +0.13 was the `c2x` column.

### 5.2 Mixtures against the eight points

Grid over (`c1`, `c2`, `dA`, `dB`) in steps of 0.05 (1,771 weight vectors). A share of molecules that no channel
reaches only scales the level and is left out of the grid; `dB` itself is close to it (0.085).

**(i) Is there a plausible mixture with v7b − v4b ≤ 0 and the right sign for v3b − v2 and v5a − v4b?** For v7b yes,
robustly. For all three together only in the zero-forward replay on all 380 molecules, by margins smaller than the
errors; on the 169-molecule prefix no weight vector passes, with or without ICEBERG, because v3b − v2 is never negative
there. **Plainly: the regime explains the v7b pair and does not explain v3b.**

| Weight vectors (of 1,771) with | zero-forward, 380 | zero-forward, prefix 169 | ICEBERG, prefix 169 |
|---|---|---|---|
| v7b − v4b ≤ 0 | 1,546 | 1,595 | 1,718 |
| v3b − v2 < 0 | 418 | 0 | 0 |
| v5a − v4b < 0 | 1,633 | 1,488 | 1,771 |
| all three | 299 (`c1` 0 to 0.55, `c2` 0.25 to 1.0, `dA` 0 to 0.30, `dB` 0 to 0.50) | 0 | 0 |
| all three, engine-2's fingerprint features zeroed | 529 | 7 | 0 |
| all three, truth keeps its popularity in the PubChem channel | 10 | | |
| best Spearman, pairs with an LB gap ≥ 0.008 ordered correctly | +0.80, 11 / 12 | +0.73, 9 / 12 | +0.77, 10 / 12 |
| old harness (`c1` + `c2xs`, 21 mixtures): v7b − v4b ≤ 0; best Spearman | 1; −0.04, 4 / 12 | 2; −0.27, 3 / 12 | 2; +0.77, 9 / 12 |

Named mixtures, with a share of unreachable molecules chosen so that v4b lands near 0.409 (`mix_se.py`). A and B are
the first reading of the directions review (class 1 about 16%, class 2 in the pool 36 to 46%), C the second (class 1
33%); "fit" is the best zero-forward weight vector without an unreachable share.

| Mixture (`c1` / `c2` / `dA` / `dB` / unreachable), replay | v4b | v7b − v4b | v3b − v2 | v5a − v4b | v4b − v1 | v1 − v1 without engine-2 | Spearman, pairs |
|---|---|---|---|---|---|---|---|
| LB | 0.409 | −0.004 | −0.014 | −0.016 | +0.010 | (+0.015) | |
| fit 0.20 / 0.50 / 0 / 0.30 / 0, zero-forward 380 | 0.564 | −0.000 ± 0.003 | −0.002 ± 0.003 | −0.003 ± 0.004 | +0.018 ± 0.006 | +0.008 ± 0.005 | +0.80, 11 / 12 |
| A 0.15 / 0.36 / 0 / 0.22 / 0.27, zero-forward 380 | 0.411 | −0.000 ± 0.002 | −0.002 ± 0.003 | −0.002 ± 0.003 | +0.013 ± 0.004 | +0.006 ± 0.003 | +0.80, 11 / 12 |
| A, ICEBERG prefix | 0.424 | −0.001 ± 0.005 | +0.002 ± 0.004 | −0.005 ± 0.006 | +0.015 ± 0.008 | +0.005 ± 0.007 | +0.76, 10 / 12 |
| B 0.15 / 0.33 / 0.04 / 0.22 / 0.26, zero-forward 380 | 0.400 | −0.003 ± 0.002 | −0.001 ± 0.002 | −0.003 ± 0.003 | +0.013 ± 0.004 | +0.010 ± 0.003 | +0.33, 8 / 12 |
| B, ICEBERG prefix | 0.418 | −0.008 ± 0.005 | +0.003 ± 0.004 | −0.006 ± 0.006 | +0.015 ± 0.007 | +0.015 ± 0.007 | +0.29, 7 / 12 |
| C 0.33 / 0.15 / 0.04 / 0.22 / 0.26, zero-forward 380 | 0.416 | −0.002 ± 0.002 | +0.000 ± 0.003 | −0.002 ± 0.003 | +0.010 ± 0.005 | +0.008 ± 0.003 | −0.12, 4 / 12 |
| C, ICEBERG prefix | 0.438 | −0.011 ± 0.004 | +0.003 ± 0.006 | −0.006 ± 0.008 | +0.015 ± 0.009 | +0.010 ± 0.007 | +0.36, 7 / 12 |
| old harness (`c1` 0.3, `c2xs` 0.7), zero-forward 380 | 0.612 | **+0.116 ± 0.010** | +0.007 ± 0.007 | +0.006 ± 0.007 | −0.004 ± 0.009 | −0.211 ± 0.010 | −0.53, 2 / 12 |

**(ii) Which mixture fits best, and how sharp is the optimum?** Zero-forward, 380 molecules: Spearman +0.80 with 11 of
12 pairs for `c1` 0 to 0.20, `c2` 0.50 to 0.65, `dA` 0, `dB` 0.30 to 0.35. 24 weight vectors reach 11 of 12 pairs
(`c1` 0 to 0.40, `c2` 0.35 to 0.75, `dA` 0, `dB` 0.25 to 0.40), 13 are within 0.05 of the best Spearman, and over the
whole grid Spearman is −0.21 / −0.10 / +0.46 at the 10th / 50th / 90th percentile. ICEBERG prefix: best +0.77 with 10
of 12 (`c1` 0.05 to 0.10, `c2` 0.70 to 0.80, `dA` 0, `dB` 0.15 to 0.20), 62 within 0.05. So the optimum is narrow in
`dA` and `dB` and flat along the `c1`-`c2` trade. **It is not evidence for a mixture.** Inside a fitting mixture the
eight local values differ by 0.000 to 0.003 with errors of 0.003 to 0.006: moving 0.04 of weight (A → B) takes the
rank correlation from +0.80 to +0.33 while every difference stays within one standard error; a subset of the
molecules changes which vectors pass; and with ICEBERG the old harness reaches +0.77 too, because v7b sits near the
top of the LB as well. The pair checks carry the information, the rank correlation does not.

**What the regime explains, and what it does not.**
- Explains: the v7b pair (−0.011 to 0.000 against −0.004; the old harness is about twelve standard errors away);
  v4b above v1 (+0.010 to +0.018 against +0.010); engine-2 being worth something (+0.005 to +0.015 against +0.015 in
  the public lineage); v5a below v4b in sign, at a third of the size or less, and only when the truth's popularity is masked
  in the PubChem channel (with the real popularity v5a wins by +0.03 to +0.06 on the truth-removed types: the fame
  artefact of the pubchem review).
- Does not explain: **v3b**. FRAG in 'all' mode with weight 0.25 against 'gap' mode with 0.5 is neutral on `c1` and
  `c2` (0.000 to ±0.007) and a gain on `dA` (+0.015 to +0.020) in both replays; on the LB it lost 0.014. Candidate
  reasons, none testable here: the parsimony score behaves differently on timsTOF spectra with minimal cleaning than
  on library Orbitrap / QTOF spectra; it interacts with GLACIER, which is missing; one and a half public-LB molecules
  are within the LB's own noise. Any decision about `FRAG_*` should ignore this regime.
- Under-predicts: the size of the v5a loss (GLACIER on [M+H]+ only and the missing library gate are part of v5a; the
  first is not replayed).

### 5.3 What engine-2 contributes (iii)

Config v4b; "v1 side" = stages A and B (base list, forward-model re-rank, PubChem merge) without engine-2
(`eng_contrib.py`). Zero-forward replay, 380 molecules:

| | `c1` | `c2` | `c2x` | `dA` | `dB` |
|---|---|---|---|---|---|
| molecules | 150 | 380 | 380 | 380 | 380 |
| truth in the v1-side top 25 / MRR | 92.7% / 0.835 | 88.9% / 0.726 | same | 41.6% / 0.127 | same |
| truth in engine-2's top 40 / first / MRR | 91.3% / 78.7% / 0.820 | 89.5% / 51.1% / 0.622 | 0 | 89.5% / 51.1% / 0.622 | 0 |
| v1-side list lacks the truth (molecules) | 11 | 42 | 42 | 222 | 222 |
| ... and engine-2's top 40 has it | 0 | 31% | 0 | **82.9%** (median engine rank 1) | 0 |
| ... then fused top 25 / top 5 / top 1; median fused rank | | 46% / 0 / 0; 16 | | **89.7% / 30.4% / 0%; 7** | |
| final MRR with the fusion | 0.832 | 0.743 | 0.455 | 0.277 | 0.085 |
| value of the fusion, paired (wins / losses) | −0.002 ± 0.008 (10 / 6) | +0.017 ± 0.007 (52 / 19) | −0.271 ± 0.015 (2 / 235) | **+0.149 ± 0.009** (280 / 10) | −0.042 ± 0.006 (5 / 126) |

ICEBERG prefix, 169 molecules (`c1` 64); in brackets the zero-forward value on the same molecules:

| | `c1` | `c2` | `c2x` | `dA` | `dB` |
|---|---|---|---|---|---|
| v1-side list lacks the truth; engine-2's top 40 has it | 2; 0 | 15; 40% | 15; 0 | 91; **84.6%** (median engine rank 2) | 91; 0 |
| ... then fused top 25 / top 5 / top 1; median fused rank | | 67% / 17% / 0; 14 | | **90.9% / 51.9% / 6.5%; 5** | |
| final MRR with the fusion | 0.853 | 0.742 | 0.540 | 0.395 | 0.131 |
| value of the fusion, paired | +0.012 ± 0.014 (+0.014) | −0.005 ± 0.015 (+0.017) | −0.207 ± 0.021 (−0.277) | **+0.267 ± 0.022** (+0.150) | +0.003 ± 0.014 (−0.046) |

- When the truth is outside the v1 pool, engine-2 usually has it first, and the reciprocal-rank fusion with
  `FUSE_ALPHA` 0.6 gives a candidate that only engine-2 lists the weight 0.6 / (3 + 1) = 0.15, the weight of v1 rank
  3.7: without a forward model it can never be first and lands at rank 7 in the median. The ICEBERG re-rank of the
  fused list then lifts it inside its formula group (rank 5 in the median, first for 6.5%), which is why the fusion
  is worth +0.27 instead of +0.15 with ICEBERG. ICEBERG has very likely seen these molecules; +0.15 is the safer
  figure.
- When the truth is in neither pool, the fusion costs 0.042 without forward models (engine-2's favourites push down
  answers that the PubChem merge and the generator had placed: 126 losses; −0.037 for the 325 PubChem answers, −0.073
  for the 55 novel ones) and nothing with ICEBERG.
- By `lib_max` of the base list (zero-forward) the value of the fusion on `dA` is +0.15 to +0.18 below 0.7 and +0.09
  to +0.12 above; on `c2` it is +0.01 to +0.02 everywhere. No `lib_max` threshold separates the types, which is why
  `FUSE_MIN_LIB` cannot work. By the similarity of the best spectral analog (`eng_by_analog.py`): `dA` +0.10 ± 0.02
  below 0.5 (35 molecules) and +0.15 above, so the value does not hinge on a very close analog, but only 9% of these
  molecules lack one.

### 5.4 What ICEBERG changes (prefix of 169 molecules, same molecules in both columns)

| Paired difference (`regime.py --first 169`) | `c1` (64) | `c2` | `c2x` | `dA` | `dB` |
|---|---|---|---|---|---|
| v4b level, ICEBERG / zero-forward | 0.853 / 0.854 | 0.742 / 0.751 | 0.540 / 0.457 | 0.395 / 0.267 | 0.131 / 0.071 |
| v7b − v4b, ICEBERG | −0.011 ± 0.008 | +0.008 ± 0.012 | +0.113 ± 0.018 | −0.179 ± 0.023 | −0.008 ± 0.009 |
| v7b − v4b, zero-forward | −0.012 ± 0.008 | −0.010 ± 0.008 | +0.166 ± 0.020 | −0.096 ± 0.013 | +0.020 ± 0.006 |
| v3b − v2, ICEBERG | +0.002 ± 0.018 | +0.002 ± 0.009 | −0.003 ± 0.007 | +0.020 ± 0.009 | +0.003 ± 0.005 |
| v3b − v2, zero-forward | +0.001 ± 0.012 | +0.003 ± 0.007 | −0.000 ± 0.010 | +0.015 ± 0.007 | +0.005 ± 0.002 |
| v5a − v4b, ICEBERG | −0.007 ± 0.022 | −0.007 ± 0.013 | −0.009 ± 0.010 | −0.013 ± 0.012 | −0.009 ± 0.008 |
| v5a − v4b, zero-forward | +0.018 ± 0.014 | −0.013 ± 0.012 | −0.017 ± 0.012 | −0.027 ± 0.007 | −0.011 ± 0.003 |
| v4a − v2 (library gate; LB 0.000), ICEBERG | +0.020 | +0.002 | +0.001 | −0.005 | −0.001 |
| v4b − v1, ICEBERG | +0.036 ± 0.026 | +0.031 ± 0.016 | +0.012 ± 0.011 | +0.003 ± 0.009 | −0.006 ± 0.004 |
| no forward models − v4b, ICEBERG | +0.001 ± 0.004 | +0.009 ± 0.017 | −0.083 ± 0.020 | −0.128 ± 0.020 | −0.060 ± 0.013 |

ICEBERG is worth nothing here where the truth is a pool row of both engines (`c1`, `c2`) and +0.06 to +0.13 where the
truth comes from the PubChem list or from engine-2 (it pulls an outside answer up inside its formula group). The
engine-2 gate therefore costs twice as much on `dA` with ICEBERG, and its small gain on `dB` disappears. The sign of
v7b − v4b under the named mixtures is unchanged (table in 5.2).

## 6. Flagged changes (step 6)

Section 5.2 does not license a calibrated regime: the v7b pair is reproduced, v3b is not. The table is therefore a
read-out per type, plus three assumed mixtures at the leaderboard level (A, B, C of section 5.2), for changes that act
through the mechanisms the regime does represent (base ranking, PubChem merge, engine-2 fusion). Left: zero-forward,
380 molecules; right: ICEBERG prefix, 169 molecules. `'ours'` is not the packaged ranker (it was trained on these
molecules) but the same recipe fitted on the 3,356 other hold-out molecules (`train_ours.py`; pre-fusion MRR on `c2`
0.821 against 0.726 for the public ranker in class-2 order). The detector is `phase5_detect.py`'s model fitted on the
416 other truth-dropped molecules (`detector.py`; out of sample AUC 0.856: natural-product catalogue 0.767, other
0.976); "absent" = below the 20% quantile of the truth-present lists and `lib_max` < 0.9.

| Change, paired against v4b | `c1` | `c2` | `dA` | `dB` | A / B / C, zero-forward 380 | A / B / C, ICEBERG prefix | Verdict |
|---|---|---|---|---|---|---|---|
| `RANKER='blend'`, `RANKER_LIB_GATE=0.7` | −0.001 ± 0.003 | +0.028 ± 0.008 | −0.001 ± 0.006 | +0.007 ± 0.005 | +0.011 / +0.011 / +0.006 (± 0.003) | +0.007 / +0.007 / +0.005 (± 0.004) | supported |
| `RANKER='public_free'`, gate 0.7 | −0.003 ± 0.002 | +0.010 ± 0.008 | +0.004 ± 0.004 | +0.010 ± 0.005 | +0.006 / +0.006 / +0.003 (± 0.003) | +0.005 / +0.005 / +0.003 (± 0.004) | supported, weaker |
| `'blend'` without the gate | +0.002 ± 0.006 | +0.044 ± 0.010 | −0.003 ± 0.006 | +0.006 ± 0.005 | +0.018 / +0.016 / +0.009 (± 0.004) | +0.015 / +0.014 / +0.010 (± 0.006) | larger here, but see below |
| `SLOTS_GENTLE=[3,6,9,12,15]` | −0.000 ± 0.001 | −0.001 ± 0.003 | +0.011 ± 0.004 | +0.011 ± 0.002 | +0.002 / +0.003 / +0.003 (± 0.001) | +0.004 / +0.004 / +0.003 (± 0.002) | supported, small |
| `PC_FWD_T1=0.0` | (prefix) 0 | −0.000 ± 0.000 | +0.000 ± 0.005 | +0.013 ± 0.005 (11 / 0) | inert | +0.003 / +0.003 / +0.003 (± 0.001) | supported where ICEBERG exists |
| both (rule C of the pubchem review) | (prefix) −0.001 ± 0.001 | +0.003 ± 0.005 | −0.004 ± 0.008 | +0.022 ± 0.006 | = slots | +0.006 / +0.005 / +0.005 (± 0.002) | supported |
| `SLOTS_GENTLE=[2,4,6,8,10]` (early slots always) | −0.001 ± 0.004 | −0.001 ± 0.004 | +0.034 ± 0.007 | +0.027 ± 0.004 | +0.005 / +0.007 / +0.007 (± 0.002) | +0.008 / +0.008 / +0.005 (± 0.004) | supported (not a preset today) |
| soft gate (`GATE_TAU` 0.8, `ICE_LAM_LIB` = `GL_LAM_LIB` = 0.5) | (prefix) −0.000 ± 0.003 | −0.012 ± 0.006 (0 / 5) | +0.001 ± 0.002 | −0.000 ± 0.001 | inert | −0.004 / −0.004 / −0.002 (± 0.002) | not supported; weak evidence |
| `FUSE_MIN_LIB` 0.3 / 0.5 / 0.7 | −0.002 / −0.002 / +0.001 | −0.006 / −0.010 / −0.014 | −0.050 / −0.088 / −0.120 | +0.010 / +0.018 / +0.025 | 0.5: −0.000 / −0.003 / −0.002 (± 0.002) | 0.5: −0.001 / −0.008 / −0.011 (± 0.005) | rejected |
| `FUSE_ALPHA` 0.8 (from 0.6) | −0.001 ± 0.001 | −0.007 ± 0.006 | +0.045 ± 0.006 | −0.015 ± 0.003 | −0.006 / −0.004 / −0.003 (± 0.002) | −0.009 / −0.006 / −0.004 (± 0.003) | rejected |
| `FUSE_ALPHA` 1.0 | −0.001 ± 0.005 | −0.019 ± 0.007 | +0.076 ± 0.007 | −0.029 ± 0.005 | −0.013 / −0.010 / −0.006 (± 0.003) | −0.012 / −0.008 / −0.002 (± 0.004) | rejected |
| `FUSE_ALPHA` 0.3 | −0.002 ± 0.006 | −0.004 ± 0.005 | −0.085 ± 0.007 | +0.022 ± 0.004 | +0.003 / −0.000 / +0.000 (± 0.002) | −0.000 / −0.004 / −0.008 (± 0.004) | no case |
| no engine-2 fusion at all | +0.002 ± 0.008 | −0.017 ± 0.007 | −0.149 ± 0.009 | +0.042 ± 0.006 | +0.003 / −0.002 / +0.001 (± 0.003) | −0.001 / −0.011 / −0.015 (± 0.007) | (reference) |
| no PubChem merge | +0.004 ± 0.003 | −0.000 ± 0.002 | −0.070 ± 0.009 | −0.031 ± 0.004 | −0.006 / −0.009 / −0.009 (± 0.001) | −0.020 / −0.022 / −0.023 (± 0.004) | (reference) |
| detector: absent → early PubChem slots | +0.002 ± 0.003 | +0.001 ± 0.001 | +0.029 ± 0.006 | +0.023 ± 0.004 | +0.006 / +0.007 / +0.007 (± 0.001) | +0.005 / +0.005 / +0.005 (± 0.002) | same as the ungated early slots |
| detector: absent → `FUSE_ALPHA` 1.0 + early PubChem slots | +0.002 ± 0.004 | +0.003 ± 0.002 | +0.089 ± 0.009 | −0.007 ± 0.004 | −0.000 / +0.003 / +0.003 (± 0.001) | −0.001 / +0.003 / +0.003 (± 0.002) | only with a large `dA` share |
| detector: absent → `FUSE_ALPHA` 1.5 + early PubChem slots | −0.001 ± 0.001 | +0.001 ± 0.005 | +0.172 ± 0.013 | −0.029 ± 0.006 | −0.006 / +0.001 / +0.000 (± 0.002) | −0.005 / +0.001 / +0.001 (± 0.002) | rejected |
| detector: present → no engine-2, absent → v4b | +0.001 ± 0.006 | −0.011 ± 0.006 | −0.032 ± 0.004 | +0.011 ± 0.003 | −0.001 / −0.002 / −0.000 (± 0.002) | +0.002 / −0.000 / −0.002 (± 0.005) | rejected |

Per-type columns are the zero-forward values on 380 molecules unless marked "(prefix)". `GL_MH_ONLY` cannot be
replayed (no GLACIER).

Reading:
- **Ranker.** Both replays support the gated blend: the gain sits on class 2 with the truth in the pool (+0.028 ±
  0.008 after the whole fusion without forward models, 33 wins and 8 losses; +0.012 ± 0.012 on the ICEBERG prefix),
  with nothing lost on class 1 or when the truth is absent. The ungated blend looks better still here, but these
  class-1 molecules are library chemistry on Orbitrap / QTOF; on the timsTOF natural products of enveda-np-examples
  the ungated ranker lost 0.055 on class 1 (ranker review 4.3), which this regime cannot see. Keep the gate.
- **PubChem slots and the forward tier.** Moving the proposals one place earlier is free where the truth is in the
  pool and worth +0.011 where it is not. `PC_FWD_T1` adds +0.013 ± 0.005 on `dB` (11 wins, no loss) and costs nothing
  on `c2` (one loss in 169); ICEBERG is optimistic on these molecules, which inflates both sides. With ICEBERG the
  PubChem merge as a whole is worth 0.02 under the mixtures, against 0.006 to 0.009 without.
- **Soft library gate.** On the prefix it loses 0.012 ± 0.006 on `c2` (lists with `lib_max` of 0.8 or more through
  analogs lose half of the ICEBERG weight) and gains nothing on `c1` (−0.000 ± 0.003), where ICEBERG does not hurt
  here (+0.001 ± 0.004 for removing it). The case for the gate came from enveda-np-examples, where the forward models
  cost class 1 0.03 to 0.05. This regime does not show that cost; it has 64 class-1 molecules, an ICEBERG that has
  seen them and no GLACIER. Weak evidence against; not a rejection.
- **Engine-2 weight.** Every rule that only turns engine-2 down or up trades `dA` against `dB` and `c2`: `FUSE_MIN_LIB`
  and a lower α lose 0.05 to 0.12 on `dA` for 0.01 to 0.025 on `dB`; a higher α does the reverse and costs `c2`. With
  the `dA` share unknown, α 0.6 is where the three mixtures are flat. `FUSE_MIN_LIB` is rejected, for the reason the
  LB gave.
- **Detector.** It separates `dA` / `dB` lists from `c2` lists well enough to lift α only where the truth is probably
  absent (`c2` +0.003 instead of −0.019 at α 1.0). That pays only if `dA` outweighs half of `dB`: with `dB` at 0.22 it
  needs `dA` above about 0.11. For the PubChem slots the detector adds nothing over moving them for everyone.

## 7. What the regime cannot represent, and what failed

- **Every molecule is a training structure with library relatives.** "Truth removed" hides a row; it does not make
  the molecule novel. Analog evidence, the generator and engine-2 are all easier here than on a really new natural
  product (pitfall 6: such simulations overstate reach about three times). The `dA` and `dB` levels are upper bounds.
- **`dA` is a construction.** It gives engine-2 an answer the v1 pool lacks by hiding a pool row; the real population
  (ChEBI / LIPID MAPS structures outside COCONUT) has other chemistry (lipids, primary metabolites) and its share of
  the hidden test is unknown. The regime prices that share (−0.09 to −0.18 per molecule for the gate, +0.15 to +0.27
  for the fusion); it cannot measure it.
- **Fame.** The hold-out truths are documented structures. The popularity prior on pool candidates (`POP_MU`) and the
  PubChem popularity list see famous answers here; the second was masked, the first was not (v4b − v1 is +0.03 on
  `c2`, three times the LB effect).
- **Forward models.** ICEBERG covers 169 of 380 molecules and has very likely seen them (it lifts outside answers by
  0.06 to 0.13 here); GLACIER is missing, so `GL_MH_ONLY`, part of v5a, v5c and v7b on the LB, is not replayed.
- **The FRAG term.** Its LB effect (v3b) is not reproduced in either replay; nothing here says why.
- **Instrument and chemistry.** 11% timsTOF (all synthetic enveda-180); the hidden test is all timsTOF natural
  products. Class-1 lists here have `lib_max` 0.97 in the median, 0.86 on enveda-np-examples.
- **Fingerprint models.** v1 and the PubChem channel run with the single fold-safe FPNet (weaker than the notebook's
  models); engine-2's own models have memorised the molecules (section 2).
- **The mixture is not identified.** Eight points with gaps of 0.004 to 0.019 and a public-LB noise of about ±0.01
  cannot fix four weights; the scan shows which weight vectors are compatible, not which is true, and a different
  subset of molecules changes which ones pass.
- The class-1 subset is 150 molecules (64 on the ICEBERG prefix), 141 of them natural-product catalogue structures.
- Failed or lost: the first engine-2 run and the first PubChem run were stopped by the 30-minute limit of background
  jobs (engine-2 at 361 of 380 without a checkpoint; a checkpoint was added). A scoring bug in the `out` variant
  crashed the first complete run at molecule 51 (fixed, score stage rerun). The first log entry counted 108k
  structures as engine-2's surplus over the v1 pool; 45k of them were a key-format difference (corrected above).
  ICEBERG reached 45% of the covered molecules.
- `INDEX.md` was not regenerated (no existing file may be edited in this pass): run `python tools/gen_index.py` when
  `prototypes/mixed_regime.py`, its test and this review are committed.

## 8. Reproduce

```bash
export CASMI_DATA='D:\casmi-data'; P=.venv/Scripts/python; S=scratch/mixed; M=D:/casmi-data/mixed_regime
$P solutions/fusion-v4n/prototypes/test_mixed_regime.py
# engine-2 inputs: kaggle datasets download prvsiyan/{casmi26-fp-models-v2,coconut-casmi26-candidates,chebi-lipidmaps-casmi26} -> $M/ds/{fpv2,coco,bio}
$P $S/s1_select.py                 # mols.parquet (380), queries.parquet
$P $S/build_base.py                # v1 candidate rows of the three versions from the ranker study
$P $S/eng2_local.py --mode c2      # engine-2, class-2 purge: in / out lists (one process: NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 --workers 1)
$P $S/eng2_local.py --mode c1      # engine-2, class-1 purge
$P $S/pc_local.py --workers 2 ; $P $S/pc_mask.py      # PubChem channel (in / drop, fz / popularity) + fame mask
$P $S/frag_local.py                # frag.json
$P $S/fwd_local.py --model ice     # ICEBERG in throttled chunks, resumable: rerun to continue after molecule 169
$P $S/fwd_local.py --model ice --only_partial         # top up molecules scored before their lists were complete
#   GLACIER (not run): $P $S/fwd_local.py --model gl ; $P $S/fwd_local.py --model gl --mh_only
$P $S/ice_prefix.py                # K = how many molecules of the order are fully scored (169)
$P $S/train_ours.py ; $P $S/detector.py                # out-of-sample ranker scores and detector scores
$P $S/regime.py --no_fwd --tag _const                  # zero-forward replay, 380 molecules (sections 5.1, 5.2)
$P $S/regime.py --first 169 --tag _ice169 ; $P $S/regime.py --first 169 --no_fwd --tag _const169    # prefix (5.2, 5.4)
$P $S/mix_se.py _const '{"A": {"c1": 0.15, "c2": 0.36, "dB": 0.22}, "B": {"c1": 0.15, "c2": 0.33, "dA": 0.04, "dB": 0.22}, "C": {"c1": 0.33, "c2": 0.15, "dA": 0.04, "dB": 0.22}}'
$P $S/eng_contrib.py --no_fwd ; $P $S/eng_contrib.py --first 169 --tag _ice169 ; $P $S/eng_diag.py ; $P $S/eng_by_analog.py
$P $S/flags.py --no_fwd --mix '<the same JSON>' ; $P $S/flags.py --first 169 --tag _ice169 --mix '<the same JSON>'    # section 6
$P $S/explore3.py ; $P $S/explore5.py ; $P $S/check_keys.py                      # candidate-set checks of section 2, key check
```

When the GPU is free, the remaining 167 ICEBERG molecules need about 35 minutes at the normal duty of 0.8 (GLACIER
was not timed); every analysis script then runs on all 380 molecules without `--first`.

Outputs: `D:\casmi-data\mixed_regime\out\` (`points*.parquet`: type, config, molecule, reciprocal rank; `mix*.json`;
`flags*.json`; `eng_contrib*.parquet`), logs in `logs\`. The channel keys used to score the final lists equal the
metric's key (`tools/metric.ik14`) for 3,000 of 3,000 sampled SMILES and for every truth.

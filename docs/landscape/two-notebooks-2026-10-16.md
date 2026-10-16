<!-- index: 2026-10-16 the two 0.451 notebooks easonyanyan/casmi26-v22-rel150-e2r2-clean (public) and omerfarukmerey/casmi26-om-stack1 (team, private) read from source: code-identical by syntax tree, both = pub-p2 / v7c (0.444) + metric-keyed engine-2 rank-2 insert (E2R2) + PubChem promotion at rel > 150 instead of 250; lineage and the public single-change score pairs behind each change; inputs and licences (same 14 datasets as pub-p2, no NIST, no new model); local e006u replay pre-registered in two arms (rel threshold literal and firing-rate-matched to Kaggle's A+B scale, E2R2, both, and our multi-spectrum PubChem selector on the same base) with per-type tables; Kaggle-scale counts from the diag67 validation run; the team's stack1-v71b submission (0.439) against stack1 (0.451); recommended base and where v71 plugs in -->
# Two notebooks at 0.451: `casmi26-v22-rel150-e2r2-clean` and `casmi26-om-stack1` (2026-10-16)

Marks: **[V]** read in the code, page data or logs, or computed here (file named); **[C]** cited from our reviews; **[I]**
inference. MRR = MRR@25 on the notebook's metric identity; "±" = one standard error of a paired per-molecule difference;
(wins / losses). **No leaderboard estimate.** One answer at rank 1 on the public part (about 124 molecules) = 0.008.

What was done: Kaggle CLI read-only (`kernels pull -m`, `kernels logs`, `kernels status`, `competitions submissions`), the
authenticated SDK's `GetKernel` for version numbers, Kaggle's anonymous web API (`ListKernelVersions`) for the public
notebooks; no browser was needed. Nothing was pushed, submitted, voted, committed or downloaded as data. Local CPU
replays: one process at a time, below-normal priority, about 45 minutes in total. Pre-registration:
`scratch/nb1016/PREREG.md` (written before any comparison; one addendum, also before). Work files: section 9.

## 0. Bottom line

1. **The two notebooks are the same program** [V]. Syntax-tree diff (`scratch/land1012/celldiff.py`): all 32 code
   cells of om-stack1 are identical to the 21 of v22 (om-stack1 stores the markdown headers as string-literal code
   cells; the only text differences are titles, credits and one comment). Their commit runs wrote byte-identical
   submissions (sha256 `2b32bc61…` in both logs). Both score **0.451**: v22 on its public page, om-stack1 as our team
   submission 57240422 ("stack1", 2026-10-15 18:19 UTC).
2. **What they are** [V]: pub-p2 (= ahmedberatozer v7c, our rerun 0.444) + **E2R2** (cell 36: engine 2's first
   candidate inserted at rank 2, compared by the official metric identity; our team's public notebook, 0.445) +
   **rel150** (cell 24: the structure-evidence clause of the PubChem promotion fires at `rel > 150.0` instead of
   `250.0`). "stack1" means "the two single changes stacked", not a model ensemble. No new model, embedding, candidate
   source or input.
3. **Public evidence** [V, pairs that differ by one line, verified by diff]: rel 250 → 150 gained +0.006 (ours,
   0.445 → 0.451), +0.004 (seyitkaangunes, 0.442 → 0.446), +0.003 (ahmedberatozer v11 vs antriksharya v7c-rank2,
   0.443 → 0.446); rel 100 gave 0.444 (+0.002). The rank-2 insert gained +0.001 (ours) and +0.002 twice (antriksharya).
4. **Local replay** [V, pre-registered]: rel150 at a firing-rate-matched scale is level with the answer's fame masked
   (+0.0002 ± 0.0014) and positive with real fame (+0.0067 ± 0.0017); E2R2 is a class trade (answers only in engine 2's
   ChEBI / LIPID MAPS library +0.016 to +0.041; PubChem-only answers −0.006 to −0.013; pooled +0.0086 ± 0.0014 masked,
   +0.0008 ± 0.0013 real). By the pre-registered rule all three contrasts are **neutral / unresolved** (a cell below
   2 SE in each); none is harmful. The leaderboard's sign is positive for both.
5. **Our selector on this base**: the team already submitted it. **stack1-v71b** (submission 57242856: stack1 +
   cd_pairmulti, 2-seed mean, as the PubChem channel's network) scored **0.439, −0.012 against stack1** [V submission
   list]. The local replay of the same swap on the stack1-like base is +0.0071 ± 0.0035 / +0.0090 ± 0.0034 [V]: the
   **local sign is opposite to the leaderboard**, as for v69a. Its code was not readable here (section 7).
6. **Base**: om-stack1 v1 (our own, 0.451, correct Apache notice and lineage). v22 is a public duplicate; nothing it
   adds. Rule flags: none new beyond pub-p2's (section 4).

## 1. Sources, versions, scores, run times

| | `easonyanyan/casmi26-v22-rel150-e2r2-clean` | `omerfarukmerey/casmi26-om-stack1` |
|---|---|---|
| Pulled | latest = version 1 (`current_version_number` 1, SDK `GetKernel`) | latest = version 1 (`current_version_number` 1). `kernels pull <ref>/<n>` is refused (403) for every n, so the requested `scriptVersionId=358764349` could not be pulled by id; since only one version exists, the pulled code is that version [I: id not readable for a private notebook] |
| Visibility | public, 8 votes, 27 views (listing of 2026-10-16 06:11 UTC) | private (team member's account) |
| Version history | v1 only, run 358828911, created 2026-10-15 23:09 UTC (`ListKernelVersions`) | v1 only |
| Public score | **0.451** (`bestPublicScore`, listing) | **0.451** = team submission 57240422, 2026-10-15 18:19 UTC, description "stack1: E2R2 (0.445 ...) + PubChem promotion rel>150 instead of 250 ...; one change vs E2R2" [V `competitions submissions`] |
| Commit run (12 visible molecules) | 1,087.8 s (`runInfo.runTimeSeconds`; log `runtime_second` 1,035.7) | log `runtime_second` 1,096.1, last log line at 1,149 s |
| Scoring run | not visible | not visible. Same pipeline family as e006: 10,566 s on 400 validation molecules in diag67 [C `reviews/diag67-2026-10-12.md`]; E2R2 adds a pass over 25-row lists |
| Commit check | ICE / GLACIER status ok, 11 / 11 scored, `merge stats {'untouched': 12, ...}` (the rel line never acts on the commit), `E2R2_APPLIED ... changed_rows 0` [V `scratch/nb1016/log_eason.txt`] | identical lines and hashes [V `log_omer.txt`] |
| Kernel files | `scratch/nb1016/eason/` | `scratch/nb1016/omer/` |

Per-version scores: `ListKernelVersions` returns no score field [V `scratch/nb1016/kver2.py`, `versions.json`]; both
notebooks have one version, so the page score is the version's.

## 2. Lineage (syntax-tree diffs, `scratch/nb1016/diff/`)

```
canqiang E006 (0.440)  --[+ rank-2 cell, score_key]--> antriksharya e006-e2rank2 (0.442) --[rel 100]--> seyitkaangunes (0.444)
       |                                                                                  `--[rel 150]--> seyitkaangunes rel-150 (0.446)
       `--[+ frag-gap, cells 33/34]--> ahmedberatozer v7c (0.441) = nursrijan (0.441) = our pub-p2 (0.444, CC BY set)
              |--[+ rank-2 cell]--> antriksharya v7c-rank2 (0.443) --[rel 150]--> ahmedberatozer v11 (0.446)
              `--[+ E2R2, metric identity] (our pub-p2 base)--> omerfarukmerey "v7c + CC BY ChEBI + engine-2 rank-2" (0.445,
                       = team E2R2 57227077) --[rel 150]--> om-stack1 (0.451, team 57240422) == easonyanyan v22 (0.451)
```

| Pair (this vs base) | code cells identical | difference |
|---|---|---|
| v22 vs om-stack1 | 32 of 32 (om-stack1's extra cells are string literals) | none |
| om-stack1 vs omer's public 0.445 notebook | 31 of 32 | cell 24 line 64: `rel > 150.0` |
| om-stack1 vs pub-p2 (`scratch/land1014/p2`, = p3 v7c) | 30 | cell 24 (rel), new cell 36 (E2R2) |
| om-stack1 vs e006r (`submissions/e006r-0.443`) | 28 | cell 24 (rel), new cell 33 (frag gap, 35 lines), cell 34 (+4 lines: frag-gap re-rank at line 90), new cell 36 |
| om-stack1 vs ahmedberatozer v11 (0.446) | 31 | v11's last cell (antriksharya's rank-2 on `chem.score_key`, 56 lines) replaced by E2R2 (196 lines); input: prvsiyan's CC BY-NC-SA ChEBI set → sergiomalv's CC BY set |
| v22 vs pub-p1 / pub-p4 | 3 of 21 / 16 of 21 | p1 is restructured (8 code cells); p4 has a window-80 meta-consensus step |
| v11 vs antriksharya v7c-rank2 (0.443) | 29 | rel line; a Python-version assert (3.12 only) |
| seyitkaangunes rel-150 (0.446) vs antriksharya e006-e2rank2 (0.442) | 30 | rel line only |

Lineage of v22 by its own credits [V cell 0 / cell 4]: "validated by Seyit Kaan Gunes sweep, LB 0.446"; E2R2 credited
to "Omer Faruk Merey, LB 0.445". Its code equals om-stack1 although om-stack1 is private; both are reproducible from two
public sources (our team's public 0.445 notebook + the one-line rel change of v11 / seyitkaangunes) [I: an independent
recombination; nothing indicates private sharing].

## 3. Every change, with cell and line (cell numbers = notebook cell index; lines 1-based in the cell)

| # | Where (v22 / om-stack1) | What | Mechanism | New input? |
|---|---|---|---|---|
| 1 | cell 24 line 63 / 64 | `if (S is not None and float(S) > S_TAU and pop_top >= POP_TAU) or (rel > 150.0):` (e006: `250.0`). `rel = fz_top − best_pool_fz` in raw f.z units of FPNet A+B (line 57); `S_TAU 6`, `POP_TAU 5` unchanged; promotion puts the channel's (simulator-re-ordered) first row at rank 1 of the engine-1 list **before** the engine-2 fusion (cell 26, `fuse2`, `1/(3+r1) + 0.6/(3+r2)`, window 40), so engine 2 can still overrule it | order (rank 1 of PubChem-only rows) | no |
| 2 | cells 35 (markdown) and 36 (196 lines) | E2R2 after the final E006 overlay: `e2r2_row` (line 21): x2 = engine 2's first usable SMILES (`ENG[mid]['smiles']`); unchanged if its metric identity equals rank 1; else every twin of x2 removed and x2 placed at rank 2, list cut to 25. Applies to **every** row, library hits included; rank 1 never changes. Keyed on `metric_identity` (cell 29: stereo stripped, tautomer-canonical IK14); antriksharya's original used `chem.score_key` (keeps stereo). Strict on the commit run, falls back to the E006 file on any error in the scoring rerun (`E2R2_FAILED`) | order (rank 2) | no (engine 2's existing list) |
| 3 | cell 33 (35 lines), cell 34 line 90 | inherited from v7c: MetFrag-lite (`frag_rescore.py`, ours, Apache 2.0) re-rank, λ 0.5, only for molecules with no GLACIER and no finite ICEBERG score (1 of 400 in the commit) | order | no |
| 4 | inputs | `sergiomalv/casmi-chebi-lmsd-structures` (CC BY 4.0) instead of `prvsiyan/chebi-lipidmaps-casmi26` (CC BY-NC-SA) | licence; coverage of engine 2 | swap |
| – | everything else | unchanged against e006r: `LIB_TAU 0.9`, `REL_TH 600`, slots, `TOPN 60`, `ICE_LAM 1.0`, `ICE_BUDGET 5400`, pool popularity 0.15, PubChem popularity 0.25 / 200, fusion window 40, ICE fallback | | |

Markdown claims checked: v22's "CC BY swap +0.001 to +0.003" matches ours (v59a − v33b +0.001; e006r − e006 page
+0.003) [C `docs/experiments.md` rows 42, 51]. Neither notebook changes the PubChem channel itself (cell 15 identical to
pub-p2's cell 14) [V].

## 4. Inputs, licences, rule flags

Both attach the same 14 datasets as pub-p2 / v70a [V `kernel-metadata.json`]: competition data, `metric/rdkit-2026-3-3-wheel`,
`prvsiyan/casmi26-ranker-features`, `prvsiyan/casmi26-fp-models-v2`, `prvsiyan/coconut-casmi26-candidates`,
`dmitriigluzdov/casmi26-pubchem-popularity-prior`, `megayak/casmi26-simulated-ranker-rows`, seven `ahmedberatozer/casmi26-*`
(fpnet-full1, glacier, iceberg, pubchem-tier, v2-pool, v3-models, v4b-models), `sergiomalv/casmi-chebi-lmsd-structures`.
No hengck23, eveliaveldrine or MassBank set (several upstream notebooks attach them). Licence status as in
`submissions/e006r-0.443/README.txt` and the audit [C]: **not cleared by the host**: `prvsiyan/casmi26-fp-models-v2` and
`-ranker-features` (CC BY-NC-SA labels) and unlicensed code inside the ahmedberatozer datasets (data and weights allowed by
the host in thread 745072, code licence unanswered). **No NIST-derived item, no new model, nothing private among the
inputs.** Code: E2R2 is our team's (public notebook, Apache 2.0, idea credited to antriksharya, Apache 2.0); the rel line is
a constant. om-stack1 is private and used inside the team (rule 3.5.d satisfied). v22's licence field was not read [not
verified]. Tuning risk, not a rule issue: the threshold 150 was picked on the public leaderboard among four values (100,
150, 250, 500) on two bases [V seyitkaangunes cell 0].

## 5. What is sensible, judged against what we measured

| Change | Acts on | Public evidence (single-change pairs) | Our earlier evidence | This study | Verdict |
|---|---|---|---|---|---|
| rel 250 → 150 | order: rank 1 for a PubChem-only row whose fit beats the pool by a margin | +0.006 (ours 0.445 → 0.451), +0.004 (0.442 → 0.446), +0.003 (0.443 → 0.446); D5 base 0.433 → 0.439 and 500: 0.430 [claimed on seyitkaangunes' page, not verified]; 100: 0.444. Each pair below one answer; four of four positive | the rel branch could not be tested locally (local rel ≤ 206, threshold 250 never crossed; 12 firings, all empty windows, 0 right) [C graft, e006imp 3.3]; at an honest scale promotion at rel > 300 moved 1 answer up and 31 down on pool answers [C pubchem-2026-10-02 2.2]; "only the leaderboard can decide this branch" [C e006imp 3.3] | Kaggle A+B scale (diag67, 359 pool-answer validation molecules below the gate, leaked): rel > 250 fires 8×, rel > 150 29× (all wrong rows by construction: the channel excludes pool keys) [V `thresholds.py`]. Of the 8 at 250, the channel's first row (before its simulator re-order) is first in the final table for 2 [V; approximate, the promoted row is the re-ordered list's first]. Local matched replay: masked +0.0002 ± 0.0014, real +0.0067 ± 0.0017 (table 6.2) | **sensible on the public evidence, fame-conditional**: it trades pool answers for famous PubChem-only ones; local level / positive; neutral by the pre-registered rule |
| E2R2 | order: rank 2 | +0.001 (ours pub-p2 0.444 → 0.445), +0.002 (e006 0.440 → 0.442), +0.002 (v7c 0.441 → 0.443) | rank 2 carries 0.072 of v59a's public MRR (2 × (0.413 − 0.377)) [C experiments rows 43–44]; engine 2 more weight = a dA / dB trade [C anatomy-2026-10-06] | local: dA +0.041 / +0.016, dB −0.006 / −0.013, c2 −0.002 (table 6.3); Kaggle A+B validation (pool answers; engine 2 lists the answer for only 59 of 267 below the gate there [C diag67]): 129 of 400 lists changed, −0.0066 ± 0.0018 (1 / 21) [V `e2r2_diag67.py`] | **neutral, keep**: positive three times on the leaderboard, a class-mix bet locally |
| frag gap (v7c) | order, molecules without forward scores | v7c 0.441 vs e006 0.440; pub-p2 0.444 vs e006r 0.443 | fragment term +0.002 ± 0.001 [C graft] | not replayed (absent from e006u) | neutral |
| CC BY swap | licence | +0.001, +0.003 | [C rows 42, 51] | – | required |
| Not changed: window 40, ICE fallback, GLACIER on [M+H]+ only | | window 80: 0.443 = e006r [C v69c]; joint ICEBERG + GLACIER 0.440 (−0.003) [C v69a]; library shield 0.443 [C v69b]; promoted row first after the fusion 0.443 [C v69d] | local: window 80 null, joint +0.007 (sign opposite to the leaderboard), shield +0.002 [C replay_table] | – | their choice to keep these is consistent with our scores |

Why promotion before the fusion is the right place (consistent with ours): seyitkaangunes measured forcing the promoted
row to rank 1 after the fusion at 0.402 against 0.433 on the D5 base [V his cell 0, not re-checked]; our v69d (promoted row
first after the fusion, S branch) was 0.443 = e006r and v67b 0.410 vs v67a 0.411 [C].

## 6. Local replay on e006u (pre-registered, `scratch/nb1016/PREREG.md`)

### 6.1 Construction
`scratch/nb1016/nb_replay.py`: `scratch/e006/e6_lists.py` unedited, transformed in memory (as v69 / pcsel); per run e006's
merge cell with `(rel > 250.0)` replaced (asserted once); after e006's chain two variants on the same objects: `off` (e006's
last cell) and `e2r2` (then cell 36 of om-stack1, verbatim; ENG = the harness's engine-2 lists; `IS_RERUN` True). Checks
[V `nb_report.py`, ALL_CHECKS_OK]: run T250/off = the stored e006u lists on 2,084 of 2,084; in every run `off` = the run's own
e006 lists; no `E2R2_FAILED`; with the new channel at 250, the lists equal pcsel's stored new-channel lists on 2,084 of 2,084.
No forward score was missing in any run. The frag-gap cell is not in the harness.

**Scale of `rel`.** Literal thresholds do not transfer (local f.z is the fold-safe FPNet's, much smaller than A+B's).
Matched thresholds (label-free, `thresholds.py`): Kaggle shares among below-gate molecules in diag67 (A+B): 8 / 359 = 2.23%
above 250, 29 / 359 = 8.08% above 150; local pool-answer molecules below the gate (547, 4 with an empty window): **t250 =
59.29, t150 = 33.02** (fire 17 and 48). Known bias: diag67 is leaked for A+B (inflated pool fit, lower rel), so the matched
local thresholds are if anything too high.

### 6.2 Composites, pooled mx (380) + panel (226) = 606 molecules [V `casmi-work/nb1016/table.md`]

| Contrast | arm m (fame masked) | arm u (real fame) | worst cell | verdict (rule) |
|---|---|---|---|---|
| R-lit: rel 150 vs 250 literal | +0.0000 ± 0.0000 (0 / 0) | +0.0003 ± 0.0003 (1 / 0) | – (1 list changed) | neutral (no effect) |
| **R-match**: t150 vs t250 | +0.0002 ± 0.0014 (18 / 29) | **+0.0067 ± 0.0017** (27 / 9) | panel dA m −0.0143 ± 0.0052 (2 / 13) | neutral / unresolved |
| **E**: e006u + E2R2 vs e006u | **+0.0086 ± 0.0014** (100 / 80) | +0.0008 ± 0.0013 (56 / 105) | panel dB u −0.0201 ± 0.0033 (0 / 39) | neutral / unresolved |
| RE-match: t150 + E2R2 vs t250 | **+0.0083 ± 0.0020** (105 / 88) | **+0.0072 ± 0.0021** (74 / 97) | panel c2 m −0.0052 ± 0.0022 (1 / 12) | neutral / unresolved |
| context: t250 vs e006u | −0.0002 ± 0.0009 | +0.0057 ± 0.0015 | | |
| **F**: v71 channel on the t150 + E2R2 base | **+0.0071 ± 0.0035** (125 / 102) | **+0.0090 ± 0.0034** (76 / 68) | mx c2 m −0.0030 ± 0.0042 | sensible (rule) — **leaderboard −0.012** (section 7) |
| F2: v71 channel at t250, no E2R2 | +0.0105 ± 0.0032 | +0.0132 ± 0.0036 | | |

### 6.3 Per list type, pooled over both sets [V `pertype.py` → `casmi-work/nb1016/pertype.txt`]

c1 = answer with a library spectrum, c2 = pool structure without spectrum, dA = answer only in engine 2's ChEBI / LIPID
MAPS library, dB = PubChem-only or novel answer [C mixed-regime-2026-10-02 table]; m / u = fame masked / real.

| Contrast | c1 m | c2 m | dA m | dB m | dA u | dB u |
|---|---|---|---|---|---|---|
| R-match | −0.0016 ± 0.0013 (0/2) | −0.0011 ± 0.0010 (1/3) | −0.0047 ± 0.0035 (9/23) | +0.0058 ± 0.0031 (16/3) | +0.0110 ± 0.0043 (17/7) | +0.0172 ± 0.0039 (24/0) |
| E | −0.0005 ± 0.0011 (3/7) | −0.0024 ± 0.0014 (6/22) | +0.0407 ± 0.0045 (96/27) | −0.0064 ± 0.0012 (0/36) | +0.0159 ± 0.0037 (51/42) | −0.0130 ± 0.0017 (0/67) |
| RE-match | −0.0024 ± 0.0021 (3/8) | −0.0041 ± 0.0018 (6/25) | +0.0361 ± 0.0057 (96/36) | −0.0019 ± 0.0035 (12/33) | +0.0262 ± 0.0056 (61/42) | +0.0044 ± 0.0042 (22/56) |
| F | +0.0010 ± 0.0013 (2/4) | −0.0011 ± 0.0028 (12/16) | +0.0120 ± 0.0069 (68/52) | +0.0133 ± 0.0077 (102/73) | +0.0185 ± 0.0070 (52/35) | +0.0147 ± 0.0080 (66/46) |

Per set and type (12 cells per contrast, lists and first rows changed, promotions, E2R2 counters): `table.md`. Promotions
at t250 → t150: mixed dA/dB m 35 → 78, panel 15 → 45; E2R2 inserts into 17 to 139 lists per cell and never changes a first row.

Reading [I]: rel150 is the same fame bet as e006's whole promotion (level masked, positive with real fame, a loss where a
pool answer is displaced); E2R2 moves rank-2 credit from PubChem-only answers to engine-2-library answers, so its sign is
the hidden share of the two; their sum is positive in both arms locally and on the leaderboard.

## 7. Which base, and how the multi-spectrum PubChem selector (v71) combines with it

**Base: om-stack1 v1** (team, 0.451, Apache 2.0 notice and lineage in cell 0, `docs/experiments.md` not yet updated). v22 is
the same code; using it adds nothing and its credits are less precise.

**Does the base already touch the PubChem channel?** No [V]: the channel (cell 15) is pub-p2's byte for byte. It changes
only what is done with the channel's first row (more promotions, rel150) and adds a rank-2 row from engine 2 (E2R2, which
never reads the channel). v71 changes the channel itself (which row is first, and `fz_top` / `best_pool_fz`, rescaled per
molecule by r_m into A+B units). So the two do not duplicate; they **compound on the same row**: v71 chooses `pc[0]`, rel150
promotes `pc[0]` more often, and the rescaled `rel` decides the extra firings. Locally the promotions at t150 rise from 78
to 85 (mixed dA m) and 101 to 114 (panel dA u) with the new channel [V table F].

**Where it plugs in** (pcsel's v71 was built on pub-p2, whose cells are identical here except 24 and 36) [V diff, I for
the build]: block A after cell 11 (commit switch), block B after cell 13 (engine 2), block C appended to cell 15 (the
PubChem cell, identical to pub-p2's cell 14, so `build_v71.py`'s five anchored substitutions should apply unchanged; not
run), block D after cell 36 so that its report and validation MRR cover the E2R2 output. Cell 24 then applies `rel > 150` to
the rescaled values.

**The leaderboard has already answered once** [V `competitions submissions`]: 57242856 "stack1-v71b: stack1 ... + our
retrained fold-safe multi-spectrum FPNet (cd_pairmulti, 2-seed mean) as the PubChem-channel network, rescaled per molecule
to the A+B gate scale; one change vs stack1" = **0.439** against stack1's **0.451** (−0.012, about 1.5 answers at rank 1).
Local replay of the same swap on the same base: +0.0071 ± 0.0035 / +0.0090 ± 0.0034 (F). The sign is opposite. Differences
that could explain it [I, none tested]: the 2-seed mean (not in pcsel's validated notebook; "post hoc, would need its own
replay" [C pcsel 7]); r_m into A+B units on Kaggle (smoke-tested on 6 molecules only [C pcsel 4]) versus into fold-safe units
locally, which matters more at rel > 150 than at 250; the local channel's stand-in network. Also: our own `casmi26-sub-v71a`
and `casmi26-val-v71-on` now report status ERROR [V `kernels status`]. **Recommendation**: do not stack v71 on this base again
before a Kaggle validation pair (on / off, same 400 molecules) shows where the 0.012 went; the first thing to read is the
v71b notebook's log (`V71_PATCH` r_m, promotions and gate shares against stack1's).

## 8. Not verified

- The scoring-run times of both notebooks; the licence field of v22's page; seyitkaangunes' D5-base sweep values (his text).
- That 358764349 is om-stack1's version 1 (one version exists; the id is not readable for a private notebook by the CLI).
- The code, log and patch counters of stack1-v71b (another team member's private notebook): what exactly it changed.
- Kaggle-scale firing counts come from a leaked validation set whose answers are all pool structures: they show the cost
  side only. The fusion's veto at 150 was not replayed at that scale (GLACIER scores of diag67 are not stored).
- e006u is a reconstruction (fold-safe stand-ins for A+B and engine 2; no frag-gap cell); the matched thresholds rest on a
  firing-share match, not on a measured unit conversion. Local replays had the leaderboard's sign only part of the time
  (v69a, now F).
- `docs/experiments.md` lacks the team rows of 2026-10-15 (E2R2 0.445, stack1 0.451, stack1-v71b 0.439, iso005-pub446 0.419,
  57231723 0.425, A3 pending) and pub-p1's score (0.444, row 64 says pending) [V submission list]. `INDEX.md` not regenerated.

## 9. Files

`scratch/nb1016/` (git-ignored): `eason/`, `omer/` (pulled notebooks + metadata), `up/` (seven upstream notebooks),
`diff/` (syntax-tree diffs), `*_flat.py`, `log_*.txt`, `kmeta.py`, `kver.py`, `kver2.py`, `versions.json`, `PREREG.md`,
`thresholds.py`, `nb_replay.py`, `nb_report.py`, `pertype.py`, `e2r2_diag67.py`.
`C:\Users\hemre\casmi-work\nb1016\`: `thresholds.json`, `T250|T150|Tm250|Tm150[-new]/lists/*.pkl`, `table.md`, `table.json`,
`pertype.txt`, `e2r2_diag67.json`, `logs/`.

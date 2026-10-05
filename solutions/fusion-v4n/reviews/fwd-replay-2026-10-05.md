<!-- index: fusion-v4n 2026-10-05 ICEBERG and GLACIER run locally for every candidate group nb/07 would score on the 380 mixed-regime molecules (v13a and v22a joins), the first full local replay of the final stage with real forward scores: per-type MRR for zero-forward / v13a / v22a, one-knob-at-a-time measurements of the forward and fusion weights with exposure counts, and stage accounting for joined answers -->
# The final stage with real forward scores (2026-10-05)

Offline only. Nothing was submitted, pushed or committed, no `kaggle` command was run, no existing file was edited, no
model was trained, the visible `test.parquet` was not read. GPU: inference only, two throttled jobs (`gpushare`, duty
0.8; no game was detected during the run), at most 2.0 GB per runner process and about 4.4 GB for both together.
Scripts: `scratch/fwd/` (git-ignored). Outputs and caches: `D:\casmi-data\fwd\` (29 MB, all created here; nothing
else under `D:\` was written or deleted).

**Every number below was computed in this study** unless a source is named. MRR = MRR@25 on the metric key; "±" is one
standard error of a paired per-molecule difference; (wins / losses) in brackets; **"ne" = within two standard errors,
no evidence**. Setting: mixed regime (`reviews/mixed-regime-2026-10-02.md`), 380 hold-out molecules, types `c1` (150),
`c2` (380), `dA` / `dB` (the 325 molecules with a PubChem record), join of 100 PubChem candidates below `lib_max` 0.7,
public ranker, fusion config of v13a / v22a (`POP_MU` 0.25, `FRAG_LAM` 0.5 `gap`, `ICE_LAM_LIB` = `GL_LAM_LIB` = 0,
everything else default). v13a = candidates selected by the fold-safe model (`D:\casmi-data\pcjoin`), v22a = selected
by the fine-tune `ft_hn` (`D:\casmi-data\hardneg\join_ft_hn`). **No leaderboard estimate is given.**

## 0. Bottom line

1. **Both forward models run locally, unchanged, with full coverage.** The runners of the Kaggle packages
   (`D:\casmi-data\coverage_study\{iceberg,glacier}`, repo `.venv`, torch 2.8 + cu126, `--any-rdkit`) scored 34,570
   of 34,572 ICEBERG and 34,572 of 34,572 GLACIER candidates in the 306 molecules that have an [M+H]+ or [M+Na]+
   spectrum: 100.0% of what `nb/07` would hand to the runners there; 81.7% of all candidates of the v22a
   configuration, the rest belonging to molecules the runners skip on Kaggle too. GLACIER: 200 predictions/s, 21
   minutes for everything; ICEBERG: 40/s, 61 minutes for the 22,541 candidates not already cached. Against the
   scores a Kaggle run dumped (valF, 30 molecules, 1,348 candidates): Pearson r 0.998 (GLACIER) and 0.997 (ICEBERG).
2. **The forward stage is a trade, and a large one.** v22a with real scores minus zero-forward: `dA` +0.047 ± 0.014,
   `dB` +0.063 ± 0.014, `c2` −0.023 ± 0.012 (ne), `c1` −0.008 ± 0.007 (ne). It nearly doubles the share of pool-answer
   lists with a joined row first (`c2`: 4.7% → 8.2%, 18 → 31 lists). Every zero-forward study of the join understated
   both the gain on PubChem-only answers and the exposure.
3. **All of the added exposure and nearly all of the `c2` cost come from one step: the re-rank after the engine-2
   fusion.** In stage A a joined row sits in a formula group of its own (the `|gen` tag of the popularity re-order),
   so the forward terms cannot move it past a pool row; after the fusion the tag is gone. Stage A with real scores:
   `c2` +0.010, exposure unchanged (5.0%). Post-fusion re-rank: `c2` −0.033 ± 0.008 (9 / 40), 11 pool answers lose the
   first place to a joined row, `dA` +0.038, `dB` +0.046.
4. **No knob is better than v22a's value on all four types.** Higher weights (2) cost `c2` 0.026 to 0.029 with more
   exposure; lower weights and `POST_*_LAM` 0 move MRR from the answer-removed types to `c2` and lower the exposure;
   `FUSE_ALPHA` trades `dA` against `dB`; `FUSE_KRR` 10 loses; `GL_MH_ONLY` and "stage B off for joined molecules" show
   no evidence in any type (v14c scored 0.431 = v13a on the leaderboard, `EXPERIMENTS.md`).
5. **Where a joined answer loses its rank now (`dB`, 177 answers):** PubChem list 0.501 → joint list 0.566 → after A
   0.591 → after B 0.586 → after the RRF 0.497 → final 0.553 (zero-forward: 0.464). The RRF still takes 0.089; the
   post-fusion re-rank returns 0.056 of it. The largest remaining loss is unchanged: the ranker's order inside the
   joint list (0.24 per record molecule), then the 72 answers outside the first 100 (0.22) and the 76 behind the gate
   (0.20).
6. **Recommendation: change nothing in the default.** The one lever with evidence on the exposure side is the
   post-fusion re-rank (`POST_ICE_LAM` = `POST_GL_LAM` = 0, or keeping joined rows in their own group there): it is
   a trade whose sign depends on the share of PubChem-only answers in the hidden test, which the regime cannot
   measure. See section 6 for the per-knob table.

## 1. Getting the models to run (task 1)

Nothing had to be installed. What already existed: local copies of both Kaggle packages with checkpoints
(`D:\casmi-data\coverage_study\iceberg`, `glacier`; the directories under `input\datasets\ahmedberatozer\` hold only
the two `.py` files each), the pure-python wheels unpacked as `coverage_study\ice_site_win`, and
`scratch/mixed/fwd_local.py`, which had scored ICEBERG for 139 molecules on 2 October and never ran GLACIER. The
runners are called as subprocesses with `--site none --any-rdkit --threads 1 --memfrac --bs 32`.

`scratch/fwd/run_fwd.py` (adapted from `fwd_local.py`): chunks of molecules (4,000 jobs ICEBERG, 8,000 GLACIER), each
chunk a separate runner process, the result merged into `D:\casmi-data\fwd\<model>.json` after every chunk, a lock
file against a second copy, a start check (GPU has room; other processes hold under 9 GB), `gpushare.Share` pause
after each chunk. ICEBERG started from the 12,031 scores of `mixed_regime\fwd\ice.json` that belong to the candidate
set (same package, same options; the file was read, not changed).

| | ICEBERG | GLACIER | GLACIER, [M+H]+ spectra only |
|---|---|---|---|
| covered molecules (of 380) | 306 | 306 | 293 |
| candidates scored | 34,570 of 34,572 | 34,572 of 34,572 | 33,095 of 33,095 |
| predictions/s inside a chunk | 23 to 59 (two chunks at 10 and 12) | 117 to 208 | 125 |
| largest GPU memory of a runner | 1.99 GB | 0.44 GB | 0.43 GB |
| wall time | 61 min (19 chunks) | 21 min (13 chunks) | 3 min (63 molecules; the others copied) |

The two ICEBERG failures are a radical and a quaternary ammonium sulfate. Two ICEBERG chunks took 679 s and 439 s:
they contain molecules of 92 and 93 heavy atoms, where the fragmentation runs on the CPU (GPU at 3%); on Kaggle the
same molecules eat the ICEBERG budget.

**Differences to Kaggle, and the check.** RDKit 2026.03.3 instead of 2025.03.6 (the runner calls that "not
panel-exact"), Windows, torch 2.8. `check_kaggle.py` re-scored the first 30 molecules of the Kaggle run valF
(`runs/valF/stage_cache`) locally:

| Model | candidates | Pearson r | within 0.001 | within 0.01 | largest difference | best-scored candidate of the molecule agrees |
|---|---|---|---|---|---|---|
| GLACIER | 1,348 | 0.99818 | 83.9% | 97.4% | 0.18 | 25 / 30 |
| ICEBERG | 1,348 | 0.99656 | 77.1% | 93.8% | 0.12 | 23 / 30 |

So the local scores are the Kaggle scores up to a small tail of candidates, not bit-exact. Only one run and 30
molecules were checked.

## 2. What was scored, and coverage (task 2)

`build_cands.py` builds, for each selection (no join, v13a, v22a) and type, the candidate list exactly as
`nb/07_rescore.py` does: `ice_candidates` of the base top 60, plus the union with engine-2's first 40 that are not in
the base top 60, plus `ice_candidates` of the 25 PubChem-list entries when `lib_max` < 0.9; only members of
same-formula groups of two or more. The runners got the union over the twelve lists (41,899 candidates, 376
molecules). At replay time the scores are **masked to the list of the configuration being replayed**, so a candidate
scored for another type or selection never leaks in. Caches: `D:\casmi-data\fwd\ice.json`, `gl.json`, `gl_mh.json`
(the notebook's `stage_cache` format, `{mid: {smiles: score | null}}`), `cands.json`.

| Selection, type | molecules with candidates | candidates | of them in molecules the runners cover | with ICEBERG score | with GLACIER score | joined rows of the base top 60 with ICEBERG score (covered molecules) |
|---|---|---|---|---|---|---|
| v22a, `c1` | 148 | 9,810 | 7,342 | 7,342 (100.0%; 74.8% of all) | 7,342 | 306 / 343 |
| v22a, `c2` | 376 | 30,633 | 25,025 | 25,023 (100.0%; 81.7%) | 25,025 | 8,382 / 8,718 |
| v22a, `dA` | 376 | 30,833 | 25,192 | 25,190 (100.0%; 81.7%) | 25,192 | 8,773 / 9,108 |
| v22a, `dB` | 374 | 30,850 | 25,219 | 25,217 (100.0%; 81.7%) | 25,219 | 8,773 / 9,108 |
| v13a, `c2` / `dB` | 376 / 374 | 30,080 / 30,331 | 24,507 / 24,743 | 24,506 / 24,742 | all | 8,312 / 8,704 and 8,704 / 9,089 |
| no join, `c2` / `dB` | 376 / 374 | 27,021 / 27,020 | 21,905 / 21,907 | all | all | |

The 18% without a score are the candidates of the 74 molecules without an [M+H]+ or [M+Na]+ spectrum (negative mode,
[M+NH4]+ only); the notebook has no score for them either and the `FRAG` term (`gap`) takes over. The joined rows
without a score (335 of 9,108) are alone in their formula group. Before this study 27% of the joined rows had an
ICEBERG score and none a GLACIER score (`reviews/join-rank-2026-10-05.md`).

**Local stand-ins that remain.** The v1 engine and the PubChem list run with the fold-safe FPNet instead of the public
A+B models; the fusion's PubChem list (slot merge) is the fold-safe model's list in every selection, because v22a
keeps the public list there. `join-rank` used the fine-tune's list for the slot merge of the new selection, so its
zero-forward `dB` 0.280 is 0.276 here; v13a reproduces its 0.256 exactly.

## 3. Selections (task 3)

`replay.py`: `fusion_core.build_submission` through `scratch/mixed/regime_lib.py`, joined rows with pool id −2.
"Zero-forward" is the replay all earlier studies used (equal scores for every candidate of a covered molecule).

| Replay | `c1` | `c2` | `dA` | `dB` | `c2`: joined rows in the first 25 | `c2`: joined row first | `c1`: joined row first |
|---|---|---|---|---|---|---|---|
| no join, zero-forward | 0.8323 | 0.7390 | 0.2678 | 0.0799 | 0 | 0 | 0 |
| no join, real scores | 0.8238 | 0.7223 | 0.4269 | 0.1906 | 0 | 0 | 0 |
| v13a, zero-forward | 0.8311 | 0.7516 | 0.4496 | 0.2562 | 8.47 | 4.2% (16) | 4.0% (6) |
| **v13a** | 0.8223 | 0.7339 | 0.5016 | 0.3096 | 9.23 | 6.6% (25) | 4.0% (6) |
| v22a, zero-forward | 0.8298 | 0.7518 | 0.4704 | 0.2763 | 8.42 | 4.7% (18) | 4.7% (7) |
| v22a, ICEBERG only | 0.8233 | 0.7408 | 0.4983 | 0.3166 | 9.09 | 6.1% (23) | 4.7% (7) |
| v22a, GLACIER only | 0.8203 | 0.7379 | 0.5118 | 0.3295 | 9.04 | 6.3% (24) | 4.7% (7) |
| **v22a** | 0.8214 | 0.7288 | 0.5170 | 0.3396 | 9.24 | 8.2% (31) | 4.7% (7) |

| Difference | `c1` | `c2` | `dA` | `dB` |
|---|---|---|---|---|
| v22a, real minus zero-forward | −0.0084 ± 0.0068 (3 / 5) ne | −0.0230 ± 0.0119 (29 / 40) ne | +0.0466 ± 0.0136 (81 / 35) | +0.0633 ± 0.0138 (78 / 26) |
| v13a, real minus zero-forward | −0.0088 ± 0.0069 (3 / 5) ne | −0.0178 ± 0.0117 (31 / 35) ne | +0.0520 ± 0.0144 (84 / 32) | +0.0534 ± 0.0133 (73 / 24) |
| no join, real minus zero-forward | −0.0086 ± 0.0068 (4 / 5) ne | −0.0167 ± 0.0127 (35 / 37) ne | +0.1592 ± 0.0176 (138 / 31) | +0.1108 ± 0.0146 (99 / 18) |
| **v22a minus v13a, real** | −0.0009 ± 0.0012 (2 / 3) ne | −0.0051 ± 0.0046 (11 / 18) ne | +0.0153 ± 0.0083 (32 / 18) ne | +0.0300 ± 0.0098 (38 / 20) |
| v22a minus v13a, zero-forward | −0.0013 ± 0.0011 (0 / 4) ne | +0.0001 ± 0.0027 (6 / 8) ne | +0.0208 ± 0.0085 (31 / 20) | +0.0201 ± 0.0077 (39 / 20) |
| v22a minus no join, real | −0.0024 ± 0.0013 (0 / 5) ne | +0.0065 ± 0.0076 (25 / 23) ne | +0.0900 ± 0.0157 (82 / 29) | +0.1490 ± 0.0166 (111 / 20) |
| v13a minus no join, real | −0.0015 ± 0.0008 (0 / 5) ne | +0.0116 ± 0.0073 (26 / 17) ne | +0.0747 ± 0.0150 (78 / 31) | +0.1190 ± 0.0148 (97 / 17) |
| v22a: ICEBERG only minus zero-forward | −0.0064 ± 0.0051 ne | −0.0110 ± 0.0086 ne | +0.0279 ± 0.0103 (72 / 29) | +0.0403 ± 0.0108 (66 / 22) |
| v22a: GLACIER only minus zero-forward | −0.0095 ± 0.0073 ne | −0.0139 ± 0.0104 ne | +0.0414 ± 0.0119 (73 / 35) | +0.0532 ± 0.0112 (77 / 17) |

Readings:

- **v22a against v13a with real scores**: `dB` +0.030 ± 0.010 (zero-forward +0.020), `dA` +0.015 ± 0.008 ne, `c2`
  −0.005 ± 0.005 ne, and more exposure: joined row first on `c2` 6.6% → 8.2% (25 → 31 lists), which the zero-forward
  replay showed as 4.2% → 4.7%. The fine-tune's candidates are more similar to the answer, and the forward models
  put more of them above a pool answer. That is the configuration that is submitted as v22a; its exposure is higher
  than the hn-variants report could see.
- **The join's value on `c2` shrinks with real scores**: v13a minus no join +0.012 ± 0.007 ne, v22a +0.007 ± 0.008
  ne (zero-forward: +0.013 for both).
- **Without a join the forward models already lift outside answers a lot** (`dA` +0.159, `dB` +0.111): they pull
  engine-2's and the PubChem list's entries up after the fusion. Part of the join's zero-forward gain was therefore
  available without it; with real scores the join adds +0.090 (`dA`) and +0.149 (`dB`) instead of +0.203 and +0.196.

## 4. One knob at a time, against v22a (task 3)

Reference v22a: `c1` 0.8214, `c2` 0.7288, `dA` 0.5170, `dB` 0.3396; `c2` exposure 9.24 joined rows in the first 25,
joined row first 8.2% (31 of 380); `c1` joined row first 4.7% (7 of 150). The defaults are `ICE_LAM` = `GL_LAM` = 1,
`POST_*` = the same, `FUSE_ALPHA` 0.6, `FUSE_KRR` 3, `TOPN` 60, `ICE_UNION_K` 40; those values are the reference row.

| Knob | `c1` | `c2` | `dA` | `dB` | `c2`: joined rows in the first 25 | `c2`: joined row first | `c1`: joined row first |
|---|---|---|---|---|---|---|---|
| `ICE_LAM` 0 | −0.0011 ± 0.0065 ne | +0.0091 ± 0.0074 (29 / 19) ne | −0.0052 ± 0.0082 ne | −0.0102 ± 0.0083 (32 / 46) ne | 9.04 | 6.3% (24) | 4.7% (7) |
| `ICE_LAM` 0.5 | +0.0052 ± 0.0050 ne | +0.0104 ± 0.0054 (25 / 14) ne | −0.0047 ± 0.0051 ne | −0.0086 ± 0.0055 ne | 9.20 | 6.6% (25) | 4.7% (7) |
| `ICE_LAM` 2 | −0.0109 ± 0.0102 (2 / 9) ne | **−0.0262 ± 0.0070 (8 / 40)** | −0.0058 ± 0.0063 ne | +0.0024 ± 0.0064 ne | 9.33 | 10.5% (40) | 4.7% (7) |
| `GL_LAM` 0 | +0.0019 ± 0.0042 ne | +0.0114 ± 0.0081 (28 / 23) ne | **−0.0192 ± 0.0084 (23 / 46)** | **−0.0230 ± 0.0093 (20 / 51)** | 9.10 | 6.1% (23) | 4.7% (7) |
| `GL_LAM` 0.5 | +0.0035 ± 0.0034 ne | +0.0089 ± 0.0057 (21 / 10) ne | +0.0000 ± 0.0050 ne | −0.0105 ± 0.0067 ne | 9.18 | 6.6% (25) | 4.7% (7) |
| `GL_LAM` 2 | −0.0105 ± 0.0083 (1 / 9) ne | **−0.0293 ± 0.0069 (7 / 41)** | −0.0085 ± 0.0066 ne | +0.0012 ± 0.0062 ne | 9.31 | 10.5% (40) | 4.7% (7) |
| `GL_MH_ONLY` | +0.0004 ± 0.0041 ne | +0.0040 ± 0.0035 (10 / 4) ne | +0.0054 ± 0.0048 ne | +0.0058 ± 0.0037 ne | 9.25 | 7.9% (30) | 4.7% (7) |
| `POST_ICE_LAM` 0 | +0.0010 ± 0.0056 ne | +0.0101 ± 0.0066 (26 / 15) ne | −0.0081 ± 0.0070 ne | **−0.0168 ± 0.0051 (18 / 38)** | 9.02 | 7.1% (27) | 4.7% (7) |
| `POST_ICE_LAM` 0.5 | +0.0013 ± 0.0011 ne | +0.0035 ± 0.0052 ne | +0.0000 ± 0.0053 ne | **−0.0088 ± 0.0042 (19 / 19)** | 9.19 | 6.8% (26) | 4.7% (7) |
| `POST_ICE_LAM` 1, set explicitly | −0.0067 ± 0.0047 (0 / 3) ne | −0.0054 ± 0.0035 ne | **+0.0065 ± 0.0029 (18 / 1)** | **+0.0015 ± 0.0007 (6 / 0)** | 9.24 | 8.2% (31) | 4.7% (7) |
| `POST_GL_LAM` 0 | +0.0027 ± 0.0035 ne | **+0.0133 ± 0.0066 (26 / 13)** | −0.0108 ± 0.0065 ne | **−0.0187 ± 0.0074 (18 / 42)** | 9.09 | 6.6% (25) | 4.7% (7) |
| `POST_GL_LAM` 0.5 | +0.0031 ± 0.0034 ne | +0.0036 ± 0.0042 ne | +0.0036 ± 0.0044 ne | **−0.0126 ± 0.0053 (18 / 27)** | 9.19 | 7.1% (27) | 4.7% (7) |
| `POST_GL_LAM` 1, set explicitly | −0.0077 ± 0.0055 ne | −0.0017 ± 0.0025 ne | **+0.0082 ± 0.0027 (18 / 1)** | +0.0011 ± 0.0006 ne | 9.24 | 8.2% (31) | 4.7% (7) |
| `FUSE_ALPHA` 0.4 | −0.0021 ± 0.0022 ne | +0.0004 ± 0.0050 ne | **−0.0318 ± 0.0060 (11 / 83)** | **+0.0127 ± 0.0029 (53 / 4)** | 9.85 | 8.9% (34) | 5.3% (8) |
| `FUSE_ALPHA` 0.8 | +0.0057 ± 0.0046 ne | −0.0053 ± 0.0055 ne | **+0.0239 ± 0.0062 (75 / 20)** | **−0.0163 ± 0.0040 (6 / 51)** | 8.68 | 5.5% (21) | 4.0% (6) |
| `FUSE_KRR` 1 | +0.0100 ± 0.0061 ne | −0.0006 ± 0.0053 ne | −0.0050 ± 0.0058 ne | +0.0027 ± 0.0045 ne | 9.18 | 8.2% (31) | 5.3% (8) |
| `FUSE_KRR` 10 | +0.0002 ± 0.0091 ne | **−0.0225 ± 0.0073 (25 / 46)** | +0.0042 ± 0.0063 ne | **−0.0240 ± 0.0055 (17 / 53)** | 9.57 | 6.3% (24) | 3.3% (5) |
| stage B off for joined molecules | −0.0011 ± 0.0013 ne | −0.0002 ± 0.0033 ne | +0.0008 ± 0.0039 ne | +0.0003 ± 0.0023 ne | 8.62 | 7.9% (30) | 4.7% (7) |

Two combined settings, run because each is a single switch in practice (not a grid):

| Setting | `c1` | `c2` | `dA` | `dB` | `c2`: joined rows in the first 25 | `c2`: joined row first |
|---|---|---|---|---|---|---|
| `ICE_LAM` 0 and `GL_LAM` 0 (real coverage; `FRAG` gap as on Kaggle) | +0.0084 ± 0.0068 ne | +0.0230 ± 0.0119 ne | **−0.0466 ± 0.0136** | **−0.0633 ± 0.0138** | 8.42 | 4.7% (18) |
| `POST_ICE_LAM` 0 and `POST_GL_LAM` 0 | +0.0080 ± 0.0061 ne | **+0.0331 ± 0.0081 (40 / 9)** | **−0.0377 ± 0.0115 (31 / 67)** | **−0.0457 ± 0.0094 (15 / 66)** | 8.36 | 5.0% (19) |

Three more that the task named as never measured (same replay, the candidate mask rebuilt for each):

| Knob | `c1` | `c2` | `dA` | `dB` | `c2`: joined row first |
|---|---|---|---|---|---|
| `TOPN` 40 | −0.0044 ± 0.0035 ne | −0.0015 ± 0.0039 ne | **+0.0051 ± 0.0022 (24 / 8)** | +0.0018 ± 0.0019 ne | 7.6% (29) |
| `ICE_UNION` off | −0.0003 ± 0.0003 ne | −0.0020 ± 0.0014 ne | **−0.0361 ± 0.0082 (9 / 38)** | +0.0030 ± 0.0017 ne | 8.2% (31) |
| `ICE_UNION_K` 20 | +0.0011 ± 0.0011 ne | −0.0016 ± 0.0013 ne | −0.0008 ± 0.0006 ne | +0.0004 ± 0.0003 ne | 8.2% (31) |

Notes on single rows:

- **`POST_*_LAM` 1 set explicitly is not the default.** With `None` the post-fusion weight follows `ICE_LAM_LIB` /
  `GL_LAM_LIB` (0 for `lib_max` ≥ 0.9); a number switches the post-fusion re-rank on for library hits as well. That is
  what the row measures: +0.007 to +0.008 on `dA` (18 / 1), −0.007 ne on `c1`. The same applies to the 0.5 rows for
  the library hits, where the weight goes from 0 to 0.5.
- **On `c2` the two forward models together cost more than either alone**: 0.7408 (ICEBERG only) and 0.7379
  (GLACIER only) against 0.7288 with both (zero-forward 0.7518); on `c1` the three are within 0.003 of each other.
- **Weight 2 is the only forward setting that is clearly worse**, and it is worse where exposure rises (`c2` joined row
  first 8.2% → 10.5%).
- **`FUSE_ALPHA`** behaves as in the zero-forward replay of the mixed-regime report: engine-2 weight helps where
  engine-2 has the answer and hurts where it cannot. With joined rows a lower alpha also raises the exposure (8.9%).
- **Stage B off for joined molecules** changes nothing measurable and removes 0.6 joined-or-PubChem rows from the
  first 25 of a `c2` list; the first place is unaffected.
- TOPN above 60 cannot be replayed: the cached base lists hold 60 rows.

## 5. Where the forward terms act: per stage

`extra.py`. v22a, all four types; "RRF" = stage C with the post-fusion weights at zero.

| Stage | scores | `c1` | `c2` | `dA` | `dB` | `c2`: joined rows in the first 25 | `c2`: joined row first | `c2`: answer first |
|---|---|---|---|---|---|---|---|---|
| after stage A | zero-forward | 0.8279 | 0.7186 | 0.3339 | 0.3339 | 10.17 | 8.9% (34) | 63.9% |
| after stage A | real | 0.8219 | 0.7233 | 0.3455 | 0.3455 | 10.12 | 8.7% (33) | 63.9% |
| after stage B | zero-forward | 0.8269 | 0.7162 | 0.3428 | 0.3428 | 11.22 | 8.9% (34) | 63.9% |
| after stage B | real | 0.8206 | 0.7208 | 0.3583 | 0.3583 | 11.14 | 8.7% (33) | 63.9% |
| after the RRF, no post re-rank | zero-forward | 0.8298 | 0.7518 | 0.4704 | 0.2763 | 8.42 | 4.7% (18) | 68.2% |
| after the RRF, no post re-rank | real | 0.8295 | 0.7619 | 0.4793 | 0.2939 | 8.36 | 5.0% (19) | 69.2% |
| final | zero-forward | 0.8298 | 0.7518 | 0.4704 | 0.2763 | 8.42 | 4.7% (18) | 68.2% |
| final | real | 0.8214 | 0.7288 | 0.5170 | 0.3396 | 9.24 | 8.2% (31) | 64.5% |

- **Stage A with real scores is a gain on every answer type that it can touch and leaves the exposure alone**:
  through the RRF it carries `c2` +0.010, `dA` +0.009, `dB` +0.018 with 18 → 19 lists topped by a joined row. The
  reason is structural: `popularity_reorder` tags generated and joined rows `|gen`, so in stage A the forward
  re-rank orders joined rows among joined rows and pool rows among pool rows, each in its own slots.
- **The post-fusion re-rank is where pool answers and joined rows are compared by the forward models.** `c2`: the
  answer is first in 263 lists before it and 245 after; 23 first places are lost, 11 of them to a joined row, 5 are
  won, none from a joined row. On `c1` the whole loss of the forward stage (−0.008) also arises here.
- **An emulated variant** (not a `fusion_core` flag; `extra.py` hands the fusion a formula function that keeps the
  `|gen` tag on joined rows after the fusion, so the post re-rank cannot move a joined row past a pool row):
  `c1` 0.0000, `c2` **+0.0158 ± 0.0044 (21 / 2)**, `dA` **−0.0272 ± 0.0090 (20 / 41)**, `dB` **−0.0367 ± 0.0080
  (13 / 57)**; `c2` joined rows in the first 25 9.24 → 8.32, joined row first 8.2% → 5.3% (20 lists). It gives back
  about half of the `c2` cost and four fifths of the `dB` gain of the post re-rank; it is a weaker version of
  `POST_*_LAM` 0, not a free repair.

## 6. Stage accounting for joined answers (task 4)

`stages.py`. Answer-removed lists of the 325 record molecules; an answer "arrives as a joined row" in 177 lists
(v22a) and 157 (v13a). 151 of the 177 answers have forward scores (the others are in molecules the runners do not
cover); of their 7,816 same-formula competitors in the base top 60 (14.8 pool rows and 31.1 joined rows per list),
6,637 have an ICEBERG score.

**v22a, the 177 joined answers:**

| Stage | zero-forward: MRR / first | real scores: MRR / first / first 5 / first 25 |
|---|---|---|
| PubChem list of the fine-tune | 0.501 / 36.7% | |
| joint base list, public ranker | 0.566 / 41.8% | |
| after stage A | 0.565 / 41.8% | 0.591 / 43.5% / 80.8% / 93.8% |
| after stage B (slot merge) | 0.556 / 41.8% | 0.586 / 43.5% / 78.5% / 93.2% |
| after the RRF, `dB` | 0.464 / 30.5% | 0.497 / 32.2% / 72.9% / 92.7% |
| **final, `dB`** | 0.464 / 30.5% | **0.553 / 39.5% / 79.1% / 92.1%** |
| after the RRF, `dA` | 0.704 / 58.8% | 0.724 / 61.0% / 89.3% / 98.9% |
| **final, `dA`** | 0.704 / 58.8% | **0.723 / 60.5% / 87.0% / 99.4%** |

Steps with real scores, per joined answer: joint list → A +0.025 ± 0.018 (40 / 22) ne; A → B −0.006 ± 0.003 (2 / 42);
B → RRF **−0.089 ± 0.013 (5 / 75)** on `dB`, +0.139 ± 0.019 (75 / 9) on `dA`; RRF → final **+0.056 ± 0.014 (46 / 13)**
on `dB`, −0.001 ± 0.015 (24 / 23) on `dA`. Real minus zero-forward, final: `dB` +0.089 ± 0.023 (60 / 21), `dA`
+0.019 ± 0.019 ne. The 150 answers in covered molecules: `dB` final 0.502 → 0.606. The 27 in uncovered molecules
stay at 0.257 (the `FRAG` term costs them 0.011 ne in stage A, the slot merge 0.021, the RRF 0.065).

- **What is above a listed answer in the final `dB` list**: 1.13 joined rows and 1.25 others (zero-forward 1.31 and
  1.98). After the RRF alone 1.21 and 2.26: the fusion puts one more pool or engine-2 row above the answer, the post
  re-rank removes it again.
- **The forward models and the answer** (145 answers scored together with competitors, mean group 47): the answer
  has the best ICEBERG score of its formula group in 32.4%, the best GLACIER score in 44.8%; it beats a single pool
  competitor 89% (ICEBERG) / 88% (GLACIER) of the time and a joined competitor 90% / 90%. Where ICEBERG puts the
  answer first (47) the final `dB` MRR is 0.870 (zero-forward 0.663); where not (98) 0.506 (0.450). GLACIER: 0.862
  (65) and 0.430 (80). Of the 74 answers first in the joint list 55 are still first at the end (zero-forward 54); 15
  others reach the first place.
- **v13a, 157 joined answers**: PubChem list 0.389, joint list 0.580, final `dB` 0.481 (zero-forward) → 0.560 (real),
  `dA` 0.729 → 0.730; steps base → A +0.025 ± 0.017, B → RRF −0.100 ± 0.015, RRF → final +0.060 ± 0.017. The same
  picture; the selection model does not change what the forward terms do to a joined answer.

**Where the rank is lost now, `dB`, in MRR per record molecule against an answer that is always first** (v22a, real
scores; zero-forward of `join-rank` section 1 in brackets):

| Loss | per record molecule |
|---|---|
| answer joined, lost in the joint list (ranker order against same-formula pool rows and joined decoys) | 177 / 325 × (1 − 0.566) = 0.236 (0.24) |
| answer not in the first 100 of the PubChem list (72; final 0.017) | 0.218 (0.22) |
| answer in the first 100 but the list is at or above the gate (76; final 0.148, zero-forward 0.083) | 0.199 (0.21) |
| answer joined, lost in the RRF | 0.048 |
| answer joined, slot merge | 0.003 |
| answer joined, regained in stage A / in the post-fusion re-rank | −0.014 / −0.031 |

So with real scores the fusion's net cost for a joined answer is 0.02 per record molecule instead of 0.05; the order
of the three large losses is unchanged and none of them is in the stage this study measured. The forward models are
not a route to the 0.24: they are right about a joined answer against one competitor nine times in ten, but with 46
same-formula competitors per list that leaves the answer first in a third to under a half of the groups.

## 7. Recommendations per knob

Exposure is treated as a hard constraint (v14a, v19a: more or better-placed joined rows scored lower on the
leaderboard than the regime predicted). "No" = keep the value of v22a.

| Knob | Change it? | Why |
|---|---|---|
| `ICE_LAM` (0, 0.5, 2) | **No** | 2 is worse on `c2` (−0.026 ± 0.007) with more exposure; 0 and 0.5 show no evidence in any type |
| `GL_LAM` (0, 0.5, 2) | **No** | 2 is worse on `c2` (−0.029 ± 0.007) with more exposure; 0 costs `dA` and `dB` 0.02; 0.5 no evidence |
| `GL_MH_ONLY` | **No** | no evidence in any type (largest +0.006 ± 0.004), exposure unchanged; leaderboard v14c = v13a |
| `POST_ICE_LAM` / `POST_GL_LAM` (0, 0.5) | **No as a default; the one lever to hold in reserve** | lowers exposure (8.2% → 5.0% for both at 0) and gains `c2` +0.033 ± 0.008, at −0.038 on `dA` and −0.046 on `dB`; the net sign depends on the unknown share of PubChem-only answers |
| `POST_*_LAM` set to a number while `ICE_LAM_LIB` is 0 | **No** | switches the post re-rank on for library hits: `c1` −0.007 ne, 0 / 3 |
| `FUSE_ALPHA` 0.4 / 0.8 | **No** | a pure trade between `dA` and `dB` (±0.013 to ±0.032), `c2` and `c1` no evidence; 0.4 raises exposure |
| `FUSE_KRR` 1 / 10 | **No** | 10 is worse on `c2` and `dB` (−0.023, −0.024); 1 no evidence |
| stage B off for joined molecules | **No** | no evidence in any type; first-place exposure unchanged |
| `TOPN` 40, `ICE_UNION` off, `ICE_UNION_K` 20 | **No** | `TOPN` 40: `dA` +0.005 ± 0.002 only; union off costs `dA` 0.036 |

If the next leaderboard result for v22a is below v13a although v22a is the better list locally, the first thing to
test is the post-fusion re-rank, because that is where v22a's extra exposure arises (6.6% → 8.2%) and where the
zero-forward studies were blind. That would be a new experiment behind the existing `POST_ICE_LAM` / `POST_GL_LAM`
knobs, with the trade stated above; this study does not recommend it on its own numbers.

## 8. Honesty and limits

- **The forward models have very likely seen these molecules.** Both checkpoints are MassSpecGym models and every
  hold-out molecule is a training structure with public spectra (`mixed-regime` section 2 and 7). The answer's
  forward score is therefore optimistic. This affects: every gain attributed to the forward terms on `dA` and `dB`
  (+0.047, +0.063, the +0.056 of the post re-rank, "the answer beats a competitor 89%"): upper bounds. It affects
  the costs on `c2` and `c1` in the other direction: here the pool answer is helped by the leak and still loses, so
  the cost of the forward terms on pool-answer lists and the exposure counts (a joined row first) are more likely
  understated than overstated. v20a (a different intensity checkpoint, −0.030 on the leaderboard) fits a stage this
  sensitive, but nothing here measures a checkpoint swap.
- **Other leaks, unchanged from the mixed-regime report**: engine-2's fingerprint models have memorised the
  molecules (`dA` and `c2` levels are optimistic, the fusion looks better than it is); the answers are famous
  (`POP_MU`); "answer removed" hides a row of a known structure, so `dA` / `dB` levels are upper bounds. Differences
  between knobs on the same lists are less affected than levels.
- **Local stand-ins**: fold-safe FPNet instead of the public A+B models in the v1 engine and in the PubChem list; the
  fine-tune's selection is the real one. `c2` uses one emulated tie order per run (`jr_lib`).
- **Composition**: 11% timsTOF, 81% of molecules covered by the forward models; the hidden test is all timsTOF
  natural products. 74 molecules contribute nothing to any forward-knob difference.
- **The regime has failed to predict the leaderboard for changes in the number or rank of joined rows.** Every knob
  here that helps `dA` / `dB` does so by ranking joined rows higher. No leaderboard estimate is given for anything.
- **Not verified**: bit-exactness of the local scores (r 0.997 to 0.998 on 30 molecules of one Kaggle run; RDKit
  differs); the public A+B selection; the ICEBERG budget on Kaggle (locally everything was scored; on Kaggle the
  runner stops at `ICE_BUDGET` 5,400 s, and large molecules took up to 11 minutes per chunk here); `TOPN` above 60;
  `ICE_FT`; the `ICE_SURROGATE` path. No Kaggle stage dump of a joined run exists locally, so the joined
  configuration itself was not compared with a Kaggle run.
- **Failed or stopped**: nothing failed. Two "wait for the job" shell commands of mine were stopped by the harness
  time limit; the scoring jobs themselves ran to the end. ICEBERG returned no score for two candidates.

## 9. Reproduce

```
export CASMI_DATA='D:\casmi-data'; P=.venv/Scripts/python
$P scratch/fwd/build_cands.py                  # candidate groups per selection and type -> D:/casmi-data/fwd/cands.json (70 s)
$P scratch/fwd/run_fwd.py --model gl           # resumable; 21 min
$P scratch/fwd/run_fwd.py --model gl --mh_only # 3 min
$P scratch/fwd/run_fwd.py --model ice          # resumable; 61 min from the 2 October cache
$P scratch/fwd/check_kaggle.py --run valF      # local scores against a Kaggle dump
$P scratch/fwd/replay.py && $P scratch/fwd/report.py   # sections 2 to 4 (2 min)
$P scratch/fwd/stages.py                       # section 6
$P scratch/fwd/extra.py                        # section 5
```

`fw_lib.py` holds the selections and the `nb/07` candidate rule. Outputs: `replay.json`, `replay_permol.parquet`,
`coverage.json`, `stages.parquet`, `report.md`, `stages.md`, `extra.md` under `D:\casmi-data\fwd\`. This file is new;
run `python tools/gen_index.py` when it is committed.

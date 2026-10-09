<!-- index: joint spectrum-structure model, phase 0: fixed leak-safe benchmark (DEV / FINAL by scaffold), evaluate(scores), leak-safe training data with isomer negatives, reference model, ablation loop -->
# Joint spectrum-structure model (phase 0: benchmark and ablation loop)

Our own model of a different kind from the fingerprint network: spectra and structures are embedded in one space and
trained contrastively with same-formula isomers as negatives; its score is meant to enter the fold-safe ranker
(`RANKER 'fs_avg'`, v33b) as new features. This folder holds the benchmark every run is judged on, the data plumbing
and a small reference model. Nothing here runs on Kaggle yet. Only `train.parquet`, the public pool tables and the
PubChem tier are read (no NIST). Report of phase 0: `reviews/joint0-2026-10-09.md`.

## Files
| File | What it is |
|---|---|
| `jpaths.py` | paths under `$CASMI_DATA` |
| `bench_build.py` | builds the fixed benchmark once (refuses to overwrite); `--runs` adds the coverage table |
| `bench.py` | loader, ranking, strata, bootstrap; **`evaluate(scores_parquet, baseline, half, as_feature)`**; baselines table |
| `cand_fp.py` | fingerprints of benchmark candidates that are not pool rows (per half, resumable) |
| `data.py` | leak-safe index, `JointData`, `BatchDataset` (molecule batches with K isomer negatives), `assert_leak_safe`, `stats` |
| `graphs.py` | SMILES to graph arrays for the GNN variant (no torch-geometric) |
| `model.py` | peak-set transformer, fingerprint MLP, GNN, InfoNCE |
| `train_joint.py` | modes `load` / `bench` / `train` / `score`; shared GPU through `../gpushare.py`, resume, GPU ledger |

Tests (no competition data): `tests/test_joint_bench.py`, `tests/test_joint_data.py`.

## Benchmark (`$CASMI_DATA/joint/bench`, fixed)
Molecules: the 5,350 hold-fold molecules of the fsranker3 table (fold 0 and the natural-product panel of the
identity-disjoint split; no fold-safe model has seen them), halved by generic Murcko scaffold: **DEV 2,675 / FINAL
2,675**, no scaffold in both. Decide on DEV; look at FINAL only when a phase ends. Lists (only those holding the
answer are stored; `runs.parquet` has the denominators):

| view | what | DEV lists | FINAL lists |
|---|---|---|---|
| `c2J` | pool window + 100 joined PubChem-only rows, answer a pool row (production list) | 1,508 | 1,504 |
| `dropJ` | answer removed from the pool, joined rows added: the answer is a PubChem-only row | 1,063 | 989 |
| `c2N` | pool window only (larger n) | 2,515 | 2,475 |

Class-1 lists (`c1N`, `c1J`) are stored for the ranker refit only. A score file is a parquet of
`(molecule_id, candidate_key, score)`; `Bench.pairs(half)` lists the pairs to score with SMILES and pool index.

```bash
export CASMI_DATA='D:\casmi-data'; P=.venv/Scripts/python; J=solutions/fusion-v4n/train/joint
$P $J/bench.py --baselines --half dev                              # coverage + random / fz / pub / ours / avg, per stratum, ± bootstrap SE
$P $J/bench.py --scores S.parquet --baseline avg --half dev        # the score alone, paired difference to a baseline
$P $J/bench.py --scores S.parquet --as_feature --half dev          # ranker refitted out of fold by scaffold with the score's four features
```
`as_feature` is the number that matters: the production recipe (165 inputs, LightGBM lambdarank, public
hyper-parameters, every list type of the training molecules) refitted on DEV with and without `jm__score`,
`jm__z`, `jm__rank`, `jm__gap`, compared alone and inside the rank average with the public ranker (v33b's base list).
The refit without the score is cached in `bench/refit/`. It reads the feature table `rankbank4/data/X.npy`.

## Training data (`$CASMI_DATA/joint/index`, 136 MB; no copy of the spectrum cache)
```bash
$P $J/data.py --build      # 6 s
$P $J/data.py --stats      # leak-safety assertions (printed counts) + negative statistics
```
Positives: the 220,443 structures (1,982,405 spectra) outside the held-out folds; a structure sharing an InChIKey
first block, metric key or raw key with a held-out one is removed (none had to be). Negatives per positive, in this
order: same-formula pool isomers (half of K), cached PubChem window negatives of `../mine_pc_negatives.py` (hard and
random), near-mass pool rows of another formula, random pool rows. Held-out structures may be drawn as negatives by
default (`neg_hold=True`): on the hidden test a class-2 answer is a pool row and is drawn as a negative too.
The loader needs the two fingerprint tables in the OS file cache (`data.warm()`, 4.9 GB of standby memory); cold, a
batch takes over a second.

## Loop
```bash
$P $J/train_joint.py --mode bench --variant fp --bss 128,256 --steps 40 --tag bench_fp     # throughput, memory, inference speed
$P $J/train_joint.py --mode train --variant fp --bs 256 --k 15 --minutes 30 --max_steps 13000 --tag abl_x --resume
$P $J/cand_fp.py --half dev                                                                 # once per half (fingerprint variant)
$P $J/train_joint.py --mode score --ckpt $CASMI_DATA/joint/runs/abl_x/last.pt --half dev
$P $J/bench.py --scores $CASMI_DATA/joint/runs/abl_x/scores_dev.parquet --as_feature
```
`--minutes` is a wall-time budget across resumes; `--max_steps` fixes the cosine schedule. Exit code 75 = the GPU was
yielded (wrap with `../guard_run.py`). `--frac` trains on a fixed subset of the molecules. Every GPU process appends
its wall seconds to `joint/gpu_ledger.jsonl`. Batch 256 with K 15 needs up to 7.9 GB under the 8 GB cap; batch 384
does not fit (GNN: batch 128).

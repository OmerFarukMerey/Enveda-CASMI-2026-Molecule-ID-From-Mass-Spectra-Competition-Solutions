<!-- index: fusion-v4n clean chain (started 2026-10-10): our own rebuild of the inputs that carry a licence problem; provenance and licence of every input, the files of this folder, commands, state -->
# Clean chain: inputs that are ours or permissively licensed

Goal (user decision of 2026-10-10): a submission that complies with the rules and is eligible for prizes. Plan,
licence table, measurements and schedule: [docs/competition/clean-chain-plan-2026-10-10.md](../../../docs/competition/clean-chain-plan-2026-10-10.md).
This folder holds the parts that need no GPU. Nothing here is wired into the notebook yet, and nothing here imports
the `casmi` / `fe_v4` packages of the public datasets (they carry no code licence).

## Files
| File | What it is |
|---|---|
| `pool_build.py` | builds our own candidate-pool tables in the v1 engine's schema from `train.parquet` and an upstream COCONUT release; stages `sources`, `structs`, `assemble`, `verify`, each resumable |
| `chemcore.py` | our own chemistry core: adduct parsing and neutral mass, spectrum cleaning, bond-cleavage fragment masses |
| `golden.py` | writes golden fixtures from stored outputs of the current chain to `$CASMI_DATA/clean/golden/` (not in git) |
| `SPEC.md` | behaviour specification of every module a clean-room rewrite has to replace, with its acceptance test |
| `../tests/test_clean_contract.py` | data-free rule tests and golden tests (skipped when the fixtures are absent) |
| `fpnet.py` | our own fingerprint network: model class, peak preparation, collate, bit unpacking, checkpoint I/O (reads and writes the field layout of the public fold-safe checkpoint) |
| `specdata.py` | our own data access: spectrum-cache builder from `train.parquet`, loader, leak check, spectrum merging, element table, negatives, samplers, worker stream |
| `train_fpnet.py` | trainer and hold-out evaluation (Apache-2.0 derived loop, see below): mixed precision, resume, GPU sharing, seeds, run ledger |
| `../tests/test_own_fpnet.py`, `../tests/test_fs_net.py` | data-free tests of the three files above and of the notebook key `RANKER_FS_NET` |
| `subform.py` | ours: subformula annotation of peaks (ion formula of any adduct, best subformula within max(15 ppm, 0.004 Da)), cache builder (`reviews/specenc-2026-10-13.md`) |
| `chemrules.py` | ours: knowledge-driven MS/MS rules (diagnostic losses / ions -> structural requirements) re-ordering same-formula rows in a list's first 10; not wired in, local verdict fail (`reviews/chemrules-2026-10-17.md`); tests `../tests/test_chemrules.py` |
| `specenc.py` | ours: spectrum encoder with configurable peak tokens (m/z = `fpnet.FPNet`, subformula, pairwise mass differences, multi-spectrum, intensity rank, external embedding token), items, data stream; `RefSignature` for the reference interface |
| `train_specenc.py` | trainer of `specenc.py` (the Apache-2.0 derived loop of `train_fpnet.py`): kept checkpoints with learning curves, true / pipeline-formula evaluation, iso top-1 |
| `../tests/test_specenc.py` | data-free tests of the three files above |

## Own fingerprint network (2026-10-10): what is whose
| Part | Status |
|---|---|
| `train_fpnet.py`: loss, optimiser groups, learning-rate schedule, hold-out window evaluation and its query groups | **derived from the Apache-2.0 loop** `train_fpnet.py` of `dmitriigluzdov/casmi26-fold-safe-fpnet` (through `train/train_fpnet.py`); notice in `THIRD_PARTY_NOTICES.md` |
| `train_fpnet.py`: worker processes, step schedule, resume, ledger, bucketed forward, sampling / augmentation / freeze / instrument options; `train/gpushare.py`, `train/guard_run.py` | ours |
| `fpnet.py`, `specdata.py` | ours; they import NumPy, pandas, PyTorch and `chemcore.py` only |
| data | `train.parquet` only (spectrum cache; pool tables for negatives and fingerprints: the reference tables for the drop-in network, `pool_build.py` tables for a network on our own bit selection) and the published split `split.parquet` (which structures are held out) |

**Clean-room limit, stated plainly.** No file of the unlicensed `casmi` package was opened or copied for these three
files. But (1) `SPEC.md`, which they follow, was written by an author who had read that package; (2) a drop-in network
must have the checkpoint's parameter layout and the engine's input conventions, and those were pinned by **observing
the reference as a black box**: the parameter names and shapes of `fpnet_fold0.pt`, the inputs and outputs of the
reference module's submodules recorded with forward hooks, and the outputs of its preparation / merge / collate
functions and of its data tables on test inputs (`scratch/ownfp/probe_*.py`, `diag*.py`); (3) one traceback printed
two source lines of the reference `forward` during probing (the concatenation of the three peak embeddings), which
showed nothing the hooks had not already shown; (4) `scratch/ranker/precompute_logits.py`, an earlier script of ours
that mirrors how the engine builds its query items, was read while preparing the ranker refit, after the three files
were written. The result is an independent implementation that is functionally identical where the interface
demands it; whether that is clean enough is the host's call, as for the rest of this folder. Decided by us without
looking (and therefore possibly different from the reference): initialisation, augmentation, how merges are drawn in
training, the fill of small candidate windows, instrument rules, hold-out structures as negatives (kept), training
only on spectra with at least one cleaned peak.

Parity (`scratch/ownfp/parity.py`, 2026-10-10): `fpnet_fold0.pt` loaded into our class gives **bit-identical logits**
to the reference class on 512 real hold-out queries on the CPU in float32 (maximum difference 0.0; 0.10 under
float16 autocast on the GPU against a logit standard deviation of 1.64); peak preparation identical on 3,000 of 3,000
spectra, merging on 1,500 of 1,500 groups, structure-to-pool map on 275,810 of 275,810, element table on all but 12
pool rows (charged formulas the reference parses wrongly); the engine's loader reads a checkpoint written by ours;
the hold-out table of `fpnet_fold0.pt` through our code equals the stored one within 0.001 top-1 on every subset.
`build_cache` on row group 0 (131,072 spectra): every metadata column and every offset identical; 266 spectra with
more than 512 raw peaks keep different peaks at the cut (the network reads the 160 most intense).

Commands (run directories on drive C:; `W` = a directory with the spectrum cache, the pool tables and `split.parquet`):
```bash
W='D:\casmi-data\directions\run\work'; O='C:\Users\hemre\casmi-work\ownfp\runs'; T=solutions/fusion-v4n/train; C=solutions/fusion-v4n/clean
# baseline of 2026-10-10: published length, 3.3 h at duty 0.8 on the RTX 4070 Ti SUPER, 6.7 GB; seeds 0 and 1
$P $T/guard_run.py --free_gb 9 -- python $C/train_fpnet.py --work $W --out $O --tag base_s0 --seed 0 --bs 256 --max_steps 82855 --resume
$P $C/train_fpnet.py --work $W --out $O --tag x_ref_fold0 --eval_only 'D:\casmi-data\directions\dl\fpnet_fold0.pt'   # hold-out table of a checkpoint
$P scratch/ownfp/run_queue.py     # the queue scratch/ownfp/queue.txt, one job at a time; ledger <O>/ledger.jsonl, hours in casmi-work/ownfp/hours.jsonl
$P solutions/fusion-v4n/tests/test_own_fpnet.py && $P solutions/fusion-v4n/tests/test_fs_net.py
```

## Provenance of every input
| Input | Where it comes from | Licence | Read by |
|---|---|---|---|
| `train.parquet` | the competition (Kaggle) | CC BY-NC 4.0 by the Rules tab; the host: "should be considered open source" (thread 745072, 2026-10-09/10; 745841) | `pool_build.py sources`, `golden.py` |
| `coconut_csv-10-2026.zip` | COCONUT 2.0, October 2026 release, downloaded by us on 2026-10-04 from https://coconut.naturalproducts.net/download (local: `$CASMI_DATA/extra_db/raw/`, 247,959,854 bytes, 738,794 rows). **Not** taken from a Kaggle dataset | "COCONUT data is released under the Creative Commons CC0 license" (download page, read 2026-10-10). Cite Chandrasekhar et al., Nucleic Acids Res. 2024, gkae1063 | `pool_build.py sources` |
| RDKit 2026.03.3 | PyPI / the metric's wheel dataset | BSD-3-Clause | everything |
| NumPy, pandas, pyarrow | PyPI | BSD-3-Clause / Apache-2.0 | everything |
| reference tables for verification only: `pool_meta.parquet`, `pool_fp.npy`, `fp_bits.npy`, `train_structs.parquet`, `pool_frag_*.npy` of `ahmedberatozer/casmi26-v2-pool`; our spectrum cache built with the public code | public Kaggle dataset, "Other" with a non-commercial source note | compared against, **never an input** of the tables we build (`pool_build.py verify`, `golden.py`; `--bits ref` is a diagnostic switch and is off by default) | `verify`, tests |

Not used: the ChEBI / LIPID MAPS set (engine 2 only; `sergiomalv/casmi-chebi-lmsd-structures`, CC BY 4.0, is already
clean), the PubChem tier (plan, section 2: rebuildable from PubChem's public-domain FTP files, not started).

## Commands
```bash
export CASMI_DATA='D:\casmi-data'; P=.venv/Scripts/python; C=solutions/fusion-v4n/clean
$P $C/pool_build.py sources                      # 10 s: 275,810 training structures, 479,690 COCONUT connectivities
$P $C/pool_build.py structs --workers 4          # about 9 CPU hours (tautomer canonicalisation), resumable per chunk
$P $C/pool_build.py assemble                     # tables -> scratch/clean/pool/tables/ (about 1.3 GB)
$P $C/pool_build.py verify                       # against the existing pool -> scratch/clean/pool/verify.json
$P $C/golden.py && $P solutions/fusion-v4n/tests/test_clean_contract.py
```
Outputs go to `scratch/clean/pool/` (drive C:, git-ignored) because drive D: has about 3 GB free.

## State (2026-10-10)
See the plan, section 4, for what was run and the coverage numbers. Not built yet: the fragment tables of the pool
(`pool_frag_*`, needs a compiled version of `chemcore.fragment_masses`), every module of `SPEC.md` sections 5 to 9.

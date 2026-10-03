<!-- index: local training of our own models (FPNet fine-tunes): data layout, commands, provenance, results -->
# Local training

Models we train ourselves, on the team's own machine (RTX 4070 Ti SUPER 16 GB). Only `train.parquet` and public,
NIST-free artefacts are used. Nothing here runs on Kaggle; a trained checkpoint goes to Kaggle as a dataset and is
selected with a config flag.

## Files
- `train_fpnet.py`: train or fine-tune the fingerprint network. Derived from the public fold-safe training loop
  (`dmitriigluzdov/casmi26-fold-safe-fpnet`, `train_fpnet.py`, Apache-2.0); see `THIRD_PARTY_NOTICES.md`.
- `guard_run.py`: supervisor for a shared machine; restarts the trainer with `--resume` after it yielded the GPU.
- `sim_iceberg.py`: simulate [M+H]+ spectra of pool structures with the packaged ICEBERG checkpoint (ms-pred, open
  MassSpecGym weights, MIT, as packaged in `ahmedberatozer/casmi26-iceberg`; not NIST-trained) and write them in the
  layout of the spectrum cache, for `train_fpnet.py --sim`. Chunked and resumable, same GPU sharing.

## Sharing the machine
The PC is also used for games. The trainer runs at below-normal priority with three data threads and sleeps between
steps so that it takes only a share of the GPU: `--duty` (0.8) normally and `--game_duty` (0.25) while a game runs. A
game is any GPU application that is not in the `DESKTOP` list of `train_fpnet.py`, or more than 5 GB of GPU memory
held by other processes. Above `--yield_gb` (9 GB held by others) the trainer saves `<tag>_resume.pt` and exits with
code 75. Measured: 13 steps/s unthrottled, 3.1 steps/s at duty 0.25, 3.8 GB of GPU memory at batch 64.

## Data layout (not in git)
A run directory with `work/` holding the spectrum cache of `train.parquet` (`spec_mz.npy`, `spec_it.npy`,
`spec_off.npy`, `spec_meta.parquet`), the pool tables of `ahmedberatozer/casmi26-v2-pool` (`pool_fp.npy`,
`pool_meta.parquet`, `fp_bits.npy`, `train_structs.parquet`) and `split.parquet` of the fold-safe dataset
(identity-disjoint folds; fold 0 = `hold`, the 265 natural-product panel structures = `np`). The `casmi` package
(`build.py`, `chem.py`, `fpnet.py`, `spectra.py`, `traindata.py`) comes from `ahmedberatozer/casmi26-v4b-models`.
Currently at `D:\casmi-data\directions\run` and `D:\casmi-data\directions\code`.

## Commands
```bash
P=.venv/Scripts/python; T=solutions/fusion-v4n/train
C="--code D:\\casmi-data\\directions\\code --run D:\\casmi-data\\directions\\run"
# hold-out table of a checkpoint (11 subsets: instrument, polarity, natural-product catalogue, np panel)
$P $T/train_fpnet.py $C --eval_only 'D:\casmi-data\directions\dl\fpnet_fold0.pt' --tag base_fold0
# fine-tune from the fold-safe checkpoint with half of the spectrum-uniform samples from NP-catalogue structures
$P $T/guard_run.py -- python $T/train_fpnet.py $C --init 'D:\casmi-data\directions\dl\fpnet_fold0.pt' \
    --np_p 0.5 --bs 64 --lr 1e-4 --warmup 500 --max_steps 20000 --duty 0.8 --game_duty 0.25 --resume --tag ft_np50
```
Batch 64 needs 3.8 GB of GPU memory and runs at 13 steps/s (830 spectra/s); 20,000 steps take about 26 minutes.

## Experiments
Question 1 (2026-10-02): does natural-product-weighted sampling improve the model where the hidden test lives?
Both runs start from the fold-safe checkpoint, which never saw fold 0 or the np panel, so the hold-out numbers are
honest. Control = same fine-tune without the weighting, to separate "more training" from "NP weighting".

| Run | Start | Sampling | Steps | hold NP-catalogue top-1 / MRR (n 3,000) | np panel top-1 / MRR (n 380) | hold not NP top-1 |
|---|---|---|---|---|---|---|
| `base_fold0` (public checkpoint) | | | | 0.389 / 0.546 | 0.347 / 0.527 | 0.706 |
| `ft_ctrl` | fold-safe | uniform | 20,000 | 0.393 / 0.549 | 0.345 / 0.525 | 0.702 |
| `ft_np50` | fold-safe | `--np_p 0.5` | 20,000 | 0.386 / 0.545 | 0.337 / 0.527 | 0.699 |

Result (2026-10-02): **no effect.** Differences are within ±0.01 (standard error of a top-1 difference about 0.013 on
3,000 groups, 0.036 on the panel). More training with the public recipe changes nothing, and drawing half of the
spectrum-uniform samples from natural-product structures changes nothing either. The model is saturated on this
data; its weakness on natural products (0.39 against 0.70) is not a sampling problem. A natural-product-weighted
fine-tune of the pipeline's checkpoints is therefore not worth a submission on this evidence. What the model lacks
is spectra of natural products it has not seen; the next experiment gives it simulated ones (question 2).

Question 2 (2026-10-02): does the model find a structure from a real spectrum more often when it was trained on a
SIMULATED spectrum of that structure? Full report: `reviews/simspec-2026-10-02.md`. ICEBERG [M+H]+ spectra at 20 /
40 / 60 eV for T (1,400 held-out natural-product-catalogue structures with real spectra), Tp (half of the np panel)
and B (10,000 catalogue structures without any spectrum); C (1,401) and Cp (the other panel half) are the same
populations, not simulated. A quarter of every batch is simulated (`ft_sim25`), otherwise the recipe of `ft_ctrl`.

```bash
S=D:/casmi-data/simspec    # structs.parquet / sim_structs.parquet / eval_sets.parquet come from scratch/simspec/select_structs.py
$P $T/guard_run.py -- python $T/sim_iceberg.py --structs $S/sim_structs.parquet --out $S/sim1      # 34,569 spectra, 20 min
$P $T/sim_iceberg.py --structs $S/sim_structs.parquet --out $S/sim1 --assemble
$P $T/guard_run.py -- python $T/train_fpnet.py $C --init 'D:\casmi-data\directions\dl\fpnet_fold0.pt' \
    --sim $S/sim1 --sim_pow 2 --sim_floor 0.001 --sim_p 0.25 --extra_sets $S/eval_sets.parquet \
    --bs 64 --lr 1e-4 --warmup 500 --max_steps 20000 --duty 0.8 --game_duty 0.25 --resume --tag ft_sim25
$P $T/train_fpnet.py $C --extra_sets $S/eval_sets.parquet --ranks_out --eval_only <ckpt.pt> --tag x_<name>   # per-group ranks
```
`--sim_pow 2`: ICEBERG returns intensities on the sqrt scale (ms-pred trains on sqrt intensities); squared, the
simulated spectra have the peak-count profile of measured timsTOF spectra.

Paired top-1 difference to `ft_ctrl` on real spectra (both 20,000 steps; standard errors clustered by structure):

| Queries | groups | control | `ft_sim25` | difference |
|---|---|---|---|---|
| T (simulated) | 2,178 | 0.407 | 0.520 | +0.113 ± 0.008 |
| T, structures outside the MassSpecGym sources | 404 | 0.334 | 0.465 | +0.131 ± 0.018 |
| np panel timsTOF, simulated half | 189 | 0.333 | 0.540 | +0.206 ± 0.033 |
| C (not simulated) | 2,185 | 0.399 | 0.413 | +0.014 ± 0.007 |
| np panel timsTOF, other half | 191 | 0.356 | 0.340 | −0.016 ± 0.024 |
| hold all / hold not NP-catalogue | 3,000 | 0.669 / 0.702 | 0.670 / 0.700 | +0.001 / −0.002 ± 0.006 |

Result (2026-10-02): **a large gain on the simulated structures, almost none elsewhere, and most of it will not
carry over to a full run.** The model learns a preference for structures it was trained on. A simulated truth beats
a never-seen competitor more often (+0.035 per pair), but a competitor that was simulated too hardly more often
(+0.010 ± 0.006), and a truth that was not simulated loses to simulated competitors (−0.027 ± 0.005). In a full run
all class-2 competitors are simulated. Projected gain there: +0.03 to +0.08 top-1 on class-2-like structures (not
measured). Next: the window arm (`scratch/simspec/run_chain5.sh`: the competitors of 431 targets simulated too),
before the 13-hour simulation of all 436,515 class-2 pool entries.


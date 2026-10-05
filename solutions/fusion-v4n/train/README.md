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


Question 3 (2026-10-05): does the model choose better among PubChem isomers when PubChem isomers are among its
training negatives? Full report: `reviews/hardneg-2026-10-05.md`. The PubChem channel hands the 100 best PubChem-only
structures by f.z to the engine, in windows of about 7,500 rows, and the model had only ever been trained against the
pool's window (about 56 rows). `mine_pc_negatives.py` collects, for each of the 219,155 training structures, 16 hard
negatives (highest ECFP4 Tanimoto in a 1,024-row sample of its ±10 ppm PubChem window; 81% share the answer's
formula) and 32 random window rows per formula (43%); `train_fpnet.py --pc_neg` adds 16 + 16 of them to the 63 pool
negatives of every sample. Otherwise the recipe of `ft_ctrl`.

```bash
H=D:/casmi-data/hardneg
$P $T/mine_pc_negatives.py --work D:/casmi-data/directions/run/work --code D:/casmi-data/directions/code \
    --tier D:/casmi-data/pubchem_study/tier --ik14 D:/casmi-data/pubchem_study/popds/pc_ik14.npy --out $H/w10   # 2.6 h, 2 workers
$P $T/mine_pc_negatives.py ... --out $H/w10 --assemble                                                        # fp.npy 4.0 GB
$P $T/guard_run.py --free_gb 12 --checks 2 --every 20 -- python $T/train_fpnet.py $C \
    --init 'D:\casmi-data\directions\dl\fpnet_fold0.pt' --pc_neg $H/w10 --pc_k 32 --pc_hard 16 \
    --bs 64 --lr 1e-4 --warmup 500 --max_steps 20000 --duty 0.8 --game_duty 0.25 --yield_gb 11 --resume --tag ft_hn
```

"Answer among the first N PubChem-only structures", both passes of the channel, truth-removed molecules with a PubChem
record that the model never saw; paired difference to `ft_ctrl`:

| Molecules | n | first 25 | first 50 | first 100 | first 200 |
|---|---|---|---|---|---|
| mixed regime, `ft_ctrl` | 325 | 0.600 | 0.655 | 0.705 | 0.751 |
| mixed regime, `ft_hn` | 325 | 0.705 (+0.105 ± 0.018) | 0.742 (+0.086 ± 0.017) | 0.778 (+0.074 ± 0.016) | 0.825 (+0.074 ± 0.015) |
| second hold set, `ft_hn` − `ft_ctrl` | 689 | +0.132 ± 0.014 | +0.102 ± 0.012 | +0.078 ± 0.011 | +0.058 ± 0.010 |
| np panel (Enveda timsTOF), `ft_ctrl` → `ft_hn` | 250 | 0.704 → 0.804 | 0.748 → 0.868 | 0.836 → 0.904 | 0.864 → 0.928 |

Pool side (hold-out table, paired): hold all top-1 0.669 → 0.685 (+0.016 ± 0.006), hold NP-catalogue 0.393 → 0.408,
np panel timsTOF 0.345 → 0.403 (+0.060 ± 0.022).

Result (2026-10-05): **the first training change that moves the model on unseen molecules; the answer is as often in
the first 25 as it was in the first 100.** Two thirds of the change of the logits is one constant vector (the same on
two halves of the molecules, cosine 0.999): a prior for what a measured structure looks like among PubChem rows.
Adding that vector to the control's logits gives +0.06 of the +0.07 at N 100. It is learned from the training
libraries; it holds on the Enveda panel, which no model here was trained on, but the hidden test is the only real check.
Production = the same fine-tune applied to public models A and B (commands in the report, section 7); not started.


Question 4 (2026-10-05): can we make our own natural-product-tuned FPNet for engine 2 (the role of
`goooman/casmi26-fp-npx4-soup`, whose base model is undocumented)? Full report: `reviews/np-ours-2026-10-05.md`.
Engine 2's network is `nb/embed/eng/pv_fp.FPNet` (prvsiyan's architecture: 6,930 bits, an instrument embedding, raw
collision energy), not the v1 network, so it has its own trainer, `train_eng_fpnet.py` (same sharing, resume and
logging). Base: the public `prvsiyan/casmi26-fp-models-v2/fp_single_s2.pt` (CC0), the per-spectrum model engine 2
already runs in every submission.

```bash
R=D:/casmi-data/np_ours/run; B=D:/casmi-data/mixed_regime/ds/fpv2/fp_single_s2.pt
# one fold: the fold's 50 Enveda NP standards and the 380 replay molecules are never trained on
$P $T/guard_run.py --free_gb 12 --checks 1 --every 5 -- python $T/train_eng_fpnet.py --run $R --init $B --mix $B \
    --hold_keys D:/casmi-data/mixed_regime/mols.parquet --hold_np_fold 0 --seed 0 --boost lib:enveda-np-examples=100 \
    --max_steps 2000 --bs 64 --lr 5e-5 --warmup 200 --w_bce 0 --resume --tag p_np100_f0        # 2.6 min, 2.7 GB
$P $T/train_eng_fpnet.py --run $R --soup $R/p_np100_f0_final.pt,...,$R/p_np100_f4_final.pt --out $R/fp_single_np100_soup.pt
```

Start-up check of the base (printed as `BASE`): logit standard deviation over the bits 12.2, f.z spread inside a
candidate list 405 to 420, **ranking loss at temperature 1: 30 to 38** (the failure of the raw recipe on public models
A / B), at the fitted temperature 90.5: 0.80 to 0.93, BCE 0.33, training top-1 0.75 to 0.78. The loss divides f.z by
that fixed temperature, so the fine-tune starts at the base's own optimum; the 200-step smoke run kept every
evaluation set within ±0.01 and the loss at 0.73 to 0.84 (the base is converged on data it has seen).

The base has not seen the Enveda NP standards (`enveda-np-examples`, 250 structures): it ranks them at MRR 0.48
against 0.77 to 0.92 on the other libraries. They are therefore a real unseen set; every fold model is evaluated on
the 50 it did not train on. Rank of the answer by f.z in its ±10 ppm pool window, mean logits over the structure's
np-examples spectra, pooled over the five folds (n 250, paired to the base):

| Arm (5 folds × 2,000 steps, lr 5e-5, batch 64) | share of samples | unseen standards, model alone | unseen, ½ (model + base) | enveda-180 (seen), alone | replay molecules (not trained on), alone |
|---|---|---|---|---|---|
| base `fp_single_s2` | | 0.478 | | 0.910 | 0.864 |
| NP standards ×100 (`p_np100`) | 4.2% | 0.613 (+0.136 ± 0.018, 114 / 23) | 0.558 (+0.080 ± 0.013) | −0.003 ± 0.003 | −0.017 ± 0.004 |
| all timsTOF rows ×4 (`p_tims4`) | 77% | 0.500 (+0.022 ± 0.009) | 0.488 (+0.010 ± 0.006) | +0.009 ± 0.003 | −0.006 ± 0.003 |
| fold 0 only (n 50): no boost / ×4 / ×20 with loss weight 4 / ×100 / ×500 | | +0.046 ± 0.029 / +0.053 ± 0.029 / +0.173 ± 0.043 / +0.233 ± 0.052 / +0.214 ± 0.051 | | | |

Result (2026-10-05): **training on 200 Enveda NP standards moves the 50 the model has not seen from 0.48 to 0.61**;
oversampling all timsTOF spectra does not. A literal 4× of the standards is 0.17% of the samples and does nothing
beyond the control. In engine 2's own mix (½ [½ (s2 + ours) + merged]) the unseen standards go 0.490 → 0.538
(+0.048 ± 0.010). On the local replay (no NP standard, no timsTOF natural product among its 380 molecules) the soup
is indistinguishable from no extra model, and below goooman's model on engine 2's `c2` list (−0.008 ± 0.004).
Soups: `D:\casmi-data\np_ours\run\fp_single_np100_soup.pt`, `fp_single_tims4_soup.pt`. GPU: 0.31 h for the ten
fold runs, 0.11 h of pilots. A model trained from scratch with the same trainer (`scr_np100`: 20,000 steps, lr 3e-4,
`--temp 90.5 --w_bce 0.2`, 0.30 h) reaches 0.68 to 0.84 on seen libraries against the public model's 0.78 to 0.92
and adds nothing as an extra model either. Neither is as good as goooman's model in the replay; the dataset
`casmi26-fp-np-ours` was not created (flag value `ENG_EXTRA_FPNET='np-ours'` and preset `v25b` exist, default off).

<!-- index: ens3 / ens3e (closed): own seeds of bgunshui's 52-epoch kit recipe as extra members of his 0.480 neural ensemble; trainer, kernel builder, member correlation -->
# ens3: own scorer seeds in bgunshui's 0.480 neural ensemble (closed)

bgunshui's public `casmi26-480-neural-ensemble` (LB 0.480) is MIDX with a second model of sanpier's spectrum-scorer kit,
retrained 52 epochs on every identity by bgunshui's public kernel `casmi26-scorer-train-public`. The final lists are
ranked by 0.5 z(public model) + 0.5 z(retrained model), with z-scores per molecule. ens3 adds three more seeds of the
same recipe as extra members. Experiments rows 124 and 125 in `docs/experiments.md`.

| File | What it does |
|---|---|
| `train_scorer_seeds.py` | our trainer (kit features and model, Apache-2.0, plus the competition's adduct names; parallel structure preparation, scaffold folds). Seed mode trained the members: `NN_INPUT=<dir with train.parquet> NN_WORK=<work dir> NN_SKIP_F=1 NN_A_RANGE=20,23 NN_EPOCHS=52 python train_scorer_seeds.py` writes `out/A20..A22/last.pt`, about 12 min per seed on a laptop RTX 5070 |
| `build_ens3.py` | builds the private kernel: bgunshui's notebook with the ensemble line replaced by `W_PUB z(public) + W_BG z(retrained) + W_OM z(mean of our seeds)`; members found under `/kaggle/input` by SHA-256, fallback to 0.5 / 0.5 if any is missing |
| `corr.py` | per-molecule Spearman and top-5 overlap between members, from a commit run's `neural_*.parquet` files |

The three checkpoints (`om_seed_A20..A22.pt`, kit format `dict(state_dict, epoch=52, stage='full_refit')`) are in the
private dataset `omerfarukmerey/casmi26-om-nn-b3`, not in git.

**Result.** ens3 (0.5 / 0.25 / 0.25) **0.469**, ens3e (equal thirds) **0.472**, against 0.480 for the base. On the 12
visible lists our seeds agree with bgunshui's model as closely as with each other (Spearman 0.88-0.89 against
0.88-0.90), so the recipe is reproduced. The 0.480 depends on bgunshui's exact checkpoint. More seeds of the recipe
are not a lever on this pipeline (the same lesson as n3, row 114). A public release of the seeds was staged and
dropped.

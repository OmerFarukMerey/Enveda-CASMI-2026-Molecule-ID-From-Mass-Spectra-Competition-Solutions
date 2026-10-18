<!-- index: GLACIER fine-tune on Enveda timsTOF spectra with the official ms-pred code: provenance, data, recipe, commands, results -->
# GLACIER fine-tune (glacierft)

Our training code for fine-tuning the GLACIER forward model (stack1's isomer re-scorer) with the **official ms-pred code** on
permitted spectra (train.parquet). The host allows official checkpoints and retraining with the authors' code on permitted data
when the training code is delivered (thread 748882); this folder is that code. Study, pre-registration and verdict:
`scratch/glacierft/PREREG.md`, `solutions/fusion-v4n/reviews/glacierft-2026-10-18.md`.

## Provenance
| Item | Source | Licence |
|---|---|---|
| Model code, losses, optimiser, featurisation, batch layout, decoy mutation | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred) @ `708148c2a8eb0c32120644436fefd2fe90cd2214`, local checkout (`$MSPRED_SRC`, default `C:\Users\hemre\casmi-work\ext\ms-pred\src`), imported unchanged | MIT |
| Starting weights | the authors' open MassSpecGym GLACIER checkpoint (ms-pred README "GLACIER", GLACIER_checkpoint.zip → best.ckpt, sha256 `5a47cecc…`), as packaged in `ahmedberatozer/casmi26-glacier` `ckpt/glacier.pt` (re-saved state_dict, sha256 `9192e4c2…`) | MIT |
| dgl / torch_scatter / LinSATNet stand-ins | `casmi26-glacier/shim/` (the same files stack1's runner uses) | own code of the package |
| `shim/pygmtools` | ours: `sinkhorn` / `hungarian` in pure torch (pygmtools not installed) | MIT (this repo) |
| pytorch_lightning 2.6.6, torchmetrics, einops | the casmi26-iceberg wheel site (`$ICE_SITE`) | Apache-2.0 / MIT |
| Training data | `train.parquet` of the competition (CC BY-NC 4.0): Enveda-180 timsTOF [M+H]+ and Orbitrap / Q-TOF [M+H]+ of MSnLib, RIKEN, MassBank, MoNA, Spectraverse, MS-DIAL; decoy structures from the team's candidate pool (casmi26-v2-pool tables). **No NIST data**, nothing downloaded | — |

## Files
- `env.py`: sys.path and import stubs for the official code; `load_model` / `save_model` (same `{hyper_parameters, state_dict}` layout as
  the package checkpoint, so `gl_runner.py --ckpt FILE` loads a fine-tuned file unchanged).
- `build_data.py`: scaffold-disjoint split (held-out H1 / H2, validation V, training) and the runner-format isomer items.
- `data.py`: one training sample = target spectrum (ms-pred's raw_spec processing) + decoys (ms-pred `mutate` + pool same-formula), ms-pred
  `IntenContrDataset.collate_fn`; conditioning exactly as stack1's runner (5-eV CE bucket, timsTOF → QTOF token).
- `train_glacierft.py`: the loop: `JointModel._common_step(batch, 'train')`, `configure_optimizers()`, gradient clip 5; validation = the
  runner's own scoring of V in process, early stopping on V isomer MRR; GPU sharing (gpushare, exit 75 → `guard_run.py`), chunked runs.
- Tests: `solutions/fusion-v4n/tests/test_glacierft.py` (no competition data).

## Recipe and its differences from the official contrastive fine-tune
Official (`configs/glacier/joint_contr_finetune_msg.yaml`): all parameters, lr 2.7e-4, batch 16, 5 of 20 decoys (half mutations, half
PubChem same-formula), contr_threshold 0.4, sk_tau 0.05, fragment loss 0.1 on MAGMa labels, early stopping on validation loss.
Ours: the same model, loss terms and optimiser, with
1. **only the intensity head trains** (`token_mapper`, `inten_decoder`, `inten_encoder`, `output_map`; 6.7 M of 15.1 M parameters). The
   Graphormer encoder and the fragment generator stay frozen in eval mode, so the fragment loss is off (weight 0) and no MAGMa labels are
   needed. Without the fragment loss an unfrozen encoder would move the fragment generator unsupervised;
2. the PubChem half of the decoys comes from our local pool (random same-formula structures, evaluation keys removed);
3. lr 5e-5, warm-up 500, early stopping on **validation isomer MRR** (the target metric) instead of the loss;
4. a plain PyTorch loop (one GPU, Windows, shared machine) instead of Lightning's Trainer.

## Commands
```bash
P=.venv/Scripts/python; T=solutions/fusion-v4n/train/glacierft
$P scratch/glacierft/r_meta.py                 # replay truths (needs the stack1-like replay lists, casmi-work/arb)
$P $T/build_data.py                            # -> casmi-work/glacierft/data/
$P scratch/glacierft/run_gpu.py train          # GPU_LOCK + guard_run; -> casmi-work/glacierft/run_ft1/best.pt
$P scratch/glacierft/run_gpu.py eval H_base H_ft mx_base mx_ft p2_base p2_ft
$P scratch/glacierft/primary.py; $P scratch/glacierft/replay_eval.py
```

## Results (2026-10-18, `reviews/glacierft-2026-10-18.md`)
Run `run_ft1`: stopped by patience at 7,000 steps, best V at 4,000 (V isomer MRR 0.919 → 0.935), 0.92 GPU-h; `best.pt` sha256
`02b326df…` (not uploaded anywhere). Held-out timsTOF isomer MRR (443 molecules, not in MassSpecGym, never trained) 0.8915 → 0.8998,
+0.008 [−0.007, +0.024]; stack1-like replay +0.002 composite (CI spans 0). Fails the pre-registration: not a Kaggle candidate.

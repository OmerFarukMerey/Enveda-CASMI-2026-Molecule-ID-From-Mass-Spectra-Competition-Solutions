<!-- index: fingerprint-conditioned structure generator (MSNovelist idea, our own): evaluation sets and leak rules, noise calibration, tokeniser, model, trainer, sampler, evaluation and the rule-based baseline -->
# fpgen: a learned fingerprint-conditioned structure generator

Our own answer to the block the pipeline cannot reach: structures that are **in no database** (about 45% of hidden molecules by the
coverage study). A sequence model writes a SMILES string conditioned on (a) a molecular fingerprint and (b) the molecular formula. It
needs no spectra to train: it learns from structures alone, with the true fingerprint **corrupted to look like a predicted one**. At test
time the fingerprint comes from the fold-safe fingerprint network and the formula from the pipeline's candidate formulas.

Nothing here runs on Kaggle. Only `train.parquet`-derived tables, COCONUT (through the public pool) and the PubChem tier are read; no
NIST data, no downloads, no installs. Report of phases 0 and 1: [`../../reviews/fpgen-2026-10-12.md`](../../reviews/fpgen-2026-10-12.md),
pre-registration `scratch/fpgen/PREREG.md`.

## Files
| File | What it is |
|---|---|
| `fpaths.py` | paths: inputs under `$CASMI_DATA`, everything written under `$FPGEN_WORK` (default `C:\Users\hemre\casmi-work\fpgen`) |
| `tok.py` | SMILES tokeniser, per-token element table, vocabulary, formula parsing and counting |
| `chemrows.py` | per-structure chemistry pass (stereo-free canonical SMILES, 8 randomised SMILES, generic Murcko scaffold, InChIKey first block), parallel and resumable |
| `sets.py` | the three evaluation sets (DEV class N, set F, scaffold hold-out), the pool rows removed from every training set, `leak_check()` |
| `fplogits.py` | fold-safe network logits of held-out molecules for the calibration (GPU, shared), and a parity check against the stored DEV logits |
| `noise.py` | per-bit calibration of the network and the corruption that turns true bits into a predicted-looking conditioning vector |
| `data.py` | training universe, vocabulary, batches (conditioning fingerprint, formula counts, token ids); the PubChem arm |
| `pubchem.py` | the PubChem natural-product-like / mass-window subset with its fingerprints and the same leak removals |
| `model.py` | the transformer decoder (fingerprint and formula as prefix tokens) and the formula-constrained sampler |
| `train.py` | trainer: GPU sharing (`../gpushare.py`), resume, ledger, leak assertions printed in every run |
| `evalgen.py` | sample a checkpoint on the evaluation sets, keep valid formula-matching structures, rank them, report the pre-registered metrics |
| `rulebase.py` | the rule-based generator of `gen3` on the same molecules with the same hit test: the baseline to beat |
| `keycheck.py` | recompute the competition metric key for the top ranked rows (check of the cheap tautomer-set hit test) |
| `queue_run.py` | one job at a time, restart after a GPU yield, ledger and GPU-hour budget |

Tests (no competition data): `../../tests/test_fpgen.py`.

## Commands
```bash
export CASMI_DATA='D:\casmi-data'; P=.venv/Scripts/python; G=solutions/fusion-v4n/train/fpgen
$P $G/chemrows.py --workers 5                       # chemistry pass over the pool (4 min)
$P $G/fplogits.py --parity 60                       # the logit path against the stored DEV logits
$P $G/fplogits.py --n 20000                         # calibration logits of held-out molecules (70 s on the GPU)
$P $G/noise.py --fit                                # per-bit calibration -> cal/noise.npz, noise.json
$P $G/sets.py                                       # evaluation sets + removal mask (3 min)
$P $G/data.py --vocab --stats                       # vocabulary (55 tokens)
$P $G/rulebase.py --sets N,F,SH                     # the rule-based baseline (add --ban_scaffold for the matched variant)
$P $G/queue_run.py --queue scratch/fpgen/queue.txt --budget_h 14    # trains and evaluates, one job at a time
```

## Facts measured here (2026-10-12, details in the review)
- Universe after the leak removals: 677,745 structures (pool = train + COCONUT); vocabulary 55 tokens; 31.9 M parameters at d 512 / 8 layers.
- Throughput on the RTX 4070 Ti SUPER at duty 0.8: **7.3 steps/s at batch 160 (1,170 structures/s), 7.7 GB**; sampling 2,200 sequences/s
  at batch 2,048 with a preallocated key-value cache (51/s without it).
- The conditioning fingerprint is weakly informative per bit (median separation of the two class-conditional logit means 0.32 standard
  deviations; Tanimoto between the thresholded prediction and the truth 0.22), which the noise model reproduces to three decimals.

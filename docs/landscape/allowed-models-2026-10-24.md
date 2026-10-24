<!-- index: 2026-10-24 desk check of the host-allowed models as NEW evidence: MS2DeepScore 2.0 (Zenodo 17826815 dual-mode model, 416 MB, CC BY 4.0, Apache-2.0 code with public training pipeline, trained on GNPS + MassBank EU + MoNA + MSnLib, no NIST in training), Kaggle copies (ranjithch, sabreenelkamash; no user notebook), offline install on Python 3.12, T4 runtime (embed 2.4 M train spectra 10-20 min, exact GPU search seconds), integration as a per-candidate "triangulation" term below the library gate; MS2Query (no), FIORA-OS (gap-only, confounded -0.010 pair), MIST; public notebooks that tried them -->
# Allowed models as new evidence, 2026-10-24

Desk research only: nothing downloaded except metadata (Zenodo and PyPI APIs, Kaggle dataset metadata, one public
notebook rendered). Marks: **[V]** read at the source today; **[I]** inference or estimate. Rulings: the host allowed
MS2DeepScore 2.0 and FIORA-OS by name (751373, see [update-2026-10-24b.md](update-2026-10-24b.md)).

## 1. MS2DeepScore 2.0 [V]

| Item | Finding | Source |
|---|---|---|
| Model | "Dual mode MS2Deepscore model (MS2DeepScore >2.6)", 2025-12-05, **CC BY 4.0**. Files: `ms2deepscore_model.pt` **416,126,840 B** (md5 `2150e186969862e276287b2b958b9273`), `embedding_evaluator.pt` 791,865 B (optional: predicts the error of an embedding, filters low-information spectra), `settings.json` 2,836 B, `embedding_model_settings.json` 273 B. The README names this record as the current pretrained model. An older dual-mode model (14290920, 2024-12, 416,126,504 B) exists | zenodo.org/records/17826815 (API) |
| Architecture | Siamese dense net: 0.1 Da bins from m/z 10 to 1000 (9,900 inputs), intensity^0.5, + ion-mode bit + precursor m/z / 1000 → 10,000 → **500-d embedding**; cosine of two embeddings predicts the **Tanimoto of the two structures** (RDKit "daylight" 4,096 bits). Peaks above m/z 1000 are ignored. Trained 2025-08-18, 250 epochs, 1/20 of InChIKey14 held out for validation and for test (seed not recorded) | `settings.json` |
| Training data | "GNPS library", "MassBank EU", "MoNA", "MS2 spectra of MSnLib": 519,580 positive spectra (36,638 molecules) and 145,594 negative spectra (18,480 molecules). The methods never use NIST for training (NIST appears only as a plasma reference material in a case study). The 2024 split is published (Zenodo 13934470, CC BY 4.0, 1.5 GB of mgf). Caveat: "GNPS library" probably includes GNPS-NIST14-MATCHES, but the host allowed that library by name (745832) and MS2DeepScore 2.0 by name (751373) | bioRxiv 10.1101/2024.03.25.586580v5; Nat. Commun. 17, 2483 (2026) |
| Modes | Positive, negative and cross-mode (cross-mode predictions above 0.9 are rare) | README, paper |
| Code, licence | `matchms/ms2deepscore`, **Apache-2.0** (PyPI), includes `train_new_model/`, so the training code is public | PyPI, GitHub |
| Package | `ms2deepscore` 2.11.0 (2026-09-08, py3-none-any, 141,694 B, Python ≥ 3.11). Its dependencies include `chemap` (pulls scikit-fingerprints) and `onnxscript`. **2.7.2** (2026-02-25, 103,655 B) loads models from 2.6 on and needs only matchms, numba, numpy ≥ 2, pandas, scikit-learn, tensorboard, torch ≥ 2.4, tqdm, matplotlib. `matchms` 0.33.1 (213,818 B) wants rdkit ≥ 2024.3.5 (2026.03.3 satisfies it), scipy < 1.17, networkx < 3.5. Offline plan [I]: `pip install --no-index --no-deps` of ms2deepscore 2.7.2, matchms 0.33.1, pickydict (6 KB), sparsestack (10 KB), pyteomics (262 KB), deprecated (22 KB); about 0.6 MB of pure-Python wheels. Never let pip touch rdkit, numpy or torch. Not tested on the Kaggle image | PyPI JSON |
| Kaggle copies | `ranjithch/casmi26-ms2deepscore-model` (09-21, label CC0-1.0, no description, `ms2deepscore_model.pt` 416,126,840 B, the same byte size as the Zenodo 2025-12 file, checksum not verified). `ranjithch/casmi26-ms2deepscore-embeddings` (CC0, `train_embeddings.npy` 480 MB = **240,000** × 500 float32 (a subset of train), `test_embeddings.npy` = 400 × 500, meta parquet). `sabreenelkamash/ms2deepscore-models` (10-16, same 416 MB file) and `sabreenelkamash/ms2deepscore` (3.27 GB, **cp311** wheels incl. torch 2.6, ms2deepscore 2.7.2, rdkit 2024.3.2: wrong Python for our image). **No notebook uses any of the four** (`kaggle kernels list --dataset` returns "Not found"). The embeddings dataset has no generating code, so it is not usable under 751373 ("we need the code that generates the intermediates") | `kaggle datasets files` / `metadata` |

**Public notebooks [V].** Only `hansamaqht/casmi-competitive-ms2deepscore` (10-22) has MS2DeepScore in its title. Its code is a
pure-Python modified cosine over the first 800,000 train spectra, with no model, and it scores **0.000**. The same
author's `casmi-dreams-embedding-retrieval` (standalone DreaMS kNN against train) scores 0.086. No public notebook in
the 781 runs MS2DeepScore. So there is no LB pair.

## 2. How MS2DeepScore could add evidence [I]

- **(a) Analogue search as a candidate source: no.** The neighbours are train structures. For class 1 the library
  channel already finds them, and for classes 2 and 3 the answer is not in train. DreaMS neighbours as candidates or
  tail fill scored below their bases (update-2026-10-02, -04). Engine 2's analog propagation is already a spectral
  analogue mechanism.
- **(b) Per-candidate "triangulation" term (recommended).** The model predicts Tanimoto(query structure, reference
  structure) for every reference spectrum. For candidate *c*, compare those predictions with the exact
  Tanimoto(c, ref) over the query's top K (about 200) neighbours, one per InChIKey14, plus a few hundred random
  references of the same polarity. Score = -Σ w_j (pred_j - T(c, ref_j))², with w_j growing with pred_j and summed
  over the molecule's spectra, optionally weighted by the embedding evaluator. The score is defined for **every
  candidate**, including PubChem-only (class 2) and generated (class 3) structures. It uses a model trained on
  different libraries with a different objective from our FPNets, DreaMS heads and forward models. Fingerprints
  must be RDKit `RDKFingerprint`, 4,096 bits, as in training (check the exact parameters in `fingerprint_utils.py`).
- **Where to add it.** In the final merge of the s1 base, only below the library gate (`lib_max` < 0.7), as a same-formula
  z-scored term at low weight, or as an RRF list at weight ≤ 0.25. Re-ranks of existing candidates have been
  sign-unstable (sanpier +0.001 at 0.25, -0.021 at 0.5; shuheitak -0.032). Library hits stay bit-identical.
  The term should sit behind a default-off flag.
- **Local validation trap.** MS2DeepScore was trained on most public-library molecules in train.parquet (gnps,
  massbank, mona, pluskal_ms2), and its 2025 split is unpublished. Evaluate on **enveda-180 / enveda-np-examples
  molecules absent from those four libraries**, with the query molecule's own spectra removed from the reference set.
  Otherwise the predicted similarities are memorised and the gain will be optimistic.

**T4 runtime (estimate).** The model has about 104 M weights (9,902 × 10,000 + 10,000 × 500), so about 0.2 GFLOP
per spectrum. Embedding all 2.43 M train spectra is about 0.5 PFLOP: 1 to 3 min of GPU in fp16/fp32. Total wall
time is about **10 to 20 min**, dominated by parquet reading and binning. With vectorised binning it is closer to 10 min;
matchms `Spectrum` objects take about 20 min of CPU. An "~1 M reference" subset halves that. The embeddings take
2.4 GB fp16 on the GPU. Search: 1,400 queries × 2.43 M × 500 is about 1.7 TFLOP, so an **exact chunked matmul with
top-k finishes in seconds** and no ANN index is needed. The triangulation needs RDKit fingerprints for about 430 × 300
candidates plus about 90 k neighbour structures: about 5 CPU minutes [I]. Total cost is about 15 to 25 min of the
9 h budget. Alternative: precompute train embeddings in a public notebook (code public, so the intermediate is allowed)
and attach them as a 2.4 GB fp16 dataset.

## 3. Other allowed models, briefly

| Model | Status | Evidence | Verdict |
|---|---|---|---|
| **MS2Query** 1.5.4 (Apache-2.0) | Pins `ms2deepscore==2.0.0`, `torch<2.6`, `onnxruntime<1.16`, `matchms<=0.26.4`, which clash with the image. Its library is GNPS, already in train. Not named by the host | none on Kaggle | **no**; rebuild the idea with MS2DeepScore and train as the library (section 2) |
| **FIORA-OS** v1.0.0 (MIT, MSnLib v7 only, CC BY 4.0) | `ahmedberatozer/casmi26-fiora-os-v1` 64.6 MB (weights, pinned source, cp312 wheels, scorer); plan in [forward-models-2026-10-07.md](forward-models-2026-10-07.md) §4 | lehau007 v30 0.413 → v32 FIORA tri-forward **0.403** (confounded with ICEBERG/GLACIER budget changes); nursrijan copy 0.420 (with pair-tail). Only as an all-molecules third forward term | **gap-only** for negative-only molecules (about 7% of hidden-like molecules), never for covered ones |
| **MIST** (coleygroup, MIT, MassSpecGym) | Zenodo 23303289 (10-20); Kaggle `ahmedberatozer/casmi26-mist-msg-v1` 96.7 MB, `biohack44/casmi26-mist` (10-22). Host answer pending (751373) | third FP lists elsewhere +0.005 / -0.005 to -0.012 | wait for the ruling |
| DreaMS SSL (MIT / CC BY 4.0) | already in the stack via `fe_D.pt` | neighbours as candidates: below base | nothing new |

## 4. Downloads to approve (none done)

| # | File | Source | Size | For |
|---|---|---|---|---|
| 1 | `ms2deepscore_model.pt` (+ `settings.json`, optional `embedding_evaluator.pt`) | zenodo.org/records/17826815, or attach `ranjithch/casmi26-ms2deepscore-model` on Kaggle (same size; licence label CC0 versus Zenodo CC BY 4.0, so prefer our own upload with attribution) | 416.1 MB (+ 2.8 KB, + 0.8 MB) | local check and Kaggle |
| 2 | Wheels: `ms2deepscore-2.7.2`, `matchms-0.33.1`, `pickydict-0.5.0`, `sparsestack-0.7.1`, `pyteomics-5.0.1`, `deprecated-3.0.0` (py3-none-any) | PyPI | about 0.6 MB | offline install |
| 3 | Optional: MS2DeepScore 2024 split (to check which test molecules the model has seen) | zenodo.org/records/13934470 | 1.5 GB | leak check only |

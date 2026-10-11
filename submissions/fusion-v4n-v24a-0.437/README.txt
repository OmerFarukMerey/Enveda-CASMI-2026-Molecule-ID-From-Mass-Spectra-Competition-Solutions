fusion-v4n v24a: public LB 0.437
==========================================================

Submission
----------
Public LB        0.437   (MRR@25, tautomer-canonical InChIKey14)
Date             2026-10-05 17:35 UTC
Kaggle id        56858227
Kaggle notebook  huseyinemreaksoy/casmi26-sub-v24a (private), version 1
Submitted by     Hemre
Config name      v24a-v13a-fwd075
Built on         v13a (0.431), see ../fusion-v4n-v13a-0.431/
Scoring run      a little over 4 h on a T4 (limit 9 h)

Files in this folder
--------------------
notebook.ipynb   the exact notebook that was submitted (File > Import Notebook on Kaggle)
config.json      the overrides on top of solutions/fusion-v4n/nb/00_config.py
README.txt       this file

What changed against v13a
-------------------------
Two numbers: ICE_LAM 1.0 -> 0.75 and GL_LAM 1.0 -> 0.75. The two forward models (ICEBERG, GLACIER) re-rank
candidates of the same formula; their scores now count three quarters of what they did, in the first re-rank and in
the re-rank after the engine fusion (POST_ICE_LAM / POST_GL_LAM follow the base weights). Nothing else changes.

Why
---
Every submission that let more PubChem candidates overtake pool candidates lost (200 candidates 0.406, gate 0.8
0.416, ranker blend 0.417, fine-tuned selection 0.423). The forward models are where joined PubChem rows gain rank
over answers that are pool rows. Lowering their weight is the first change that reduces that and gains.

What the pipeline does (solutions/fusion-v4n)
---------------------------------------------
1. v1 engine: candidates from a pool of 710,701 structures (training structures + COCONUT), fingerprint model
   (FPNet), public LightGBM ranker.
2. Engine 2 (prvsiyan): COCONUT + ChEBI / LIPID MAPS + training structures.
3. PubChem-only channel: FPNet score over the +-10 ppm mass window of a PubChem snapshot.
4. PubChem join (nb/embed/pc_join.py): the best 100 PubChem-only candidates are added to the v1 engine's candidates
   and ranked by the same ranker when the molecule has no good library match (lib_max < 0.7).
5. ICEBERG + GLACIER re-rank candidates of the same formula, with weight 0.75 each (was 1.0).
6. Slot merge of the remaining PubChem list, then rank fusion with engine 2 (alpha 0.6, k 3).

Config (config.json)
--------------------
POP_MU 0.25                popularity prior on pool candidates
FRAG_LAM 0.5, FRAG_MODE gap    MetFrag-style re-score only for molecules the forward models cannot score
ICE_LAM_LIB 0, GL_LAM_LIB 0    forward models off for library hits (lib_max >= 0.9)
PC_JOIN_N 100              PubChem candidates joined into the v1 engine
PC_JOIN_MAX_LIB 0.7        join only when lib_max < 0.7
ICE_LAM 0.75, GL_LAM 0.75  forward-model weights (new)
Everything else: defaults (the public 0.399 notebook).

Kaggle inputs (15 datasets + the competition)
---------------------------------------------
Same as v13a: prvsiyan/casmi26-fp-models-v2, prvsiyan/casmi26-ranker-features,
megayak/casmi26-simulated-ranker-rows, ahmedberatozer/casmi26-fpnet-full1, ahmedberatozer/casmi26-glacier,
ahmedberatozer/casmi26-iceberg, ahmedberatozer/casmi26-pubchem-tier, ahmedberatozer/casmi26-v2-pool,
ahmedberatozer/casmi26-v3-models, ahmedberatozer/casmi26-v4b-models, prvsiyan/chebi-lipidmaps-casmi26,
prvsiyan/coconut-casmi26-candidates, metric/rdkit-2026-3-3-wheel, dmitriigluzdov/casmi26-pubchem-popularity-prior,
dmitriigluzdov/casmi26-fold-safe-fpnet (validation mode only).
All public. No NIST data and no weights derived from it. Internet off, GPU T4, pinned docker image.

Reproduce
---------
Either import notebook.ipynb on Kaggle and attach the inputs above, or from the repo root:
  python solutions/fusion-v4n/tools/build_notebook.py --user <kaggle user> --slug casmi26-sub-v24a \
      --config solutions/fusion-v4n/configs/v24a-v13a-fwd075.json --out scratch/sub_v24a
  kaggle kernels push -p scratch/sub_v24a
  kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k <kaggle user>/casmi26-sub-v24a -v 1 \
      -f submission.csv -m "fusion-v4n v24a"

Local evidence before the submission
------------------------------------
Full local replay with real ICEBERG / GLACIER scores, 380 hold-out molecules, difference to v13a (MRR@25):
  close to a library match (150)                    +0.005 +- 0.004  (5 better / 1 worse)
  answer in the pool without spectra (380)          +0.011 +- 0.004  (22 / 5)
  answer removed from the pool (325)                +0.001 +- 0.005
  answer removed from the pool and engine 2 (325)   +0.001 +- 0.006
  pool-answer lists with a PubChem row on top       25 -> 22 of 380  (answer lost first place: 8 -> 5)
Enveda timsTOF panel (250 molecules, four model arms): the two pool-answer types gain in every arm (+0.007 to
+0.027), the answer-removed type loses 0.004 to 0.009.
A random search over 1,764 configurations had found "lower forward weights" as its only consistent signal.
Expected 0.428 to 0.437 (submitted as an information probe); scored 0.437.

What we learned from it
-----------------------
- The exposure rule works in both directions: fewer pool answers overtaken by PubChem rows gained 0.006.
- The local sets cannot predict the size of a leaderboard change (no local score orders our eleven earlier results),
  so single changes that look harmless locally are tested on the leaderboard one at a time.
- Next on the same axis: weights 0.5 (replay: +0.026 on pool answers without spectra, -0.004 / -0.007 on
  answer-removed molecules, 19 exposed lists).

Details: solutions/fusion-v4n/EXPERIMENTS.md, docs/experiments.md (row 27), solutions/fusion-v4n/reviews/
(combo-, param-search-, calib-, fwd-replay-, panel2-2026-10-05.md).

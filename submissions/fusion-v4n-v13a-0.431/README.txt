fusion-v4n v13a: public LB 0.431 (team best on 2026-10-05)
==========================================================

Submission
----------
Public LB        0.431   (MRR@25, tautomer-canonical InChIKey14)
Date             2026-10-03 08:35 UTC
Kaggle id        56793694
Kaggle notebook  huseyinemreaksoy/casmi26-sub-v13a (private), version 1
Submitted by     Hemre
Config name      v13a-v12a-n100
Built on         v12a (0.420), see ../fusion-v4n-v12a-0.420/
Git commit       75a70f6  "Join 100 PubChem candidates: public LB 0.431 (v13a)"
Scoring run      3 to 4 h on a T4 (limit 9 h)

Files in this folder
--------------------
notebook.ipynb   the exact notebook that was submitted (File > Import Notebook on Kaggle)
config.json      the overrides on top of solutions/fusion-v4n/nb/00_config.py
README.txt       this file

What changed against v12a
-------------------------
One number: PC_JOIN_N 50 -> 100. The best 100 (was 50) PubChem-only candidates enter the main engine.

What the pipeline does (solutions/fusion-v4n)
---------------------------------------------
1. v1 engine: candidates from a pool of 710,701 structures (training structures + COCONUT), fingerprint model
   (FPNet), public LightGBM ranker.
2. Engine 2 (prvsiyan): COCONUT + ChEBI / LIPID MAPS + training structures.
3. PubChem-only channel: FPNet score over the +-10 ppm mass window of a PubChem snapshot.
4. PubChem join (nb/embed/pc_join.py): the best PC_JOIN_N PubChem-only candidates are added to the v1 engine's
   candidates and ranked by the same ranker, but only when the molecule has no good library match
   (lib_max < PC_JOIN_MAX_LIB).
5. ICEBERG + GLACIER (forward models) re-rank candidates of the same formula.
6. Slot merge of the remaining PubChem list, then rank fusion with engine 2 (alpha 0.6, k 3).

Config (config.json)
--------------------
POP_MU 0.25                popularity prior on pool candidates
FRAG_LAM 0.5, FRAG_MODE gap    MetFrag-style re-score only for molecules the forward models cannot score
ICE_LAM_LIB 0, GL_LAM_LIB 0    forward models off for library hits (lib_max >= 0.9)
PC_JOIN_N 100              PubChem candidates joined into the v1 engine
PC_JOIN_MAX_LIB 0.7        join only when lib_max < 0.7
Everything else: defaults (the public 0.399 notebook).

Kaggle inputs (15 datasets + the competition)
---------------------------------------------
prvsiyan/casmi26-fp-models-v2, prvsiyan/casmi26-ranker-features, megayak/casmi26-simulated-ranker-rows,
ahmedberatozer/casmi26-fpnet-full1, ahmedberatozer/casmi26-glacier, ahmedberatozer/casmi26-iceberg,
ahmedberatozer/casmi26-pubchem-tier, ahmedberatozer/casmi26-v2-pool, ahmedberatozer/casmi26-v3-models,
ahmedberatozer/casmi26-v4b-models, prvsiyan/chebi-lipidmaps-casmi26, prvsiyan/coconut-casmi26-candidates,
metric/rdkit-2026-3-3-wheel, dmitriigluzdov/casmi26-pubchem-popularity-prior,
dmitriigluzdov/casmi26-fold-safe-fpnet (validation mode only).
All public. No NIST data and no weights derived from it. Internet off, GPU T4, pinned docker image
(solutions/fusion-v4n/tools/build_notebook.py, DOCKER_IMAGE).

Reproduce
---------
Either import notebook.ipynb on Kaggle and attach the inputs above, or from the repo root:
  python solutions/fusion-v4n/tools/build_notebook.py --user <kaggle user> --slug casmi26-sub-v13a \
      --config solutions/fusion-v4n/configs/v13a-v12a-n100.json --out scratch/sub_v13a
  kaggle kernels push -p scratch/sub_v13a
  kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k <kaggle user>/casmi26-sub-v13a -v 1 \
      -f submission.csv -m "fusion-v4n v13a"

Local evidence before the submission
------------------------------------
380 hold-out molecules ("mixed regime", solutions/fusion-v4n/reviews/pcjoin-2026-10-02.md), N 100 minus N 50, public
ranker: answers that exist only in PubChem +0.016 +- 0.007 without forward models and +0.033 +- 0.014 with ICEBERG;
answers in the pool without spectra +0.005 +- 0.008; molecules close to a library match -0.001 to -0.005.
Predicted 0.423 (0.412 to 0.432); scored 0.431.

What we learned from it
-----------------------
- The join is the only change of this pipeline that moved the leaderboard: no join 0.409, N 50 0.420, N 100 0.431.
- More is not better: N 200 scored 0.406 (v14a), the gate at 0.8 scored 0.416 (v14b). Anything that puts more
  PubChem rows above pool candidates loses.
- Neutral on top of v13a: GLACIER on [M+H]+ only 0.431 (v14c), PubChem shortlist 20,000 0.431 (v15b), confidence
  gate 0.430 (v17b). Worse: our ranker blend 0.417 (v15a), fine-tuned ICEBERG 0.401 (v20a), N 200 with the
  confidence gate 0.409 (v19a).
- The local hold-out explains the direction but does not predict the size of leaderboard changes.

Details: solutions/fusion-v4n/EXPERIMENTS.md, docs/experiments.md (row 17).

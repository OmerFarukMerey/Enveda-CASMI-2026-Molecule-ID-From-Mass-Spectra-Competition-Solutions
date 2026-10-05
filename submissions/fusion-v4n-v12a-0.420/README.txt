fusion-v4n v12a: public LB 0.420 (first PubChem join)
=====================================================

Submission
----------
Public LB        0.420   (MRR@25, tautomer-canonical InChIKey14)
Date             2026-10-03 00:02 UTC
Kaggle id        56784505
Kaggle notebook  huseyinemreaksoy/casmi26-sub-v12a (private), version 1
Submitted by     Hemre
Config name      v12a-v4b-pcjoin50-lib07
Built on         v4b (0.409; public notebook
                 https://www.kaggle.com/code/huseyinemreaksoy/casmi26-sub-v4b-0-409-on-the-public)
Git commit       8300c2d  "Rank PubChem candidates with the v1 engine: public LB 0.420 (v12a)"
Scoring run      3 to 4 h on a T4 (limit 9 h)
Superseded by    v13a (0.431), see ../fusion-v4n-v13a-0.431/

Files in this folder
--------------------
notebook.ipynb   the exact notebook that was submitted (File > Import Notebook on Kaggle)
config.json      the overrides on top of solutions/fusion-v4n/nb/00_config.py
README.txt       this file

What changed against v4b
------------------------
The PubChem join (new code: solutions/fusion-v4n/nb/embed/pc_join.py). Before, PubChem-only candidates could only
reach the final list through a few fixed slots after the ranking. Now the best 50 of them are added to the main
engine's candidates and ranked by the same public ranker as the pool candidates, when the molecule has no good
library match (lib_max < 0.7). Flags: PC_JOIN_N 50, PC_JOIN_MAX_LIB 0.7.

What the pipeline does (solutions/fusion-v4n)
---------------------------------------------
1. v1 engine: candidates from a pool of 710,701 structures (training structures + COCONUT), fingerprint model
   (FPNet), public LightGBM ranker.
2. Engine 2 (prvsiyan): COCONUT + ChEBI / LIPID MAPS + training structures.
3. PubChem-only channel: FPNet score over the +-10 ppm mass window of a PubChem snapshot.
4. PubChem join: see above.
5. ICEBERG + GLACIER (forward models) re-rank candidates of the same formula.
6. Slot merge of the remaining PubChem list, then rank fusion with engine 2 (alpha 0.6, k 3).

Config (config.json)
--------------------
POP_MU 0.25                popularity prior on pool candidates
FRAG_LAM 0.5, FRAG_MODE gap    MetFrag-style re-score only for molecules the forward models cannot score
ICE_LAM_LIB 0, GL_LAM_LIB 0    forward models off for library hits (lib_max >= 0.9)
PC_JOIN_N 50               PubChem candidates joined into the v1 engine
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
  python solutions/fusion-v4n/tools/build_notebook.py --user <kaggle user> --slug casmi26-sub-v12a \
      --config solutions/fusion-v4n/configs/v12a-v4b-pcjoin50-lib07.json --out scratch/sub_v12a
  kaggle kernels push -p scratch/sub_v12a
  kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k <kaggle user>/casmi26-sub-v12a -v 1 \
      -f submission.csv -m "fusion-v4n v12a"

Local evidence before the submission
------------------------------------
380 hold-out molecules ("mixed regime", solutions/fusion-v4n/reviews/pcjoin-2026-10-02.md), join at N 50 against no
join, public ranker, without forward models, MRR@25 difference:
  answer removed from the pool and from engine 2 (325 with a PubChem record)   +0.194 +- 0.018 (153 better / 27 worse)
  answer removed from the pool only                                            +0.196 +- 0.019
  answer in the pool without spectra (380)                                     +0.013 +- 0.006
  close to a library match (150)                                               -0.006 +- 0.006
The gate lib_max < 0.7 removes most of the cost on the last group and keeps 83 to 87% of the gain.
Predicted 0.415 (0.403 to 0.44); scored 0.420.

What we learned from it
-----------------------
- First change of this pipeline that acts on answers outside the main pool, and the first real leaderboard gain
  (+0.011) after our earlier submissions, which only re-ordered pool candidates and stayed between 0.390 and 0.409.
- The same step again (N 50 -> 100) gave another +0.011 (v13a).

Details: solutions/fusion-v4n/EXPERIMENTS.md, docs/experiments.md (row 16).

fusion-v4n v33b: public LB 0.442 (team best on 2026-10-06)
==========================================================

Submission
----------
Public LB        0.442   (MRR@25, tautomer-canonical InChIKey14)
Date             2026-10-06 15:08 UTC
Kaggle id        56884062
Kaggle notebook  huseyinemreaksoy/casmi26-sub-v33b (private), version 1
Submitted by     Hemre
Config name      v33b-v24a-fsranker-gate07-rankavg
Built on         v24a (0.437), see ../fusion-v4n-v24a-0.437/
Scoring run      about 4 h 20 min on a T4 (limit 9 h)

Files in this folder
--------------------
notebook.ipynb   the exact notebook that was submitted (File > Import Notebook on Kaggle)
config.json      the overrides on top of solutions/fusion-v4n/nb/00_config.py
README.txt       this file

What changed against v24a
-------------------------
For molecules without a good library match (lib_max < 0.7) the base list is built from two rankers instead of one:
the public LightGBM ranker, and a LightGBM ranker of our own that was fitted to the features of the fold-safe FPNet on
molecules that FPNet has never seen. Their ranks are averaged (RANKER=fs_avg, RANKER_LIB_GATE=0.7). Molecules at or
above the gate are processed exactly as in v24a. Code: solutions/fusion-v4n/nb/embed/fs_pass.py.

Why
---
The public fingerprint models were trained on all of train.parquet and the public ranker was fitted to their features
on those same molecules. On the hidden test every molecule is new to the models, so the ranker meets fingerprint
features from a different distribution than the one it learned on (locally: the z-score of the answer in the
fz_softmax feature is 6.6 on seen molecules and 3.3 on unseen ones). Our ranker is fitted in the unseen-molecule
regime: the fingerprint bank is the fold-safe FPNet and the training lists come from the hold-out fold of that model.
Most of our loss on pool answers happens at the ranker, and half of it in nitrogen-free molecules with many isomers,
where this ranker gains most.

What the pipeline does (solutions/fusion-v4n)
---------------------------------------------
1. v1 engine: candidates from a pool of 710,701 structures (training structures + COCONUT), fingerprint model
   (FPNet), public LightGBM ranker.
2. Engine 2 (prvsiyan): COCONUT + ChEBI / LIPID MAPS + training structures.
3. PubChem-only channel and PubChem join: the best 100 PubChem-only candidates are ranked together with the pool
   candidates when lib_max < 0.7.
4. New: below the gate a second engine pass with the fold-safe FPNet; its candidates are scored by our ranker and the
   ranks of the two rankers are averaged.
5. ICEBERG + GLACIER re-rank candidates of the same formula with weight 0.75 each.
6. Slot merge of the remaining PubChem list, then rank fusion with engine 2 (alpha 0.6, k 3).

Config (config.json)
--------------------
POP_MU 0.25, FRAG_LAM 0.5, FRAG_MODE gap, ICE_LAM_LIB 0, GL_LAM_LIB 0     as in the 0.409 version
PC_JOIN_N 100, PC_JOIN_MAX_LIB 0.7                                       the PubChem join (0.431)
ICE_LAM 0.75, GL_LAM 0.75                                                forward-model weights (0.437)
RANKER fs_avg, RANKER_LIB_GATE 0.7                                       new

Kaggle inputs (16 datasets + the competition)
---------------------------------------------
The 15 public datasets of v24a (see ../fusion-v4n-v24a-0.437/README.txt), of which
dmitriigluzdov/casmi26-fold-safe-fpnet is now used in scoring (fpnet_fold0.pt), plus one of ours:
  huseyinemreaksoy/casmi26-ours-ranker-fs  (private)  four LightGBM boosters, 165 features, fitted on 6,687 lists
      built from train.parquet only; features from the fold-safe FPNet and the public feature models.
      How it was built: solutions/fusion-v4n/reviews/fsranker-2026-10-06.md.
No NIST data and no weights derived from it. The dataset card of the fold-safe FPNet carries no licence statement; it
is a public, train-derived artefact of another competitor (the hosts allow those). Internet off, GPU T4, pinned
docker image.

Reproduce
---------
  python solutions/fusion-v4n/tools/build_notebook.py --user <kaggle user> --slug casmi26-sub-v33b \
      --config solutions/fusion-v4n/configs/v33b-v24a-fsranker-gate07-rankavg.json --out scratch/sub_v33b
  kaggle kernels push -p scratch/sub_v33b
  kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k <kaggle user>/casmi26-sub-v33b -v 1 \
      -f submission.csv -m "fusion-v4n v33b"
The private ranker dataset must exist under the same user (team members: ask Hemre to share it inside the team).

Local evidence before the submission
------------------------------------
Full replay with real ICEBERG / GLACIER scores, difference to v24a (which has seen every local molecule, so it is an
upper bound here; our ranker has seen none of them):
                                              Enveda panel (250)     mixed regime (380)
  close to a library match                    -0.009 +- 0.004        -0.001
  answer in the pool without spectra          +0.009 +- 0.005        +0.007 +- 0.006
  answer removed from the pool                +0.030 +- 0.008        +0.009 +- 0.006
  answer removed from the pool and engine 2   +0.023 +- 0.008        +0.009 +- 0.008
  pool-answer lists topped by a PubChem row   not raised             not raised
At ranker level, on unseen molecules, our ranker beats the public ranker on the same features by +0.038 to +0.057
(nitrogen-free molecules +0.07 to +0.10). Expected 0.435 to 0.448; scored 0.442.

What we learned from it
-----------------------
- Second structural gain after the PubChem join: the ranker must be fitted in the regime it is used in.
- Better fingerprint models alone did not help (five attempts); a ranker fitted to unseen-molecule features did.
- Open: the same change without the average (RANKER=fs, preset v33a), other gates, and a ranker fitted with more
  training lists.

Details: solutions/fusion-v4n/reviews/rankbank-2026-10-06.md, fsranker-2026-10-06.md, docs/experiments.md (row 31).

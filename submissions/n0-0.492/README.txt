n0: public LB 0.492 (team best, 2026-11-16)
===========================================

Submission
----------
Public LB        0.492   (MRR@25; tautomer-canonical InChIKey first block)
Date             2026-11-16
Kaggle notebook  casmi-n0-lam35 (private; owner name left out of this repo)
Submitted by     Arsokan (Fable session)
Run mode         COMMIT_MODE = 'smoke' (the commit run validates 12 visible molecules and asserts all stages)

What it is
----------
NOT our pipeline. A copy of the public notebook mathleteutd "0.492 Test-Set Network Rerank": the public 0.487 line
(Peaks to Structures scorer + external-library overlay) plus cell N-1c, an in-test molecular-network rerank that runs
before the external-library overlay, with LAM 0.35 (the analyticaobscura original used 0.25; ax13n, 0.488).

All Kaggle inputs are public (kernel-metadata.json; the "id" owner is replaced by YOUR_KAGGLE_USERNAME).
Licence note: takumuhata/casmi26-extlib is attached; its source / licence must be confirmed by the team before the
final selection (docs/plan/roadmap.md, Open questions). Datasets are not redistributed here.

Local evidence
--------------
Offline network harness (solutions/arsokan-fable/offline/net_*.py): the LAM curve has the same shape as the public
scores (peak at 0.35-0.5); 22 design variants (TAU, PW, TOPN, neighbour weight, several rounds, ...) are within
noise; ceiling about +0.005, which this version already takes.

Follow-up
---------
t0 = this notebook + one cell (TA-1, solutions/arsokan-fable/stages/train_analog_rerank.py) right after N-1c.
Notebook: solutions/arsokan-fable/kaggle/t0/. Public score 0.461 (-0.031): harmful, not for final selection
(docs/experiments.md row 141). n0 stays the team best.

Files in this folder
--------------------
README.txt
notebook.ipynb         the exact submitted notebook
kernel-metadata.json   Kaggle inputs and machine settings (owner redacted)

Reproduce
---------
Set "id" in kernel-metadata.json to your account, rename notebook.ipynb to
casmi26-0-492-test-set-network-rerank.ipynb, kaggle kernels push -p <folder>, then submit its submission.csv.

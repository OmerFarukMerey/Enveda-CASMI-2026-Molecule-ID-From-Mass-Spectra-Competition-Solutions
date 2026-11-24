gnx11: public LB 0.505 (team best, 2026-11-24)
==============================================

Submission
----------
Public LB        0.505   (MRR@25; tautomer-canonical InChIKey first block)
Date             2026-11-24 05:56 UTC
Kaggle id        58814261
Kaggle notebook  huseyinemreaksoy/casmi26-sub-gnx11 (private), version 1
Submitted by     Hemre
Run mode         fast commit (the commit run writes placeholder SMILES; the full pipeline runs only in the scoring rerun)

What it is
----------
The public 0.505 line (kozykappa "CASMI26 LB 0.505" = Ömer's gfp5 + goodpjw2008's E180X + kozykappa's NPX cell) with
two changes:
  1. the P5 tail cell is fpn's (12 parents / 300 products + NetGen network parents; see submissions/fpn-0.496/);
  2. E180X 1.1 (goodpjw2008's "Stack + NPX + E180X 1.1" last cell) in place of E180X 1.0.
NPX: a LightGBM model of p(natural product | structure), trained on the natural-product tables against the enveda-180
synthetic library; re-orders ranks 2-25 and moves rank 1 only when the leader looks clearly synthetic.
Stage order: v13 -> N-1c network -> library overlay -> P5 -> FR 3-25 -> APRIOR -> UNIQ -> NPX -> E180X 1.1.
The 117 GNPS-NIST14-MATCHES rows are kept (host-allowed by name, thread 745832).

Evidence and caveats
--------------------
Clean hold-out kernel casmi26-vh2-gnx11 v4 (300 train molecules held out, main engine on fpnet_fold0, which never saw
them): final MRR nphard 0.716 / npref0 0.862 / e180 0.859; NPX off is -0.005 on nphard (2 up / 6 down). The hold-out
cannot predict the board for rank-1 rules: every hold-out answer is still marked as a train structure in the static
pool, and the main engine ranks marked structures far higher. gnxds (this notebook + a rule that trusts the main-engine
leader at ranks 1-2, even gated to equal provenance) scored public 0.478 on the same day.
On the same hold-out the public 0.506 notebook (pablodiegoo) was not better: nphard -0.001, e180 -0.024.

Inputs and licences
-------------------
See kernel-metadata.json: all public, real weights. Base notebook Apache-2.0 (taeyangg4 / goodpjw2008 / kozykappa
lineage). taeyangg4's two datasets and samartalwar MassBank-harmonized are CC BY-NC-SA 4.0. Datasets are not
redistributed here.

Files in this folder
--------------------
README.txt
notebook.ipynb         the exact submitted notebook
kernel-metadata.json   Kaggle inputs and machine settings
build_gnx.py           builder (reads pulled kernels from scratch/, not in git)

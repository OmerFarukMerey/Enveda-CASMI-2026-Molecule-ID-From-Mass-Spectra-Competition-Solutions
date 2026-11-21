fpn: public LB 0.496 (team best, NIST-free, 2026-11-21)
=======================================================

Submission
----------
Public LB        0.496   (MRR@25; tautomer-canonical InChIKey first block)
Date             2026-11-20 18:00 UTC
Kaggle id        58627226
Kaggle notebook  huseyinemreaksoy/casmi26-sub-fpn (private), version 1
Submitted by     Hemre (merge of Ömer's kernels)
Run mode         fast commit (the commit run writes dummy SMILES and smoke-runs P5 on 12 molecules; the full pipeline
                 runs only in the scoring rerun)

What it is
----------
Ömer's three public-0.496 kernels in one, on taeyangg4's public 0.495 notebook (see submissions/tp5-0.495/):
  fp5  (Kaggle 58584361, 0.496)  the FR 3-25 learned fragment re-ranker runs AFTER the P5 tail, so it also orders the
                                 generated candidates P5 inserts into ranks 3-25 (ranks 1-2 locked; session guard 8.2 h)
  p12f (Kaggle 58584371, 0.496)  fp5 + P5 runner with up to 12 transformable-first analog parents and 300 products (6 / 150)
  n5f  (Kaggle 58584379, 0.496)  fp5 + NetGen: the top-2 guesses of each molecule's N-1c network neighbours become extra
                                 P5 generation parents when the mass gap is a <= 2-step casmi.derive transform
The kernel is n5f's notebook with p12's runner edit re-applied (the two edits touch different lines of p5_generate;
new_keys() counts both p12's extra parents and n5's negative parent ids as new). The embedded runner's sha256 is
re-pinned; the builder asserts that fp5 + the p12 edit reproduces p12f's runner byte for byte.

NIST: the external-library overlay drops the 117 GNPS-NIST14-MATCHES rows, as in tp5. Ömer's fp5 / p12f / n5f (and
cf5 / g90 / cg) used the source library unchanged, i.e. WITH those rows, against the team rule; do not pick them as
finals. This kernel is the compliant version of the 0.496 line.

Commit smoke run: 12 molecules, 0 errors, 230 new P5 rows (= p12f's smoke), NetGen parents for 114 molecules, log line
"dropped 117 NIST-matched reference rows".

Evidence and caveats
--------------------
No local harness for this pipeline. Each of the three changes scored 0.496 alone and the merge also scored 0.496, so the
public part (~124 molecules) cannot separate them; ranks 1-2 are locked, so the changes only move ranks 3-25.

Inputs and licences
-------------------
As tp5 (see kernel-metadata.json): all public, real weights. taeyangg4's two datasets and samartalwar MassBank-harmonized
are CC BY-NC-SA 4.0. Datasets are not redistributed here. Base notebook Apache-2.0 (taeyangg4).

Files in this folder
--------------------
README.txt
notebook.ipynb         the exact submitted notebook
kernel-metadata.json   Kaggle inputs and machine settings
build_fpn.py           builder (reads Ömer's pulled kernels from scratch/land1120/om/, not in git)

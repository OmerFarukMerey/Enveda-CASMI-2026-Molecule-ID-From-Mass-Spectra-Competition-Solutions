tp5: public LB 0.495 (team best, 2026-11-20)
============================================

Submission
----------
Public LB        0.495   (MRR@25; tautomer-canonical InChIKey first block)
Date             2026-11-19 21:59 UTC
Kaggle id        58580969
Kaggle notebook  huseyinemreaksoy/casmi26-sub-tp5 (private), version 1
Submitted by     Hemre
Run mode         fast commit (the commit run writes dummy SMILES; the full pipeline runs only in the scoring rerun)

What it is
----------
NOT our pipeline. A copy of the public notebook taeyangg4/casmi26-lb-0-495-network-fr-p5-tail (Apache-2.0, 0.495 on
its own page) with one change and one notice cell. It is the source of the 0.495 family (mathleteutd "10 train anchors"
and alexchilton "60 arms" attach the same inputs). Its line: alexchilton v13 + N-1c in-test network rerank (n0, 0.492)
+ an FRNet fragment re-ranker on ranks 3-25 (taeyangg4/casmi26-fr3to25-clean, W 3.0, ranks 1-2 kept) + a P5 tail step
+ a larger external library for the rank-1 overlay (taeyangg4/casmi26-extlib-gnps-full).

The one change: the external-library overlay drops the reference spectra whose source_library names NIST, i.e. the
117 rows of GNPS-NIST14-MATCHES (GNPS user spectra whose structure labels come from NIST14 matching). Team rule: no NIST
or NIST-derived data. 198,464 of 198,581 rows kept; the run log prints "dropped 117 NIST-matched reference rows".
The score equals the source notebook's, so those rows made no difference on the public part.

Inputs and licences
-------------------
All public, real weights (not demo uploads), see kernel-metadata.json. taeyangg4/casmi26-extlib-gnps-full and
taeyangg4/casmi26-fr3to25-clean are CC BY-NC-SA 4.0 (uploaded 2026-11-09 / 11-14); samartalwar MassBank-harmonized is
CC BY-NC-SA. Datasets are not redistributed here.

Files in this folder
--------------------
README.txt
notebook.ipynb         the exact submitted notebook
kernel-metadata.json   Kaggle inputs and machine settings
build_tp.py            builder (reads the pulled source notebook from scratch/land1120/, not in git)

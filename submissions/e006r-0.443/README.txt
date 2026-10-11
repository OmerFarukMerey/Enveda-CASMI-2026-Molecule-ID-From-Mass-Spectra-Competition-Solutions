e006r: public LB 0.443 (team best in the scoring regime in force since 2026-10-07; baseline from 2026-10-11)
==========================================================================================================

Submission
----------
Public LB        0.443   (MRR@25; stereo-stripped, tautomer-canonical InChIKey first block)
Date             2026-10-11 06:48 UTC
Kaggle id        57065388
Kaggle notebook  huseyinemreaksoy/casmi26-e006r (private), version 1
Submitted by     Hemre
Scoring run      not public; estimated 3 h 40 min on a T4 (limit 9 h)

What it is
----------
NOT our pipeline. It is the public notebook canqiang/casmi26-e006-window-40-ice-fallback-true (version 1, Apache 2.0,
0.440 on its own page), code unchanged, with one input swapped: the ChEBI / LIPID MAPS structures come from
sergiomalv/casmi-chebi-lmsd-structures (CC BY 4.0) instead of prvsiyan/chebi-lipidmaps-casmi26 (CC BY-NC-SA 4.0), plus
one notice cell. e006 itself is the public notebook flexonafft v80 (v18 family: metric-key votes, tail fill) plus an
ICEBERG-only re-rank for molecules that have no [M+H]+ spectrum. Analysis: docs/landscape/canqiang-e006-2026-10-10.md,
solutions/fusion-v4n/reviews/e006-2026-10-10.md.

Why it is here
--------------
On 2026-10-07 Kaggle changed the scoring. Our own best notebook (v33b, 0.442 before the change) scores 0.433 in the new
regime and its licence-clean version v59a 0.434. This rerun scores 0.443, so it is the reference that every later
experiment is compared with, and the base of the v67 line (e006's chain fed with our base list). The difference to
v59a (0.009) is about one answer at rank 1 on the public part (about 124 molecules, 0.008 each).

Files in this folder
--------------------
notebook.ipynb         the exact notebook that was submitted
kernel-metadata.json   its Kaggle inputs
README.txt             this file

Licences
--------
Code: Apache 2.0 (notice in solutions/fusion-v4n/reference/e006/ and THIRD_PARTY_NOTICES.md). Inputs: the same public
datasets as our own notebooks. Not cleared by the host as of 2026-10-11: prvsiyan/casmi26-fp-models-v2 and
prvsiyan/casmi26-ranker-features (CC BY-NC-SA labels), and code without a licence inside the ahmedberatozer datasets
(docs/competition/clean-chain-plan-2026-10-10.md). The attached hengck23 dataset is never read by the code.

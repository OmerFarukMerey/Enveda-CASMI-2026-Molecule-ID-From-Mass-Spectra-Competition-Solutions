ax13n: public LB 0.488 (team best, 2026-11-13)
=============================================

Submission
----------
Public LB        0.488   (MRR@25; tautomer-canonical InChIKey first block)
Date             2026-11-13 17:16 UTC
Kaggle id        58316277
Kaggle notebook  huseyinemreaksoy/casmi26-sub-ax13n (private), version 1
Submitted by     Hemre
Run mode         fast commit (the commit run writes dummy SMILES; the full pipeline runs only in the scoring rerun)

What it is
----------
NOT our pipeline. A copy of the public notebook analyticaobscura/casmi-2026 (Apache-2.0, 0.488 on its own page), code
unchanged, plus one notice cell. That notebook is alexchilton/casmi26-lb-0-487-peaks-to-structures-scorer v13
(Apache-2.0, 0.487) with one added cell, N-1c: an in-test molecular-network rerank that runs before the external-library
overlay. Test spectra with modified cosine >= 0.6 over >= 6 matched peaks are neighbours; candidates that are
Tanimoto-similar (Morgan) to a neighbour's top 5 move up, weight lambda 0.25, z-scored against -log(rank).

All Kaggle inputs are public (kernel-metadata.json). Licence note: samartalwar/casmi-2026-spectral-library-massbankharmonized
is CC BY-NC-SA (the audit flags NC); takumuhata/casmi26-extlib is CC0 (GNPS + MassBank + MoNA, no NIST per its card).
Datasets are not redistributed here.

Local evidence
--------------
N-1c on the 600 clean held-out lists (scratch/land1113/test_n1c.py, not in git): MRR 0.6029 -> 0.6035 (+0.0006),
12 wins / 6 losses, top-1 never changed. lambda 0.5 loses (-0.0077). The Kaggle submission message says 17 / 6 by mistake.

What did not beat it (all on top of ax13n or ax13, see docs/experiments.md rows 129-133)
-----------------------------------------------------------------------------------------
ax13   (alexchilton v13 copy)                         0.487
ax13t  (train-blind guard in the overlay)             0.481
ax13n2 (rank-2 tier for library cosine in [0.70,0.80)) 0.486
ax13g  (novelty classifier gate on the ranks 2-25 popularity prior) 0.487
Every change to the external-library overlay so far has lost on public.

Files in this folder
--------------------
README.txt
notebook.ipynb         the exact submitted notebook
kernel-metadata.json   Kaggle inputs and machine settings
build_ax13n.py         builder (reads the pulled source notebook from scratch/land1113/ao/; --t adds the ax13t patch)

Reproduce
---------
kaggle kernels push -p <folder with notebook.ipynb renamed to casmi26-sub-ax13n.ipynb and kernel-metadata.json>
then submit its submission.csv (version 1) to the competition.

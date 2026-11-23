gfpe: public LB 0.501 (team best, level with the best public notebooks, 2026-11-23)
====================================================================================

Submission
----------
Public LB        0.501   (MRR@25; tautomer-canonical InChIKey first block)
Date             2026-11-22 21:45 UTC
Kaggle id        58743587
Kaggle notebook  huseyinemreaksoy/casmi26-sub-gfpe (private), version 2 (identical to version 1)
Submitted by     Hemre
Run mode         fast commit (the commit run writes placeholder SMILES; the full pipeline runs only in the scoring rerun)

What it is
----------
Ömer's gfp5 (Kaggle 58692718, public 0.500: goodpjw2008's public 0.498 stack with the FR 3-25 re-ranker moved after
the P5 tail) with two changes:
  1. E180X, goodpjw2008's last cell (public 0.501 in goodpjw2008 v5 and in koushikrudra's copy), appended after APRIOR
     and UNIQ. Structures of the train library that are not in the natural-product tables are demoted: tier A
     (enveda-180 structures) to the end of the list including rank 1, tier B (other train-only structures) only within
     ranks 2-25.
  2. The P5 tail cell (code cell 18) is fpn's: 12 parents / 300 products plus NetGen network parents
     (see submissions/fpn-0.496/).
Stage order: v13 -> N-1c network -> library overlay -> P5 -> FR 3-25 -> APRIOR -> UNIQ -> E180X.

NIST: the 117 GNPS-NIST14-MATCHES rows of the external library are kept. The host allowed that library by name for
library matching and training (thread 745832; docs/competition/gnps-nist14-2026-10-12.md). NIST itself stays banned.

Size: Kaggle rejects notebooks over 1 MiB, so the builder writes compact JSON and keeps only the first markdown cell.

Evidence and caveats
--------------------
Local harness scratch/land1123/harness.py (250 np-examples, lists of older pipelines, not gfpe's own): E180X never
loses a correct answer, +0.000..+0.005. The public part is about 124 molecules, so 0.501 against gfp5's 0.500 is one
molecule moving a few ranks; the P5 change alone was neutral on public (fpn 0.496 = fp5 0.496). Every public
0.500-0.501 notebook is gfp5 + E180X; nothing public scores higher on 2026-11-23.

Inputs and licences
-------------------
See kernel-metadata.json: all public, real weights. Base notebook Apache-2.0 (taeyangg4), goodpjw2008's cells
Apache-2.0. taeyangg4's two datasets and samartalwar MassBank-harmonized are CC BY-NC-SA 4.0. Datasets are not
redistributed here.

Files in this folder
--------------------
README.txt
notebook.ipynb         the exact submitted notebook
kernel-metadata.json   Kaggle inputs and machine settings
build_gfpe.py          builder (reads pulled kernels from scratch/land1120 and scratch/land1122, not in git)

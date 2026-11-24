gnxv2: public LB 0.509 (team best, 2026-11-24)
==============================================

Submission
----------
Public LB        0.509   (MRR@25; tautomer-canonical InChIKey first block)
Date             2026-11-24 14:03 UTC (scored 20:51 UTC)
Kaggle id        58833544
Kaggle notebook  huseyinemreaksoy/casmi26-sub-gnxv2 (private), version 1
Submitted by     Hemre
Run mode         fast commit (the commit run writes placeholder SMILES; the full pipeline runs only in the scoring rerun)

What it is
----------
The public 0.509 notebook (ishankarnick "CASMI26 LB 0.509: 47.7M virtual natural products" = kozykappa's 0.505 stack +
a last cell, the virtual natural-product tier) with two changes:
  1. E180X 1.1 (goodpjw2008) in place of E180X 1.0;
  2. our NP-extra block (cell_np_extra.py) right before the tier: 20,113 real natural products of
     natural_products.parquet and qwertykeypadapple/casmi26-lotus-candidates that are not in the 710,701-structure
     candidate pool, written in the tier's row layout and handed to the tier as one extra row set (fixed route prior
     -1.80). They pass through the tier's unchanged scoring, de-duplication and merge. Three small edits in the tier
     cell: extra row set in the config, absolute paths accepted, one retry without the extras if the overlay fails.
Virtual tier: candidates from 47.7M one-step derive.py derivatives (45 transforms) of the pool, scored by three FPNets
plus a learned route prior, top 40 re-ordered by two scorers, merged with margins (rank 1 needs +1.5), gated to lists
with a weak library match.
Stage order: v13 -> N-1c network -> library overlay -> P5 -> FR 3-25 -> APRIOR -> UNIQ -> NPX -> E180X 1.1 -> NP-extra
-> virtual tier. Markdown cells after the first were dropped: a notebook push fails with HTTP 400 near 1.04 MB.

Evidence and caveats
--------------------
Same public score as the unchanged public notebook (0.509), so the NP-extra block shows no measurable gain on the
public part; with about 125 public molecules one rank 2 -> 1 move is 0.004 and reruns vary by 0.002.
Coverage study (scratch, vt_coverage.py): answers treated as novel are one-step reachable for 89% of np-examples, 65%
of vh2 nphard and 43% of natural-product structures outside the pool. Extra rows per mass window: median 0, p90 9.
Harm check on 145 hold-out lists: 3 changed, no correct leader displaced.
Two-step derivatives (net-zero transform pairs on the tier's top parents) were measured afterwards on 400 hold-outs:
0 of 19 two-step-only answers reached, and the cell's merge rule displaced 13 correct leaders. Not built.
The tier's ranking is the weak point: of the one-step-reachable hold-out answers only a minority is in its top 8.

Inputs and licences
-------------------
See kernel-metadata.json: all public. Base notebook Apache-2.0 lineage (taeyangg4 / goodpjw2008 / kozykappa /
ishankarnick). ishankarnick/casmi26-vt0, casmi26-vtlp and casmi-scorer-x1full are CC BY-NC-SA 4.0;
qwertykeypadapple/casmi26-lotus-candidates is CC0. Datasets are not redistributed here.

Files in this folder
--------------------
README.txt
notebook.ipynb         the exact submitted notebook
kernel-metadata.json   Kaggle inputs and machine settings
cell_np_extra.py       our NP-extra cell
build_gnxv2.py         builder (reads pulled kernels from scratch/, not in git)

gfp5: public LB 0.500 (team best, 2026-11-22; NOT NIST-free, see below)
========================================================================

Submission
----------
Public LB        0.500   (MRR@25; tautomer-canonical InChIKey first block)
Date             2026-11-22 00:03 UTC
Kaggle notebook  omerfarukmerey/casmi26-om-gfp5, version 1 (submitted). Made public on 2026-11-22 as
                 https://www.kaggle.com/code/omerfarukmerey/casmi26-0-500-aprior-stack-fr-after-p5-tail
                 (version 2 = the public write-up in cell 0; code cells byte-identical to version 1)
Submitted by     Ömer (Claude session)
Run mode         fast commit (the commit run writes a placeholder submission.csv; the full pipeline runs only in the
                 scoring rerun, ~5.5 h)

What it is
----------
goodpjw2008's public notebook "Enveda CASMI | Stack + APRIOR + UNIQ | LB 0.498"
(goodpjw2008/enveda-casmi-stack-aprior-uniq-lb-0-498) with one cell moved: taeyangg4's FR 3-25 learned fragment
re-ranker runs AFTER the P5 tail (our fp5 order, public 0.496 on the 0.495 base, row 148):

    was   ... overlay checks -> FR 3-25 -> P5 tail -> APRIOR -> UNIQ
    now   ... overlay checks -> P5 tail -> FR 3-25 -> APRIOR -> UNIQ

goodpjw2008's stack = taeyangg4 0.495 (GNPS extlib overlay, FR 3-25, P5 tail, N-1c D-L1 network) + overlay checks
(threshold 0.80, rank-2 hedge) + APRIOR (LightGBM prior on which adducts a structure is recorded with, trained on 60k
enveda-180 molecules; re-orders ranks 2-25 by -1.5 ln(rank) + 0.5 log-likelihood, rank 1 kept) + UNIQ (one structure
for one molecule). Built by build_om.py (variant gfp5 = p_fp5 on base g498).

NIST: like the source notebooks, the external-library overlay keeps the 117 GNPS-NIST14-MATCHES rows. The host allows
them (forum thread 745832, 2026-10-09: "Allowed for library matching and training"), but the team rule is no NIST.
gfp5n (= gfp5 + the tp5/fpn NIST-row filter, build_om.py variant gfp5n, kernel omerfarukmerey/casmi26-om-gfp5n) is the
compliant version: queued for 2026-11-23. Dropping the rows was neutral twice on public (tp5 0.495 = source; kchuang's
"E274 APRIOR UNIQ nistclean" 0.498 = goodpjw2008). Pick gfp5n, not gfp5, as a final once it scores.

Same day, same base
-------------------
gfp5a75   gfp5 + APRIOR weight 0.75 (was 0.5)                                         0.499
o95       fp5 (0.496) + overlay only on near-identical spectra (cos >= 0.95, or >= 0.90 with 8 peaks)   0.489
ooff      fp5 (0.496) + overlay off                                                    0.489
The overlay is worth ~0.007 on public; a stricter gate is as bad as none (real timsTOF test spectra match library
spectra of the same structure at cosine 0.80-0.85).

Offline APRIOR study (vh_aprior.py, Kaggle CPU kernel casmi-vh-aprior)
----------------------------------------------------------------------
Visible test vs train: 1,209 of 1,213 test spectra match train rows exactly. The test keeps a recorded [M+H]+ 97% of
the time, [M-H]- 83%, formate 71%, [M+Na]+ 73%, and ~58% of each molecule's spectra. (These rates describe how the
placeholder was drawn from train, not necessarily the hidden test; nothing submitted was tuned on them: the
observation-aware variant that used them was rejected, and the weight was chosen on the board.) On 5,062 held-out enveda-180
molecules (thinned that way) against random same-formula decoys: MRR gain +0.018 / +0.024 / +0.030 / +0.032 at weight
0.35 / 0.5 / 0.75 / 1.0; training on all enveda-180 molecules with 600 trees beat 60k / 300 at every weight; an
observation-aware likelihood (expects adducts to be missing at the rates above) was worse at every weight.
The board disagreed on the weight (0.75: 0.499 < 0.5: 0.500): random decoys are far easier than real answer lists.
Follow-up gfp5ab (all molecules, 600 trees, weight 0.5) queued for 2026-11-23.

Inputs and licences
-------------------
See kernel-metadata.json: all public, real weights; datasets are not redistributed here. taeyangg4's two datasets and
samartalwar MassBank-harmonized are CC BY-NC-SA 4.0. Base notebook Apache-2.0 (goodpjw2008, on taeyangg4's).

Files in this folder
--------------------
README.txt
notebook.ipynb         the exact submitted notebook (version 1, pulled from Kaggle)
kernel-metadata.json   Kaggle inputs and machine settings
build_om.py            builder (reads the pulled public notebooks from the session scratchpad, not in git)
publish.py             makes the public version: cell 0 replaced by title + write-up + credits, code cells asserted unchanged
vh_aprior.py           the offline APRIOR study (private Kaggle CPU kernel, never submitted)

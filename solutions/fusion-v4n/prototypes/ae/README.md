<!-- index: AE, spectral analog edits at rank 2 (own code after heliosli's r3p idea): modified-cosine analog search over train.parquet, single-site edits for 19 natural-product modifications, fragment site evidence, learned gate; offline benchmark and the Kaggle overlay cell -->
# AE: spectral analog edits at rank 2

Overlay for bgunshui's public `casmi26-480-neural-ensemble` (LB 0.480). The idea is heliosli's r3p overlay in the public
notebook `heliosli/casmi26-0-481`, whose library and site model are private; this code is ours. Experiments row 126 in
`docs/experiments.md`.

A query whose spectrum matches a training spectrum after shifting by one of 19 natural-product modifications (modified
cosine, 0.01 Da, precursor within 0.01 Da of the shifted mass) is probably that training structure with the
modification at one site. Every single-site edit of the analog is enumerated (additions by SMARTS site + group,
removals by cutting one acyclic single bond; products checked by exact mass), edits that are training structures are
dropped, and sites are ranked by fragment evidence (moved peaks from 1-2-cut fragments that hold the site, unmoved
peaks from fragments without it). A LightGBM gate predicts P(top edit = truth).

| File | What it does |
|---|---|
| `ae_core.py` | modified cosine (numba), modification table, edit enumeration, fragments and site score, `molecule_candidates` (gate features), `build_library` (in memory), `test_queries` |
| `build_lib.py` | the library as files: [M+H]+ / [M-H]- / [M+Na]+, top 32 peaks <= precursor + 1 Da, sqrt intensity, <= 3 spectra per structure and adduct |
| `eval_ae.py` | first look: best analog only |
| `eval2.py` | held-out benchmark rows: each sampled enveda-180 structure's own spectra against every other structure; top 5 analogs with edits, features and labels |
| `gate.py` | 5-fold (by molecule) LightGBM gate, per-molecule fire rate / precision by threshold, final fit |
| `check_same.py` | `molecule_candidates` reproduces `eval2.py` rows |
| `ae_cell.py`, `build_ae.py` | the Kaggle overlay cell (placeholders filled by the builder) and the builder that inserts it after MIDX's final layer |
| `test_cell.py` | runs a filled cell locally on a mock (training molecules as the test, random lists) |

Offline (7,858 held-out enveda-180 structures; products that are other training structures dropped): the truth is an
edit of one of the top 5 analogs for 650 molecules (8.3%); out-of-fold, the gate at p >= 0.25 fires on 7.1% with
precision 0.468, at p >= 0.5 on 3.7% with precision 0.569. The gate model lives in the private dataset
`omerfarukmerey/casmi26-om-ae` (not in git). The library is built from train.parquet in 45-50 s.

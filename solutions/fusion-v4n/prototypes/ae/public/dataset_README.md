<!-- index: README.md of the public Kaggle dataset omerfarukmerey/casmi26-analog-edit-gate (gate model + AE code) -->
# CASMI26 analog edits: gate model and code

Code and gate model for spectral analog edits: a rank-2 candidate built from a training structure whose spectrum
matches the query after a natural-product modification shift. Built by team Arsokan for the Enveda CASMI 2026
competition and used by the notebook *CASMI26 | Neural ensemble + spectral analog edits*. The idea comes from
heliosli's r3p overlay in the public notebook `heliosli/casmi26-0-481`. Its library and site model are private, and
this is an independent implementation.

| file | what it is |
|---|---|
| `ae_gate_v1.txt` | LightGBM gate (20 features, binary): P(the top single-site edit of an analog is the true structure). The notebook loads it by SHA-256 (`08aad80dc0f4f1a2...`) |
| `ae_core.py` | modified cosine, 19 modifications, edit enumeration, fragment site evidence, gate features (`molecule_candidates`), the analog library built in memory from train.parquet (`build_library`) |
| `build_lib.py` | the same library written to files, for offline work |
| `eval2.py` | the held-out benchmark: each sampled enveda-180 structure's own spectra are the queries and every other structure is the library |
| `gate.py` | 5-fold (by molecule) cross-validation of the gate, then the final fit |

**Method.**
1. **Library.** [M+H]+, [M-H]- and [M+Na]+ training spectra. Each keeps its top 32 peaks at or below the precursor
   + 1 Da, with intensities as sqrt(intensity / max). Up to 3 spectra are kept per structure and adduct.
2. **Analog search.** A query spectrum is compared with library spectra whose precursor differs by one of 19
   modifications, in either direction, within 0.01 Da. The modifications are CH2, O, CH2O, acetyl, CO2, prenyl,
   sulfate, malonyl, benzoyl, pentose, deoxyhexose, coumaroyl, galloyl, hexose, caffeoyl, glucuronide, feruloyl,
   sinapoyl and dihexose. The comparison is the modified cosine at 0.01 Da: a greedy one-to-one match of peaks that
   are equal or shifted by the modification mass.
3. **Edits.** For the top 5 analogs (distinct structure and modification), every single-site edit is enumerated.
   Additions put the group on a matching site (by SMARTS); removals cut one acyclic single bond whose removed piece
   has the modification's mass. Every product is checked by exact mass. Products that are training structures are
   dropped, because they would be library matches.
4. **Site evidence.** Each matched peak pair is explained by the analog's fragments left after cutting 1 or 2 bonds.
   Peaks that moved by the modification mass should come from fragments that hold the site, and unmoved peaks from
   fragments without it. The weighted share of consistent peaks ranks the edits.
5. **Gate.** 20 features: the analog's modified cosine and its margin over the next analog, the number of spectra
   that agree, the shifted and unshifted share of matched intensity, the site score and its margin, the number of
   edits, the modification and direction. Trained on 22,060 candidate rows of 7,858 held-out enveda-180 structures.

**Offline (out of fold, best candidate per molecule, 7,858 held-out structures).**

| gate p >= | molecules fired | precision (top edit = truth) |
|---:|---:|---:|
| 0.15 | 9.6% | 0.406 |
| 0.25 | 7.1% | 0.468 |
| 0.50 | 3.7% | 0.569 |

The truth is an edit of one of the top 5 analogs for 8.3% of the held-out structures. The gate is well calibrated:
in the top decile, the mean p is 0.457 and the hit rate 0.423.

**Retraining.** Run `python build_lib.py train.parquet LIB`, then `python eval2.py LIB 1000 SEED OUT.parquet` for
several seeds, then `python gate.py ae_gate.txt OUT*.parquet`. This needs RDKit, numba, LightGBM and pyarrow. Each
1,000 molecules take about 90 s on a laptop CPU. Only `train.parquet` is read.

Licence: CC0 1.0.

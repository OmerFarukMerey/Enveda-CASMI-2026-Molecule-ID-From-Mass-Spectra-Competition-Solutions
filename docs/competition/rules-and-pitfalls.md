<!-- index: allowed and forbidden data (NIST banned), host clarifications, traps that cost other teams submissions -->
# Rules, eligibility and pitfalls

Last reviewed 2026-09-30. Links point to the Kaggle forum. Host answers are authoritative.

## Allowed and forbidden inputs
| Item | Status | Source |
|---|---|---|
| NIST spectra, or any model trained on NIST | **Forbidden, disqualifies** (not freely available; licence restricts derivatives incl. weights) | host, [744470](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744470) |
| Open models / artefacts trained on `train.parquet` (incl. other people's public Kaggle datasets) | Allowed ("fair game") | host, [742265](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742265) |
| Enveda-180 public Zenodo release | discussed in [the forum](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion?sort=recent-comments), check before use | forum |
| GLACIER / ICEBERG weights | Public ms-pred **MassSpecGym** checkpoints (MIT) as packaged by ahmedberatozer. Not NIST-trained per the dataset cards. An open eligibility thread about GLACIER exists | dataset cards |
| P100 GPU | retired, use T4 | host, 742265 |

Before adding any external artefact, write its provenance (data it was trained on, licence) in the solution README.

## Pitfalls other teams paid for
1. **Visible `test.parquet` is a placeholder** drawn from train. Statistics or scores on it say nothing about the LB ([743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254)).
2. **Static artefacts keyed to visible-test masses or formulas are dead at scoring time.** The pipeline does not crash, it
   silently returns the baseline score. Build anything target-dependent *inside* the scored run ([744172](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744172)).
3. The commit run and the scoring rerun differ. Detect the rerun (`KAGGLE_IS_COMPETITION_RERUN`, or a test-file signature) and
   keep commit runs short.
4. Deduplicate by the **metric key** (tautomer-canonical IK14), not raw InChIKey. About 8% of molecules have a tautomer duplicate.
5. **Noise floor**: one molecule ≈ 0.0025 MRR; GBDT reseeding moves holdout MRR by ±0.007; LB differences below ~0.005
   (single submission ~±0.02) are not evidence.
6. "Delete the truth and regenerate it" novelty simulations overstate reach about 3×. Validate on truly absent molecules.
7. Curated-DB membership flags (ChEBI / LIPID MAPS / NPAtlas) look great on public holdouts and hurt on the LB.
8. Train has exact-duplicate spectra with conflicting labels ([744446](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744446)), and a possible
   one-electron m/z offset in positive mode is under discussion ([743395](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743395)).
9. Filler guesses that fail to parse consume a rank but do not zero the score (metric v13).

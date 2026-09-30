<!-- index: actionable forum findings: isomer ranking is the lever, MetFrag-lite tuning, candidate sources, validation -->
# Forum insights (actionable)

Paraphrased; follow the links for detail. Snapshot 2026-09-30.

## Where the score is
- **Ranking, not recall, is the bottleneck.** When the truth is in the list but not first, the winner has the same molecular
  formula ~98% of the time. Separating constitutional isomers is the lever, not mass windows or formula scoring.
  Use a same-formula "isomer panel" (truth + best-scoring isomers) as the development metric
  ([743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254), [743974](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743974)).
- Structure-aware evidence (in-silico fragmentation, forward spectrum prediction such as ICEBERG / GLACIER) is where gains come
  from once the formula is right.
- Mass window 8.5 → 10 ppm changes almost nothing on timsTOF (99% within ~5 ppm).

## MetFrag-lite tuning (NeckBeard, 241 enveda-np-examples isomer panel, paired t-tests; reply in 743254)
Helped: down-weight 2-bond cleavages ×0.6 (+0.030), linear instead of sqrt intensity (+0.017), tolerance 0.005 Da (+0.005),
H shifts −2..+3 (+0.010). **All four: +0.065 offline**, but the LB did not move (0.341 → 0.336, within noise).
Did not help: hand-written bond-propensity weights, 3-bond cleavages (−0.031), penalising multiply-reachable fragments.
Takeaway: parsimony ("how few cleavages") matters, hand-written chemistry does not.
Also: M+1 / M0 isotope ratios in the MS2 peak list track carbon count and can be a formula filter.

## Candidate sources
- PubChem ∪ COCONUT covers ~96% of held-out answers but windows hold ~5–10k candidates.
- Naive full-PubChem expansion (top-50 isomers by fingerprint) dropped one team from 0.335 to 0.205. Gating matters
  (our baseline's `LIB_TAU` / `REL_TH` slots do this).
- ChEBI / LIPID MAPS / NPAtlas added no coverage beyond PubChem ∪ COCONUT for one top-5 team.
- İlker's measurement: a train ∪ COCONUT pool alone scored lower than keeping both sources.

## Validation
- CV on the 250 enveda-np-examples structures correlates poorly with the LB ([741597](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741597)).
- Track two proxies: a public-library class-2-style holdout and a small timsTOF NP panel. They can disagree, and for NP-specific
  changes the timsTOF panel predicted the LB better.
- Train/test shift: test is 100% timsTOF, train 46% ([744011](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744011)).

## Ideas floated
- Train a SMILES → MS/MS forward predictor as well and use it for re-scoring ([744343](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744343), hengck23).
- A reference guide to reading CASMI spectra: few peaks carry the signal ([741641](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741641)).

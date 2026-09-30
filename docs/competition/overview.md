<!-- index: task, exact MRR@25 matching rule, submission format, 9 h offline limits, timeline, prizes -->
# Competition overview

Source: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview (read 2026-09-30).

## Task
Predict the 2D structure (SMILES) of small molecules, mostly natural products and NP-like compounds, from LC-MS/MS
spectra. A molecule may have several spectra (different collision energies or adducts), and predictions are **per
`molecule_id`**, so evidence must be aggregated across its spectra.

## Metric: MRR@25
`MRR = mean_u 1/rank_u`, where `rank_u` is the position of the first correct guess (0 if none in the top 25).
- **Correct** = same tautomer-canonical InChIKey14 (RDKit `TautomerEnumerator().Canonicalize`, then `MolToInchiKey()[:14]`),
  RDKit pinned to **2026.03.3**. Stereo and tautomer form are ignored.
- Official code: https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank (v13, identical to the live grader
  per the host). A cheap heavy-atom-composition gate runs before canonicalization. Unparseable guesses just consume a rank.
- Deduplicate candidates by this key: tautomers of the same structure otherwise waste slots.

## Submission
`submission.csv`: header `molecule_id,smiles`. `smiles` holds ≤25 guesses joined by `;`, best first. Every
`molecule_id` exactly once, no nulls, no empty lists.

## Code requirements
Notebook-only; CPU or GPU ≤ **9 h**; **internet disabled**; free, public external data and pretrained models are
allowed; the output file must be named `submission.csv`. P100 GPUs are retired, use T4.

## Timeline (23:59 UTC)
- Start: 2026-09-14
- Entry and team-merger deadline: **2026-12-07**
- Final submission: **2026-12-14**

Prizes: $50k total (16k / 12k / 9k / 7k / 6k). The public LB uses ~33% of the test set and the final uses the other 67%.

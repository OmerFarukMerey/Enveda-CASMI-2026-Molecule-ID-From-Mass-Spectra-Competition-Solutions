<!-- index: 2026-10-26 ownneural: our own spectrum <-> structure bi-encoder trained for the top-1-locked ranks-2-25 slot (listwise against same-formula isomers, PubChem window rows and near-mass analogues; Enveda-weighted; scaffold-disjoint from the s1val / s1b answers and the clean chain's H / E), scored on s1's real final lists against isoxe / fold-safe FPNet / joint1: KILL and FAIR-KILL (movable-row MRR 0.625 vs FPZ 0.626 on 230 lists; 0.402 vs J1 0.466 on the 86 lists unseen by every baseline; NP panel 0.19); not worth plugging -->
# ownneural: our own model for ranks 2-25 (2026-10-26)

Marks: **[V]** computed here. Rules: `scratch/ownneural/PREREG.md` (+ addendum 1: H exclusion; addendum 2: fair rule, written after a
smoke run with an untrained checkpoint and before any trained-model score). State: `scratch/ownneural/STATE.md`. Code:
`train/ownneural/`, `clean/onnet.py`, `clean/onnet_infer.py`, tests `tests/test_ownneural.py` (ALL OK). Data and runs:
`D:/casmi-data/ownneural/` (eval: `eval_final/eval.md|json`, `rows.parquet`). No Kaggle call, no notebook, no git, no download, no NIST.

## Design
- Bi-encoder: per-spectrum peak transformer (Fourier m/z + neutral loss, metadata token; 6 layers, d 256), attention pooling over ALL
  the molecule's spectra; fingerprint MLP on the pool's 10,226 bits; cosine. 16.3 M parameters.
- Trained on the slot's decision: per molecule a listwise softmax over the truth and 31 negatives (6 most similar + 4 random same-formula
  pool isomers, 8 hard + 5 random PubChem-window rows, 4 most similar + 2 random near-mass pool rows of another formula), plus 0.5 x
  in-batch InfoNCE. Enveda emphasis: x1.5 sampling of structures with Enveda spectra, Enveda-only spectrum views (p 0.7) weighted x1.5.
- train.parquet only. Excluded from positives and negatives: the 675 answers of both dumps + their Murcko scaffolds (23,888 structures),
  the clean chain's frozen H (600) + E (21,917). Model selection on 1,500 scaffold-held-out Enveda structures (synthetic lists).
- Two seeds, 60 GPU-busy minutes each (~30k steps); best validation MRR 0.924 / 0.923.

## Result [V] (primary = MRR of the answer among ranks 2..n of s1's final list, 230 lists with the answer at rank 2-25)

| scorer | P (230) | P n U3 (86, unseen by XE, FPZ, J1) | enveda-180 (179) | NP panel (51) | locked RRF Delta s1val / s1b |
|---|---|---|---|---|---|
| **ON (ours, new)** | 0.625 | 0.402 | 0.750 | 0.189 | +0.0006 / +0.0054 |
| XE isoxe | 0.578 | 0.410 | 0.684 | 0.205 | -0.0002 / +0.0057 |
| FPZ fold-safe FPNet f.z | 0.626 | 0.438 | 0.712 | 0.325 | +0.0008 / +0.0050 |
| J1 joint1 | 0.759 | 0.466 | 0.888 | 0.307 | +0.0027 / +0.0077 |
| base order (leaked chain) | 0.745 | 0.712 | | | |

- ON - FPZ on P -0.000 [-0.050, +0.049]; ON - XE +0.048 [+0.000, +0.096]; ON - J1 -0.134 [-0.179, -0.090].
- Fair comparisons (each baseline on the lists it never saw): vs XE (207) +0.045 [-0.004, +0.096]; vs FPZ (88) -0.035 [-0.113, +0.041];
  vs J1 (88) -0.063 [-0.138, +0.005].
- **Verdicts: KILL (section 4) and FAIR-KILL (addendum 2).**
- J1's 0.759 is memorisation: joint1 trained on most enveda-180 answers; on the lists it never saw it falls to 0.47. The smoke run showed
  this before ON was scored, hence addendum 2.
- Locked RRF on s1b is positive for every one of our scorers (+0.005 to +0.008): s1b's answers were purged, so the leaked base order is
  weaker there. On s1val, where the base is leaked, nothing moves. These numbers describe; they do not decide.

## Reading
1. Training on the slot's decision did not lift the model above what we have. It ties FPZ, beats isoxe slightly, and loses to every
   baseline on the 86 lists none of them saw. All of them are far below the leaked chain's own order (0.71 on U3).
2. Where it is good, the task is easy: unseen enveda-180 drug-like molecules (0.750, above FPZ's 0.712, which saw part of them).
3. On Enveda natural products it is at chance level: NP panel 0.19, with random about 0.16. This is joint1's failure again
   (`joint1-2026-10-10.md`). The hidden test is NP-heavy.
4. The synthetic validation lists (0.92) did not predict the real-list difficulty. Real ranks 2-25 are the chain's own near-misses.

## Recommendation
Do not plug ON into the clean instrument or a ranks-2-25 patch. No scorer trained on train.parquet alone has yet ordered NP isomers in
real lists better than FPZ. A third bi-encoder variant (aux head, more steps) would not change that. GPU: about 2.8 h wall / 2.3 h busy.

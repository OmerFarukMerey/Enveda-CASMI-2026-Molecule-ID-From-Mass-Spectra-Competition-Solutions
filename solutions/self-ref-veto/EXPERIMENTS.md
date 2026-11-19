<!-- index: self-ref-veto experiment log: public-notebook survey of 2026-11-16, leak-free NP measurements of the veto (thresholds, reference caps, similarity measures), dropped ideas, submissions -->
# self-ref-veto: experiments

## 2026-11-16 survey of the public notebooks (what the board responds to)

- The reproducible top is **0.492**: the public 0.487 recipe + N-1c in-test network rerank at weight 0.35
  (mathleteutd, rogersjohnson l4-lam035, alexchilton "59 arms"; the team's verbatim copy n0 scored 0.492). heliosli's
  0.494 notebook runs on demo weights since 11-16 and cannot be reproduced.
- The notebooks sorted above the 0.492 copies (heetp0, evgendvorkin, ozertuu apex-gold) embed the **identical** frozen
  pipeline (sha d1e757c2...) and differ only in N-1c weight (0.35 / 0.45) or overlay knobs (K_PEAKS 3, 7.5 ppm,
  heavy-peak rule, LOTUS fill): ties at 0.492 sorted arbitrarily.
- Every N-1c variant around 0.35 (0.38 / 0.40 / 0.42 / 0.45, top-10 neighbours, all-adduct edges) sorts below the
  0.35 copies; prvsiyan's "Fox" is the same pipeline with pinned paths. The field has converged: a knob will not move
  the board (alexchilton: 4 of 59 one-knob arms gained more than 0.002); a new signal has to.

## Ideas checked and dropped (data in hand, no submission)

| Idea | Measurement | Verdict |
|---|---|---|
| One 2D answer per test molecule (demote a structure that leads several lists) | enveda-np-examples: 8 of 240 formulas hold >= 2 structures (18 of 250); the placeholder test never maps one IK14 to two molecule_ids; break-even owner accuracy 77% | too few conflicts to matter |
| Adduct bridge for the overlay / network | visible test: 9 of 400 molecules lack an `[M+H]+` / `[M-H]-` spectrum | exposure about 2% |
| Train-analog evidence rerank (team's TA-1, t0) | leak-free NP isomer test: top-1 0.615 vs 0.269 by chance (strong signal); **board: t0 0.461 (-0.031)** | promotion by family-level evidence moves too many correct leaders |

## The veto: leak-free NP measurements (`analysis/`)

Queries: the 250 `enveda-np-examples` timsTOF natural products (234 with `[M+H]+` / `[M-H]-` spectra). References:
train spectra from all other libraries. Truth pairs: query vs its own structure (232 have same-adduct references).
Isomer pairs: query vs up to 40 same-formula training structures (2,362 pairs).

Merged spectra (TA / N-1c preparation), 6 references per adduct (`np_veto_power2.py`):

| threshold | truth vetoed | isomer vetoed by Tanimoto to truth: < 0.4 / 0.4-0.6 / 0.6-0.8 / >= 0.8 |
|---|---|---|
| 0.15 | 1.7% | 43% / 29% / 13% / 0% |
| 0.20 | 3.0% | 55% / 36% / 17% / 0% |
| 0.25 | 3.0% | 65% / 42% / 20% / 0% |
| 0.30 | 3.9% | 73% / 48% / 25% / 0% |

Peak-count guards (>= 5 or >= 8 peaks in both spectra) change nothing: the low-cosine truths have 10-120 peaks.

Similarity measures at matched truth error (`np_veto_measures.py`, 6 references per adduct):

| measure | truth error 2% | truth error 3% |
|---|---|---|
| merged cosine | 43 / 29 / 14% | 66 / 42 / 20% |
| spectral entropy | 49 / 32 / 15% | 70 / 45 / 23% |
| **best single pair** | **66 / 44 / 22%** | **78 / 53 / 26%** |

Best single pair, references per adduct (same script, cap argument):

| cap | threshold | truth vetoed | isomers < 0.4 / 0.4-0.6 / 0.6-0.8 |
|---|---|---|---|
| 6 | 0.30 | 2.2% (5/232) | 65% / 44% / 22% |
| 6 | 0.40 | 3.0% (7/232) | 77% / 52% / 26% |
| 30 | 0.30 | 1.3% (3/232) | 64% / 43% / 20% |
| 30 | 0.40 | 2.2% (5/232) | 75% / 50% / 23% |

List simulation (truth pairs + 6 known isomers + 1-4 unknown candidates, 20 shuffles of every query):

| rule | wrong known leader, truth at ranks 2-5 | truth leading |
|---|---|---|
| best pair 0.30, cap 6 | mean dRR +0.230; better 73%, worse 2% | -0.020; worse 2.6% |
| + confirmation 0.85 / 0.20 | +0.344; worse 2.8% | -0.021; worse 2.9% |

## Submissions

| Submission | Change on the 0.492 notebook | Commit | Public LB |
|---|---|---|---|
| vpd | best-pair veto < 0.30, 6 refs / adduct, after N-1c; dedup after the overlay | smoke healthy: every stage ran; veto inactive on the 12 visible library hits | **0.492** (= base: no public rank-1 change) |
| w30c | best-pair veto < 0.40, 30 refs / adduct + confirmation 0.85 / 0.20; dedup | smoke healthy | **0.484** (−0.008: correct leaders demoted; the team's harness reproduces it, a cross-lab class-1 truth with an Orbitrap-only reference) |
| nls | leader-safe network: rank 1 exactly as n0 (LAM 0.35), ranks 2-25 at LAM 0.70 | smoke healthy | **0.487** (−0.005: right answers moved down within ranks 2-25) |

## Leader-safe network (nls), 2026-11-17/18

The board rewarded splitting the neural weight by rank (w_top 0.25, w_mid 0.75: +0.004). N-1c uses one weight for every
rank and its public optimum (0.35) is a rank-1 optimum. nls keeps rank 1 exactly as n0 (verified identical on 400 / 400
local lists) and orders ranks 2-25 at LAM 0.70. Simulation on the NP network (real N-1c edges among the 250
enveda-np-examples, lists drawn around the truth, imperfect neighbour lists), mean change of the truth's reciprocal rank
per networked list: LAM_MID 0.5 / 0.7 / 1.0 = +0.005 / +0.009 / +0.011 (better 8 / 11 / 13%, worse 3 / 5 / 6%).

## 2026-11-18 results: rank-2-25 re-orderings (rank 1 identical to n0)

| Submission | Change | Public |
|---|---|---|
| nls | network weight 0.70 on ranks 2-25 | 0.487 |
| tas | TA-1 train-analog evidence on ranks 2-25 only | 0.487 |
| nlt | both | 0.483 |

Reading: n0's order of ranks 2-25 (w_mid 0.75, documentation prior, mass prior) is tuned to the public part; both new signals move right answers down there, and the losses add. Together with vpd (0.492) and w30c (0.484): every change to n0's lists tried here was neutral or negative on the board.

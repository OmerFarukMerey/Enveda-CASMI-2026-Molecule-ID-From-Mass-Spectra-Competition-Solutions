<!-- index: 2026-10-17 knowledge-driven chemistry-rules scorer (diagnostic neutral losses / ions tied to SMARTS-level requirements, absence penalties, adduct rules, hard unexplained-peak veto) as a gated re-order of same-formula rows in stack1's first 10; pre-registered dev / eval split of the 606 replay molecules; 6 of 22 rules dropped on dev (adduct, tropylium, immonium, NH3, unexplained-peak rules), the frozen 16-rule scorer +0.0013 / +0.0027 on eval with CIs over 0, about 0.1 public answer; no candidate -->
# Chemistry rules for same-formula isomer order (2026-10-17)

Marks: **[V]** computed here; **[C]** cited; **[I]** inference. MRR = MRR@25 on the notebook's metric identity. Composite as in
stack1err (per-molecule mean over list types, then over molecules). Arms: **m** = answer's popularity masked, **u** = real.
CPU only, one process, below-normal priority; nothing submitted, pushed, downloaded or committed; the visible `test.parquet`
was not read; no NIST data. Pre-registration: `scratch/chemrules/PREREG.md` (written after the development split, before the
evaluation split was scored).

## 0. Bottom line

- **No candidate.** The frozen 16-rule scorer gains **+0.0013 ± 0.0013 (m) / +0.0027 ± 0.0020 (u)** on the 397-molecule
  evaluation split (95% CI [−0.0012, +0.0038] / [−0.0007, +0.0073]); pool-answer lists **+0.0005 ± 0.0020**. It fails criterion 2
  (CI) and criterion 4 (size: about **0.09 public answers** expected against the 1 required). It is safe (wins / losses 15 / 5 and
  19 / 4, worst case 0.12 RR) but almost never decides rank 1 (16 first rows changed in 1,377 lists per arm).
- **Why it cannot be large** [V, I]: the isomers above an answer are almost always positional isomers that carry the same
  functional groups (both glycosides, both caffeoyl esters, both acids, same methoxy count). Diagnostic-loss rules separate a
  top-10 same-formula group in 23% of lists but carry enough evidence (θ = 1) to move rows in only 7%. The rules that do separate
  often (adduct-implied groups, tropylium / immonium ions, an "every strong peak must be explainable" veto) are the ones that
  are chemically too coarse and lose, on dev and again on eval.
- **Do not build a patch.** Keep `clean/chemrules.py` as a tested, CPU-trivial module (2 ms per list) in case a future
  candidate generator produces class-level isomers (glycoside vs aglycone ester, acid vs lactone), where it would bite.

## 1. Set-up [V]

- Base: B*, the stack1-faithful lists (`casmi-work/stack1err/bstar.pkl`: e006u replay + stack1's frag-gap branch + E2R2), 606
  molecules (mixed regime 380, Enveda panel 226), types c1 / c2 (pool answers) and dA / dB; spectra from the regime query tables.
- Split by `sha1(molecule_id) mod 3`: **development** = 0 (209 molecules, 1,077 lists), **evaluation** = 1, 2 (397 molecules,
  2,109 lists). Rules, thresholds and the dropped rules were chosen on dev only.
- Re-order: in B*[:10], each same-formula group (≥ 2 rows) is permuted among its own slots by violation tier
  floor((v − v_min) / θ), θ = 1, then old order. A rule counts only if it separates the group. Rows of other formulas never move,
  nothing enters or leaves the 25. The scorer reads SMILES and spectra only (no row source, no popularity, no label).
- Coordinator note (finalrank study): dA / dB answers were removed from the pool on purpose, so results are reported separately
  for pool-answer lists, and criterion 3 requires the gain to hold there.

## 2. Rules (`solutions/fusion-v4n/clean/chemrules.py`)

Evidence is the maximum relative intensity over all spectra of the molecule (cross-spectrum pooling; absence = absent in every
informative spectrum); ≥ 0.2 → weight 1, ≥ 0.05 → 0.5; absence penalties 0.5; tolerance 0.005 Da + 5 ppm; low-resolution
spectra (most peaks at ≤ 2 decimals) are ignored; losses are taken from the precursor and from the core ion ([M+NH4]+ → [M+H]+,
formate / chloride adducts → [M-H]-), sugar losses also in chains.

| Rule | Evidence | Requirement on the candidate | Default |
|---|---|---|---|
| L_HEX / L_DHEX / L_PENT / L_URON | loss 162.0528 / 146.0579 / 132.0423 / 176.0321 (neg. also +H) | O-linked pyranose / furanose of that class (ring parser) | on |
| L_CGLY | loss 120.0423 or 90.0317 | C-glycoside | on |
| I_OXO | 163.0601 / 147.0652 / 133.0495 (pos.) | glycoside of that class | on |
| A_GLY | no sugar loss or oxonium anywhere | (penalises an O-glycoside) | on |
| L_ACYL / A_ACYL | ketene or acid loss of galloyl, coumaroyl, caffeoyl, feruloyl, sinapoyl, malonyl, benzoyl, cinnamoyl, quinic acid | acyclic ester / amide whose acyl piece has that composition / (penalises a named acyl whose loss is absent) | on |
| L_SULF / A_SULF, L_PHOS | SO3, H2SO4 loss, HSO4- / HPO3, H3PO4 loss, H2PO4-, PO3- | sulfate or sulfonic acid / phosphate | on |
| L_CO2N | CO2 loss in negative mode | C(=O)O (acid, ester, lactone) | on |
| L_CH3 | CH3• loss 15.0235 | CH3 on O / N / S | on |
| L_TMA, I_CHOL | loss 59.0735; 104.1070 / 184.0733 | N(CH3)3; choline | on |
| L_NH3 | NH3 loss from [M+H]+ | NH2 | **off** (dev) |
| I_TROP | 91.0542 | benzyl-type ring | **off** (dev) |
| I_IMM | Phe / Tyr / Trp / His immonium ions | that side chain on C-N | **off** (dev) |
| AD_H2O | adduct [M-H2O+H]+ / [M-2H2O+H]+ / [M-H2O-H]- | ≥ 1 / 2 aliphatic OH or COOH | **off** (dev) |
| AD_NEG | adduct [M-H]- | an acidic H | **off** (dev) |
| XPK | a peak ≥ 0.3 not a ≤ 2-cleavage piece ± H (MetFrag-lite pieces as a hard veto) | explains it | **off** (dev) |

Not implemented, with reasons: NP-likeness (tested 2026-10-08, `nplike`: no signal among pool rows, every replay cell failed);
ring / double-bond equivalents and elements (identical inside a same-formula group); a soft MetFrag-lite score for every molecule
(already stack1's frag-gap branch; the isomer top-25 fragment re-rank scored 0.419 vs 0.446 on the leaderboard [C stack1err]);
flavonoid retro-Diels-Alder ions (C7H5O4+ 153.0182 is also galloyl+, i.e. composition-ambiguous; the ion is a ≤ 2-cleavage
piece that MetFrag-lite already credits).

## 3. Development split (choices) [V `cr_eval.py --split dev`]

| Variant | arm | touched | w / l | composite | pool lists (c1 + c2 w / l) |
|---|---|---|---|---|---|
| all 22 rules, θ 1 | m / u | 286 / 282 | 24 / 58, 27 / 72 | −0.026 / −0.032 | −0.056 (1 / 41) |
| XPK alone | m / u | 164 | 14 / 33, 14 / 41 | −0.016 / −0.019 | −0.038 (0 / 24) |
| AD_NEG alone | m / u | 73 / 72 | 6 / 7, 8 / 7 | −0.003 | −0.005 |
| I_TROP alone | m / u | 24 | 2 / 8, 1 / 14 | −0.004 / −0.006 | −0.008 |
| AD_H2O, I_IMM, L_NH3 alone | m | 29, 12, 12 | 0 / 5, 0 / 4, 2 / 4 | −0.001 to −0.002 each | all negative |
| **frozen 16 rules, θ 1** | m / u | 36 / 35 | 6 / 1, 6 / 1 | +0.0006 / +0.0016 | −0.0006 (1 / 1) |
| frozen, θ 0.5 | m / u | 94 | 9 / 15, 8 / 21 | −0.006 / −0.008 | −0.013 (1 / 11) |

Typical failures read on dev examples: [M-H]- labels on lactones and sulfones (no acidic H); [M-H2O+H]+ on guaiacol and
lactones; C7H7+ from indole alkaloids, terpenoids and phenethyl amides; m/z 110.0718 (His immonium) from caffeine; NH3 loss from
emetine's secondary amine; XPK vetoing the true structure because its strongest ions are rearrangement products.

## 4. Evaluation split (pre-registered) [V `cr_eval.py --split eval --boot --single`, `casmi-work/chemrules/res_eval_frozen.json`]

| Arm | lists | group-level firing | touched (mx w/l, p2 w/l) | first rows | w / l | composite ± SE [95% CI] | pool (c1 + c2) | dA / dB | worst case |
|---|---|---|---|---|---|---|---|---|---|
| m | 1,377 | 312 | 101 (38: 8 / 4, 63: 7 / 1) | 16 | 15 / 5 | **+0.0013 ± 0.0013** [−0.0012, +0.0038] | +0.0005 ± 0.0020 (5 / 2) | +0.0019 ± 0.0011 (10 / 3) | 0.12 RR |
| u | 1,377 | 309 | 100 (38: 11 / 4, 62: 8 / 0) | 16 | 19 / 4 | **+0.0027 ± 0.0020** [−0.0007, +0.0073] | +0.0005 ± 0.0020 (5 / 2) | +0.0052 ± 0.0025 (14 / 2) | 0.11 RR |

Per type (m): c1 −0.0019 ± 0.0039 (2 / 1), c2 +0.0022 ± 0.0022 (3 / 1), dA +0.0030 ± 0.0021 (5 / 1), dB +0.0009 ± 0.0008 (5 / 2).
Criteria: 1 (≥ 15 touched) pass; **2 (CI > 0) fail** in both arms; 3 (pool gain, wins > losses) pass; **4 (size ≥ +0.0133) fail**
(weaker arm +0.0013 → 0.09 public answers); 5 (worst case ≤ 1 RR) pass. **Verdict: fail.**

Per rule on eval (group-level firing / in moved groups; the rule alone, arm m, w / l, composite):

| Rule | fires | in moved groups | alone: touched, w / l | alone: composite |
|---|---|---|---|---|
| L_HEX | 65 | 47 | 27, 2 / 3 | −0.0003 |
| L_DHEX | 14 | 13 | 8, 2 / 0 (u 3 / 0) | +0.0002 / +0.0005 |
| L_PENT | 11 | 5 | 3, 0 / 3 | −0.0005 |
| L_URON | 6 | 6 | 4, 0 / 1 | −0.0000 |
| L_CGLY | 18 | 12 | 3, 0 / 0 | 0 |
| I_OXO | 15 | 14 | 8, 2 / 4 | −0.0006 |
| A_GLY, A_ACYL, A_SULF | 107, 180, 11 | 16, 23, 0 | 0 (weight 0.5 < θ alone) | 0 |
| L_ACYL | 6 | 6 | 4, 1 / 0 | +0.0000 |
| L_PHOS | 5 | 5 | 3, 0 / 0 | 0 |
| L_CO2N | 95 | 40 | 29, 4 / 0 (u 5 / 0) | +0.0005 / +0.0006 |
| L_CH3 | 47 | 39 | 22, 3 / 0 | +0.0002 |
| L_SULF, L_TMA, I_CHOL | 0 | 0 | 0 | 0 |
| *dropped:* XPK | | | 303, 22 / 90 | **−0.030** (u −0.036) |
| *dropped:* I_TROP | | | 65, 2 / 22 | −0.008 |
| *dropped:* AD_NEG | | | 174, 23 / 13 | −0.003 (pool −0.007) |
| *dropped:* AD_H2O | | | 50, 9 / 2 (u 6 / 5) | +0.0001 / −0.0003 (pool −0.001) |
| *dropped:* L_NH3, I_IMM | | | 12, 1 / 4; 4, 0 / 0 | −0.0003; 0 |

Readings [I]: the dev decisions hold on eval (every dropped rule loses or is null; the hard unexplained-peak veto is again the
worst, which agrees with the leaderboard loss of the fragment re-rank). The frozen rules' gain is consistent in sign (both sets,
both arms, 34 wins / 9 losses over the two arms) but small: they move mostly rows 2-10 (16 first rows of 101 touched lists).
The coordinator's caveat does not bite here: the rules read no source, the pool-answer part is positive (5 / 2), and the larger
dA / dB part comes from the same rules (L_CO2N, L_CH3, L_DHEX) [V, per-type rows].

## 5. Runtime [V]

Profiles (RDKit SMARTS, ring parser) and evidence for 2,109 lists took 3.6 s in one process (2 ms per list, profiles cached by
SMILES). A Kaggle T4 notebook with ~400 molecules × 10 rows: about 1-2 s CPU, no GPU, no new dataset (NumPy + RDKit).

## 6. Recommendation

No patch on stack1. The knowledge rules are right when they speak but speak too rarely where the loss is (positional isomers
with identical functional groups). If the team wants a near-free probe anyway, the frozen scorer's local expectation is +0.001 to
+0.003 composite (≈ 0.1 public answer): below the leaderboard noise floor, so a submission would not be informative.

## 7. Files

`solutions/fusion-v4n/clean/chemrules.py` (module), `solutions/fusion-v4n/tests/test_chemrules.py` (8 data-free tests).
`scratch/chemrules/` (git-ignored): `PREREG.md`, `STATE.md`, `cr_data.py`, `cr_explore.py`, `cr_eval.py`, `cr_diag.py`.
`C:\Users\hemre\casmi-work\chemrules\`: `recs.pkl`, `res_dev_*.json`, `res_eval_frozen.json`, `eval_frozen.log`, `summ.py`.

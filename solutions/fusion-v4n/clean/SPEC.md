<!-- index: fusion-v4n clean chain: behaviour specification (inputs, outputs, formulas, acceptance test) of every third-party module without a code licence that a rewrite would have to replace: pool tables, chemistry core, spectrum cache, fragmenter, library and analog search, fingerprint network, v1 engine, fe_v4 feature families, ranker, forward-model runners; which parts are pinned by golden tests and which are accepted on the DEV benchmark -->
# Clean chain: behaviour specification of the modules to be rewritten

Written 2026-10-10 from the BEHAVIOUR of the current chain: file formats, the docstrings and signatures of the modules,
and the stored outputs. It states what goes in, what comes out and by which formula, so that an implementation can be
written without the original source. **Limit, stated plainly:** the author of this file read the original modules
(`code/casmi/**`, `code/fe_v4/**` of `ahmedberatozer/casmi26-v4b-models`, which carry no code licence) in order to write
it. A strict clean-room process separates the person who reads from the person who implements. Sections 1 to 4 were
then implemented by the same author (`pool_build.py`, `chemcore.py`), from this text and RDKit / NumPy documentation,
without copying; whether that is enough is a question for the host or a lawyer, not for this file. For the larger
modules (5 to 9) the recommendation is that the implementer works from this file and the golden fixtures only.

**Depth.** Sections 1 to 4 are complete and verified: the implementation passes the golden tests. Sections 5 to 9 are
at contract level: they were written from the modules' docstrings and signatures, the dataset manifests and our own
reviews, not from a line-by-line reading, and constants given there (similarity exponents, layer counts, window
widths) are the reference values as documented, **not re-verified here**. They say what a replacement has to provide,
not how to reproduce the reference bit for bit.

Two kinds of acceptance test:
- **pinned**: the rewrite must reproduce stored outputs of the current chain (`clean/golden.py` writes the fixtures to
  `$CASMI_DATA/clean/golden/`; `tests/test_clean_contract.py` compares). Used where our own models and tables must stay
  interchangeable with existing caches.
- **benchmark**: no parity is wanted (our own models are refitted on the new features anyway); the module is accepted
  when the chain built on it reaches the stated number on the DEV half of the joint benchmark
  (`train/joint/bench.py`; FINAL untouched).

Sizes of the originals (lines, for effort estimates): `casmi` 15 files / 2,626 lines in the v3 copy; the v4b copy adds
`dreams_model.py` 628 and `fpnet6.py` 299; `fe_v4` 11 files / 3,863 lines; forward-model runners `ice_runner.py` 760,
`fuse.py` 160, `gl_runner.py` 896, `gl_fuse.py` 139, `shim/**` 1,085.

## 1. Pool tables (pinned; implemented in `pool_build.py`)

One directory, read by the engine at start-up:

| File | Content |
|---|---|
| `pool_meta.parquet` | one row per candidate structure, **sorted by `mass` ascending**; columns `key` (str, the metric key: first 14 characters of the InChIKey of the tautomer-canonical form, RDKit 2026.03.3 `TautomerEnumerator().Canonicalize`), `smiles` (stereo-free canonical SMILES), `formula` (`CalcMolFormula`), `mass` (float64, `ExactMolWt`), `n_heavy` (int16), `src` (int8: 1 = training structure only, 2 = COCONUT only, 3 = both), `train_sid` (int64, `sid` of the training structure or −1), `np_likeness` (float64, COCONUT's field or NaN), `pid` (int64 = row number). `key` is unique |
| `pool_fp.npy` | uint8 `(n_pool, ceil(nbits / 8))`: the selected fingerprint bits of every row, `numpy.packbits` (big-endian bit order), row-aligned with `pool_meta` |
| `fp_bits.npy` | int32 `(nbits,)`: indices of the selected bits inside the raw fingerprint (reference pool: 10,226 of 16,551) |
| `train_structs.parquet` | one row per `inchikey14` of `train.parquet`, in order of first appearance; `sid` = row number; columns `inchikey14, smiles, formula_given, n_spec, n_tims, libmask, formula, mass, n_heavy, key, rawkey, sid`. `n_spec` = spectra of the structure, `n_tims` = those whose `instrument_type` contains "timstof" (case-insensitive), `libmask` = bit i set if the structure occurs in library i of the list `enveda-180, pluskal_ms2, riken, gnps, massbank, mona, spectraverse, msdial, drug_plus, enveda-np-examples, masaryk`; `rawkey` = InChIKey first block without tautomer canonicalisation |
| `train_fp_sel.npy` | uint8 `(n_train_structs, ceil(nbits / 8))`: selected bits of the training structures, row = `sid` |
| `pool_frag_off.npy`, `pool_frag_mass.npy` | int64 `(n_pool + 1,)` offsets and float32 fragment masses of every pool row (section 4) |

Standardisation of an input SMILES: parse; keep the largest fragment (`rdMolStandardize.LargestFragmentChooser`);
neutralise (`Uncharger`); a structure that still has a net formal charge is not a COCONUT pool row; remove stereo;
canonical SMILES. All other fields are computed from that SMILES.

Raw fingerprint = concatenation, in this order, of RDKit bit vectors (`rdFingerprintGenerator`, default settings unless
stated): Morgan radius 2, 4,096 bits; Morgan radius 3, 4,096; Morgan radius 2 with feature atom invariants, 2,048;
RDKit path fingerprint `maxPath=6`, 2,048; atom pair, 2,048; topological torsion, 2,048; MACCS keys, 167. Total 16,551.
Bit selection: the bits whose frequency over the structures lies in [0.005, 0.995] (the reference took a random
sample of 200,000 rows; `pool_build.py` uses all rows, so its own selection may differ by a few bits: `verify` reports
the overlap; our own models are trained on our own selection).

Pool = training structures ∪ COCONUT structures, one row per `key`; where both sources have the key the training row's
SMILES is kept and `src` = 3. Pins: `test_gold_pool_rows` (300 reference rows: SMILES, key, formula, mass and all
10,226 reference bits identical), `pool_build.py verify` (full tables).

## 2. Chemistry core (pinned; implemented in `chemcore.py`)

- Adduct string `[nM±aX±bY…]z±` → `(n, z, delta, sign)` with `m/z · z = n · M + delta`,
  `delta = Σ ± k · mass(formula) − sign · z · m_e`. Neutral mass `M = (m/z · z − delta) / n`; NaN if the string does
  not parse. The ten test adducts: `[M+H]+ [M+NH4]+ [M-H2O+H]+ [M-2H2O+H]+ [M+Na]+ [M+K]+ [M-H]- [M-H2O-H]-
  [M+CH2O2-H]- [M+Cl]-`. Adduct index for the networks: the ten test adducts first (index 0 to 9), then every other
  adduct of the vocabulary in a fixed order, last index = unknown (reference vocabulary: 60 entries; a rewrite fixes
  its own list and stores it in the checkpoint).
- Pin: `test_gold_neutral_mass` (three spectra of every adduct in `train.parquet`, |ΔM| < 2e-6 Da).

## 3. Spectrum cache of `train.parquet` (pinned; cleaning implemented in `chemcore.clean_peaks`)

Per spectrum: keep peaks with `0 < m/z ≤ precursor + 2.0` and intensity ≥ 0.001 × the largest intensity in that range;
if more than 512 remain keep the 512 most intense; sort by m/z; intensities divided by that largest intensity;
float32. Files: `spec_off.npy` int64 `(n + 1,)`, `spec_mz.npy`, `spec_it.npy` float32 (CSR), `spec_meta.parquet` with one
row per spectrum in file order: `adduct, precursor_mz, precursor_error_ppm, collision_energy_orig_units,
base_peak_intensity, num_peaks, ce_mean, ce_min, ce_max, ce_n` (statistics of the list `collision_energy_ev`; NaN / 0
if empty), `sid` (training-structure id, −1 if none), `lib` (index in the library list of section 1), `instr`
(0 timsTOF, 1 Orbitrap / FT, 2 other TOF, 3 ion trap, 4 triple quadrupole, 5 other; substring rules on the lower-cased
`instrument_type` in that order of precedence), `adduct_ix`, `mode` (+1 / −1), `nm` (neutral mass), `n_clean`.
Pin: `test_gold_clean_peaks` (400 spectra of row group 0, m/z identical, intensities within 1e-7).

## 4. Fragmenter (pinned for the mass lists; implemented in `chemcore.fragment_masses`)

Heavy-atom graph of the SMILES; atom weight = monoisotopic element mass + its hydrogens. Fragment set = the whole
molecule and every connected component after removing one bond or two bonds (any bond, ring bonds included). Masses
rounded to 0.001 Da (`floor(1000 m + 0.5) / 1000`), unique, ascending; at most 200,000 values before rounding.
Peak explanation: a peak at `p` of a spectrum with adduct `a` is explained if a fragment mass `f` satisfies
`|p − (f + d + s · m_H)| ≤ max(0.01 Da, 15 ppm)` for an integer `s` with `|s| ≤ 2` and an ion offset `d` of the adduct
(positive mode: +proton; additionally Na⁺, K⁺ or NH4⁺ when the adduct contains them; negative mode: −proton;
additionally Cl⁻ or formate when the adduct contains them). Features per candidate (fragment channel): intensity-weighted
share, mean share, count share and top-10 share of explained peaks; the same four for sub-formula explanation (a peak
is explained if some sub-formula of the candidate's formula, with a hydrogen count compatible with its heavy atoms,
has that ion mass). Query peaks for this channel: cleaned with floor 0.002 and the 128 most intense.
Pin: `test_gold_fragments` (100 pool rows against `pool_frag_mass.npy`, same length, |Δ| < 6e-4). The explanation
features are **benchmark**-accepted. `chemcore.fragment_masses` is pure Python (seconds for 60 heavy atoms): the
production version needs a compiled loop (numba), about 60 lines.

## 5. Library and analog search (benchmark; not implemented)

- Input: the spectrum cache; a query = list of cleaned spectra with mode and a target neutral mass (median over the
  molecule's spectra).
- Weights of a spectrum for similarity: intensity^power normalised to sum 1, optional entropy weighting (Li et al.
  2021: spectra with entropy below 3 are sharpened with exponent 0.25 + 0.25 · S). Similarities between two m/z-sorted
  weighted peak lists with tolerance `max(tol_da, ppm)`: spectral entropy similarity
  `1 − (2 S_merged − S_a − S_b) / ln 4`, weighted cosine, and a "hybrid" variant in which a library peak may match a
  query peak directly or shifted by the difference of the neutral masses.
- Direct search (class 1): library spectra whose neutral mass is within the ppm window of the target and with the same
  polarity; per structure: best similarity, mean of the best few, number of hits. `lib_max` of a molecule = the best
  direct similarity over all structures; the pipeline's library gates read it (`LIB_TAU 0.9`, join gate 0.7).
- Analog search: hybrid similarity against one representative spectrum per (structure, polarity[, library]) (the one
  with most cleaned peaks) within a mass window of the target; the top analogs `(sid, similarity, mass shift)`.
  Analog propagation to a candidate: similarity-weighted Tanimoto between the candidate's and the analogs' selected
  fingerprint bits (`ap`, best Tanimoto, Tanimoto to the top analog, weighted mean).
- Acceptance: on DEV `c2N`, the analog prior alone and inside the refitted ranker (it is the largest feature group of
  our ranker: gain share 0.36, permutation loss −0.12 on `c2J`, `reviews/fsranker2-2026-10-07.md` 2.3).

## 6. Fingerprint network and model bank (benchmark for accuracy, pinned for the interface; not implemented)

- Item of one spectrum: peaks cleaned as in section 3 but with the top `max_peaks` by intensity, **sqrt** intensities;
  precursor m/z; adduct index; collision energy with a known flag; number of merged spectra; mode. A merged item joins
  up to eight spectra of one structure and polarity (peaks within a tolerance merged, maximum intensity kept).
- Model: a transformer encoder over peak tokens (multi-scale sinusoidal embedding of m/z plus an intensity embedding)
  with a global token built from precursor, adduct, collision energy, merged count and mode; reference size d 512,
  8 layers, 8 heads; output: `nbits` logits `z` and a 10-way element-presence head.
- Score of a candidate with fingerprint bits `f`: `f · z`. Training loss: softmax ranking of the true structure against
  K = 63 pool structures inside ±10 ppm of its mass (plus the element head), mixed precision.
- Molecule logits: `0.5 · (mean over spectra + mean over the merged items per polarity)`, averaged over the models
  of the bank.
- Our training loop `train/train_fpnet.py` (derived from the Apache-2.0 loop) imports the model class, the collate
  function and the training-data access of `casmi`: **these three (211 + 176 lines) are what must be rewritten** before
  an own network counts as clean. Our `train/joint/model.py` (ours) already has a peak-set transformer and a loader
  (`train/joint/data.py`) that can serve as the base.
- Acceptance: hold-out table of `train_fpnet.py` (fold 0, never seen): top-1 0.669 all / 0.389 NP-catalogue / 0.345
  panel for the published fold-safe checkpoint (`train/README.md`); the own network must be within ±0.013 (one SE).

## 7. Evidence engine (benchmark; not implemented)

`run(spectra, target)` → candidate set, feature matrix (94 columns), names. Candidates = pool rows inside ±`ppm` of
the target mass (reference 10 ppm) plus generated derivatives of the strongest analogs (`pid = −1`): reaction-SMARTS
transformations (one or two steps) whose exact mass change equals `target − mass(analog)` within tolerance, applied at
every site, de-duplicated by metric key. Feature groups (each raw value plus rank inside the list normalised to
[0, 1], z-score, gap to the list maximum): library (section 5), analog propagation, fingerprint model (`f · z` of the
bank, per-spectrum and merged views, bit-count-normalised score, element-head agreement), fragmenter and sub-formula
(section 4), list descriptors (log candidate count, isomer count, flags for generated rows), crosses
(analog × fingerprint rank, library × fingerprint agreement). Ranks must not depend on the row order (average ranks on
ties): the reference breaks ties by position, which leaks the pool's source order (`reviews/ranker-2026-10-02.md`).
The join of PubChem-only rows (`nb/embed/pc_join.py`, ours) calls the same feature code for rows that are not pool
rows; a rewrite keeps that entry point (`features for explicit SMILES`).

## 8. Feature families `fe_v4` (benchmark; not implemented; candidates for removal)

Five families, 66 columns in the public ranker and 158 in our rows: `derivation` (evidence that a candidate is a
biosynthetic derivative of an analog, with transformation and site priors counted on training pairs), `frag2`
(discriminative peak explanation inside the list and inside same-formula groups, cleavage coverage, diagnostic neutral
losses against substructure capabilities), `analog_struct` (relations on the heavy-atom element graph between a
candidate and the analogs: shared ring systems, core containment, scaffold hashes), `model_views` (agreement of several
fingerprint models, per-polarity and per-block scores), `fragnet` (a count-based log-likelihood-ratio model of which
annotated 1- and 2-bond fragments match peaks, fitted on training spectra; 9.4% of the hold-out answers are in its
fitted roster, `reviews/fsranker2-2026-10-07.md`). Static tables under `fe_v4/res/` were fitted by the author from
`train.parquet`; their build scripts are not public, so a rebuild needs our own fit.
Measured weight in our ranker (permutation inside each list, an upper bound; DEV-type lists, same review): `fragnet`
−0.104 ± 0.009 on `c2J`, `derivation` −0.033, `analog_struct` −0.008, `model_views` / `frag2` −0.003 to −0.005.
Recommendation: rewrite `fragnet` (916 lines, the one family that carries weight) as our own count model on the
fragmenter of section 4, with out-of-fold tables; drop the other four and refit; accept on DEV `c2J` / `dropJ`
against the `ours` baseline (0.788 / 0.655, `scratch/clean/abl/dev_rankers.py`).

## 9. Forward-model runners (pinned on scores; not implemented)

`ice_runner.py`, `gl_runner.py` (subprocess runners), `fuse.py`, `gl_fuse.py` (notebook glue), `shim/**` (pure-torch
stand-ins for DGL / torch_scatter / LinSATNet): the dataset notices call them "own code" and give no licence. The
ms-pred source and the MassSpecGym checkpoints they drive are MIT.
- Contract (from the runner's usage text): `runner <package_dir> <in.json> <out.json> --device --budget SEC`; input a
  JSON list of groups `{mid, cands: [SMILES], spectra: [{mz, it, prec, adduct, mode, instrument, ce_ev, ce_orig,
  ce_units}]}`; output per molecule `{SMILES: score or null}` where the score is the mean entropy
  similarity between the candidate's predicted spectrum and each covered measured spectrum (ICEBERG: positive mode,
  `[M+H]+` and the adducts its checkpoint supports, collision energy required; GLACIER: `[M+H]+`). `null` when nothing
  is covered or the budget ran out. Fusion: inside same-formula groups of a list the z-scored forward scores are added
  to the z-scored ranker score with weights `ICE_LAM`, `GL_LAM` (`fusion_core`, ours).
- Pin: stored scores of the local replay (`$CASMI_DATA/fwd/`, `reviews/fwd-replay-2026-10-05.md`: local scores equal
  the Kaggle dump). A rewritten runner must reproduce them within 1e-4 on 200 groups.
- Permissive starting point instead of a rewrite from nothing: `runner/fwd_runner.py` of
  `prvsiyan/casmi26-forward-reference-stack` (16,254 bytes, "our own code, MIT" per its card) with the unmodified
  ms-pred source; it needs DGL / torch-scatter wheels (the ahmedberatozer packages avoid them through the shim).

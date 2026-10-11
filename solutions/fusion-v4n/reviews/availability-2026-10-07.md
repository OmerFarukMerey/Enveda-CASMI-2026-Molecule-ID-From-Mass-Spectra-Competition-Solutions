<!-- index: fusion-v4n 2026-10-07 scoping of a commercial-availability prior (vendor / reference-standard depositors) without any large download: what the popularity arrays hold and how POP_MU enters; model-free in-group enrichment of the answer among its same-formula isomers by substance count, PubMed count, their sum, patents and chance (Enveda panel, other-library NP-catalogue answers, Enveda-180; nitrogen-free; group size; pool rows and joined PubChem rows); COCONUT collections and vendor ids in synonyms as a local stand-in for vendor membership and what they add beyond popularity (conditional logit, failure analysis, a one-stage proxy); what the Sls2 tier-2 table is (PubChem-only rows, no pool rows, depositor list unpublished); PubChem depositor categories from the live source table; compliance view; download and build plan with sizes -->
# A commercial-availability prior: scoping without the download (2026-10-07)

**First line, as asked.** The crude signals we already have enrich the answer among its same-formula isomers
**strongly** on natural products (substance count alone: top-1 0.71 on the Enveda panel against 0.07 by chance, 0.31
against 0.10 on other-library NP-catalogue answers; below chance on Enveda-180). So the idea is not dead on arrival.
**But the sharper signal the hypothesis asks for adds little on top of it in everything measurable here**: the local
stand-in for vendor membership (COCONUT collections + vendor ids in synonyms) is weaker than the substance count on
every set, adds +0.011 ± 0.014 MRR on the panel when combined with it, and is negative in the one-stage proxy with a
ranker. And the table of team Sls2 does not contain a single pool structure. Nothing here is a leaderboard estimate.

Marks: **[M]** measured here (scripts `scratch/avail/a1..a4`, logs `scratch/avail/a*.log`, tables
`D:\casmi-data\avail\`), **[V]** read from a primary source today (Kaggle CLI listing or card, NCBI FTP listing, the
PubChem source table), **[H]** hypothesis or inference.

Machine: CPU only, one process at below-normal priority, no GPU. Fetched: seven card / script files of
`dmitriigluzdov/casmi26-pubchem-popularity-prior` (largest 12 KB), two NCBI FTP directory listings, `README-Extras`,
the PubChem source table (231 KB). Nothing larger. No model was run; no `kaggle` write; nothing committed; the
visible `test.parquet` was not read.

## 1. What we have today

**The arrays** [V README of `dmitriigluzdov/casmi26-pubchem-popularity-prior`; M shapes]. Built from the PubChem
Compound Extras of 2026-09-27 (`CID-SID.gz`, `CID-PMID.gz`, `CID-Patent.gz`), one value per structure:

| Array | Rows | Value | Local |
|---|---:|---|---|
| `pool_lsid.npy` | 710,701 | log(1 + PubChem **substance records**), summed over the CIDs that share the InChIKey connectivity block | yes |
| `pool_lpmid.npy` | 710,701 | log(1 + distinct PubMed articles) | yes |
| `pool_np.npy` | 710,701 | bits 1 COCONUT, 2 LOTUS, 4 NPAtlas | yes |
| `pool_lpatent.npy` | 710,701 | log(1 + patent documents) | yes (`pubchem_study/popds`) |
| `pc_lsid / pc_lpmid / pc_np / pc_ik14.npy` | 105,875,489 | the same for the PubChem tier, in tier row order | yes (`pubchem_study/popds`) |
| `pool_cid.npy` (2.8 MB), `pc_cid.npy` (424 MB), `pc_lpatent.npy` | | representative CID, patents | **no** |

Pool rows are keyed by position (`pid` = row of `casmi26-v2-pool/pool_meta.parquet`); 679,471 of 710,701 pool rows
(95.6%) have a PubChem record [V]. A substance record is one deposit by one depositor, so `lsid` is **a count of
depositors that list the compound**, vendors and everything else together.

**How `POP_MU` enters** [V `nb/08_submit.py`, `nb/embed/fusion_core.py::popularity_reorder`]: `POOL_POP = pool_lsid +
pool_lpmid`. In stage A, before the forward-model re-rank, the first 60 base candidates get `f = z(ranker score over
the list) + POP_MU * POOL_POP[pid]`; pool rows are re-sorted among the pool slots; generated and joined rows
(`pid < 0`) keep their slots, get no popularity and are put into a formula group of their own. `POP_MU` = 0.25 in
v33b; with `POP_LIB_OFF` it is skipped at `lib_max ≥ GATE_TAU`. A typical `POOL_POP` gap between two isomers is
several units (panel answers: median 13.2; all COCONUT pool rows: median 2.5 [M]), so 0.25 × gap is of the order of
one to three ranker standard deviations: the prior is already strong where the counts differ.

**PubChem depositors** [V `rest/pug/sourcetable/substance/CSV`, fetched today]: 933 sources, 349.9 M live substances.
544 sources are in the category "Chemical Vendors" and hold 166.5 M substances (**47.6%**); the rest are R&D (283
sources), curation efforts (43, 41.3 M), governmental (47), journal publishers (19), subscription services (11), NIH
initiatives (26). So roughly half of what `lsid` counts are vendor listings already. NP / bioactive / standards
vendors that deposit: Ambinter 10.1 M, Molport 6.9 M, TimTec 1.1 M, Specs 439 k, MuseChem 433 k, LGC Standards 349 k,
Biosynth 275 k, Sigma-Aldrich 258 k, MedChemExpress 117 k, TargetMol 102 k, TCI 31 k, Selleck 15.8 k, BioCrick 14.6 k,
ChemFaces 9.7 k, AbMole 9.4 k, Biopurify 1.1 k, Planta Analytica 173. Not found as depositors under these names:
Extrasynthese, PhytoLab, AnalytiCon, Greenpharma. Two depositors need care: **NIST Mass Spectrometry Data Center**
(266 k substances) and the spectral libraries **MoNA** (1.32 M) and MassBank Europe (20.6 k).

## 2. Enrichment of the answer among its same-formula isomers (model-free) [M]

For each answer: the group = the answer plus every other candidate with the same molecular formula. Rank the group
by one signal; report the answer's expected reciprocal rank and top-1 under random tie-breaking, and the AUC (share
of isomers the answer beats, ties one half). Two universes: **P** = every pool row of that formula (no model at
all), **L** = the cached candidate lists of the c2 regime (answer hidden from the library; fold-safe bank; feature
caches `D:\casmi-data\rankbank\feat\c__*.npz`, which carry the lists of `panel2`, `mixed_regime`, `trainplan` and the
hold-out), pool rows only and pool + the 100 joined PubChem rows. The two universes agree to the third digit for
pool rows, because the candidate search returns the whole formula group.

Sets: **Enveda panel** = `enveda-np-examples` (248 of 250 structures in the pool); **other-library NP** = train
structures outside the Enveda libraries with a natural-product catalogue flag (25,784; 20,000 sampled in P; the 1,360
hold-out ones in L); **Enveda-180** as the synthetic contrast.

| Set (universe) | n | median group | chance MRR / top-1 | substances MRR / top-1 / AUC | PubMed | current prior (sum) | patents |
|---|---:|---:|---|---|---|---|---|
| Enveda panel (P) | 239 | 44 | 0.171 / 0.067 | **0.819 / 0.707 / 0.984** | 0.800 / 0.683 / 0.980 | 0.814 / 0.699 / 0.984 | 0.740 / 0.611 / 0.966 |
| panel, N-free | 131 | 71 | 0.111 / 0.036 | 0.788 / 0.664 / 0.989 | 0.748 / 0.617 / 0.981 | 0.769 / 0.641 / 0.987 | 0.678 / 0.538 / 0.965 |
| panel, with N | 108 | 21 | 0.245 / 0.105 | 0.856 / 0.759 / 0.979 | 0.863 / 0.764 / 0.979 | 0.869 / 0.769 / 0.980 | 0.816 / 0.699 / 0.967 |
| other-library NP (P) | 18,068 | 30 | 0.220 / 0.095 | 0.464 / 0.309 / 0.823 | 0.431 / 0.285 / 0.743 | 0.473 / 0.316 / 0.827 | 0.407 / 0.265 / 0.728 |
| other-library NP, N-free | 9,060 | 52 | 0.153 / 0.058 | 0.381 / 0.226 / 0.835 | 0.344 / 0.203 / 0.744 | 0.386 / 0.229 / 0.839 | 0.315 / 0.181 / 0.732 |
| hold-out NP (L, pool rows) | 1,360 | 30 | 0.226 / 0.100 | 0.469 / 0.318 / 0.832 | 0.428 / 0.278 / 0.747 | 0.472 / 0.312 / 0.836 | 0.405 / 0.265 / 0.725 |
| hold-out NP, N-free (L) | 692 | 51 | 0.158 / 0.061 | 0.386 / 0.235 / 0.849 | 0.334 / 0.188 / 0.745 | 0.383 / 0.221 / 0.851 | 0.307 / 0.174 / 0.726 |
| other-library, not NP (P) | 16,868 | 18 | 0.271 / 0.124 | 0.317 / 0.165 / 0.526 | 0.292 / 0.145 / 0.516 | 0.311 / 0.160 / 0.520 | 0.298 / 0.151 / 0.522 |
| **Enveda-180 (P)** | 19,379 | 47 | 0.169 / 0.065 | **0.133 / 0.039 / 0.426** | 0.148 / 0.049 / 0.471 | 0.133 / 0.040 / 0.426 | 0.154 / 0.052 / 0.484 |
| hold-out, no PubChem record (L) | 450 | 12 | 0.321 / 0.151 | 0.180 / 0.031 / 0.104 | 0.263 / 0.094 / 0.416 | 0.180 / 0.031 / 0.104 | 0.253 / 0.085 / 0.393 |
| panel, pool + joined rows (L) | 250 | 118 | 0.071 / 0.018 | 0.824 / 0.716 / 0.991 | 0.803 / 0.690 / 0.988 | 0.822 / 0.712 / 0.991 | (not local) |
| mixed-regime NP, pool + joined (L) | 241 | 102 | 0.112 / 0.042 | 0.484 / 0.349 / 0.905 | 0.438 / 0.314 / 0.815 | 0.491 / 0.344 / 0.913 | |
| trainplan NP, pool + joined (L) | 222 | 109 | 0.109 / 0.038 | 0.459 / 0.320 / 0.894 | 0.390 / 0.260 / 0.783 | 0.462 / 0.315 / 0.895 | |

By group size, current prior, lists with pool rows (MRR / top-1; chance MRR in brackets):

| Isomers in the group | Enveda panel | hold-out NP | hold-out Enveda-180 |
|---|---|---|---|
| 2 to 3 | 1.000 / 1.000 (0.69), n 15 | 0.864 / 0.740 (0.69), n 169 | 0.623 / 0.306 (0.69), n 24 |
| 4 to 9 | 0.943 / 0.917 (0.41), n 24 | 0.717 / 0.537 (0.42), n 202 | 0.324 / 0.090 (0.40), n 59 |
| 10 to 29 | 0.854 / 0.738 (0.19), n 65 | 0.550 / 0.370 (0.20), n 307 | 0.164 / 0.033 (0.21), n 100 |
| 30 and more | 0.755 / 0.613 (0.06), n 137 | 0.268 / 0.113 (0.06), n 682 | 0.031 / 0.003 (0.06), n 304 |

Reading [M, then H]:
- Substance count alone carries almost everything; adding PubMed changes the third digit; patents are weaker.
- **Natural product or not decides the sign.** AUC 0.98 (panel), 0.83 (other NP), 0.52 (other, not NP), 0.43
  (Enveda-180), 0.10 for answers without a PubChem record.
- The panel is not an ordinary sample of natural products: its answers have a median popularity of 13.2 against 5.3
  for other-library NP answers and 2.5 for all COCONUT pool rows; 37% are in the PhytoLab phyproof reference
  collection (0.2% of COCONUT), 84% carry a standards / bioactive-vendor mark, and **all 248 are also in another
  public library**. `EXPERIMENTS.md` already calls them "deliberately common compounds".
- Among joined rows the AUC rises (0.91 against 0.84), because PubChem-only rows are mostly obscure. This is the
  quantity that lost on the leaderboard when used (section 4).

**What bounds the reading.** Every answer we can label is a train structure, i.e. it is in a public spectral
library, and those are built from compounds somebody could buy or was given. The hidden test's class 2 (no public
spectrum) and class 3 (not in PubChem) are by definition not such molecules. The numbers above are therefore an
upper reading for class 1-like molecules and say nothing measured about class 2. One more bias on the non-Enveda
sets: MoNA and MassBank are themselves PubChem depositors, so a library answer gets a substance record from its own
library.

## 3. A sharper signal: what is local, and what it adds

**`pool_meta.parquet` has no source columns** [M]: `key, smiles, formula, mass, n_heavy, src, train_sid,
np_likeness, pid` (`src` 1 = train only 248,440; 2 = COCONUT only 436,515; 3 = both 25,746). **But the full COCONUT
dump is local** (`D:\casmi-data\extra_db\raw\coconut_csv-10-2026.zip`, 738,794 rows, 44 columns) with `collections`,
`dois`, `organisms`, `synonyms`, `cas`, `annotation_level`. Joined by InChIKey14 it covers 461,417 pool rows (99.8%
of the COCONUT rows, 0.2% of train-only rows). No joined PubChem row has a COCONUT record (they are PubChem-only by
construction), so this stand-in exists for pool candidates only.

Stand-in for vendor membership [M counts over COCONUT rows]: collections InterBioScreen 59,236, AnalytiCon Discovery
NPs 3,786, **PhytoLab phyproof Reference Substances 1,086**, Specs Natural Products 615, Indofine 39, ZINC NP 78,670;
vendor ids in synonyms: AKOS 88,643, MFCD 14,110, `HY-` (MedChemExpress) 11,457, Specs 4,016, TargetMol-like 791,
Selleck-like 273. ChemFaces (`CFN`) and BioCrick (`BCN`) ids: none, MolPort 27: the synonym field is not a vendor
catalogue. `vend_n` = number of these marks (0 to 10; 22% of COCONUT structures have one), `std` = PhytoLab or
MedChemExpress or Selleck or TargetMol (2.5%).

Enrichment as in section 2 (MRR / top-1 / AUC; pool formula groups):

| Signal | Enveda panel (n 239) | other-library NP (n 18,068) | Enveda-180 (n 19,379) |
|---|---|---|---|
| chance | 0.171 / 0.067 / 0.5 | 0.220 / 0.095 / 0.5 | 0.169 / 0.065 / 0.5 |
| current prior | 0.814 / 0.699 / 0.984 | 0.473 / 0.316 / 0.827 | 0.133 / 0.040 / 0.426 |
| number of COCONUT collections | 0.771 / 0.677 / 0.955 | 0.361 / 0.216 / 0.687 | 0.138 / 0.046 / 0.425 |
| `vend_n` | 0.682 / 0.563 / 0.927 | 0.371 / 0.230 / 0.678 | 0.153 / 0.054 / 0.476 |
| number of DOIs | 0.711 / 0.598 / 0.904 | 0.325 / 0.189 / 0.606 | 0.162 / 0.061 / 0.492 |
| `std` | 0.602 / 0.437 / 0.899 | 0.333 / 0.201 / 0.609 | 0.167 / 0.063 / 0.499 |
| MedChemExpress id | 0.514 / 0.374 / 0.798 | 0.327 / 0.198 / 0.600 | |
| PhytoLab collection | 0.430 / 0.314 / 0.686 | 0.242 / 0.116 / 0.516 | |
| has CAS | 0.452 / 0.280 / 0.876 | 0.337 / 0.199 / 0.651 | |
| in COCONUT at all | 0.196 / 0.084 / 0.569 | 0.252 / 0.122 / 0.568 | 0.138 / 0.046 / 0.425 |

Every stand-in enriches natural-product answers, and **every one is weaker than the substance count**. That is
expected if half of the substance records are vendor listings: correlation of `lsid` with `vend_n` over COCONUT
rows 0.51; mean `lsid` 2.1 at `vend_n` 0, 3.3 at 1, 5.2 at 3, 6.0 at 5 or more.

**What it adds on top of the current prior** [M]:

| Test | Result |
|---|---|
| Fixed combination `pop + 2·vend_n` against `pop`, paired | panel +0.011 ± 0.014 MRR, +0.033 ± 0.024 top-1 (N-free +0.014 ± 0.023); hold-out NP +0.010 ± 0.005 (N-free with 30+ isomers +0.013 ± 0.007); other-library NP, 18,068 groups: −0.000 ± 0.001; Enveda-180 −0.002 ± 0.001; **other-library, not NP: −0.029 ± 0.001**; hold-out "other" −0.033 ± 0.006 |
| Conditional logit inside formula groups, fitted on 5,399 other-library NP groups (5-fold) | `pop` 0.476 MRR; `pop` + in-COCONUT flag 0.489; + `vend_n` 0.490; + `std` 0.490; + all COCONUT fields 0.487. Applied unchanged to the panel: 0.814 → 0.820 → 0.829 → 0.829 → 0.818 (standard error of each about 0.019). The individual vendor flags get mixed signs (PhytoLab −0.41, MFCD −0.41, AKOS +0.51, MedChemExpress +0.26) |
| The same fitted inside the panel (5-fold, 239 groups) | `pop` 0.814 ± 0.019; + `vend_n` 0.833; + all COCONUT fields 0.857 ± 0.018 (N-free 0.769 → 0.810). The only cell where the gain exceeds one standard error; fitted on the panel itself |
| Where the current prior fails on the panel | another isomer is more popular in 72 of 239 groups. That isomer itself carries the `std` mark in 69% and a vendor mark in 81% of them, and is a train structure in 82%. A `std` flag could separate the answer from **all** more popular isomers in 11 groups and points the wrong way in 9; `vend_n > 0`: 7 separable. Other-library NP: 3,645 failures of 5,399; separable 120, wrong way 1,169 |
| One-stage proxy: `z(public ranker on fold-safe features, top 60) + 0.25·pop + ν·signal`, MRR@25, c2 lists, pool rows | **every** vendor term is zero or negative on every set: `vend_n` ν 0.25: panel −0.001 ± 0.001, hold-out NP −0.008 ± 0.002, Enveda-180 −0.004 ± 0.002; `std` ν 1: −0.002, −0.013 ± 0.003, −0.002; number of collections ν 0.25: +0.002 ± 0.002, −0.000 ± 0.002, −0.003 ± 0.002. In the same proxy `POP_MU` itself: panel 0.956 → 0.985 (0.15) → 0.982 (0.25) → 0.949 (1.0); hold-out NP 0.804 → 0.819 → 0.816 → 0.688; no PubChem record 0.844 → 0.829 → 0.811 → 0.393; Enveda-180 0.986 → 0.985 → 0.985 → 0.667 |

The proxy is one stage on lists whose ranker is close to its ceiling there (0.96 on the panel) and it is not a
replay; `uregime-2026-10-07.md` showed that such numbers do not order leaderboard results. It is reported because it
is the only place where the stand-in meets a spectral score, and there it does not help.

Reading [H]: the isomer that beats the answer under the popularity prior is usually **another commercially
available natural product** (quercetin against morin). Availability cannot separate two compounds that are both
sold. A complete depositor table would fix the stand-in's gaps (ChemFaces, BioCrick, Biopurify are missing from
it), but not this.

## 4. The Sls2 table [V, Kaggle CLI today]

`ngdminh31/casmi26-pubchem-tier2-fp-public`, CC0-1.0, three files: `tier2_fp.npy` 6,269,506,016 bytes,
`tier2_mass.npy` 57,850,240 bytes (float64: **7,231,264 rows**; 867 bytes of fingerprint per row checks),
`tier2_meta.pkl` 618,764,447 bytes. Card: PubChem FTP snapshot 2026-09-15, compounds deposited by "56 selected
sources (natural-product/metabolite databases incl. Wikidata; literature collections: Nikkaji, Thieme, Springer
Nature, Wiley, ChEMBL, BindingDB; natural-product/bioactive vendors; public screening collections)", neutral, single
fragment, CHNOPS / halogen, 150 to 1,200 Da.

- **Keys**: `tier2_meta.pkl` holds InChIKey14, SMILES, CID and a **group bitmask: 1 = NP / metabolite databases, 2 =
  literature, 4 = vendors, 8 = screening**. Four groups, not 56 flags.
- **The 56 depositors are not listed** anywhere: not in the card, and there is no small file. The card names nine.
  The card says "no depositor annotations are redistributed".
- **Coverage: PubChem-only rows.** The table excludes everything "already present in COCONUT, train.parquet,
  PubChemLite-2026 or ChEBI/LIPID MAPS". Our pool is train + COCONUT, so **no pool structure is in it**. It can label
  joined PubChem rows and nothing else; it cannot say which of two pool isomers a vendor sells, which is what the
  hypothesis needs.
- Its purpose by its layout (6,930-bit fingerprints in engine 2's bit selection) is a candidate source for engine 2
  [H]. That Sls2 uses the vendor bit as a prior is not shown by anything public; the table is of 09-19 and the team
  rose on 10-06.
- `tier2_meta.pkl` is a pickle from a third party: loading it executes whatever it contains. If it is ever fetched,
  it must be read with a restricted unpickler or not at all. Building our own table from the NCBI files avoids it.

## 5. Compliance view

- **What it is.** PubChem depositor information is public domain and distributed by NCBI [V]. A per-compound count
  of vendor depositors is general knowledge about a compound, of the same kind as the substance count we use and as
  the reference counts MetFrag has used for years. It is computed for all 710,701 pool structures (and, if wanted,
  the PubChem tier) before any test molecule is seen and is not keyed to any test mass, formula or id.
- **What separates it from tuning on the test distribution.** The table does not; the **weight** would. `POP_MU`
  went 0.15 → 0.25 on a leaderboard comparison. A second weight chosen the same way is a second parameter fitted to
  about 130 public molecules (one answer moving from place 2 to 1 is 0.0038). The defensible form is a weight fixed
  from the stage-level measure before the submission, one submission, no line search.
- **Pitfall 7 applies in full.** Curated-database membership flags looked great on public hold-outs and hurt other
  teams on the leaderboard. A vendor flag is a membership flag. Section 2 shows the mechanism of the trap: on the
  Enveda panel popularity alone reaches top-1 0.70, on ordinary natural products 0.31, on synthetic answers it is
  below chance, and our hold-outs are all library molecules.
- **Depositors to exclude on principle.** NIST Mass Spectrometry Data Center and NIST Chemistry WebBook are PubChem
  depositors. A vendor-only count does not touch them; a flag "deposited by NIST MS Data Center" would be membership
  of the NIST library and must not be built. (The substance count we use today includes those records as one
  depositor among many: public PubChem data, not NIST spectra; worth stating in the README.) MoNA / MassBank flags
  mean "has a public spectrum", i.e. class 1, and are equally not availability.
- **Fragility on the hidden set** [H]: (a) class 3 molecules are not in PubChem and cannot gain; a stronger prior
  pushes real candidates above their generated ones only through the pool slots, since generated rows keep theirs;
  (b) class 2 means "no public spectrum", which is negatively related to being a common standard: the more popular
  isomer is a library compound in 82% of the panel's failures, and if it were the answer the molecule would be
  class 1; (c) a host who built the test to probe structure elucidation may have chosen isolates on purpose; the
  data page calls the train examples "common NPs" and makes no such statement about the test.
- **Our own leaderboard evidence** [V `docs/experiments.md`]: pool popularity 0.15 gave +0.005 (0.399 → 0.404);
  0.25 against 0.15 on the gated configuration gave +0.005 again (v4a → v4b, 0.409); a popularity prior **inside the PubChem channel lost** twice (v5a 0.393 against a 0.399 reference, v5c 0.405
  against 0.409). Both gains are at the size of the noise of one submission and both have the same sign. No value
  above 0.25 was submitted according to the log lines read. Locally, `POP_MU` 0.4 was +0.014 ± 0.006 on the
  panel-like set and −0.011 ± 0.006 on ordinary molecules.

## 6. Plan, if a download is approved

**Recommendation on what to fetch.** Not the Sls2 pickle (no pool rows, four coarse groups, unsafe format). The
primary source instead:

| File | Source | Size [V listing of 2026-09-27] | Why |
|---|---|---:|---|
| `SID-Map.gz` (+ `.md5`) | `ftp.ncbi.nlm.nih.gov/pubchem/Substance/Extras/` | **3.3 GB** | every live substance: SID, source name, registry id, CID. The only file that says who deposited what |
| `Source-Names` | same directory | 70 KB | source id → display name |
| source table CSV | `pubchem.ncbi.nlm.nih.gov/rest/pug/sourcetable/substance/CSV` | 231 KB (already in `scratch/avail/web/`) | source → category ("Chemical Vendors" etc.) |
| `pool_cid.npy` | `dmitriigluzdov/casmi26-pubchem-popularity-prior` | 2.8 MB | representative CID of each pool row |
| `pc_cid.npy` | same dataset | 424 MB | the same for the PubChem tier; only if joined rows are to be labelled |
| optional: `CID-InChI-Key.gz` | `pubchem/Compound/Extras/` | 6.9 GB | to sum vendor deposits over all stereo-variant CIDs of a connectivity block, as the popularity arrays do. Without it the feature uses the representative CID only (the most-deposited variant) and undercounts a little |

Minimum to answer the question: `SID-Map.gz` + `pool_cid.npy` = 3.3 GB. `tier2_meta.pkl` (619 MB) would only be a
cross-check of what Sls2 labelled, on joined rows.

**Build** (CPU, one pass over about 350 M lines, an estimated 20 to 40 minutes at low priority; `scratch/avail/`
then `tools/` if it survives): keep lines whose CID is a pool (or tier) CID; map the source to its category; per
structure write
1. `n_vendor`: number of distinct "Chemical Vendors" depositors, aggregators (Molport, Ambinter, ZINC, AKos-type)
   counted apart;
2. `np_std`: listed by one of a fixed, documented set of natural-product / reference-standard vendors (ChemFaces,
   BioCrick, Biopurify, MedChemExpress, TargetMol, Selleck, LGC Standards, Planta Analytica, Sigma-Aldrich, TCI,
   Biosynth; the list is written down before any enrichment is looked at);
3. `in_stock`: listed by Molport or Ambinter (ZINC's own in-stock subsets are a separate, larger download and are
   not needed for a first answer);
4. `n_nonvendor`: the rest of the substance count, so that the current prior can be split into its two halves.
   NIST depositors are dropped from every column.

**Where it would enter.** Stage A, pool candidates only, as a second term in `popularity_reorder`: `f = z + POP_MU ·
pop + AVAIL_MU · avail[pid]`, behind a new flag `AVAIL_MU` (default 0, so v33b stays reproducible), under the same
`POP_LIB_OFF` gate. The first thing to test is not an added term but the **split**: `μ_v · log1p(n_vendor) + μ_o ·
log1p(n_nonvendor) + μ_p · lpmid` against today's sum, because sections 1 and 3 say the vendor share is already
inside `lsid`. Joined PubChem rows: **no term** in the first candidate. Extra weight on PubChem rows has lost on the
leaderboard five times and the PubChem-channel popularity lost twice; a label for joined rows would at most be used
to order the 50 joined rows among themselves, and only after its own stage measure.

**Validation before any submission.**
1. The in-group enrichment of section 2 with the new columns, on the same groups, with the panel as the cleanest
   proxy of the acquisition process and the other-library NP answers as the sober one. **Pass condition, fixed
   now**: the split or the added term beats today's prior by more than two paired standard errors on the hold-out NP
   set's nitrogen-free groups with 30 or more isomers (where our loss is; today 0.244 MRR, n 443) **and** is not
   negative on Enveda-180 and on answers without a PubChem record. The stand-in reached +0.013 ± 0.007 there, i.e.
   it would not have passed.
2. The failure analysis of section 3 repeated: in how many of the panel's 72 failures the answer has a vendor
   listing that every more popular isomer lacks. With the stand-in: 11 (and 9 the wrong way). If the full table
   does not at least double the net, stop.
3. Only then the mandatory local full replay against v33b (panel and mixed regime, with the forward stages), and a
   proposal for one submission, which needs the user's approval.

**Cost on Kaggle.** A dataset of ours with three `uint8` arrays in `pid` order: 3 × 710,701 bytes = **2.1 MB**
(`pool_np.npy` is 710,829 bytes). For the PubChem tier, one `uint8` per row is 105.9 MB per column (`pc_np.npy` is
105,875,617 bytes): 106 to 212 MB, or a sparse table of the listed CIDs of a few MB; not needed for the first
candidate. Run time: one `np.load` and an indexed add per molecule; nothing measurable. Provenance for the README:
NCBI PubChem Substance Extras of 2026-09-27, public domain; the source-category table; the fixed vendor list.

## 7. What could not be established

- Whether the hidden test's molecules are commercially available standards. Nothing local can show it: every
  labelled answer is a library molecule, and the panel is a set of common compounds.
- What Sls2 does with its table, and its 56 depositors.
- What a complete depositor table adds over the COCONUT stand-in (the stand-in has no ChemFaces / BioCrick /
  Biopurify marks). Needs `SID-Map.gz`.
- Patent counts and any COCONUT-type annotation for joined PubChem rows (`pc_lpatent.npy` is not local; COCONUT has
  no record of them by construction).
- Any leaderboard effect. The two popularity gains (+0.005 each) are the only unbiased evidence that a prior of
  this family helps on the hidden set at all, and each is within the noise of one submission.

## 8. Verdict

Measured: popularity already separates natural-product answers from their isomers very well, and the substance
count is to a large part a vendor count. Measured: the available vendor marks are weaker than that count and add
about a hundredth of MRR on the panel within its error, nothing on 18,068 other natural products, and lose on
non-NP answers and in the stage proxy. Structural: the Sls2 table has no pool rows. Hypothesis that remains open:
a complete depositor split (vendor against non-vendor substance records) is a better-shaped prior than their sum.
That is a 3.3 GB download and an hour of CPU for a question whose stand-in answer is "small"; it is worth doing only
if nothing with a better stage measure is waiting, and it should be judged by the pass condition written above.

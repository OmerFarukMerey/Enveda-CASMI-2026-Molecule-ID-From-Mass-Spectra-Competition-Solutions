<!-- index: literature survey: how structure identification from MS/MS is solved outside Kaggle, with numbers and what transfers to our pipeline -->
# Literature survey, 2026-10-05: structure identification from MS/MS outside Kaggle

Web search and page fetches on 2026-10-05; nothing was run, trained or submitted. Pages were read through a fetch
tool that returns extracted text, so every number is "the page says so". **Marks.** **[P]** read on the paper,
repository or contest page. **[S]** seen only in a search-result snippet or abstract listing: **unverified**. **[I]** my
inference. Publisher pages of Nature, Springer, ChemRxiv, C&EN and some PMC pages refused the fetch; what rests on them
is [S]. Reference numbers in brackets point to the list at the end.

## 1. Summary

1. **Without a reference spectrum, the best published systems put the right structure first for roughly a third of
   compounds when the candidates come from PubChem or ChemSpider** (CASMI 2016: 34%; CSI:FingerID structure-disjoint:
   39%; ICEBERG with the true formula: 40%), and for about two thirds when candidates come from a biological database
   (68.5% [S]). CASMI 2022, the closest real-world analogue, ended at 25 to 38% correct 2D structures among the
   challenges each team attempted. [P/S, section 3]
2. **De novo generation is far from that.** With the true formula given, the best top-1 exact-match rates are 16 to 25%
   (FRIGID) and 24 to 30% (MS-GPT, July 2026) on benchmarks whose molecules are known compounds; a 2026 audit shows
   such numbers fall when close analogues are removed from the decoder's corpus. Enveda's own MS2Mol reaches exact
   matches in single digits and reports "close match" instead. [P/S]
3. **The three levers with the largest reported effects all act on candidates the model has no spectra for:** training
   the spectrum-to-fingerprint model on simulated spectra of database molecules (MIST bonus retrieval 9.6% to 53.8%
   top-1 in the clean setting), contrastive fine-tuning against same-formula PubChem decoys (ICEBERG / GLACIER top-1
   0.385 to 0.525 on a random split), and knowing the formula (MIST mass-based retrieval 26.6% with a predicted formula,
   51.5% with the true one). [P]
4. **Database priors are the strongest single signal in every contest with known compounds and the least transferable
   one here.** Reference counts alone ranked 143 of 208 CASMI 2016 challenges first; the PubChem return order alone
   gives 50% top-1 on MassSpecGym without looking at the spectrum. Our own submissions with such priors lost. [P]

## 2. Method families

**(a) Fingerprint prediction + database ranking.** CSI:FingerID / SIRIUS (fragmentation tree, kernel or deep kernel
predictors, Bayesian scoring) is still the reference; it is trained on GNPS, MassBank and NIST and its structure search
runs through a web service [S][20]. MIST (Nat. Mach. Intell. 2023) replaces trees by a transformer over peak
sub-formulas; it predicts fingerprints better than CSI:FingerID on most spectra but CSI:FingerID keeps the higher
top-1 retrieval rate [S][10]. MIST's published recipe already contains the two additions that matter: augmentation
with simulated spectra and a contrastive head [S][10]. De Waele et al. (2026) show that fingerprint accuracy and
retrieval accuracy are different objectives and trade off, depending on how similar the candidates are [P][12].

**(b) Joint embeddings.** JESTR (Bioinformatics 2025): 15.6% rank@1 on MassSpecGym mass-based candidates, 11.9%
formula-based [S][6]. MVP (Anal. Chem. 2026): 26.4% / 11.1%, and 36.0% / 14.0% when the query is a consensus of the
molecule's spectra [S][7]. FLARE (bioRxiv 2026, peak-to-atom alignment): 43.2% / 22.7% [P][8]. MSAlign (arXiv 2026):
frozen DreaMS + ChemBERTa with 4 M trainable parameters, candidate-based contrastive training, 20 minutes on one
V100; 53.8% R@1 on a formula split but only 16.2% on the MCES split against FLARE's 27.2% [P][9]. The MassSpecGym
audit found that SMILES-string encoders exploited canonicalisation artefacts (82.4% falling to 6.7%) [P][2], so
language-model molecule towers need care.

**(c) Forward prediction for ranking.** ICEBERG 2 (2025): 40.0% top-1 and 92% top-10 on NIST'20 [M+H]+ against
same-formula PubChem candidates, 33.5% on a scaffold split [P][4]. MARASON (ICML 2025) adds retrieval of reference
spectra of similar structures (27% against 19% top-1 [S][21]). GLACIER (arXiv 2026): MassSpecGym recall@1 69.95%
(mass challenge) and 49.88% (formula challenge), 7.95 times faster than ICEBERG (1.3 ms per spectrum) [P][3]. The audit
lists ICEBERG at 36.2% on the full formula-challenge test split, half of which has no collision energy [P][2]: the
ms-pred numbers are for the subset the simulator can serve [I]. FraGNNet (TMLR 2025) and FIORA (Nat. Commun. 2025) are
the alternatives; FIORA has open MSnLib weights and negative mode [S][22][23].

**(d) De novo.** MSNovelist (2022): 25 to 26% first rank on GNPS and CASMI 2016 [S][18], on splits a 2026 review calls
leak-prone [P][17]. FRIGID's table (true formula given) [P][5]:

| Method | NPLIB1 top-1 / top-10 | MassSpecGym top-1 / top-10 |
|---|---|---|
| MIST + MSNovelist | 5.40 / 11.04 | 0.00 / 0.00 |
| MADGEN | 2.10 / 2.39 | 1.31 / 1.54 |
| MS-BART | 7.45 / 10.99 | 1.07 / 1.11 |
| DiffMS (ICML 2025) | 8.34 / 15.44 | 2.30 / 4.25 |
| MIST + MolForge (corrected) | 2.24 / 5.11 | 10.73 / 14.48 |
| FRIGID-base | 19.80 / 24.41 | 16.09 / 18.19 |
| FRIGID + ICEBERG-guided scaling | 25.03 / 33.37 | 18.29 / 22.00 |
| MS-GPT (arXiv 2607.23607) [P][15] | 29.8 / 41.1 | 23.9 / 28.7 |
| Test-time-tuned language model [S][16] | 12.88 / – | 3.16 / – |

MARLIN (arXiv 2026) is the first to drop the true formula, with a mass constraint instead; its abstract gives no
numbers [P][14]. The audit shows decoder accuracy depends on analogues (Tanimoto ≥ 0.7) of test molecules in the
pretraining corpus, and that a batching bug inflated MIST + MolForge threefold [P][2].

**(e) Representations and analog search.** Spec2Vec and MS2DeepScore are library-similarity models. DreaMS (Nat.
Biotechnol. 2025): self-supervised transformer on GeMS (GNPS spectra), 1024-d embeddings; fine-tuned for fingerprints
it is on par with MIST without peak formulas [S][11]. Enveda's PRISM: masked-peak BERT on 1.2 billion spectra (half
Enveda's own), +23% relative in spectral matching, +7 to 16% relative R² on property prediction; no code or weights
[P][24].

**(f) Formula from MS/MS.** SIRIUS trees; ZODIAC re-ranks formulas jointly across a dataset (up to 16-fold fewer errors
on a plant extract above 700 Da) [S][28]. BUDDY (Nat. Methods 2023): bottom-up candidate generation + learned
ranking; 93.0% below m/z 400, 63.8% at or above [S][26]. MIST-CF (JCIM 2023): 71% top-1 on NPLIB1 against 48% for
SIRIUS, MS/MS only, positive mode only [S][25][P repo]. Cost of a wrong formula, from the audit [P][2]: MIST on the
mass challenge 26.6% top-1 with MIST-CF formulas, 51.5% with true ones.

**(g) Confidence.** COSMIC (Nat. Biotechnol. 2022): calibrated score + SVM on score gaps; 57 correct of 123 queries at
FDR below 10% in the biomolecule database, 16 of 127 at FDR 0 in ChemSpider, where other tools annotated nothing at
FDR 40 to 70% [S][27]. Two 2026 papers on MassSpecGym: retrieval-level confidence (score gaps over the candidate list)
is the best rejection signal and fingerprint-level uncertainty is poor [P][13]; conformal sets keep 90% coverage but
swell to about 80% of the candidates under distribution shift, where top-1 is about 10% [P][29].

## 3. Numbers on comparable tasks

| Task, candidate set | Formula given | Best published top-1 | Source |
|---|---|---|---|
| MassSpecGym mass challenge (≤ 256 candidates within mass tolerance, PubChem-derived; test molecules MCES ≥ 10 from train) | no | MIST 14.6; JESTR 15.6; MVP 26.4 (36.0 consensus); FLARE 43.2; ICEBERG 2 64.0 and GLACIER 70.0 (ms-pred evaluation) | [1][6][7][8][3] |
| MassSpecGym bonus (same-formula candidates) | **yes** | MIST 9.6; FLARE 22.7; ICEBERG 36.2 (full split); GLACIER 49.9; MIST trained with ICEBERG-simulated PubChem spectra 53.8; **PubChem order, no spectrum, 50.0** | [2][3][8] |
| NIST'20 [M+H]+, PubChem same-formula | **yes** | ICEBERG 40.0 (top-10 92); scaffold split 33.5; GLACIER + contrastive 52.5 / 46.0 | [4][3] |
| CSI:FingerID structure-disjoint, PubChem vs. biological database | no | 39.4 vs. 68.5 | [S][19] |
| CASMI 2016 category 2 (ChemSpider ± 5 ppm, median 816 candidates, 208 challenges) | no | CSI:FingerID 34% (top-10 63%); six tools combined 39% (top-10 80%) | [30] |
| CASMI 2016 category 3 (metadata allowed) | no | MS-FINDER + databases 70%; all tools + references + MoNA 79% (top-10 94%) | [30] |
| CASMI 2017 category 4 | no | CSI:FingerID 66 of 198 (33.3%) | [S][32] |
| CASMI 2022 priority set (raw LC-MS/MS, no library entry) | no | 2D structure: 38% (21/55), 25% (58/228, Nothias), 26% (57/222, SIRIUS team [S]); formula 94% | [31] |
| De novo, MassSpecGym / NPLIB1 | **yes** | 23.9 / 29.8 (MS-GPT); 18.3 / 25.0 (FRIGID) | [15][5] |
| EnvedaDark (226 natural products in no public database) | no | MS2Mol: 62% "meaningful" (Tanimoto ≥ 0.4), 21% "close"; CSI:FingerID 43% / 11% | [S][33] |

## 4. How contests were won

- **CASMI 2016** [P][30]. Fragmentation only: CSI:FingerID and its IOKR variant, 30 to 34% first. With metadata the
  winner (MS-FINDER) reached 70% by adding library hits and database presence. Signals alone: reference count 143 of
  208 first, MoNA library 70, "lowest ChemSpider ID" 113, retention time 1. Equal-weight consensus of six fragmenters
  beat each of them (39% first, 80% in ten).
- **Blaženović et al. 2017** [P][34]: four fragmenters alone 10 to 17% first, voting 22%, with database presence and
  library matches (weighted four times) 87 to 93%. Its conclusion: in silico tools alone do not solve true unknowns;
  boosting solves "known unknowns".
- **CASMI 2017** [S][32]: SIRIUS / CSI:FingerID won the natural-product category.
- **CASMI 2022** [P][31]: formulas are nearly solved with MS1 (94%), structures are not (25 to 38% of attempted).
  MIST-CF later matched the winning formula entry in positive mode from MS/MS alone [S][25].

Lesson: priors decide contests whose answers are documented compounds; they cannot be validated for novel ones, and
the audit calls reliance on PubChem ordering a shortcut [P][2].

## 5. What Enveda has published

- **Competition framing** [P][35]: about 400 test molecules, natural or natural-like and their analogues, unpublished
  spectra; three novelty classes (library knowns, known structures with new spectra, novel structures); training data
  = open libraries + Enveda-180 (1.16 M spectra of 184,330 synthetic molecules). Enveda says about 10% of signals in
  an untargeted study match a known molecule. No class shares are published.
- **MS2Mol** (ChemRxiv 2023) [S][33]: sequence-to-sequence transformer, spectrum to SMILES, with a confidence model.
  CSI:FingerID beats it on exact matches (13% vs. 9% on CASMI, 11% vs. 2% on EnvedaLight); it wins on approximate
  matches for structures in no database. Training-data provenance not verified.
- **MS2Prop** (bioRxiv 2022) [S][36]: properties straight from spectra; states that 2 to 20% of metabolite chemistry can
  be annotated today.
- **PRISM** [P][24]: section 2(e). Not released.

Reading [I]: Enveda treats novel structures as an approximate-match problem; our metric is exact connectivity, so the
novel class is close to lost for every team and the contest is decided on classes 1 and 2.

## 6. Availability

| Method | Code | Weights | Trained on; NIST | Licence | Hardware |
|---|---|---|---|---|---|
| SIRIUS / CSI:FingerID, COSMIC, ZODIAC | yes | web service only | GNPS, MassBank, **NIST: yes** [S] | academic use | unusable offline |
| MIST (MassSpecGym release) | yes | Zenodo | MassSpecGym; NIST: no | MIT / CC BY 4.0 | minutes on T4 (our review) |
| MIST-CF | yes | public model | NPLIB1; NIST: no (a NIST model exists on request) | MIT | not stated; positive mode only [P][25] |
| msbuddy | yes | bundled LightGBM | GNPS + **NIST: yes** [S][26] | Apache-2.0 | CPU; candidate generation is rule-based and usable, the ranker is not |
| ICEBERG 2.1, GLACIER | yes | MassSpecGym checkpoints | MassSpecGym; NIST: no (NIST weights only with a licence) | MIT code [P][37]; paper page says CC BY-NC-ND | training: two 24 GB GPUs; ICEBERG 6.2 + 9.2 GPU-h, contrastive 35.6 GPU-h on A5000 [P][4] |
| MARASON | yes | none public | NIST'20, MassSpecGym | MIT | as ICEBERG |
| FIORA | yes | MSnLib model | default NIST'17 + MS-DIAL (**yes**); open model MSnLib (no) [S] | MIT | small GNN |
| FraGNNet | yes | unknown | NIST'20 in the paper | unknown | – |
| JESTR, MVP, FLARE | yes | unknown | MassSpecGym (JESTR also NIST) | unknown | single GPU |
| MSAlign | promised | none | MassSpecGym, Spectraverse, NPLIB1 | – | 20 min on a V100 [P][9] |
| DreaMS | yes | Zenodo, Hugging Face | GeMS (GNPS); NIST: no for pretraining, fine-tune data not verified | MIT | 1024-d encoder |
| DiffMS, FRIGID | yes | yes | NPLIB1, MassSpecGym, unlabeled structures; NIST: no | Apache-2.0 / weights CC BY-NC | FRIGID 6.6 s per spectrum on RTX 6000 Ada, 86 s with scaling [P][5] |
| MS-GPT | yes | yes [P abstract] | not verified | not verified | – |
| MS2Mol, PRISM, LSM-MS2 | no | no | proprietary | – | – |

## 7. Synthesis: eight directions, ranked

What we know constrains the ranking: order changes inside our lists do not move the score, outside candidates help up
to 100 and hurt beyond, and about half of the answers are in no list. So a direction counts only if it changes
**which outside candidates enter** or **when they may pass pool candidates**.

1. **Train the fingerprint model on simulated spectra of database molecules.** Literature: MIST fingerprint Tanimoto
   0.262 to 0.581 and formula-challenge top-1 9.6% to 53.8% with ICEBERG spectra of PubChem molecules, test molecules
   excluded [P][2]; MIST's original recipe [10]. Transfer: it is the only reported lever that improves retrieval of
   molecules *without* spectra, which is our PubChem-only class; our FPNet starts from ten times more structures than
   MIST, so expect a fraction of that gain [I]. Needs: GLACIER simulation of PubChem isomers near our mass windows
   (369 spectra per second on an A5000) and a fine-tune of the FPNet, 15 to 20 GPU-hours by our own estimate. Risk:
   the simulators know [M+H]+ on Orbitrap / QTOF only, the same domain gap that sank MIST on timsTOF.
2. **A formula step that weights, not cuts, the PubChem window.** Literature: true formula doubles MIST's mass-based
   top-1 (26.6 to 51.5) [P][2]; MIST-CF 71% and BUDDY 93% below m/z 400 [S]. Transfer: our MIST study saw 33 to 53% in
   the first 100 with the true formula, and fewer wrong-formula decoys is exactly what v14a lacked. Needs: our own
   ranker (formula labels are free for 2.5 M spectra; a LightGBM over sub-formula explanation features trains on CPU);
   msbuddy's ranker is NIST-trained and forbidden, MIST-CF's public model is positive mode only. Risk: accuracy drops
   above 400 Da where natural products sit; use the top three formulas with weights.
3. **A contrastive selector with PubChem same-formula hard negatives, used for the join selection only.** Literature:
   contrastive fine-tuning lifts ICEBERG / GLACIER top-1 from 0.385 to 0.525 (random) and 0.384 to 0.460 (scaffold)
   but adds under 1.5 points on MassSpecGym [P][3]; MSAlign shows a candidate-based contrastive head is cheap, and
   also that it generalises worse than FLARE on the MCES split [P][9]. Transfer: as a final ranker it repeats work
   we closed; as the model that picks the 100 it attacks recall. Needs: negatives mined from our PubChem tier, a few
   GPU-hours. Risk: gains shrink on novel scaffolds; De Waele's trade-off says tune for retrieval, not fingerprint
   loss [12].
4. **Confidence-gated mixing of pool, PubChem and generated candidates.** Literature: list-level score gaps are the
   best trust signal [P][13]; COSMIC's features are of the same kind [S][27]. Transfer: this is our `PC_JOIN_CONF`;
   the literature supports extending it to decide, per molecule, how many outside rows may rise. Cost: none. Risk:
   calibration cannot be checked locally (pitfall 10).
5. **Fine-tune a forward model on our instrument and adducts.** Literature: instrument and adduct shift is named as
   a main failure mode [P][17]; GLACIER is single-stage and fast [P][3]. Transfer: gives negative mode and [M+NH4]+
   coverage and makes direction 1 realistic. Needs: more than a day on 16 GB (upstream uses two 24 GB cards). Risk:
   cost; isomer re-ordering alone has not paid.
6. **De novo candidates in tail slots, formula from direction 2.** Literature ceiling: 16 to 30% top-1 with the true
   formula on known compounds, single digits for exact matches on novel natural products [P/S]. Transfer: a hit at
   rank 15 to 25 is worth at most 0.07 for that molecule, so even 15% recall on a 30% class adds under 0.003 [I];
   it pays only if the gate of direction 4 lets it rise where retrieval has clearly failed.
7. **Database priors.** Literature: the largest effects on record (143 of 208; 50% spectrum-blind) [P]. Transfer:
   poor so far (0.393, pitfall 7), and the answers here are not reference standards. Only as a tie-breaker inside
   the PubChem tier, checked by one submission.
8. **Self-training on the test spectra within the run.** Literature: no retrieval result; test-time tuning helps a
   weak generator (+27% and +67% relative, to 3.2% and 12.9%) [S][16]; under shift, confident sets do not exist
   [P][29]. With about 400 molecules the pseudo-labels are few and wrong. Not recommended; the related cheap idea is
   ZODIAC-style joint formula inference across test spectra [S][28].

**Ceiling.** Published top-1 for compounds without reference spectra is 34 to 40% against PubChem-scale candidates and
at most 53% with the true formula and simulated-spectra training; novel structures are essentially not recovered
exactly. If half the hidden answers are outside our lists and most of those are PubChem-only, closing a third of that
class at mean reciprocal rank 0.4 is worth roughly +0.03 to +0.06 [I]: 0.47 is plausible through directions 1 to 3;
nothing in the literature supports much more than 0.50 without the novel class.

## References

1. Bushuiev et al., "MassSpecGym: A benchmark for the discovery and identification of molecules", NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2410.23326
2. Liu, Bushuiev, Lightheart et al., "MassSpecGym in the Wild: Uncovering and Correcting Evaluation Pitfalls in AI-Driven Molecule Discovery", arXiv 2026. https://arxiv.org/abs/2606.19624
3. Wang, Wang, Coley, "GLACIER: Rethinking Mass Spectrum Prediction as an Object Detection Problem", arXiv 2026. https://arxiv.org/abs/2606.29161
4. Wang et al., "Neural Spectral Prediction for Structure Elucidation with Tandem Mass Spectrometry" (ICEBERG 2), bioRxiv 2025. https://www.biorxiv.org/content/10.1101/2025.05.28.656653v1
5. Bohde et al., "FRIGID: Scaling Diffusion-Based Molecular Generation from Mass Spectra at Training and Inference Time", ICML 2026. https://arxiv.org/abs/2604.16648
6. Kalia et al., "JESTR: Joint Embedding Space Technique for Ranking candidate molecules for the annotation of untargeted metabolomics data", Bioinformatics 2025. https://academic.oup.com/bioinformatics/article/41/7/btaf354/8165426
7. Chen, Hassoun, "Learning from All Views: A Multiview Contrastive Framework for Metabolite Annotation" (MVP), Analytical Chemistry 2026. https://github.com/HassounLab/MVP
8. Chen, Rushing, Hassoun, "FLARE: Fine-grained Learning for Alignment of spectra-molecule REpresentation Enhances Metabolite Annotation", bioRxiv 2026. https://www.biorxiv.org/content/10.64898/2026.01.27.702086v1
9. Krzakala et al., "MSAlign: Aligning Molecule and Mass Spectra Foundation Models for Metabolite Identification", arXiv 2026. https://arxiv.org/abs/2605.19752
10. Goldman et al., "Annotating metabolite mass spectra with domain-inspired chemical formula transformers" (MIST), Nature Machine Intelligence 2023. https://www.nature.com/articles/s42256-023-00708-3 ; https://github.com/samgoldman97/mist
11. Bushuiev et al., "Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS", Nature Biotechnology 2025. https://www.nature.com/articles/s41587-025-02663-3 ; https://github.com/pluskal-lab/DreaMS
12. De Waele et al., "Small molecule retrieval from tandem mass spectrometry: what are we optimizing for?", arXiv 2026. https://arxiv.org/abs/2602.16507
13. Jürgens et al., "When should we trust the annotation? Selective prediction for molecular structure retrieval from mass spectra", arXiv 2026. https://arxiv.org/abs/2603.10950
14. Che, Du, Xu, "MARLIN: De Novo Molecular Structure Elucidation from Tandem Mass Spectra without a Ground-Truth Formula", arXiv 2026. https://arxiv.org/abs/2607.04774
15. Zhao et al., "MS-GPT: Rethinking MS/MS De Novo Structure Elucidation as Spectrum-Induced Posterior Querying of a Molecule-Language Model", arXiv 2026. https://arxiv.org/abs/2607.23607
16. Mismetti et al., "Test-Time Tuned Language Models Enable End-to-end De Novo Molecular Structure Generation from MS/MS Spectra", arXiv 2025. https://arxiv.org/abs/2510.23746
17. Schneider et al., "De Novo Structure Prediction from Tandem Mass Spectra: Algorithms, Benchmarks, and Limitations", Molecules 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC12985711/
18. Stravs et al., "MSNovelist: de novo structure generation from mass spectra", Nature Methods 2022. https://www.nature.com/articles/s41592-022-01486-3
19. Dührkop et al., "Searching molecular structure databases with tandem mass spectra using CSI:FingerID", PNAS 2015. https://www.pnas.org/doi/10.1073/pnas.1509788112
20. SIRIUS documentation, account and licence. https://v6.docs.sirius-ms.io/account-and-license/
21. Wang et al., "Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning" (MARASON), ICML 2025. https://arxiv.org/abs/2502.17874
22. Young et al., "FraGNNet: A Deep Probabilistic Model for Tandem Mass Spectrum Prediction", TMLR 2025. https://arxiv.org/abs/2404.02360
23. Nowatzky et al., "FIORA: Local neighborhood-based prediction of compound mass spectra from single fragmentation events", Nature Communications 2025. https://www.nature.com/articles/s41467-025-57422-4 ; https://github.com/BAMeScience/fiora
24. Enveda, "PRISM: A foundation model for life's chemistry", company post. https://enveda.com/prism-a-foundation-model-for-lifes-chemistry/
25. Goldman et al., "MIST-CF: Chemical Formula Inference from Tandem Mass Spectra", JCIM 2023. https://arxiv.org/abs/2307.08240 ; https://github.com/samgoldman97/mist-cf
26. Xing et al., "BUDDY: molecular formula discovery via bottom-up MS/MS interrogation", Nature Methods 2023. https://www.nature.com/articles/s41592-023-01850-x ; https://github.com/Philipbear/msbuddy
27. Hoffmann et al., "High-confidence structural annotation of metabolites absent from spectral libraries" (COSMIC), Nature Biotechnology 2022. https://www.nature.com/articles/s41587-021-01045-9
28. Ludwig et al., "Database-independent molecular formula annotation using Gibbs sampling through ZODIAC", Nature Machine Intelligence 2020. https://www.biorxiv.org/content/10.1101/842740v1
29. "Reliable Molecular Retrieval from Mass Spectra using Conformal Prediction", bioRxiv 2026. https://www.biorxiv.org/content/10.64898/2026.03.12.711424v1.full
30. Schymanski et al., "Critical Assessment of Small Molecule Identification 2016: automated methods", Journal of Cheminformatics 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/
31. Fiehn Lab, "CASMI 2022 Results", contest page. https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results
32. Böcker group, "CASMI 2017 results are out", 2017. https://bio.informatik.uni-jena.de/2017/11/casmi-2017-results-are-out/
33. Butler et al., "MS2Mol: A transformer model for illuminating dark chemical space from mass spectra", ChemRxiv 2023. https://chemrxiv.org/engage/chemrxiv/article-details/6492f28ea2c387fa9ab2a465
34. Blaženović et al., "Comprehensive comparison of in silico MS/MS fragmentation tools of the CASMI contest: database boosting is needed to achieve 93% accuracy", Journal of Cheminformatics 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5445034/
35. "Enveda Is Launching the Next AlphaFold Contest", press article, September 2026. https://finance.yahoo.com/healthcare/articles/enveda-launching-next-alphafold-contest-120700676.html ; C&EN, https://cen.acs.org/analytical-chemistry/mass-spectrometry/competition-identify-small-molecules-capture/104/web/2026/09 (not fetchable)
36. Voronov et al., "MS2Prop: A machine learning model that directly predicts chemical properties from mass spectrometry data for novel compounds", bioRxiv 2022. https://www.biorxiv.org/content/10.1101/2022.10.09.511482v3
37. coleygroup/ms-pred, repository. https://github.com/coleygroup/ms-pred

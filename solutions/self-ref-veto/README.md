<!-- index: self-ref-veto (closed): the public 0.492 notebook + a self-reference contradiction veto and leader-safe re-orderings of ranks 2-25; leak-free NP measurements, builder, five submissions (best 0.492 = base) -->
# self-ref-veto

Owner: Ömer (built in a Claude session, 2026-11-16) · Base: the public 0.492 notebook
([mathleteutd/casmi26-0-492-test-set-network-rerank](https://www.kaggle.com/code/mathleteutd/casmi26-0-492-test-set-network-rerank),
Apache-2.0, = analyticaobscura/casmi-2026 with the N-1c network weight 0.35), kept verbatim in `reference/`.

## Idea

Every model in the public pipeline was trained on `train.parquet` and prefers the structures it has seen (the
memorisation bias the team measured many times: "a better stand-alone model is a worse fusion input"). For a hidden
molecule whose answer has no public spectrum (classes 2 and 3) the pipeline therefore tends to put a **known** isomer
first; the documentation prior even adds a bonus for candidates with spectral-library records.

A known structure carries its own evidence: its measured spectra are in `train.parquet`. If none of them resembles any
spectrum of the query, that structure is not the compound that was measured. The veto uses exactly this and nothing
else:

- lists with `library_max >= 0.9` are untouched (class-1 hits);
- in the other lists, a candidate that is a training structure with spectra of the query's adduct (`[M+H]+`,
  `[M-H]-`) whose best single spectrum pair with the query has sqrt-intensity cosine below `V1_TAU` moves below every
  candidate that is not contradicted (order otherwise kept);
- candidates without own spectra (where class-2 answers live) are never moved down; known structures whose spectra
  agree are never moved down;
- optional confirmation (`V2_ON`): a known leader that matches its own spectra only weakly gives way to a known
  candidate in ranks 2-10 that matches its own spectra strongly (>= 0.85 and 0.20 better);
- the external-library overlay still runs afterwards and keeps the last word; a metric-key deduplication follows it
  (`cells/dedup_after_overlay.py`: the overlay inserts reference SMILES strings and can leave a dead duplicate lower
  in a list).

It is the opposite of the train-analog rerank TA-1 (team submission t0, 0.461): TA-1 *promotes* candidates by
family-level evidence and moved many correct leaders; the veto only *demotes*, on identity-level evidence.

## Leak-free measurement (before any submission)

`analysis/np_veto_measures.py`: the 250 `enveda-np-examples` natural products (timsTOF, the closest public set to the
hidden test) against (a) the spectra of their own structure from **other** libraries (Orbitrap / QTOF: the
cross-instrument case of a class-1 molecule with a weak library match) and (b) up to 40 same-formula training isomers
each. Share of pairs vetoed:

| similarity, references per adduct | threshold | true structure vetoed | wrong isomer vetoed, Tanimoto to truth < 0.4 / 0.4-0.6 / 0.6-0.8 |
|---|---|---|---|
| merged spectra, 6 | 0.25 | 3.0% | 66% / 42% / 20% |
| best single pair, 6 (**vpd**) | 0.30 | 2.2% | 65% / 44% / 22% |
| best single pair, 30 | 0.30 | 1.3% | 64% / 43% / 20% |
| best single pair, 30 (**w30**, **w30c**) | 0.40 | 2.2% | 75% / 50% / 23% |

Isomers at Tanimoto >= 0.8 are never vetoed (near-identical spectra). The five true structures vetoed at cap 6 all
have rich spectra (13-278 references in 4-6 libraries): genuine instrument / energy disagreement, not sparse data.

List simulation with these pairs (`analysis/` notes in EXPERIMENTS.md): a wrong known leader with the truth at ranks
2-5 is improved in 73% of lists and worsened in 2%; a correct leader is worsened in 2.6% of lists.

What this does **not** show: how often the hidden test's lists hold contradicted known candidates above the answer.
That is the board's question.

## Results

| Submission | Change on the 0.492 notebook | Public LB |
|---|---|---|
| vpd | veto, best pair < 0.30, 6 refs / adduct + dedup | 0.492 (= base) |
| w30c | veto, best pair < 0.40, 30 refs / adduct + confirmation + dedup | 0.484 (−0.008) |
| nls | leader-safe network: rank 1 as n0 (LAM 0.35), ranks 2-25 at LAM 0.70 | 0.487 (−0.005) |
| tas | the team's TA-1 train-analog evidence on ranks 2-25 only (rank 1 frozen) | 0.487 (−0.005) |
| nlt | nls + tas | 0.483 (−0.009) |

**Outcome (2026-11-18): closed.** No variant beat the 0.492 base. The veto is neutral when weak and loses when strong; re-ordering ranks 2-25 with rank 1 untouched loses too, because n0's order of ranks 2-25 is itself tuned to the public part. The team's validation harness (`solutions/arsokan-fable/VALIDATION-HARNESS.md`) reached the same conclusion: re-ordering n0's lists is exhausted; a gain needs new evidence for molecules without reference spectra.

Details and every measurement: [EXPERIMENTS.md](EXPERIMENTS.md); team log: `docs/experiments.md` rows 143-145.

## Build and run

```bash
python -I solutions/self-ref-veto/tools/build.py vpd          # variants: vpd w30 w30c nls tas nlt -> kaggle/<slug>/
kaggle kernels push -p solutions/self-ref-veto/kaggle/casmi26-om-vpd    # smoke commit: full pipeline on 12 molecules
kaggle competitions submit -c enveda-CASMI26-molecule-id-mass-spectra -k <owner>/casmi26-om-vpd -v 1 -f submission.csv -m "..."
python -I solutions/self-ref-veto/tools/getout.py <owner>/casmi26-om-vpd scratch/out_vpd submission.csv,v1_report.json
```

`tools/mkharness.py` + `tools/runcell.py` run any cell locally against a stand-in `/kaggle/working` built from the
visible test (plumbing only: the visible test is drawn from train, so its scores are leaked).

## Data provenance

No new external data and no new model: the cell reads `train.parquet` (competition data) and the files the public
pipeline writes. All Kaggle inputs are those of the public 0.492 notebook (`reference/kernel-metadata.json`); see the
licence notes in `docs/competition/` for the fork line's inputs (MassBank-harmonized is CC BY-NC-SA, takumuhata extlib
CC0). No NIST data or NIST-derived model.

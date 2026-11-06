<!-- index: markdown header of the public AE notebook (placeholders __SCORE__, __DATASET__, __RUNS__, __TAU__ filled by build_public_ae.py) -->
# CASMI26 | Neural ensemble + spectral analog edits

**Public LB __SCORE__.** This notebook is bgunshui's
[CASMI26 0.480 neural ensemble](https://www.kaggle.com/code/bgunshui/casmi26-480-neural-ensemble) with one cell
added after its final layer, headed *Overlay (team Arsokan): spectral analog edits at rank 2*. Every other cell runs
as bgunshui published it. Below this header, bgunshui's notebook follows as published, except that garbled characters
in its markdown were repaired.

**The idea** comes from heliosli's r3p overlay in [casmi26-0.481](https://www.kaggle.com/code/heliosli/casmi26-0-481).
Its analog library and site model are private datasets, so that notebook cannot be run by others. This is an
independent implementation in which every part is public: the analog library is built in the notebook from
`train.parquet`, and the gate model and all the code are in [__DATASET__](https://www.kaggle.com/datasets/__DATASET__)
(CC0).

## What the cell does

Many hidden answers are in no candidate source. Some of them are close relatives of a training structure, for
example the same scaffold with one more methyl, hydroxyl or sugar. Their spectra then match the relative's spectrum
after shifting by the modification mass. That is the idea behind molecular networking.

1. **Analog search.** Each query spectrum ([M+H]+, [M-H]-, [M+Na]+) is compared with every training spectrum whose
   precursor differs by one of 19 natural-product modifications, in either direction: CH2, O, CH2O, acetyl, CO2,
   prenyl, sulfate, malonyl, benzoyl, pentose, deoxyhexose, coumaroyl, galloyl, hexose, caffeoyl, glucuronide,
   feruloyl, sinapoyl and dihexose. The comparison is the modified cosine at 0.01 Da.
2. **Edits.** For the 5 best analogs, every single-site edit is generated: the group added at a matching site, or a
   piece of that mass cut off. Every product is checked by exact mass. Edits that are training structures are
   dropped, because they would already be library matches.
3. **Site evidence.** Each matched peak is explained by the analog's fragments after cutting 1 or 2 bonds. Peaks
   that moved by the modification mass should come from fragments that hold the site, and peaks that did not move
   should come from fragments without it. The share of consistent peak intensity ranks the edits.
4. **Gate.** A LightGBM model trained by us predicts the probability that the top edit is the true structure. It was
   trained on 7,858 held-out enveda-180 structures of `train.parquet`: each structure's own spectra were the queries
   and every other structure was the library.

**The rule.** The cell only touches lists whose library match is below 0.95. When the gate's probability is at least
`AE_TAU` (__TAU__ here), the top edit goes to rank 2. Rank 1 and the order of the rest stay as they were, and the list
stays at 25. If the edit is already lower in the list, it moves up to rank 2. Any error restores the answer file of
the final layer.

**Offline** (out of fold, best candidate per molecule):

| gate p >= | molecules fired | precision (top edit = truth) |
|---:|---:|---:|
| 0.15 | 9.6% | 0.406 |
| 0.25 | 7.1% | 0.468 |
| 0.50 | 3.7% | 0.569 |

A correct edit at rank 2 is worth 0.5 for that molecule. A wrong one costs only when the true answer sat at rank 2
or lower: it moves down one place, which costs 1/6 at rank 2 and less further down.

## Every submission of this overlay

| run | gate threshold | public LB |
|---|---:|---:|
__RUNS__

For comparison, bgunshui's notebook alone scores 0.480. The public split is about 125 molecules, so one answer gained
at rank 2 is worth about 0.004.

The commit run ranks only the 12 visible molecules. These are copies of training spectra and library matches, so the
cell leaves them alone. A dry pass with the library gate off also runs on them. `ae_overlay_report.json` lists every
molecule the gate fired on, with the modification, the analog and the action taken. Building the library takes about
2 minutes on Kaggle, and the edits a few seconds.

## Run it yourself

Copy & Edit, keep GPU T4 and internet off, and Save Version. The commit run takes about 25 minutes. Then submit that
version; the scoring rerun takes 4-5 hours. The switches are at the top of the overlay cell: `AE_ON`, `AE_TAU`,
`AE_LIB_TAU` and `AE_BUDGET_S`.

## Credits

- **heliosli**: the analog-edit idea (r3p in [casmi26-0.481](https://www.kaggle.com/code/heliosli/casmi26-0-481)),
  which we reimplemented.
- **bgunshui**: the [0.480 neural ensemble](https://www.kaggle.com/code/bgunshui/casmi26-480-neural-ensemble) this
  notebook extends, and its [training kernel](https://www.kaggle.com/code/bgunshui/casmi26-scorer-train-public).
- **baidalinadilzhan**: [MIDX](https://www.kaggle.com/code/baidalinadilzhan/casmi26-midx-exact-dedup-fill-0-479),
  with its documentation prior on ranks 2-25, metric-exact InChIKey-14 dedup and outsider fill.
- **busyaprime**: the 0.477 notebook that MIDX extends. Its sections are kept as written below.
- **Saner Turhaner (sanpier)**: the spectrum scorer kit. **Roman Tamrazov**: Spectral Parallax and the fragment
  check. **Dmitrii Gluzdov**, **Roman Rozen** and **alex chilton**, and everyone credited in the base notebooks'
  own credits sections below.

The overlay cell, the gate model and the code are ours (team Arsokan), CC0.

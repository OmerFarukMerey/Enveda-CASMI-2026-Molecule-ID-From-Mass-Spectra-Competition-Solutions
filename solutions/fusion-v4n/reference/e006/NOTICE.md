<!-- index: origin, licence and attribution of the public notebook snapshot that the E006_FUSE option runs as a second pipeline -->
# reference/e006: snapshot of a public notebook (Apache License 2.0)

| | |
|---|---|
| Notebook | [CASMI26 E006 Window 40 ICE Fallback True](https://www.kaggle.com/code/canqiang/casmi26-e006-window-40-ice-fallback-true), `canqiang/casmi26-e006-window-40-ice-fallback-true`, version 1 (run 2026-10-10 06:52 UTC, public score 0.440) |
| Licence | "This Notebook has been released under the Apache 2.0 open source license" (notebook page, read 2026-10-10) |
| Files | `casmi26-e006-window-40-ice-fallback-true.ipynb` and `kernel-metadata.json` as returned by `kaggle kernels pull -m` on 2026-10-10, unchanged. sha256 of the .ipynb: `ee55b55c7f5bdb08f1a8154396b18fdb0203fb7178b3a52429e0999e69ad7e53` |
| Its own credits | cells 2 to 4 of the notebook (kept): obstacledeveloper (V1.1, direct source), flexonafft (v80), bobthebot369 (v18), ahmedberatozer (engine, ranker, pool, PubChem, FPNet, simulator packages), prvsiyan (analog propagation, fingerprint models), dmitriigluzdov (popularity prior), samanyu1808 (ICEBERG where GLACIER has no score), the official metric notebook (Apache-2.0), ms-pred (MIT) |

How it is used. `tools/build_notebook.py` embeds the code cells of the .ipynb, unchanged, into a notebook built with `E006_FUSE`;
`nb/embed/e006_stage.py` turns them into one script at run time with three marked text patches (work folder, engine-2 lists shared
with our own run, full run on request) and a no-op `display`. The stand-alone rerun `casmi26-e006r` (`scratch/e006/e006r/`) is the
.ipynb itself plus one notice cell.

Changes against the original run, in both uses: the attached input `prvsiyan/chebi-lipidmaps-casmi26` (CC BY-NC-SA 4.0) is replaced
by `sergiomalv/casmi-chebi-lmsd-structures` (CC BY 4.0; same file names). Inside our combined notebook the inputs
`hengck23/hengck23-dreams-enveda-casmi26` and `eveliaveldrine/rdkit-wheels` are not attached: the notebook reads neither on the pinned
Python 3.12 image (the first lacks the file its tail-fill branch looks for, the second is a fallback wheel for Python 3.13).

Datasets are not redistributed here; each keeps its own licence (`reviews/e006-2026-10-10.md`, section 1).

"e006 improved" (`E006_BASE`, `nb/e006imp_stage.py`, `reviews/e006imp-2026-10-10.md`): the same cells run as the main pipeline with four more
marked patches of `e006_stage.script_improved` (our base lists in place of its engine call; optional: another threshold of the `rel` branch of
its promotion, GLACIER on every covered spectrum, the popularity-branch promotion kept in the final list). Every changed line is tagged
`# ours: patch N`; the snapshot files here stay unchanged.

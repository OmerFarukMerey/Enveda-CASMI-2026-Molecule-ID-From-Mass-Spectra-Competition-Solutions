<!-- index: fusion-v4n detailed experiment log -->
# fusion-v4n experiment log

One row per change. Local = `tools/refuse.py` on the visible placeholder test (plumbing check, optimistic), LB = public leaderboard.
LB rows are also appended to `../../docs/experiments.md`.

| # | Date | Version | Change | Local | LB | Note |
|---|---|---|---|---|---|---|
| 0 | 2026-09-30 | ours-v1-baseline | Reproduction of the reference v4n (default CFG) | – | – (ref 0.399) | Kaggle commit v1 (notebook095c91ef58): COMPLETE in 1089 s on T4, 12-molecule smoke run. All stages ran: engine-2 12/12, PubChem 12/12, base lists 0 errors, derivation self-test 0 mismatches, ICE and GL 11/11 covered molecules scored (755/773 candidates). submission.csv valid (400 rows, 23–25 parseable guesses per smoke molecule). Not submitted. |
| 0b | 2026-09-30 | ours-v1-baseline | + cell ids, drop ice_site at the end (output hygiene only) | 0.958 (12 visible molecules; all exact library hits, plumbing check only) | **0.399** (submitted as Kaggle v3 = v1 code from the editor) | reproduces the reference exactly |

Queue of next experiments: [../../docs/plan/roadmap.md](../../docs/plan/roadmap.md) (P1–P4).
| 1 | 2026-09-30 | v2-pop015-fraggap05 | POP_MU=0.15, FRAG_LAM=0.5, FRAG_MODE=gap (Kaggle v4). Commit run: all stages OK in 1023 s, popularity applied 12/12, frag re-rank used 2×. Chosen before the held-out validation finished (user asked for a pre-reset submission); frag gains measured only on a same-formula isomer panel (+0.056 ± 0.015, optimistic) | 0.958 (12 visible) | pending | submitted 23:16 |

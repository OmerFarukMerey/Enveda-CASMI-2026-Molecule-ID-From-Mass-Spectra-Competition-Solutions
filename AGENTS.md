<!-- index: AI-agent entry point: reading order, layout, hard rules (no data in git, no NIST, no visible-test tuning), conventions -->
# AGENTS.md: guide for AI coding agents

Read this first. It is the entry point for Claude Code, Codex, Cursor, Copilot and similar tools. Humans: see [README.md](README.md).

## What this repo is

Team solutions for the Kaggle code competition **Enveda CASMI 2026** (predict molecular structures from MS/MS
spectra; submit ≤25 ranked SMILES per `molecule_id`; metric MRR@25 on tautomer-canonical InChIKey14).
Submissions run **offline in a Kaggle notebook, ≤9 h, T4 GPU**. Competition facts: [docs/competition/](docs/competition/).

## How to navigate (read in this order, stop when you have enough)

1. [INDEX.md](INDEX.md): generated map of every tracked file with a one-line purpose.
2. [docs/competition/rules-and-pitfalls.md](docs/competition/rules-and-pitfalls.md): constraints you must not break.
3. [solutions/README.md](solutions/README.md), then the solution you are working on: its `AGENTS.md`, `README.md`,
   `solution.json` (machine-readable: entry points, commands, Kaggle inputs, status).
4. [docs/plan/roadmap.md](docs/plan/roadmap.md) and [docs/experiments.md](docs/experiments.md) before proposing experiments.
5. [docs/landscape/](docs/landscape/) for what other competitors do.

## Layout

```
AGENTS.md  CLAUDE.md  README.md  INDEX.md  llms.txt      entry points (INDEX.md is generated)
docs/competition/   task, data, rules, pitfalls            (facts; cite the Kaggle page when you change them)
docs/landscape/     leaderboard + public notebooks + forum (dated snapshots)
docs/plan/          roadmap and open questions
docs/experiments.md team-wide submission log (append-only)
solutions/<name>/   one self-contained approach each
submissions/        one folder per notable submission: README.txt, exact notebook, config (see submissions/README.md)
data/               NOT in git. data/input/ mirrors /kaggle/input (competitions/<slug>/, datasets/<owner>/<slug>/)
tools/              repo-level scripts (gen_index.py)
```

## Hard rules

- Never commit data, weights, run outputs or anything under `data/`, `runs/`, `scratch/` (see `.gitignore`).
- Never use NIST or other non-free data or weights derived from it: it disqualifies the team. Only freely, publicly
  available data and models. Record the provenance of any new external artifact in the solution's README.
- Do not tune on the visible `test.parquet`. It is a placeholder drawn from train and is replaced at scoring time.
  Anything keyed to its masks, masses or formulas silently does nothing on the hidden test.
- RDKit must be `2026.03.3` wherever keys are compared (the metric pins it).
- Keep a solution's default config reproducible. Put new ideas behind a config flag and log results in
  `docs/experiments.md` (LB) and the solution's `EXPERIMENTS.md` (details).
- Code copied from public notebooks and datasets stays under its licence. Add it to `THIRD_PARTY_NOTICES.md`.
- After adding, removing or renaming files, run `python tools/gen_index.py` and commit the updated `INDEX.md`.

## Conventions

- Python ≥3.12, plain scripts plus small modules. Each file starts with a docstring whose first line says what it is
  (INDEX.md is built from it).
- A solution's tools resolve paths through its own `tools/paths.py` (repo root `data/`, overridable with `$CASMI_DATA`).
- Tests live next to the solution (`solutions/<name>/tests/`) and must run without competition data.
- Commit messages: imperative, say what changed and why. One logical change per commit.

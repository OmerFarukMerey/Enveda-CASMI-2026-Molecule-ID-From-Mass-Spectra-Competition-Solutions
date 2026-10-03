<!-- index: list of solutions and the per-solution layout contract (README, AGENTS, solution.json, …) -->
# solutions/

One self-contained approach per folder. Each approach can be copied to Kaggle on its own.

| Folder | Owner | Summary | Best public LB |
|---|---|---|---|
| [fusion-v4n/](fusion-v4n/) | Hemre | Public 0.399 notebook restructured: config-driven, stage dumps, tested fusion core; PubChem candidates ranked by the main engine | **0.420** |
| *(branch `research/competition-analysis`)* | İlker | From-scratch pipeline + leak-aware validation. Suggested home when merged: `solutions/research-pipeline/` | 0.260 |

## Layout contract (every solution)

```
solutions/<name>/
  README.md        humans: idea, how to run, results, data provenance
  AGENTS.md        agents: entry points, invariants, what not to touch
  solution.json    machine-readable manifest (see below)
  EXPERIMENTS.md   detailed per-solution log (LB rows also go to docs/experiments.md)
  tools/           scripts; tools/paths.py resolves <repo>/data
  tests/           run without competition data
  kaggle/          kernel-metadata.json (+ generated notebook, git-ignored)
```

`solution.json` keys: `name, owner, status, summary, based_on, entrypoints{build, push, test, ...}, kaggle{kernel_slug,
competition, datasets[]}, config, best_lb, provenance[]`.

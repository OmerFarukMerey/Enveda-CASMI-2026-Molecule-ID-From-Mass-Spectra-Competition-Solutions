<!-- index: arsokan-fable agents guide: what each folder is, invariants, what not to touch -->
# AGENTS.md: arsokan-fable

- Not a buildable notebook: `stages/*.py` are cells pasted into a copy of the public n0 notebook (mathleteutd 0.492).
  TA-1 goes right after the network cell (n1c), before the external-library overlay.
- `stages/pc_wide_rerank_NEGATIVE.py` is a negative result (w0–w3). Do not add it to a submission.
- `offline/` and `lab/` scripts are kept exactly as they ran. Absolute `/root/...` paths belong to the author's
  machine; adjust locally, do not "fix" them in commits without a reason.
- Results: `EXPERIMENTS.md` here; LB rows also in `docs/experiments.md`.
- Never commit Kaggle credentials; `offline/dl_out.py` relies on the local Kaggle API config.

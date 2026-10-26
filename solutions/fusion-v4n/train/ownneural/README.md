<!-- index: ownneural (2026-10-26): our own spectrum <-> structure model for the top-1-locked ranks-2-25 slot: split scaffold-disjoint from the s1val / s1b evaluation dumps and the clean chain's H / E, hard-negative tables, listwise trainer, evaluation on s1's real final lists against XE / FPZ / J1 -->
# ownneural: our spectrum <-> structure model for ranks 2-25

Bi-encoder (clean/onnet.py): a multi-spectrum peak-set transformer pooled over ALL a molecule's spectra, a fingerprint MLP for the
candidate, cosine score; trained listwise against the kinds of candidates s1 puts at ranks 2-25 (same-formula pool isomers, PubChem window
rows, near-mass analogues). train.parquet only; no NIST, no external model or weights. Pre-registration, state and verdict:
`scratch/ownneural/PREREG.md`, `STATE.md`; review `reviews/ownneural-2026-10-26.md`. Inference module: `clean/onnet_infer.py`.

| File | What |
|---|---|
| `onpaths.py` | paths ($CASMI_DATA, the two evaluation dumps, the clean chain's scaffold file, GPU_LOCK) |
| `build_split.py` | role of every training structure: eval / eval_scaf (identity + Murcko scaffold of the dump answers), H / H_scaf (clean chain's frozen H + E), val (Enveda scaffold groups), train |
| `build_negs.py` | per positive: 16 most similar same-formula pool isomers, 8 most similar near-mass pool rows of another formula; banned pool rows |
| `ondata.py` | batches (Enveda-weighted sampling, Enveda-only spectrum views, 31 typed negatives), fixed Enveda validation lists, list MRR |
| `train_on.py` | trainer (GPU-busy budget, chunks, yield, best checkpoint by validation MRR, ledger) |
| `eval_lists.py` | scores s1's final lists of both dumps with ON and joint1, primary metric (MRR among ranks 2..n on lists with the answer at rank 2-25), baselines XE / FPZ / J1 / GL / IC / MF, unseen subsets, top-1-locked RRF, verdicts |

Provenance: candidate structures and their fingerprints come from the pool tables (`directions/run/work`, the casmi26-v2-pool layout;
our `clean/pool_build.record` reproduces their bits) and the PubChem tier negatives of `../mine_pc_negatives.py`; the bit selection is
`fp_bits.npy` of that pool. Tests: `../../tests/test_ownneural.py` (no competition data).

```bash
export CASMI_DATA='D:/casmi-data'; P=.venv/Scripts/python; O=solutions/fusion-v4n/train/ownneural
$P $O/build_split.py --h C:/Users/hemre/casmi-work/clean/H.parquet --e C:/Users/hemre/casmi-work/clean/E.parquet
$P $O/build_negs.py --workers 4
$P scratch/ownneural/run_gpu.py "--tag on_s0 --seed 0 --comp_minutes 60 --chunk_minutes 60"      # GPU_LOCK + guard_run
$P $O/eval_lists.py --on D:/casmi-data/ownneural/runs/on_s0/best.pt,D:/casmi-data/ownneural/runs/on_s1/best.pt --out D:/casmi-data/ownneural/eval_final
```

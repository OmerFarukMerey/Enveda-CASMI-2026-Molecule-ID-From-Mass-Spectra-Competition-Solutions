"""Inference of the ownneural model (clean/onnet.py): test.parquet-style spectrum rows + candidate SMILES -> per-candidate compatibility scores.

Licence: ours. Imports only clean/onnet.py, clean/chemcore.py, clean/specdata.py, clean/pool_build.py (ours), NumPy, pandas, PyTorch, RDKit.

Spectra (exactness with training): each raw spectrum is cleaned with chemcore.clean_peaks (the training cache's rule), the adduct mapped by
specdata.ADDUCT_IX (59 = other), the instrument by specdata.instrument_class, polarity from the ionisation mode or a trailing '-' of the
adduct, the collision energy = mean of the finite values of collision_energy_ev (unknown -> flag 0). All spectra of a molecule are pooled
(both polarities, every adduct), as in training.
Candidates: pool_build.standardise + record (stereo-free largest neutral fragment; the pool tables' fingerprint), the raw 18,599 bits
selected by the checkpoint's bit list (fp_bits.npy of the training pool). A SMILES that fails gets NaN.
Score: cosine of the molecule and candidate embeddings, averaged over the given checkpoints (seeds).

usage (Python):
    S = OnScorer(['on_s0/best.pt', 'on_s1/best.pt'], fp_bits='.../fp_bits.npy', device='cuda')
    scores = S.score_lists({mid: spectra_rows_df}, {mid: [smiles, ...]})      # -> {mid: np.array of scores}
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import chemcore  # noqa: E402
import onnet  # noqa: E402
import specdata  # noqa: E402


def _ce_mean(ce):
    if ce is None:
        return np.nan
    a = np.atleast_1d(np.asarray(ce, dtype=np.float64)); a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def _mode(mode, adduct):
    neg = str(mode).lower().startswith('neg') if isinstance(mode, str) else (mode is not None and float(mode) < 0)
    return 0 if (neg or str(adduct).endswith('-')) else 1


def spectrum_fields(mz, it, prec, adduct, mode, ce, instrument):
    """One raw spectrum -> (cleaned m/z, intensity, prec, adduct index, mode 1/0, ce mean or NaN, instrument class)."""
    m, i = chemcore.clean_peaks(mz, it, float(prec))
    return m, i, float(prec), specdata.ADDUCT_IX.get(adduct, specdata.ADDUCT_OTHER), _mode(mode, adduct), _ce_mean(ce), specdata.instrument_class(instrument)


def arrays_from_rows(groups, max_peaks=96):
    """groups: list (one per molecule) of DataFrames with the test.parquet columns -> onnet spectrum arrays (owner = molecule index).
    Spectra left without a peak after cleaning are dropped; a molecule with none keeps a single empty spectrum (embedding of its metadata)."""
    peaks, prec, add, mode, ce, instr, own = [], [], [], [], [], [], []
    for k, df in enumerate(groups):
        got = 0
        for r in df.itertuples():
            f = spectrum_fields(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz, r.adduct, r.ionization_mode, r.collision_energy_ev, r.instrument_type)
            if len(f[0]) == 0 and (got or len(df) > 1):
                continue
            peaks.append((f[0], f[1])); prec.append(f[2]); add.append(f[3]); mode.append(f[4]); ce.append(f[5]); instr.append(f[6]); own.append(k); got += 1
        if not got:
            peaks.append((np.zeros(1, np.float32), np.zeros(1, np.float32))); prec.append(0.0); add.append(specdata.ADDUCT_OTHER); mode.append(1); ce.append(np.nan); instr.append(5); own.append(k)
    return onnet.spectra_arrays(peaks, prec, add, mode, ce, instr, own, max_peaks)


class CandBits:
    """SMILES -> packed selected fingerprint bits (cached). bits = indices into the raw 18,599-bit fingerprint."""

    def __init__(self, fp_bits):
        import pool_build
        self.pb = pool_build; self.bits = np.asarray(fp_bits, np.int64); self.nbytes = (len(self.bits) + 7) // 8; self.cache = {}

    def __call__(self, smiles):
        out = np.zeros((len(smiles), self.nbytes), np.uint8); ok = np.zeros(len(smiles), bool)
        for j, s in enumerate(smiles):
            if s not in self.cache:
                v = None
                try:
                    st = self.pb.standardise(s); rc = self.pb.record(st[0]) if st else None
                    if rc is not None:
                        raw = np.unpackbits(rc[1])[:self.pb.FP_RAW]; v = np.packbits(raw[self.bits])
                except Exception:
                    v = None
                self.cache[s] = v
            v = self.cache[s]
            if v is not None:
                out[j] = v; ok[j] = True
        return out, ok


class OnScorer:
    def __init__(self, ckpts, fp_bits, device='cpu', max_peaks=96):
        import torch
        self.torch = torch; self.dev = device; self.max_peaks = max_peaks
        self.models = [onnet.load_ckpt(c, device) for c in ckpts]
        bits = np.load(fp_bits) if isinstance(fp_bits, str) else np.asarray(fp_bits)
        assert all(m.nbits == len(bits) for m in self.models), 'checkpoint and bit list disagree'
        self.cb = CandBits(bits)

    def score_lists(self, spectra, cands, chunk=64):
        """spectra: {mid: DataFrame of its rows}; cands: {mid: [smiles]} -> {mid: float32 array (NaN = unparseable SMILES)}."""
        torch = self.torch; mids = [m for m in cands if m in spectra]; out = {}
        for s in range(0, len(mids), chunk):
            ms = mids[s:s + chunk]; arr = arrays_from_rows([spectra[m] for m in ms], self.max_peaks)
            smi = [x for m in ms for x in cands[m]]; own = np.concatenate([np.full(len(cands[m]), i) for i, m in enumerate(ms)]) if smi else np.zeros(0, int)
            fp, ok = self.cb(smi); sc = np.zeros(len(smi), np.float64)
            with torch.no_grad():
                for mdl in self.models:
                    zs, _ = onnet.embed_molecules(mdl, arr, len(ms), self.dev); ze = onnet.embed_structs(mdl, fp, self.dev)
                    sc += (zs[torch.as_tensor(own, device=self.dev)] * ze).sum(1).double().cpu().numpy()
            sc /= len(self.models); sc[~ok] = np.nan; p = 0
            for i, m in enumerate(ms):
                n = len(cands[m]); out[m] = sc[p:p + n].astype(np.float32); p += n
        return out

"""Adduct surrogate for the forward models: rewrite [M+NH4]+ and water-loss spectra so they score as [M+H]+.

ICEBERG's packaged checkpoint only has trained weights for [M+H]+ and [M+Na]+ (reviews/coverage-2026-10-02.md), so
molecules measured only as other adducts get no isomer evidence. An [M+NH4]+ spectrum is mostly the [M+H]+ spectrum
below the [M+H]+ m/z (NH3 leaves first), and a water-loss spectrum is the [M+H]+ spectrum below its own precursor.
On val1 the surrogate reached the native [M+H]+ score on the same molecules (in-group MRR 0.34 against 0.33, n=18).
[M+K]+ and negative-mode spectra are left alone: both were at chance.
"""
import numpy as np

NH3 = 17.026549
WATER_LOSS = ('[M-H2O+H]+', '[M-2H2O+H]+')
ADDUCTS = ('[M+NH4]+',) + WATER_LOSS


def rewrite(df, margin=0.5):
    """Rows of a test.parquet-like frame that can be scored as [M+H]+, rewritten; other rows are dropped.
    [M+NH4]+: precursor_mz -= NH3 and peaks above the new precursor + margin removed. Water loss: only relabelled."""
    pos = df['ionization_mode'].astype(str).str.lower().eq('positive') if 'ionization_mode' in df else True
    out = df[df['adduct'].isin(ADDUCTS) & pos].copy()
    if not len(out):
        return out
    mzs, its, prec = [], [], []
    for r in out.itertuples(index=False):
        mz = np.asarray(r.ms2_mzs, np.float64); it = np.asarray(r.ms2_normalized_intensities, np.float64)
        p = float(r.precursor_mz)
        if r.adduct == '[M+NH4]+':
            p -= NH3
            keep = mz <= p + margin
            mz, it = mz[keep], it[keep]
        mzs.append(mz); its.append(it); prec.append(p)
    def col(v):                                    # object column of arrays (equal lengths must not become 2-D)
        s = np.empty(len(v), dtype=object); s[:] = v
        return s
    out['ms2_mzs'], out['ms2_normalized_intensities'], out['precursor_mz'] = col(mzs), col(its), prec
    out['adduct'] = '[M+H]+'
    return out[np.array([len(m) > 0 for m in mzs])]


def extend_items(items, extra_items, mode='gap'):
    """Append surrogate spectra to the runner items. mode 'gap': only molecules without a native covered spectrum;
    'all': every molecule. Returns (new items, number of molecules that received surrogate spectra)."""
    extra = {it['mid']: it['spectra'] for it in extra_items if it['spectra']}
    out, n = [], 0
    for it in items:
        add = extra.get(it['mid'], [])
        if add and (mode == 'all' or not it['spectra']):
            it = dict(it, spectra=list(it['spectra']) + list(add)); n += 1
        out.append(it)
    return out, n

# !! WARNING (2026-11-16): submission t0 (n0 + this stage) scored 0.461 on the hidden test against 0.492 for n0 (-0.031).
# !! Harmful as written; do not add to a submission. Most likely reason (hypothesis): training structures at >= 0.85 are
# !! left out of the evidence, so a right candidate that IS a training molecule (class 1) gets none from its own spectra
# !! while wrong candidates 0.40-0.85 similar to it collect evidence from them; the gate (rank 1 a training molecule with
# !! self-match >= 0.6, found by Morgan >= 0.999) can fail to fire on real test spectra. See stages/README.md.
# ── Train-analog evidence rerank (Arsokan TA-1) ─────────────────────────────────────────────────────────────
# For every candidate, look up its closest *training* structures (Morgan Tanimoto 0.40-0.85, top 8) and ask whether
# their measured spectra resemble the query spectrum (modified cosine). A right candidate's analogs fragment alike.
# Near-identical training molecules (>= 0.85) are ignored, so a candidate is not rewarded for being a training molecule.
# Lists are re-ordered by z(-log rank) + TA_LAM * z(evidence). Molecules whose rank 1 is a training molecule whose own
# spectrum matches the query (>= TA_GATE) are left untouched. Measured on 400 held-out class-2 molecules with the full
# pipeline's lists: MRR 0.617 -> 0.660 (gate on), top-1 45% -> 52%. Any failure leaves submission.csv unchanged.
TA_LAM, TA_GATE, TA_LO, TA_HI, TA_K = 0.75, 0.6, 0.40, 0.85, 8
def _ta_run(work='/kaggle/working', inp='/kaggle/input', sub_path=None, test_path=None, train_path=None):
    import glob, os, shutil, time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as pads
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    t0 = time.time(); sp_ = sub_path or os.path.join(work, 'submission.csv')
    if not os.path.exists(sp_): print('[ta] no submission.csv'); return
    pick = lambda name: sorted(glob.glob(inp + '/**/' + name, recursive=True), key=lambda p: ('mass-spectra' not in p.lower(), len(p)))[0]
    test_path = test_path or pick('test.parquet'); train_path = train_path or os.path.join(os.path.dirname(test_path), 'train.parquet')
    te = pd.read_parquet(test_path); sub = pd.read_csv(sp_, dtype=str, keep_default_na=False)
    L = {m: [x for x in s.split(';') if x] for m, s in zip(sub.molecule_id.astype(str), sub.smiles)}
    ADD = ('[M+H]+', '[M-H]-')
    def build_sp(df, idcol):
        SP = {}
        for (m, ad), g in df.groupby([idcol, 'adduct']):
            if str(ad) not in ADD: continue
            pr = float(np.median(pd.to_numeric(g.precursor_mz, errors='coerce')))
            mz = np.concatenate([np.asarray(x, float) for x in g.ms2_mzs])
            it = np.concatenate([np.asarray(x, float) / max(float(np.max(x)) if len(x) else 1.0, 1e-9) for x in g.ms2_normalized_intensities])
            k = mz < pr - 1.5; mz, it = mz[k], it[k]
            if not len(mz): continue
            u, inv = np.unique(np.round(mz, 2), return_inverse=True); w = np.zeros(len(u)); np.maximum.at(w, inv, it)
            o = np.argsort(-w)[:120]; u, w = u[o], np.sqrt(w[o]); oo = np.argsort(u)
            SP.setdefault(str(m), {})[str(ad)] = (u[oo], (w / max(np.linalg.norm(w), 1e-12))[oo], pr)
        return SP
    def mcos(a, b):
        (ma, ia_, pa_), (mb, ib, pb) = a, b; sh = pa_ - pb; cand = []
        for off in ((0.0, sh) if abs(sh) > 0.02 else (0.0,)):
            lo = np.searchsorted(mb, ma - off - 0.0101); hi = np.searchsorted(mb, ma - off + 0.0101, 'right')
            for i in np.flatnonzero(hi > lo):
                for j in range(lo[i], hi[i]): cand.append((ia_[i] * ib[j], i, j))
        cand.sort(reverse=True); ui, uj, s, n = set(), set(), 0.0, 0
        for v, i, j in cand:
            if i in ui or j in uj: continue
            ui.add(i); uj.add(j); s += v; n += 1
        return s, n
    SPQ = build_sp(te.assign(_m=te.molecule_id.astype(str)), '_m')
    ds = pads.dataset(train_path)
    st = ds.to_table(columns=['inchikey14', 'normalized_smiles']).to_pandas().drop_duplicates('inchikey14').reset_index(drop=True)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); fps, tkeys = [], []
    for k, s in zip(st.inchikey14.astype(str), st.normalized_smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None: fps.append(gen.GetFingerprint(m)); tkeys.append(k)
    print(f'[ta] train structures {len(fps)}, queries with spectrum {len(SPQ)}/{len(L)}, {time.time() - t0:.0f}s', flush=True)
    NB = {}
    for m, lst in L.items():
        if m not in SPQ: continue
        for c in lst[:25]:
            if c in NB: continue
            mol = Chem.MolFromSmiles(c)
            if mol is None: NB[c] = []; continue
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(mol), fps)); o = np.argsort(-sims)[:TA_K]
            NB[c] = [(tkeys[i], float(sims[i])) for i in o if sims[i] >= TA_LO]
    need = sorted({k for v in NB.values() for k, _ in v})
    print(f'[ta] candidates {len(NB)}, analog molecules {len(need)}, {time.time() - t0:.0f}s', flush=True)
    rows = []
    flt = pads.field('inchikey14').isin(pa.array(need)) & pads.field('adduct').isin(pa.array(list(ADD)))
    for b in ds.scanner(columns=['inchikey14', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'], filter=flt, batch_size=4096,
                        batch_readahead=0, fragment_readahead=0, use_threads=False).to_batches():
        if b.num_rows: rows.append(b.to_pandas().groupby(['inchikey14', 'adduct']).head(6))
    SPT = build_sp(pd.concat(rows).groupby(['inchikey14', 'adduct']).head(6), 'inchikey14') if rows else {}
    del rows
    print(f'[ta] analog spectra for {len(SPT)} molecules, {time.time() - t0:.0f}s', flush=True)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9); nch = nt1 = ngate = nev = 0; out = []
    for m, s in zip(sub.molecule_id.astype(str), sub.smiles):
        lst = L[m][:25]; q = SPQ.get(m)
        if q is None or len(lst) < 2: out.append(s); continue
        ev = np.zeros(len(lst)); selfc = -1.0
        for i, c in enumerate(lst):
            for t, sim in NB.get(c, []):
                tt = SPT.get(t)
                if tt is None: continue
                for ad in set(q) & set(tt):
                    sc, n = mcos(q[ad], tt[ad])
                    if sim >= 0.999:
                        if i == 0: selfc = max(selfc, sc if n >= 4 else 0.0)
                        continue
                    if sim >= TA_HI or n < 4: continue
                    ev[i] += sim * sc
        if selfc >= TA_GATE: ngate += 1; out.append(s); continue
        if ev.max() <= 0: out.append(s); continue
        nev += 1
        o = np.argsort(-(z(-np.log(np.arange(1, len(lst) + 1))) + TA_LAM * z(ev)), kind='stable'); nl = [lst[i] for i in o]
        if nl != lst: nch += 1; nt1 += nl[0] != lst[0]
        out.append(';'.join(nl + L[m][25:]))
    assert len(out) == len(sub)
    shutil.copy(sp_, os.path.join(os.path.dirname(sp_), 'submission_pre_ta.csv'))
    sub['smiles'] = out; tmp = sp_ + '.tmp_ta'; sub.to_csv(tmp, index=False); os.replace(tmp, sp_)
    print(f'[ta] SUCCESS molecules {len(L)}, with evidence {nev}, gated {ngate}, changed lists {nch}, top1 changed {nt1}, {time.time() - t0:.0f}s', flush=True)

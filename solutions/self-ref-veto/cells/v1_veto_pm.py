# ===== V-1p (Arsokan): self-reference contradiction veto (best spectrum pair) ======================================================
# The models in this pipeline were trained on train.parquet and prefer the structures they have seen. For a molecule
# whose answer has no public spectrum (classes 2 and 3) they therefore tend to put a KNOWN isomer first. A known
# structure, however, has its own measured spectra in train.parquet: when those spectra (same adduct, merged over all
# energies and libraries; best single spectrum pair) disagree with the query (cosine < V1_TAU), that structure cannot be the compound that was
# measured. Such contradicted candidates move below every candidate that is not contradicted (order otherwise kept).
# Candidates without own spectra (where class-2 answers live) are never moved down; a known structure whose spectra
# agree is never moved down either. Lists with a strong library match (library_max >= V1_LIB_TAU) are untouched, and
# the external-library overlay that follows keeps the last word. Any failure leaves submission.csv unchanged.
V1_TAU = globals().get('V1_TAU', 0.30)   # best single-spectrum-pair sqrt cosine
V1_LIB_TAU = globals().get('V1_LIB_TAU', 0.90)
V1_MIN_PEAKS = 3          # the query needs this many peaks below the precursor to judge
def _v1(work='/kaggle/working', inp='/kaggle/input'):
    import glob, os, shutil, time, json, numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as pads
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    t0 = time.time(); sp = os.path.join(work, 'submission.csv')
    if not os.path.exists(sp): print('[v1] no submission.csv'); return
    tp = sorted(glob.glob(inp + '/**/test.parquet', recursive=True), key=lambda p: ('mass-spectra' not in p.lower(), len(p)))
    te = pd.read_parquet(tp[0]); train_path = os.path.join(os.path.dirname(tp[0]), 'train.parquet')
    sub = pd.read_csv(sp, dtype=str, keep_default_na=False)
    L = {m: [x for x in s.split(';') if x] for m, s in zip(sub.molecule_id.astype(str), sub.smiles)}
    ev_path = os.path.join(work, 'frozen_candidate_evidence.parquet')
    if not os.path.exists(ev_path):
        print('[v1] no frozen_candidate_evidence.parquet (library confidence unknown) -> skipped'); return
    ev = pd.read_parquet(ev_path, columns=['molecule_id', 'library_max'])
    LIB = {str(m): float(v) for m, v in ev.groupby('molecule_id').library_max.max().items()}
    if not LIB:
        print('[v1] empty evidence -> skipped'); return
    ADD = ('[M+H]+', '[M-H]-')
    def build_sp(df, idcol):
        SP = {}
        for m, ad, pr, mz, it in zip(df[idcol].astype(str), df.adduct.astype(str), df.precursor_mz, df.ms2_mzs, df.ms2_normalized_intensities):
            if ad not in ADD: continue
            try:
                mz = np.asarray(mz, float); it = np.asarray(it, float); pr = float(pr)
            except Exception:
                continue
            if not len(mz): continue
            it = it / max(float(it.max()), 1e-9); k = mz < pr - 1.5; mz, it = mz[k], it[k]
            if len(mz) < V1_MIN_PEAKS: continue
            o = np.argsort(mz); w = np.sqrt(it[o]); w = w / max(np.linalg.norm(w), 1e-12)
            SP.setdefault(m, {}).setdefault(ad, []).append((mz[o], w))
        return SP
    def cos1(a, b):
        (ma, ia_), (mb, ib) = a, b; cand = []
        lo = np.searchsorted(mb, ma - 0.0101); hi = np.searchsorted(mb, ma + 0.0101, 'right')
        for i in np.flatnonzero(hi > lo):
            for j in range(lo[i], hi[i]): cand.append((ia_[i] * ib[j], i, j))
        cand.sort(reverse=True); ui, uj, s_ = set(), set(), 0.0
        for v, i, j in cand:
            if i in ui or j in uj: continue
            ui.add(i); uj.add(j); s_ += v
        return s_
    def cos(qa, ra):
        return max((cos1(a, b) for a in qa for b in ra), default=0.0)
    SPQ = build_sp(te.assign(_m=te.molecule_id.astype(str)), '_m')
    active = [m for m in L if m in SPQ and LIB.get(m, 0.0) < V1_LIB_TAU and len(L[m]) > 1]
    kc = {}
    def k14(s):
        if s not in kc:
            try:
                mol = Chem.MolFromSmiles(s); kc[s] = Chem.MolToInchiKey(mol)[:14] if mol is not None else None
            except Exception:
                kc[s] = None
        return kc[s]
    want = {k14(c) for m in active for c in L[m]} - {None}
    ds = pads.dataset(train_path)
    tk = set(ds.to_table(columns=['inchikey14']).column('inchikey14').to_pylist())
    known = sorted(want & tk)
    print(f'[v1] active lists {len(active)}, candidate structures {len(want)}, known to train {len(known)}, {time.time() - t0:.0f}s', flush=True)
    if not known:
        print('[v1] no known candidate structures -> nothing to judge'); return
    rows = []
    flt = pads.field('inchikey14').isin(pa.array(known)) & pads.field('adduct').isin(pa.array(list(ADD)))
    for b in ds.scanner(columns=['inchikey14', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'], filter=flt,
                        batch_size=4096, batch_readahead=0, fragment_readahead=0, use_threads=False).to_batches():
        if b.num_rows: rows.append(b.to_pandas().groupby(['inchikey14', 'adduct']).head(6))
    SPR = build_sp(pd.concat(rows).groupby(['inchikey14', 'adduct']).head(6), 'inchikey14') if rows else {}
    del rows
    nlist = nmoved = nveto = nlead = 0; out = []; report = []
    for m, s in zip(sub.molecule_id.astype(str), sub.smiles):
        lst = L[m]
        if m not in active: out.append(s); continue
        q = SPQ[m]; bad = []
        for i, c in enumerate(lst):
            k = k14(c); rr = SPR.get(k) if k else None
            if not rr: continue
            ads = set(q) & set(rr)
            if not ads: continue
            best = max(cos(q[ad], rr[ad]) for ad in ads)
            if best < V1_TAU: bad.append(i)
        if not bad: out.append(s); continue
        nlist += 1; nveto += len(bad)
        bs = set(bad); nl = [c for i, c in enumerate(lst) if i not in bs] + [lst[i] for i in bad]
        if nl != lst:
            nmoved += 1; nlead += nl[0] != lst[0]
            report.append(dict(molecule_id=m, vetoed=len(bad), old_rank1_vetoed=0 in bs))
        out.append(';'.join(nl))
    if nmoved:
        shutil.copy(sp, os.path.join(work, 'submission_pre_v1.csv'))
        sub['smiles'] = out; tmp = sp + '.tmp_v1'; sub.to_csv(tmp, index=False); os.replace(tmp, sp)
    json.dump(dict(active=len(active), lists_with_veto=nlist, vetoed=nveto, lists_changed=nmoved, leaders_changed=nlead,
                   tau=V1_TAU), open(os.path.join(work, 'v1_report.json'), 'w'))
    print(f'[v1] lists with a contradicted candidate {nlist}, candidates moved down {nveto}, lists changed {nmoved}, '
          f'leaders changed {nlead}, {time.time() - t0:.0f}s', flush=True)
try:
    _v1()
except Exception:
    import traceback; traceback.print_exc(); print('[v1] FAILED -> submission unchanged', flush=True)

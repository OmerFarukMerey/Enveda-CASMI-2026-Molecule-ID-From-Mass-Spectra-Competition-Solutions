# all molecules -> base lists: BASE[mid] = [smiles, keys, lib_max, ranker scores, formulas] (top TOPN, key-deduped)
te = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
te['nm'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
mols = list(te.groupby('molecule_id', sort=False))
BASE, n_err = {}, 0
for gi, (mid, sub) in enumerate(mols):
    nms = sub.nm.values[np.isfinite(sub.nm.values)]
    smis, keys, lib_max, scs, forms = [], [], 0.0, [], []
    if len(nms):
        target = float(np.median(nms))
        spectra = []
        for r in sub.itertuples():
            ce = r.collision_energy_ev
            ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                                mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct,
                                prec=float(r.precursor_mz), ce=float(ce_arr.mean()) if len(ce_arr) else np.nan,
                                ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
        try:
            C, X, F, names, info = V.run(spectra, target)
            lib_max = float(info.get('lib_max', 0.0))
            if len(C.pid):
                score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))
                order = np.argsort(-score, kind='stable')
                seen = set()
                for i in order:
                    k = C.key[i]
                    if k in seen: continue
                    seen.add(k); smis.append(C.smiles[i]); keys.append(k); scs.append(float(score[i])); forms.append(C.formula[i])
                    if len(smis) >= CFG['TOPN']: break
        except Exception as e:
            print('ERROR', mid, repr(e)); n_err += 1
    BASE[mid] = [smis, keys, lib_max, scs, forms]
    if (gi + 1) % 25 == 0 or gi == len(mols) - 1:
        log(f'  {gi+1}/{len(mols)} molecules')
log('base lists done, errors', n_err)
dump('base', BASE)

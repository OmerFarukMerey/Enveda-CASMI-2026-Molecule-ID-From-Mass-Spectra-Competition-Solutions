# ── PC-wide rerank (Arsokan): score the whole PubChem mass window with FPNet, rescore its top-200 with the two
# spectrum-candidate models, and hedge the best proposals into fixed slots of the final list.
# Held-out check (1,300 timsTOF molecules unseen by both models): FPNet alone MRR@25 0.380 -> blend 0.496.
# Fail-safe: any exception leaves submission.csv exactly as it was.
PCW_SLOTS = [2, 4, 6, 8, 10]      # slots for the best new proposals (1-based)
PCW_WEAK_FIRST = 3                # slot of proposal 1 when its margin is below PCW_M2
PCW_M1, PCW_M2, PCW_W, PCW_TOP = 3.0, 1.5, 2.0, 200
_PCW_SFEAT = None
def _pcw_fp(s):
    try: return _PCW_SFEAT.structure(s, 2048)[-1]
    except Exception: return None
def _pcw_run():
    import glob, sys, json, time, importlib.util, os
    from pathlib import Path
    import numpy as np, pandas as pd, torch
    from multiprocessing import Pool
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    t00 = time.time()
    F = lambda pat: sorted(glob.glob('/kaggle/input/**/' + pat, recursive=True))
    sub_path = ROOT / 'submission.csv'
    sub = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    te = pd.read_parquet(COMP / 'test.parquet')
    root = Path(F('INDEX.json')[0]).parent
    sp = importlib.util.spec_from_file_location('pcw_fingerprint_index', str(root / 'fingerprint_index.py'))
    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    cv = mod.CandidateView(str(root)); fi = mod.FingerprintIndex(str(root), collection='pubchem', cache_parts=2)
    NB = int(fi.spec['bits'])
    saved_path, saved_mods = list(sys.path), {k: v for k, v in sys.modules.items() if k == 'casmi' or k.startswith('casmi.') or k in ('features', 'model', 'inference')}
    for k in saved_mods: sys.modules.pop(k, None)
    try:
        code = [str(Path(p).parent.parent) for p in F('code/casmi/__init__.py')]
        cdir = next(c for c in code if Path(c, 'casmi', 'fpnet.py').is_file() and Path(c, 'casmi', 'chem.py').is_file())
        sys.path.insert(0, cdir)
        from casmi import chem, fpnet
        from casmi.spectra import merge_spectra
        fps = [fpnet.load_model(p, 'cuda') for p in F('fpnet_fold*.pt')]
        assert len(fps) == 5 and all(m.nbits == NB for m in fps), (len(fps), NB)
        bundle = Path(F('casmi_spectrum_model_v1/inference.py')[0]).parent; sys.path.insert(0, str(bundle))
        import features as sfeat
        from model import SpectrumCandidateModel
    finally:
        pass
    scorers = []
    for ck, cf in ((bundle / 'final_model.pt', bundle / 'config.json'),) + tuple((Path(p), Path(p).parent / 'config_used.json') for p in F('casmi_spectrum_model_ours_v80/final_model.pt')):
        cfg = json.loads(Path(cf).read_text()); m = SpectrumCandidateModel(cfg).to('cuda')
        m.load_state_dict(torch.load(ck, map_location='cuda', weights_only=True)['state_dict']); m.eval(); scorers.append((m, cfg))
    assert len(scorers) == 2, len(scorers)
    dtype = fpnet.autocast_dtype('cuda')
    def k14(s):
        try:
            mm = Chem.MolFromSmiles(s); return Chem.MolToInchiKey(mm)[:14] if mm is not None else None
        except Exception: return None
    global _PCW_SFEAT
    _PCW_SFEAT = sfeat
    pool = Pool(4)
    lists = {r.molecule_id: [x for x in r.smiles.split(';') if x] for r in sub.itertuples()}
    stats = dict(molecules=0, promoted=0, strong=0, weak=0, inserted=0, moved=0, skipped=0, errors=0)
    audit = []
    with torch.no_grad():
        for mid, g in te.groupby('molecule_id', sort=False):
            if mid not in lists: continue
            stats['molecules'] += 1
            try:
                spectra = []
                for s in g.itertuples():
                    try: ce = float(s.collision_energy_orig)
                    except Exception: ce = np.nan
                    spectra.append(dict(mz=np.asarray(s.ms2_mzs, np.float64), it=np.asarray(s.ms2_normalized_intensities, np.float64), prec=float(s.precursor_mz),
                                        adduct=str(s.adduct), mode=1 if 'pos' in str(s.ionization_mode).lower() else -1, ce=ce, ce_ev=getattr(s, 'collision_energy_ev', None)))
                mass = float(np.nanmedian(chem.neutral_mass(np.array([x['prec'] for x in spectra]), np.array([x['adduct'] for x in spectra], dtype=object))))
                if not np.isfinite(mass) or mass < 50: stats['skipped'] += 1; continue
                w = cv.window(mass, ppm=10, fields=('lsid',), smiles=True)
                origin = np.asarray(w['origin']); src = np.asarray(w['source_row']); pos = np.flatnonzero(origin == 0)
                if len(pos) < 3: stats['skipped'] += 1; continue
                items = []
                for x in spectra[:8]:
                    pm, pi = fpnet.prep_peaks(x['mz'], x['it'], x['prec']); ok = not np.isnan(x['ce'])
                    items.append(dict(mz=pm, it=pi, prec=x['prec'], adduct_ix=chem.adduct_index(x['adduct']), ce=x['ce'] if ok else 0.0, ce_known=1.0 if ok else 0.0, n_merged=1, mode=float(x['mode'])))
                for md in (1, -1):
                    grp = [x for x in spectra if x['mode'] == md][:8]
                    if not grp: continue
                    mm_, ii_ = merge_spectra([(x['mz'], x['it']) for x in grp]); prec = float(np.median([x['prec'] for x in grp]))
                    pm, pi = fpnet.prep_peaks(mm_, ii_, prec); ces = [x['ce'] for x in grp if not np.isnan(x['ce'])]; ads = [x['adduct'] for x in grp]
                    items.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(max(set(ads), key=ads.count)), ce=float(np.mean(ces)) if ces else 0.0,
                                      ce_known=1.0 if ces else 0.0, n_merged=min(len(grp), 8), mode=float(md)))
                inp = fpnet.collate(items, 'cuda')
                with torch.autocast('cuda', dtype=dtype):
                    z = torch.stack([m(*inp)[0].float() for m in fps]).mean(0).mean(0)
                o = fi.get(src[pos].astype(np.int64), view='public')
                sc = fpnet.unpack_gpu(torch.as_tensor(np.asarray(o[0]), device='cuda'), NB) @ z
                top = torch.topk(sc, min(PCW_TOP, len(sc))).indices.cpu().numpy(); fz = sc[top].cpu().numpy().astype(np.float64)
                smi = w['smiles']; csm = [str(smi[pos[j]]) for j in top]
                cfp = pool.map(_pcw_fp, csm, chunksize=16); valid = [i for i, f in enumerate(cfp) if f is not None]
                if len(valid) < 3: stats['skipped'] += 1; continue
                fpt = torch.as_tensor(np.array([cfp[i] for i in valid]), device='cuda', dtype=torch.float32)
                zs_all = []
                for m, cfg in scorers:
                    recs = []
                    for x in spectra[:16]:
                        try:
                            p, n = sfeat.peaks(x['mz'], x['it'], x['prec'], cfg['max_peaks'])
                            me, a = sfeat.metadata(x['prec'], x['adduct'], 'positive' if x['mode'] == 1 else 'negative', x['ce_ev']); recs.append((p, n, me, a))
                        except Exception: pass
                    T = [torch.as_tensor(np.array([r[j] for r in recs]), device='cuda') for j in range(4)]
                    ze, pe = m.encode_spectra(T[0].float(), T[1].long(), T[2].float(), T[3].long()); ze, pe = ze.mean(0), pe.mean(0)
                    s_ = ((m.encode_molecules(fpt) @ ze) / cfg['temperature'] + cfg['fp_score_weight'] * (fpt @ pe) / cfg['fp_bits']).cpu().numpy().astype(np.float64)
                    full = np.full(len(csm), s_.min()); full[valid] = s_
                    zs_all.append((full - full.mean()) / (full.std() + 1e-6))
                blend = (fz - fz.mean()) / (fz.std() + 1e-6) + PCW_W * np.mean(zs_all, 0)
                order = np.argsort(-blend); margin = float(blend[order[0]] - blend[order[1]])
                old = lists[mid]; okeys = [k14(s) for s in old]; new = list(old); nkeys = list(okeys)
                def place(smiles, key, slot):
                    i = slot - 1
                    if key in nkeys:
                        j = nkeys.index(key)
                        if j <= i: return 'kept'
                        s0 = new.pop(j); nkeys.pop(j); new.insert(i, s0); nkeys.insert(i, key); return 'moved'
                    new.insert(min(i, len(new)), smiles); nkeys.insert(min(i, len(nkeys)), key); return 'inserted'
                slots = list(PCW_SLOTS)
                if margin >= PCW_M1: slots = [1] + PCW_SLOTS[1:]; stats['promoted'] += 1
                elif margin >= PCW_M2: stats['strong'] += 1
                else: slots = [PCW_WEAK_FIRST] + [s + 1 for s in PCW_SLOTS[1:]]; stats['weak'] += 1
                acts = []
                for slot, j in zip(slots, order[:len(slots)]):
                    key = k14(csm[j])
                    if key is None: continue
                    a = place(csm[j], key, slot); acts.append(a)
                    if a in ('moved', 'inserted'): stats[a] += 1
                lists[mid] = new[:25]
                audit.append(dict(molecule_id=mid, window=int(len(pos)), margin=round(margin, 3), slots=slots, acts=acts, top=csm[order[0]]))
            except Exception as e:
                stats['errors'] += 1
                if stats['errors'] <= 5: print('[pc-wide] molecule error', mid, repr(e)[:200])
    pool.close()
    sys.path[:] = saved_path
    for k in [k for k in sys.modules if k == 'casmi' or k.startswith('casmi.') or k in ('features', 'model')]: sys.modules.pop(k, None)
    sys.modules.update(saved_mods)
    assert stats['errors'] <= max(8, stats['molecules'] // 20), stats
    out = sub.copy(); out['smiles'] = [';'.join(lists[m]) for m in out.molecule_id]
    assert len(out) == len(sub) and (out.molecule_id == sub.molecule_id).all() and out.smiles.str.split(';').str.len().max() <= 25
    import shutil
    shutil.copyfile(sub_path, ROOT / 'submission_before_pcwide.csv')
    out.to_csv(ROOT / 'submission.tmp_pcw.csv', index=False); os.replace(ROOT / 'submission.tmp_pcw.csv', sub_path)
    (ROOT / 'pcwide_audit.json').write_text(json.dumps(dict(stats=stats, rows=audit)))
    print('[pc-wide] SUCCESS', json.dumps(stats), '| changed lists', int((out.smiles != sub.smiles).sum()), '| changed rank 1', int((out.smiles.str.split(';').str[0] != sub.smiles.str.split(';').str[0]).sum()), '| sec', round(time.time() - t00))
if RUN_PIPELINE:
    try:
        _pcw_run()
    except Exception as _e:
        import traceback; print('[pc-wide] FAILED, submission untouched:', repr(_e)[:300]); traceback.print_exc()
else:
    print('[pc-wide] skipped (fast commit)')

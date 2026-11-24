# ===== VH: score every stage snapshot per group (metric key = InChIKey-14 after RDKit tautomer canonicalisation) =====
try:
    import time as _sc_t
    if _sc_t.time() - float(globals().get('NB_T0', _sc_t.time())) > 8.0 * 3600:
        raise TimeoutError('session past 8.0 h: scoring skipped, lists are saved')
    import csv as _sc_csv, json as _sc_json
    from pathlib import Path as _SP
    import pandas as _sc_pd
    from rdkit import Chem as _SC, RDLogger as _SL
    from rdkit.Chem.MolStandardize import rdMolStandardize as _SM
    _SL.DisableLog('rdApp.*')
    _te = _SM.TautomerEnumerator(); _kc = {}

    def _sc_key(s):
        if s not in _kc:
            try:
                m = _SC.MolFromSmiles(s)
                _kc[s] = _SC.MolToInchiKey(_te.Canonicalize(m))[:14] if m is not None else None
            except Exception:
                _kc[s] = None
        return _kc[s]
    _ans = _sc_pd.read_csv(VH_OUT / 'answers.csv', dtype=str)
    _truth = {r.molecule_id: _sc_key(r.smiles) for r in _ans.itertuples()}
    _grp = dict(zip(_ans.molecule_id, _ans.group))
    _res = {}
    for _f in [p for p in sorted(VH_OUT.glob('sub_*.csv')) + [_SP('/kaggle/working/submission.csv')] if p.is_file()]:
        _rr = {g: [] for g in sorted(set(_grp.values()))}
        for _r in _sc_csv.DictReader(open(_f, encoding='utf-8')):
            _ks = [_sc_key(x) for x in _r['smiles'].split(';')[:25]]
            _t = _truth.get(_r['molecule_id'])
            _rr[_grp[_r['molecule_id']]].append(1.0 / (_ks.index(_t) + 1) if _t in _ks else 0.0)
        _res[_f.name] = {g: dict(n=len(v), mrr=round(sum(v) / max(1, len(v)), 4), top1=sum(x == 1 for x in v)) for g, v in _rr.items()}
        print('[vh] score', _f.name, _sc_json.dumps(_res[_f.name]), flush=True)
    # the one-sided test: duel and gated MAINLEAD replayed on the final list, per group (main engine = fpnet_fold0 if VH_CLEAN_MAIN)
    import shutil as _sc_sh
    _evp = _SP('/kaggle/working/frozen_candidate_evidence.parquet')
    if _evp.is_file():
        _sc_sh.copy2(_evp, VH_OUT / 'frozen_candidate_evidence.parquet')
        from rdkit.Chem import rdMolDescriptors as _SD
        _ev = _sc_pd.read_parquet(_evp, columns=['molecule_id', 'smiles', 'main_rank', 'library_max'])
        _main1 = _ev[_ev.main_rank == 1].drop_duplicates('molecule_id').set_index('molecule_id').smiles.to_dict()
        _lib = _ev.groupby('molecule_id').library_max.first().to_dict()
        _fc = {}

        def _form(s):
            if s not in _fc:
                m = _SC.MolFromSmiles(s); _fc[s] = _SD.CalcMolFormula(m) if m is not None else None
            return _fc[s]

        def _duel(mid, c):
            if len(c) > 1 and _main1.get(mid) == c[1] and _form(c[0]) == _form(c[1]):
                return [c[1], c[0]] + c[2:]
            return c

        def _mlg(mid, c):
            k = _main1.get(mid)
            if k in c[1:] and _lib.get(mid, 1.0) < 0.7 and _form(k) == _form(c[0]):
                return [k] + [x for x in c if x != k]
            return c
        _fin = [p for p in [_SP('/kaggle/working/submission.csv')] + sorted(VH_OUT.glob('sub_i_*.csv')) if p.is_file()][0]
        _rule = {}
        for _name, _fn in (('duel', _duel), ('mainlead_sameformula_lib07', _mlg), ('mainlead_all', lambda m, c: ([_main1[m]] + [x for x in c if x != _main1[m]]) if _main1.get(m) in c[1:] else c)):
            _agg = {}
            for _r in _sc_csv.DictReader(open(_fin, encoding='utf-8')):
                _m = _r['molecule_id']; _c = _r['smiles'].split(';')[:25]; _n = _fn(_m, _c); _t = _truth.get(_m)
                _ks0 = [_sc_key(x) for x in _c]; _ks1 = [_sc_key(x) for x in _n]
                _a = 1.0 / (_ks0.index(_t) + 1) if _t in _ks0 else 0.0; _b = 1.0 / (_ks1.index(_t) + 1) if _t in _ks1 else 0.0
                g = _agg.setdefault(_grp[_m], dict(n=0, mrr0=0.0, mrr1=0.0, up=0, down=0, changed=0))
                g['n'] += 1; g['mrr0'] += _a; g['mrr1'] += _b; g['up'] += _b > _a; g['down'] += _b < _a; g['changed'] += _n != _c
            for g in _agg.values():
                g['mrr0'] = round(g['mrr0'] / g['n'], 4); g['mrr1'] = round(g['mrr1'] / g['n'], 4); g['delta'] = round(g['mrr1'] - g['mrr0'], 4)
            _rule[_name] = _agg
            print('[vh] rule', _name, _sc_json.dumps(_agg), flush=True)
        _main_acc = {}
        for _m, _s in _main1.items():
            g = _main_acc.setdefault(_grp.get(_m, '?'), [0, 0]); g[0] += 1; g[1] += _sc_key(_s) == _truth.get(_m)
        _res['_rules_on_final'] = dict(file=_fin.name, rules=_rule, main_engine_leader_correct={k: f'{v[1]}/{v[0]}' for k, v in _main_acc.items()},
                                       clean_main=__import__('os').environ.get('VH_CLEAN_MAIN'))
        print('[vh] main-engine leader correct', _sc_json.dumps(_res['_rules_on_final']['main_engine_leader_correct']), flush=True)
    (VH_OUT / 'scores.json').write_text(_sc_json.dumps(_res, indent=1))
except Exception as _sc_exc:
    print('[vh] scoring failed (lists are saved; score offline):', repr(_sc_exc)[:300], flush=True)

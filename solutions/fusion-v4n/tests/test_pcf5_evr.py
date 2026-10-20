"""Tests of the PCF5 cell (nb/embed/pcf5.py, cell 37 of stack1-v72a / v74a), its q1g variant (nb/embed/pcf5_q1g.py, last cell of
casmi26-om-q1-g130-pcf5) and the EVR cell (nb/embed/evr.py, cell 38 of the unsubmitted stack1-v72b-evr); no competition data, toy
lower-case keys instead of RDKit.

Covers: every tag of pcf5_row / evr_row (short lists, the 25 cut, metric twins, placeholders), the cell-24 segment mirror (with the dual
gate in the q1g variant), pcf5_rows and evr_rows on a toy final file, and the whole PCF5 cell (both variants) run in a temp dir: the commit
path (IS_RERUN False) writes the expected file, the scoring-rerun path restores the input bytes on an error, the commit path raises.
Run with pytest, or by hand: python solutions/fusion-v4n/tests/test_pcf5_evr.py"""
import contextlib, io, json, os, sys, tempfile, time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); SOL = os.path.dirname(HERE)
MARK = '# ==== CELL BODY (executes) ===='
PH = 'CCO'


def cell_src(name):
    return open(os.path.join(SOL, 'nb', 'embed', name), encoding='utf-8').read()


def defs(name):
    """The cell's functions only (the part above the cell body, which reads ./submission.csv)."""
    src = cell_src(name)
    assert src.count(MARK) == 1
    ns = {}
    exec(compile(src.split(MARK)[0], name, 'exec'), ns)
    return ns


P5, EV, P5G = defs('pcf5.py'), defs('evr.py'), defs('pcf5_q1g.py')
IDENT = lambda s: s.strip().lower() if isinstance(s, str) and s.strip() else None   # toy metric identity: 'X' and 'x' are twins


def e2r2_make_key(identity, placeholder=PH):   # = stack1 cell 36 (the cells build their key with it)
    ph = identity(placeholder)

    def key(smiles):
        if not isinstance(smiles, str) or not smiles.strip() or smiles.strip() == placeholder:
            return None
        k = identity(smiles)
        return None if k is None or k == ph else k

    return key


KEY = e2r2_make_key(IDENT)


# ---- pcf5_row ----
def test_pcf5_row_tags():
    row = P5['pcf5_row']
    L = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    assert row([], 'X', KEY) == ([], 'skipped', None)
    assert row([PH], 'X', KEY) == ([PH], 'skipped', None)
    assert row(L, None, KEY)[1] == 'invalid' and row(L, 'X;Y', KEY)[1] == 'invalid' and row(L, '  ', KEY)[1] == 'invalid'
    assert row(L, PH, KEY) == (L, 'invalid', None)                                      # placeholder identity is never inserted
    assert row(L, 'X', KEY) == (['A', 'B', 'C', 'D', 'X', 'E', 'F', 'G'], 'inserted', 'absent')
    assert row(L, 'F', KEY) == (['A', 'B', 'C', 'D', 'F', 'E', 'G'], 'moved', '6')
    for i, t in enumerate(L[:4]):
        assert row(L, t, KEY) == (L, 'already_above', str(i + 1))                       # ranks 1-4 never change
    assert row(L, 'e', KEY) == (L, 'already_above', '5')                                # its twin already sits at rank 5


def test_pcf5_row_twins_short_lists_and_cut():
    row = P5['pcf5_row']
    new, tag, frm = row(['A', 'B', 'C', 'D', 'E', 'x', 'F', 'X'], ' X ', KEY)
    assert (new, tag, frm) == (['A', 'B', 'C', 'D', 'X', 'E', 'F'], 'moved', '6')       # every twin dropped, the leader once
    assert row(['A', 'B'], 'X', KEY) == (['A', 'B', 'X'], 'inserted', 'absent')         # a short list gets the leader appended
    full = [f'T{i}' for i in range(25)]
    new, tag, _ = row(full, 'X', KEY)
    assert tag == 'inserted' and len(new) == 25 and new[4] == 'X' and new[:4] == full[:4] and new[5:] == full[4:24]
    new, tag, _ = row(full, 'T24', KEY, pos=3)
    assert new[:2] == full[:2] and new[2] == 'T24' and len(new) == 25


# ---- the cell-24 segment mirror ----
def pc(lead, rel=0.0, S=0.0, pop=0.0, nan_pool=False, **kw):
    return dict(pc=[lead, lead + 'b'], pc_fz=[rel, 0.0], fz_top=None, best_pool_fz=float('nan') if nan_pool else 0.0,
                S=S, top_pop=pop, **kw)


TAU = dict(LIB_TAU=0.9, S_TAU=6.0, POP_TAU=5.0)


def test_segments():
    for seg in (P5['pcf5_segment'], EV['evr_segment']):
        PC = dict(c=pc('L'), o=pc('L', rel=150.0), r=pc('L', rel=151.0), s=pc('L', S=7, pop=5), sp=pc('L', S=7, pop=4),
                  n=pc('L', nan_pool=True), e=dict(pc=[]))
        BASE = dict(c=([], [], 0.95), o=([], [], 0.5), r=([], [], 0.5), s=([], [], 0.5), sp=([], [], 0.5), n=([], [], 0.5),
                    e=([], [], 0.5))
        got = {m: seg(m, PC, BASE, **TAU)[0] for m in list(PC) + ['missing']}
        assert got == dict(c='gate_closed', o='open', r='promoted', s='promoted', sp='open', n='promoted', e='no_pc',
                           missing='no_pc'), got
    PC = dict(c=pc('L', pc_form=['C6']))
    try:
        P5['pcf5_segment']('c', PC, dict(c=([], [], 0.95)), **TAU)
        raise RuntimeError('no error')
    except AssertionError as e:
        assert 'gate-closed PubChem list was re-ordered' in str(e)


# ---- the whole PCF5 cell on a toy final file ----
def toy():
    """Final file, PubChem channel, BASE and cell 24's stats for seven molecules (m6 is a placeholder row, not processed)."""
    rows = [('m1', 'A;B;C;D;E;F'), ('m2', 'a2;b2;c2;d2;e2;f2;g2;Lead2'), ('m3', 'a3;b3;c3;d3;e3;f3'),
            ('m4', 'a4;b4;c4;d4;e4;f4'), ('m5', 'a5;Lead5;c5;d5;e5;f5'), ('m6', PH), ('m7', 'x7;y7')]
    PC = dict(m1=pc('Lead1'), m2=pc('Lead2', rel=100.0), m3=pc('Lead3', rel=200.0), m5=pc('Lead5'), m7=pc('Lead7'))
    BASE = dict(m1=([], [], 0.95), m2=([], [], 0.5), m3=([], [], 0.5), m4=([], [], 0.5), m5=([], [], 0.5), m7=([], [], 0.99))
    stats = dict(untouched=2, gentle=2, aggressive=1, no_pc=1, promoted=1)
    expected = dict(rows)
    expected.update(m1='A;B;C;D;Lead1;E;F', m2='a2;b2;c2;d2;Lead2;e2;f2;g2', m7='x7;y7;Lead7')
    return rows, PC, BASE, stats, expected


def test_pcf5_rows():
    rows, PC, BASE, _, expected = toy()
    active = {m for m, _ in rows} - {'m6'}
    out, seg, tags, audit = P5['pcf5_rows'](rows, active, PC, BASE, KEY, **TAU)
    assert dict(out) == expected
    assert dict(seg) == dict(gate_closed=2, open=2, promoted=1, no_pc=1)
    assert dict(tags) == {'gate_closed:inserted': 2, 'open:moved': 1, 'open:already_above': 1, 'promoted:untouched': 1,
                          'no_pc:untouched': 1}
    assert len(audit) == 6


def run_cell(is_rerun, stats_patch=None, name='pcf5.py'):
    """Run a PCF5 cell file as a notebook cell in a fresh temp dir. Returns (stdout, final file bytes, input bytes, stats json, error)."""
    rows, PC, BASE, stats, _ = toy()
    stats.update(stats_patch or {})
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        comp = os.path.join(d, 'comp'); os.makedirs(comp)
        pd.DataFrame(dict(molecule_id=[m for m, _ in rows], smiles=PH)).to_csv(os.path.join(comp, 'sample_submission.csv'), index=False)
        work = os.path.join(d, 'work'); os.makedirs(work)
        pd.DataFrame(rows, columns=['molecule_id', 'smiles']).to_csv(os.path.join(work, 'submission.csv'), index=False)
        before = open(os.path.join(work, 'submission.csv'), 'rb').read()
        scope = dict(COMP=comp, metric_identity=IDENT, e2r2_make_key=e2r2_make_key, IS_RERUN=is_rerun, PC=PC, BASE=BASE,
                     stats=stats, T0=time.time(), te=pd.DataFrame(dict(molecule_id=[m for m, _ in rows if m != 'm6'])), **TAU)
        out, err = io.StringIO(), None
        os.chdir(work)
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(cell_src(name), name, 'exec'), scope)
        except Exception as e:  # noqa: BLE001
            err = e
        finally:
            os.chdir(cwd)
        after = open(os.path.join(work, 'submission.csv'), 'rb').read()
        st = json.load(open(os.path.join(work, 'pcf5_stats.json'))) if os.path.exists(os.path.join(work, 'pcf5_stats.json')) else None
        return out.getvalue(), after, before, st, err


def test_cell_commit_path():
    out, after, before, st, err = run_cell(False)
    assert err is None, err
    _, _, _, _, expected = toy()
    got = pd.read_csv(io.BytesIO(after), dtype=str, keep_default_na=False)
    assert dict(zip(got.molecule_id, got.smiles)) == expected and list(got.molecule_id) == [m for m in expected]
    assert out.count('PCF5_APPLIED') == 1 and 'PCF5_FAILED' not in out
    assert st['status'] == 'applied' and st['changed_rows'] == 3 and st['is_rerun'] is False
    assert st['stats_control']['got'] == st['stats_control']['expected']
    assert st['changed_by_segment'] == dict(gate_closed=2, open=1)


def test_cell_rerun_restores_input_on_error():
    out, after, before, st, err = run_cell(True, dict(untouched=99))     # control vs cell 24's stats fails
    assert err is None and after == before
    assert 'PCF5_FAILED' in out and 'PCF5_APPLIED' not in out
    assert st['status'] == 'failed' and 'segment replica differs' in st['error']


def test_cell_commit_raises_on_error():
    out, after, before, st, err = run_cell(False, dict(untouched=99))
    assert isinstance(err, AssertionError) and after == before


# ---- the q1g variant: the same cell, its cell-24 mirror extended by the dual gate (rel > 130 and pop >= 2.5) ----
def test_q1g_variant_differs_only_in_the_rule():
    a, b = cell_src('pcf5.py').splitlines(), cell_src('pcf5_q1g.py').splitlines()
    import difflib
    changed = [l for l in difflib.unified_diff(a, b, lineterm='', n=0) if l[:1] in '+-' and not l.startswith(('+++', '---'))]
    assert len(changed) == 7, changed                                   # header, the rule (1 -> 2 lines), the stats version
    assert any('rel > 130.0 and pop_top >= 2.5' in l for l in changed if l.startswith('+'))
    assert any("version='q1g'" in l for l in changed if l.startswith('+'))


def test_q1g_segments():
    PC = dict(a=pc('L', rel=140.0, pop=3.0), b=pc('L', rel=140.0, pop=2.4), c=pc('L', rel=130.0, pop=9.0), d=pc('L', rel=151.0),
              g=pc('L', rel=140.0, pop=3.0))
    BASE = {m: ([], [], 0.5) for m in PC}
    BASE['g'] = ([], [], 0.95)
    got = {m: P5G['pcf5_segment'](m, PC, BASE, **TAU)[0] for m in PC}
    assert got == dict(a='promoted', b='open', c='open', d='promoted', g='gate_closed'), got
    assert P5['pcf5_segment']('a', PC, BASE, **TAU)[0] == 'open'           # the stack1 mirror does not know the dual gate


def test_q1g_cell_paths():
    out, after, before, st, err = run_cell(False, name='pcf5_q1g.py')
    assert err is None, err
    _, _, _, _, expected = toy()
    got = pd.read_csv(io.BytesIO(after), dtype=str, keep_default_na=False)
    assert dict(zip(got.molecule_id, got.smiles)) == expected and st['version'] == 'q1g' and st['changed_rows'] == 3
    out, after, before, st, err = run_cell(True, dict(untouched=99), name='pcf5_q1g.py')
    assert err is None and after == before and st['status'] == 'failed' and 'PCF5_FAILED' in out


# ---- EVR ----
def test_evr_row_tags():
    row = EV['evr_row']
    L = ['P', 'A', 'B', 'C', 'D', 'E', 'O', 'F']
    assert row([], 'O', 'P', KEY)[1] == 'skipped' and row([PH], 'O', 'P', KEY)[1] == 'skipped'
    assert row(L, None, 'P', KEY)[1] == 'invalid_O' and row(L, 'O;X', 'P', KEY)[1] == 'invalid_O'
    assert row(L, 'O', None, KEY)[1] == 'invalid_P'
    assert row(L, 'p', 'P', KEY)[1] == 'same_as_P'
    assert row(L, 'O', 'Q', KEY)[1] == 'no_P'
    assert row(L, 'O', 'P', KEY) == (['P', 'A', 'B', 'C', 'O', 'D', 'E', 'F'], 'moved', '7')
    assert row(L, 'X', 'P', KEY) == (['P', 'A', 'B', 'C', 'X', 'D', 'E', 'O', 'F'], 'inserted', 'absent')
    assert row(['X', 'Y'] + L, 'O', 'P', KEY) == (['X', 'Y', 'P', 'A', 'O', 'B', 'C', 'D', 'E', 'F'], 'moved', '9')  # k = 5
    L2 = ['A', 'B', 'C', 'D', 'E', 'P', 'G', 'O']
    assert row(L2, 'O', 'P', KEY) == (['A', 'B', 'C', 'D', 'E', 'P', 'O', 'G'], 'moved', '8')   # k = rank(P) + 1 = 7
    assert row(['A', 'B', 'C', 'O', 'P'], 'O', 'P', KEY) == (['A', 'B', 'C', 'O', 'P'], 'already_above', '4')


def test_evr_rows_only_promoted_lists_change():
    rows = [('p1', 'P1;a;b;c;d;e;O1'), ('p2', 'P2;a;b;c;d;e;O2'), ('p3', 'P3;a;b;c;d;e;O3'), ('g', 'a;b;c;d;e;f')]
    PC = dict(p1=dict(pc('P1', rel=200.0), pc=['P1', 'O1']), p2=dict(pc('P2', rel=200.0), pc=['P2', 'O2']),
              p3=dict(pc('P3', rel=200.0), pc=['P3', 'O3']), g=pc('L'))
    BASE = dict(p1=([], [], 0.5), p2=([], [], 0.5), p3=([], [], 0.5), g=([], [], 0.95))
    ORIG = dict(p1=dict(pc=['O1', 'P1']), p2=dict(pc=['O2', 'Z']), p3=dict(pc=['O3', 'P3']))
    PRE = dict(p1=['P1', 'a'], p2=['P2', 'a'], p3=['a', 'P3'])
    out, fired, tags, audit = EV['evr_rows'](rows, {m for m, _ in rows}, PC, BASE, ORIG, PRE, KEY, **TAU)
    assert fired == ['p1', 'p2', 'p3']
    assert dict(tags) == dict(moved=1, not_permutation=1, prefusion_mismatch=1)
    assert dict(out) == dict(p1='P1;a;b;c;O1;d;e', p2=rows[1][1], p3=rows[2][1], g=rows[3][1])


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print('ok', t.__name__)
    print(f'{len(tests)} passed')

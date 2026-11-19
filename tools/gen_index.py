"""Regenerate INDEX.md: every git-tracked (or to-be-added) file with a one-line purpose, grouped by folder.

The purpose comes from a Python module docstring, a Markdown `<!-- index: ... -->` first line (else H1 + first sentence), a JSON "summary", or the
OVERRIDES table below. Run after adding, removing or renaming files:  python tools/gen_index.py [--check]
"""
import ast, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERRIDES = {
    'solutions/arsokan-fable/vh/cell_lmg.py': 'LMG-1 notebook cell (candidate, not submitted): undo the merge step leader move when both engines ranked the old leader first and library_max >= 0.7',
    'solutions/arsokan-fable/offline/an_eval2.py': "TA-1 offline evaluation, earlier variant (paths: author machine)",
    'solutions/arsokan-fable/offline/an_eval4.py': "TA-1 offline evaluation of variants: K, lower bound, sum vs best, adduct fallback (paths: author machine)",
    'solutions/arsokan-fable/offline/an_prep30.py': "TA-1 offline prep with 30 analogs per candidate (paths: author machine)",
    'solutions/arsokan-fable/offline/ta_local.py': "runs the TA-1 stage locally on harness files (paths: author machine)",
    'solutions/arsokan-fable/offline/net_bench_prep.py': "builds the network-harness inputs from pccand.parquet (paths: author machine)",
    'solutions/arsokan-fable/offline/pc_ranker.py': "first LightGBM ranker over PubChem-window candidates (paths: author machine)",
    'solutions/arsokan-fable/offline/libscan.py': "best extlib2 spectrum match per test molecule (paths: author machine)",
    'solutions/arsokan-fable/offline/sim_gen.py': "GEN simulation: positional isomers of listed candidates on a validation replay, MRR before/after",
    'solutions/arsokan-fable/offline/lab_mass.py': "tests romanrozen's timsTOF mass prior on a validation replay's final lists",
    'solutions/arsokan-fable/lab/pcwin_lab.py': "Kaggle lab: are the validation truths our lists miss present in the public PubChem mass index?",
    'solutions/arsokan-fable/lab/pcw_stage_standalone_test.py': "standalone Kaggle test of the PubChem-wide rerank stage (w0-w3)",
    'solutions/arsokan-fable/kaggle/t0/notebook.ipynb': "exact t0 notebook: n0 + TA-1 cell after N-1c (Kaggle id 58419263)",
    'solutions/arsokan-fable/kaggle/t0/kernel-metadata.json': "Kaggle inputs and machine settings of t0 (owner redacted)",
    'submissions/n0-0.492/notebook.ipynb': 'the exact notebook submitted as n0 (public LB 0.492, mathleteutd copy, N-1c LAM 0.35); import into Kaggle as is',
    'submissions/n0-0.492/kernel-metadata.json': 'Kaggle inputs and machine settings of n0 (owner redacted)',
    'submissions/n0-0.492/README.txt': 'n0, public LB 0.492 (team best): what it is, inputs, local evidence, follow-up t0',
    'solutions/arsokan-fable/stages/train_analog_rerank.py': 'TA-1 notebook cell: rerank lists by spectral evidence from similar training molecules (Tanimoto 0.40-0.85), gated',
    'solutions/arsokan-fable/stages/pc_wide_rerank_NEGATIVE.py': 'NEGATIVE: PubChem-mass-window FPNet + spectrum-model rerank hedged into fixed slots (w0-w3, no hidden-test gain)',
    'solutions/arsokan-fable/offline/net_bench.py': 'offline harness of the test-set network rerank: lists, spectra, fingerprints, modified cosine (paths: author machine)',
    'solutions/arsokan-fable/offline/net_eval.py': 'offline evaluation of network-rerank design variants on the harness (paths: author machine)',
    'solutions/arsokan-fable/offline/net_grid.py': 'grid over network-rerank variants and LAM values (uses net_eval)',
    'solutions/arsokan-fable/offline/an_prep.py': 'TA-1 offline prep: nearest training analogs of every candidate (paths: author machine)',
    'solutions/arsokan-fable/offline/an_eval3.py': 'TA-1 offline evaluation on the harness lists (paths: author machine)',
    'solutions/arsokan-fable/offline/an_eval5.py': 'TA-1 offline evaluation, top-30 analogs, raw evidence dump (paths: author machine)',
    'solutions/arsokan-fable/offline/pc_ranker2.py': 'LightGBM ranker over PubChem-window candidates, fold-wise (paths: author machine)',
    'solutions/arsokan-fable/offline/libval.py': 'match accuracy of the second spectrum library (nakhun/casmi26-extlib2) on training molecules',
    'solutions/arsokan-fable/offline/sim_frigid.py': 'scores FRIGID generations against validation lists',
    'solutions/arsokan-fable/offline/dl_out.py': 'downloads a Kaggle kernel output with retries (local Kaggle API config, no credentials here)',
    'solutions/arsokan-fable/lab/pcrank_heldout_lab.py': 'Kaggle lab: rank of the true structure when FPNet scores the whole PubChem mass window',
    'solutions/arsokan-fable/lab/frigid_lab.py': 'Kaggle lab: FRIGID de novo generation on validation lists (internet on)',
    'LICENSE': 'MIT licence of the repository',
    'CLAUDE.md': 'Claude Code entry: imports AGENTS.md',
    'llms.txt': 'llms.txt: curated link list for LLM tools',
    '.gitignore': 'keeps data, weights, runs, secrets and venvs out of git',
    '.gitattributes': 'line-ending normalisation',
    '.editorconfig': 'editor whitespace settings',
    'INDEX.md': 'this file (generated by tools/gen_index.py)',
    'solutions/fusion-v4n/requirements.txt': 'local Python dependencies (RDKit pinned to the metric version)',
    'solutions/fusion-v4n/reference/v4n/notebook_main_stubbed.txt': 'reference notebook code with embedded sources elided (used by the equivalence test)',
    'solutions/fusion-v4n/reference/v4n/notebook_code.py': 'reference notebook code, all cells, verbatim (do not edit)',
    'solutions/fusion-v4n/kaggle/casmi26-ours.ipynb': 'GENERATED Kaggle notebook (build_notebook.py); import into Kaggle via File > Import Notebook',
    'solutions/fusion-v4n/configs/v1-0.399.json': 'CFG preset of v1 (reference defaults), public LB 0.399',
    'solutions/fusion-v4n/configs/v2-0.404.json': 'CFG preset of v2 (popularity μ=0.15 + gap-mode MetFrag re-score), public LB 0.404',
    'solutions/fusion-v4n/kaggle/submitted/v2-0.404.ipynb': 'the exact notebook submitted as v2 (public LB 0.404); import into Kaggle as is',
    'submissions/fusion-v4n-v12a-0.420/notebook.ipynb': 'the exact notebook submitted as v12a (public LB 0.420, PubChem join); import into Kaggle as is',
    'submissions/fusion-v4n-v12a-0.420/config.json': 'CFG overrides of v12a (copy of configs/v12a-v4b-pcjoin50-lib07.json)',
    'submissions/fusion-v4n-v12a-0.420/README.txt': 'v12a, public LB 0.420: what changed, config, inputs, reproduction, local evidence, lessons',
    'solutions/fusion-v4n/configs/v12a-v4b-pcjoin50-lib07.json': 'CFG preset of v12a (v4b + PubChem join, N 50, lib_max < 0.7), public LB 0.420',
    'submissions/fusion-v4n-v33b-0.442/notebook.ipynb': 'the exact notebook submitted as v33b (public LB 0.442, ranker fitted on unseen molecules averaged in below the gate); import into Kaggle as is',
    'submissions/fusion-v4n-v33b-0.442/config.json': 'CFG overrides of v33b (copy of configs/v33b-v24a-fsranker-gate07-rankavg.json)',
    'submissions/fusion-v4n-v33b-0.442/README.txt': 'v33b, public LB 0.442 (team best): what changed, config, inputs, reproduction, local evidence, lessons',
    'submissions/fusion-v4n-v24a-0.437/notebook.ipynb': 'the exact notebook submitted as v24a (public LB 0.437, forward-model weights 0.75); import into Kaggle as is',
    'submissions/fusion-v4n-v24a-0.437/config.json': 'CFG overrides of v24a (copy of configs/v24a-v13a-fwd075.json)',
    'submissions/fusion-v4n-v24a-0.437/README.txt': 'v24a, public LB 0.437: what changed, config, inputs, reproduction, local evidence, lessons',
    'submissions/fusion-v4n-v13a-0.431/notebook.ipynb': 'the exact notebook submitted as v13a (public LB 0.431, PubChem join with 100 candidates); import into Kaggle as is',
    'submissions/fusion-v4n-v13a-0.431/config.json': 'CFG overrides of v13a (copy of configs/v13a-v12a-n100.json)',
    'submissions/fusion-v4n-v13a-0.431/README.txt': 'v13a, public LB 0.431: what changed, config, inputs, reproduction, local evidence, lessons',
    'solutions/fusion-v4n/configs/v13a-v12a-n100.json': 'CFG preset of v13a (v12a with PC_JOIN_N 100), public LB 0.431',
    'solutions/fusion-v4n/lb_points.json': 'our public-LB results with their CFG overrides: the calibration set for tools/calibrate.py',
    'solutions/fusion-v4n/kaggle/kernel-metadata.json': 'Kaggle CLI push metadata (generated by build_notebook.py)',
    'solutions/fusion-v4n/prototypes/ae/public/dataset-metadata.json': 'Kaggle CLI metadata of the public dataset omerfarukmerey/casmi26-analog-edit-gate (CC0)',
    'solutions/self-ref-veto/cells/veto.py': 'notebook cell: self-reference contradiction veto with optional confirmation (best spectrum pair; V1_TAU, V1_CAP, V2_ON)',
    'solutions/self-ref-veto/cells/v1_veto_pm.py': 'notebook cell exactly as submitted in vpd: best-pair veto < 0.30 with 6 references per adduct (frozen, do not edit)',
    'solutions/self-ref-veto/cells/dedup_after_overlay.py': 'notebook cell: metric-key deduplication after the external-library overlay, refilled from the pre-overlay list',
    'solutions/self-ref-veto/kaggle/casmi26-om-vpd/casmi26-om-vpd.ipynb': 'GENERATED notebook of vpd (tools/build.py vpd): the 0.492 notebook + veto + dedup, the exact submitted version',
    'solutions/self-ref-veto/kaggle/casmi26-om-vpd/kernel-metadata.json': 'Kaggle CLI push metadata of vpd (private, T4, internet off, the inputs of the 0.492 notebook)',
    'solutions/self-ref-veto/kaggle/casmi26-om-w30c/casmi26-om-w30c.ipynb': 'GENERATED notebook of w30c (tools/build.py w30c): veto with 30 references per adduct, threshold 0.40, confirmation, dedup',
    'solutions/self-ref-veto/kaggle/casmi26-om-w30c/kernel-metadata.json': 'Kaggle CLI push metadata of w30c (private, T4, internet off)',
    'solutions/self-ref-veto/kaggle/casmi26-om-nls/casmi26-om-nls.ipynb': 'GENERATED notebook of nls (tools/build.py nls): network weight 0.70 on ranks 2-25, rank 1 as the 0.492 notebook (public 0.487); the exact submitted version',
    'solutions/self-ref-veto/kaggle/casmi26-om-nls/kernel-metadata.json': 'Kaggle CLI push metadata of nls (private, T4, internet off)',
    'solutions/self-ref-veto/kaggle/casmi26-om-tas/casmi26-om-tas.ipynb': 'GENERATED notebook of tas (tools/build.py tas): the team TA-1 train-analog evidence on ranks 2-25 only (public 0.487); the exact submitted version',
    'solutions/self-ref-veto/kaggle/casmi26-om-tas/kernel-metadata.json': 'Kaggle CLI push metadata of tas (private, T4, internet off)',
    'solutions/self-ref-veto/kaggle/casmi26-om-nlt/casmi26-om-nlt.ipynb': 'GENERATED notebook of nlt (tools/build.py nlt): nls + tas (public 0.483); the exact submitted version',
    'solutions/self-ref-veto/kaggle/casmi26-om-nlt/kernel-metadata.json': 'Kaggle CLI push metadata of nlt (private, T4, internet off)',
    'solutions/self-ref-veto/cells/ta_param.py': "notebook cell: the team's TA-1 stage (arsokan-fable) with a rank-1 freeze and margin switch (TA_LAM, TA_FREEZE_TOP1, TA_TOP1_MARGIN)",
}
REF_NOTE = 'verbatim upstream snapshot of the reference notebook (do not edit)'


def files():
    out = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split('\n')
    return sorted(f for f in out if f)


def first_sentence(text, limit=150):
    text = ' '.join(text.split())
    m = re.match(r'(.+?[.!?])(\s|$)', text)
    s = m.group(1) if m else text
    return s if len(s) <= limit else s[:limit - 1] + '…'


def purpose(rel):
    if rel in OVERRIDES:
        return OVERRIDES[rel]
    if '/reference/' in rel:
        return REF_NOTE
    p = os.path.join(ROOT, rel)
    try:
        src = open(p, encoding='utf-8').read()
    except (UnicodeDecodeError, OSError):
        return ''
    if rel.endswith('.py'):
        try:
            doc = ast.get_docstring(ast.parse(src))
        except SyntaxError:
            doc = None
        if doc:
            return first_sentence(doc)
        m = re.match(r'\s*#\s*(.+)', src)
        return first_sentence(m.group(1)) if m else ''
    if rel.endswith('.md'):
        m = re.match(r'\s*<!--\s*index:\s*(.+?)\s*-->', src)
        if m:
            return m.group(1)
        body = [l for l in src.split('\n') if l.strip()]
        h1 = next((l.lstrip('# ').strip() for l in body if l.startswith('# ')), '')
        para = next((l for l in body if not l.startswith(('#', '|', '>', '```', '-', '!')) and len(l) > 20), '')
        return h1 + (f': {first_sentence(para, 110)}' if para else '')
    if rel.endswith('.json'):
        try:
            d = json.loads(src)
            return first_sentence(d.get('summary', '')) if isinstance(d, dict) else ''
        except json.JSONDecodeError:
            return ''
    return ''


def build():
    groups = {}
    for f in files():
        groups.setdefault(os.path.dirname(f) or '.', []).append(f)
    lines = ['# INDEX: file map', '',
             '_Generated by `python tools/gen_index.py`. Do not edit by hand._ Entry points: [AGENTS.md](AGENTS.md) (agents), '
             '[README.md](README.md) (humans).', '']
    for d in sorted(groups, key=lambda x: (x != '.', x)):
        lines += [f'## `{d}/`' if d != '.' else '## repo root', '', '| File | Purpose |', '|---|---|']
        for f in groups[d]:
            lines.append(f'| [{os.path.basename(f)}]({f}) | {purpose(f).replace("|", "/")} |')
        lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    text = build()
    out = os.path.join(ROOT, 'INDEX.md')
    if '--check' in sys.argv:
        cur = open(out, encoding='utf-8').read() if os.path.exists(out) else ''
        sys.exit(0 if cur == text else 'INDEX.md is stale: run python tools/gen_index.py')
    open(out, 'w', encoding='utf-8', newline='\n').write(text)
    print(f'wrote INDEX.md ({text.count(chr(10))} lines)')

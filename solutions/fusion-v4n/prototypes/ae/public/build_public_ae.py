"""Public version of a scored analog-edit ticket (build_ae.py): our header first, the private notice dropped, garbled
markdown repaired, the overlay cell pointed at the public dataset.
usage: build_public_ae.py TICKET_DIR SCORE RUNS_MD
TICKET_DIR is a build_ae.py output; RUNS_MD holds the rows of the runs table (markdown, one row per line). Writes
scratch/kernels/<slug>/ at the repo root."""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src, score, runs = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]).read_text(encoding='utf-8').strip()
PRIV, DS = 'omerfarukmerey/casmi26-om-ae', 'omerfarukmerey/casmi26-analog-edit-gate'
nb = json.loads(next(src.glob('*.ipynb')).read_text(encoding='utf-8'))
meta = json.loads((src / 'kernel-metadata.json').read_text(encoding='utf-8'))
cells = nb['cells']
assert ''.join(cells[1]['source']).startswith('**Team Arsokan private ticket.**')
del cells[1]

# UTF-8 text that was once decoded as cp1251 ("В·" for "·", "Г–" for "Ö"): re-encode each suspicious run
RUN = re.compile('[Ѐ-џҐґ -»–-™]{2,}')


def repair(s):
    def fix(m):
        try:
            return m.group(0).encode('cp1251').decode('utf-8')
        except UnicodeError:
            return m.group(0)
    return RUN.sub(fix, s)


fixed = 0
for c in cells:
    if c['cell_type'] == 'markdown':
        s = ''.join(c['source']); r = repair(s)
        if r != s:
            fixed += 1; c['source'] = r.splitlines(keepends=True)

ov = [c for c in cells if c['cell_type'] == 'code' and 'def _ae(' in ''.join(c['source'])]
assert len(ov) == 1
s = ''.join(ov[0]['source'])
old = f'dataset {PRIV} ('
assert s.count(old) == 1
s = s.replace(old, f'public dataset {DS} (')
tau = re.search(r'^AE_TAU = (\S+)$', s, re.M)[1]
ov[0]['source'] = s.splitlines(keepends=True)

head = re.sub(r'\A<!-- index:.*?-->\n', '', (HERE / 'header_ae.md').read_text(encoding='utf-8'))  # the INDEX.md line
for k, v in dict(__SCORE__=score, __DATASET__=DS, __RUNS__=runs, __TAU__=tau).items():
    head = head.replace(k, v)
assert not re.search(r'__[A-Z_]+__', head), re.findall(r'__[A-Z_]+__', head)
cells.insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': head.splitlines(keepends=True)})
for c in cells:
    if c['cell_type'] == 'code':
        c['outputs'] = []; c['execution_count'] = None

title = head.splitlines()[0].lstrip('# ').strip()
slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
out = HERE.parents[4] / 'scratch' / 'kernels' / slug; out.mkdir(parents=True, exist_ok=True)  # repo-root scratch/ (not in git)
text = json.dumps(nb, ensure_ascii=False, indent=1)
(out / f'{slug}.ipynb').write_text(text, encoding='utf-8')
assert PRIV in meta['dataset_sources']
ds = [d for d in meta['dataset_sources'] if d != PRIV] + [DS]
newmeta = dict(meta, id=f'omerfarukmerey/{slug}', title=title, code_file=f'{slug}.ipynb', is_private=False, dataset_sources=ds)
(out / 'kernel-metadata.json').write_text(json.dumps(newmeta, indent=2, ensure_ascii=False), encoding='utf-8')
print(out, '| title:', title, f'({len(title)} chars) | repaired markdown cells:', fixed, '| datasets:', len(ds), '| tau', tau,
      f'| {len(text.encode()):,} bytes')

"""Per-molecule Spearman correlation of neural_score between the ensemble members (commit-run outputs, data only).
usage: corr.py RUN_DIR (the neural_*.parquet files of an ens3 commit run)."""
import sys, itertools
from pathlib import Path
import pandas as pd
d = Path(sys.argv[1])
names = dict(pub='neural_public', bg='neural_ours', A20='neural_om_seed_A20', A21='neural_om_seed_A21', A22='neural_om_seed_A22')
cols = {k: pd.read_parquet(d / f'{v}.parquet').set_index(['molecule_id', 'smiles'])['neural_score'] for k, v in names.items()}
df = pd.DataFrame(cols).dropna(); n = df.groupby(level=0).size(); df = df[df.index.get_level_values(0).isin(n[n >= 10].index)]
print('rows', len(df), 'molecules', df.index.get_level_values(0).nunique())
res = {}
for a, b in itertools.combinations(names, 2):
    r = df.groupby(level=0).apply(lambda g: g[a].rank().corr(g[b].rank()))
    res[(a, b)] = r.mean()
for (a, b), v in res.items():
    print(f'{a:>4} vs {b:<4} mean per-molecule Spearman {v:.3f}')
# top-5 overlap between members, mean over molecules
def top(g, c, k=5): return set(g[c].nlargest(k).index.get_level_values(1))
for a, b in [('pub', 'bg'), ('bg', 'A20'), ('bg', 'A21'), ('bg', 'A22'), ('A20', 'A21'), ('A20', 'A22'), ('A21', 'A22'), ('pub', 'A20')]:
    ov = df.groupby(level=0).apply(lambda g: len(top(g, a) & top(g, b)) / 5).mean()
    print(f'top-5 overlap {a} / {b}: {ov:.2f}')

"""Download competition data + the reference notebook's input datasets into <repo>/data/input/, mirroring Kaggle's
/kaggle/input layout (competitions/<slug>/, datasets/<owner>/<slug>/). Needs Kaggle auth (`kaggle auth login`) and the
competition rules accepted on kaggle.com. Already-downloaded folders are skipped.

usage:
  python tools/download_data.py                 # competition + code/small datasets needed for local re-fusion
  python tools/download_data.py --all           # every input of the notebook (~16 GB)
  python tools/download_data.py owner/slug ...  # specific datasets
  python tools/download_data.py --run user/kernel-slug [--name NAME]   # a notebook run's output -> <solution>/runs/NAME/
"""
import argparse, os, sys, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_notebook import COMPETITION, DATASETS
from paths import INPUT, RUNS
# what tools/refuse.py needs: fuse.py (ICEBERG), gl_fuse.py (GLACIER), casmi.chem (score_key)
LOCAL_MIN = ['ahmedberatozer/casmi26-iceberg', 'ahmedberatozer/casmi26-glacier', 'ahmedberatozer/casmi26-v4b-models']


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi(); a.authenticate()
    return a


def done(d):
    return os.path.isdir(d) and any(os.scandir(d))


def get_competition(a):
    d = os.path.join(INPUT, 'competitions', COMPETITION)
    if done(d):
        print('skip (exists)', d); return
    os.makedirs(d, exist_ok=True)
    print('downloading competition', COMPETITION, '(~3 GB)', flush=True)
    a.competition_download_files(COMPETITION, path=d, quiet=False)
    for z in [f for f in os.listdir(d) if f.endswith('.zip')]:
        with zipfile.ZipFile(os.path.join(d, z)) as zf:
            zf.extractall(d)
        os.remove(os.path.join(d, z))


def get_dataset(a, ref):
    d = os.path.join(INPUT, 'datasets', *ref.split('/'))
    if done(d):
        print('skip (exists)', d); return
    os.makedirs(d, exist_ok=True)
    print('downloading', ref, flush=True)
    a.dataset_download_files(ref, path=d, unzip=True, quiet=False)


def get_run(a, kernel, name):
    d = os.path.join(RUNS, name or kernel.split('/')[-1])
    os.makedirs(d, exist_ok=True)
    print('downloading output of', kernel, '->', d, flush=True)
    # page_size 200 (max): a run can leave thousands of output files and the default 20 hits the API rate limit
    a.kernels_output(kernel, path=d, page_size=200)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*'); ap.add_argument('--all', action='store_true')
    ap.add_argument('--run'); ap.add_argument('--name')
    args = ap.parse_args()
    a = api()
    if args.run:
        get_run(a, args.run, args.name); sys.exit()
    refs = args.datasets or (DATASETS if args.all else LOCAL_MIN)
    if not args.datasets:
        get_competition(a)
    for r in refs:
        get_dataset(a, r)
    print('done ->', INPUT)

"""Paths of the fingerprint-conditioned structure generator (fpgen): inputs under $CASMI_DATA, everything written under $FPGEN_WORK (drive C:)."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.dirname(HERE); SOLUTION = os.path.dirname(TRAIN); REPO = os.path.dirname(os.path.dirname(SOLUTION))
DATA = os.environ.get('CASMI_DATA', os.path.join(REPO, 'data')).replace(chr(92), '/')
WORK = DATA + '/directions/run/work'          # spectrum cache, pool tables (pool_meta, pool_fp, fp_bits), split.parquet
CODE = DATA + '/directions/code'              # public `casmi` package: the pool's fingerprint and metric-key definition (local study only)
FOLD0 = DATA + '/directions/dl/fpnet_fold0.pt'  # the public fold-safe fingerprint network (never saw fold 'hold' or the 'np' panel)
BENCH = DATA + '/joint/bench'                 # fixed benchmark (train/joint): mols.parquet with DEV / FINAL halves, spectra.parquet
COV = DATA + '/coverage2/dev_cov.parquet'     # natural class (N = in no database) of the DEV molecules
G2 = REPO + '/scratch/gen2/cache'             # z_dev.npz / z_fair.npz: stored fold-safe logits; eval_fair.parquet: set F
G3 = REPO + '/scratch/gen3/cache'             # lists.pkl: base lists and their formulas per molecule (the pipeline's formulas)
TIER = DATA + '/pubchem_study/tier'           # PubChem tier (pc_mass / pc_off / pc_smiles)
PCNP = DATA + '/pubchem_study/popds/pc_np.npy'  # natural-product flag per tier row
OUT = os.environ.get('FPGEN_WORK', 'C:/Users/hemre/casmi-work/fpgen').replace(chr(92), '/')
CHEM = OUT + '/chem'; SETS = OUT + '/sets'; TOK = OUT + '/tok'; CAL = OUT + '/cal'; RUNS = OUT + '/runs'; EVAL = OUT + '/eval'
LEDGER = OUT + '/ledger.jsonl'; GPU_LEDGER = OUT + '/gpu_ledger.jsonl'
for _p in (TRAIN, os.path.join(SOLUTION, 'clean'), os.path.join(SOLUTION, 'nb', 'embed')):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def mk():
    for d in (OUT, CHEM, SETS, TOK, CAL, RUNS, EVAL):
        os.makedirs(d, exist_ok=True)


def low_priority():
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def watch_parent():
    """Worker initialiser part: a worker exits when its parent is gone (no orphans after a killed shell)."""
    import threading, time, psutil
    ppid = os.getppid()

    def w():
        while True:
            time.sleep(5)
            if not psutil.pid_exists(ppid):
                os._exit(0)
    threading.Thread(target=w, daemon=True).start()

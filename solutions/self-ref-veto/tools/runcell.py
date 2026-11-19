"""Run one notebook cell file locally against the harness: /kaggle/working -> <h>/working, /kaggle/input -> <h>/input.
usage: python -I runcell.py <harness_dir> <cell.py> [NAME=VALUE ...]   (globals set before the cell runs)
The harness input folder must contain test.parquet and train.parquet (train may be a path file: input/TRAIN_PATH.txt).
"""
import os, shutil, sys, time

sys.stdout.reconfigure(encoding="utf-8")
H, cell = sys.argv[1], sys.argv[2]
code = open(cell, encoding="utf-8").read()
work, inp = os.path.join(H, "working").replace("\\", "/"), os.path.join(H, "input").replace("\\", "/")
code = code.replace("'/kaggle/working'", repr(work)).replace("'/kaggle/input'", repr(inp))
g = {"__name__": "__main__"}
for kv in sys.argv[3:]:
    k, v = kv.split("=", 1)
    g[k] = eval(v)
before = open(os.path.join(work, "submission.csv"), encoding="utf-8").read()
t0 = time.time()
exec(compile(code, cell, "exec"), g)
after = open(os.path.join(work, "submission.csv"), encoding="utf-8").read()
print(f"--- ran {os.path.basename(cell)} in {time.time() - t0:.1f}s; submission changed: {before != after}")

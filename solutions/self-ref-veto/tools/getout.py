"""Fetch selected output files and the run log of a kernel's latest session (official Kaggle API).
usage: python -I getout.py <owner/slug> <out_dir> <name1,name2,...>"""
import os, sys, requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
ref, out, names = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
owner, slug = ref.split("/"); a = KaggleApi(); a.authenticate(); tok = None; found = set(); log = None
with a.build_kaggle_client() as k:
    for _ in range(200):
        req = ApiListKernelSessionOutputRequest(); req.user_name = owner; req.kernel_slug = slug
        if tok: req.page_token = tok
        r = k.kernels.kernels_api_client.list_kernel_session_output(req)
        if log is None and r.log: log = r.log
        for it in r.files:
            base = os.path.basename(it.file_name)
            if base in names and base not in found:
                data = requests.get(it.url, timeout=120).content
                open(os.path.join(out, base), "wb").write(data); found.add(base)
        tok = r.next_page_token
        if not tok or found >= names: break
if log: open(os.path.join(out, slug + ".log"), "w", encoding="utf-8").write(log)
print("found:", sorted(found), "| missing:", sorted(names - found), "| log chars:", len(log or ""))

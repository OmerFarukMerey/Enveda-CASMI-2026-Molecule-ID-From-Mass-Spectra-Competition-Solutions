"""Fetch only the vh/ files of the harness kernel output: pages the file list slowly (large pages, back-off on 429).
usage: .venv\\Scripts\\python scratch/land1123/vh_fetch.py <owner/kernel> <target_dir> [regex]"""
import os, re, sys, time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

kernel, target = sys.argv[1], sys.argv[2]
pat = re.compile(sys.argv[3] if len(sys.argv) > 3 else r'^vh/')
api = KaggleApi(); api.authenticate()
owner, slug = kernel.split('/')
token, pages, seen, got = None, 0, 0, 0
with api.build_kaggle_client() as kc:
    while True:
        req = ApiListKernelSessionOutputRequest(); req.user_name = owner; req.kernel_slug = slug
        req.page_size = 500
        if token: req.page_token = token
        for attempt in range(40):
            try:
                resp = kc.kernels.kernels_api_client.list_kernel_session_output(req); break
            except requests.HTTPError as e:
                if e.response.status_code != 429: raise
                time.sleep(30 + 15 * min(attempt, 6))
        else:
            sys.exit('still rate-limited after 40 tries')
        pages += 1
        for it in resp.files or []:
            seen += 1
            if not pat.search(it.file_name): continue
            out = os.path.join(target, it.file_name); os.makedirs(os.path.dirname(out), exist_ok=True)
            r = requests.get(it.url); r.raise_for_status(); open(out, 'wb').write(r.content); got += 1
            print('got', it.file_name, len(r.content), flush=True)
        token = resp.next_page_token
        print('page', pages, 'files seen', seen, 'matched', got, flush=True)
        if not token: break
        time.sleep(4)
print('done', seen, got)

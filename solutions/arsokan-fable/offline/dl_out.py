import time, sys
from kaggle.api.kaggle_api_extended import KaggleApi
a = KaggleApi(); a.authenticate()
kernel, dst, pat = sys.argv[1], sys.argv[2], sys.argv[3]
for i in range(8):
    try:
        tok = None
        while True:
            files, tok = a.kernels_output(kernel, dst, file_pattern=pat, force=True, page_token=tok, page_size=100)
            print('page', files, flush=True)
            if not tok: break
        print('DONE'); sys.exit(0)
    except Exception as e:
        print('try', i, str(e)[:100], flush=True); time.sleep(240)
sys.exit(1)

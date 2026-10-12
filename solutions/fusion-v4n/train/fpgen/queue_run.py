"""Job queue of fpgen: runs the lines of a queue file one at a time (never two GPU jobs), restarts a trainer that yielded the GPU, keeps a ledger and a GPU-hour budget.

A queue line is a command for the repo's interpreter, e.g. `train.py --tag ref0 ...`; `#` comments and blank lines are skipped. A line
whose exit code is 75 (the trainer yielded the GPU) is retried with `--resume` after the GPU is free again, up to --retries times.
usage: python queue_run.py --queue scratch/fpgen/queue.txt [--budget_h 14] [--status]
"""
import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpaths as P
import guard_run

HERE = os.path.dirname(os.path.abspath(__file__))
YIELD = 75


def hours():
    """GPU hours already spent by this study (sum of the ledger's wall times)."""
    h = 0.0
    for f in (P.LEDGER, P.GPU_LEDGER):
        if os.path.exists(f):
            for l in open(f):
                try:
                    h += json.loads(l).get('wall_s', 0) / 3600
                except Exception:
                    pass
    return h


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--queue', required=True); ap.add_argument('--budget_h', type=float, default=14.0)
    ap.add_argument('--retries', type=int, default=20); ap.add_argument('--status', action='store_true'); a = ap.parse_args()
    P.mk(); P.low_priority()
    if a.status:
        print(json.dumps(dict(gpu_hours=round(hours(), 2), queue=open(a.queue).read()), indent=1)); return
    done_f = P.OUT + '/queue_done.txt'; done = set(open(done_f).read().splitlines()) if os.path.exists(done_f) else set()
    for line in open(a.queue).read().splitlines():
        cmd = line.strip()
        if not cmd or cmd.startswith('#') or cmd in done:
            continue
        if hours() > a.budget_h:
            print('budget of %.1f GPU hours reached (%.2f spent), stopping' % (a.budget_h, hours()), flush=True); break
        full = [sys.executable, os.path.join(HERE, *cmd.split(' ')[0].split('/'))] + cmd.split(' ')[1:]
        for k in range(a.retries + 1):
            guard_run.wait_free(9.0, 2, 20)
            print(time.strftime('%H:%M:%S'), 'queue: start', cmd, '(attempt %d, %.2f GPU h spent)' % (k + 1, hours()), flush=True)
            rc = subprocess.call(full + (['--resume'] if k and '--resume' not in full else []))
            if rc != YIELD:
                break
            print(time.strftime('%H:%M:%S'), 'queue: job yielded the GPU, waiting', flush=True); time.sleep(60)
        print(time.strftime('%H:%M:%S'), 'queue: exit code', rc, 'for', cmd, flush=True)
        if rc == 0:
            done.add(cmd); open(done_f, 'a').write(cmd + '\n')
        else:
            print('queue: stopping on a failed job', flush=True); break
    print('queue finished, %.2f GPU hours spent' % hours(), flush=True)


if __name__ == '__main__':
    main()

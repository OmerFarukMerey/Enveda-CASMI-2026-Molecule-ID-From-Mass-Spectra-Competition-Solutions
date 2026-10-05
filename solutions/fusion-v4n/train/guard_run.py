"""Run a training command only while the GPU is free, restarting it after it yields (exit code 75).

The machine is shared with the user's own work and games. The trainer (train/train_fpnet.py) checkpoints and exits
with code 75 when other processes take GPU memory; this supervisor waits until the GPU has been free for a while and
starts the command again with --resume appended. Any other non-zero exit code stops the loop.

usage: python train/guard_run.py [--free_gb 3.0] [--checks 4] [--every 30] -- <command ...>
"""
import argparse, subprocess, sys, time

YIELD_CODE = 75


def gpu_used_gb():
    try:
        return float(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=20).stdout.split()[0]) / 1024
    except Exception:
        return None


def wait_free(free_gb, checks, every, log=print):
    """Block until GPU memory in use stayed below free_gb for `checks` consecutive polls."""
    ok, told = 0, False
    while ok < checks:
        u = gpu_used_gb()
        if u is not None and u < free_gb:
            ok += 1
        else:
            ok = 0
            if not told:
                log(time.strftime('%H:%M:%S'), f'guard: GPU busy ({u} GB in use), waiting'); told = True
        time.sleep(every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--free_gb', type=float, default=3.0); ap.add_argument('--checks', type=int, default=4)
    ap.add_argument('--every', type=float, default=30.0); ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = [c for c in a.cmd if c != '--'] if a.cmd and a.cmd[0] == '--' else a.cmd
    assert cmd, 'no command given'
    if cmd[0] == 'python':                        # same interpreter as the supervisor
        cmd[0] = sys.executable
    n = 0
    while True:
        wait_free(a.free_gb, a.checks, a.every)
        run = cmd + (['--resume'] if n and '--resume' not in cmd else [])
        print(time.strftime('%H:%M:%S'), 'guard: start', ' '.join(run), flush=True)
        rc = subprocess.call(run)
        n += 1
        if rc != YIELD_CODE:
            print(time.strftime('%H:%M:%S'), 'guard: finished with exit code', rc, flush=True)
            sys.exit(rc)
        print(time.strftime('%H:%M:%S'), 'guard: trainer yielded the GPU', flush=True)


if __name__ == '__main__':
    main()

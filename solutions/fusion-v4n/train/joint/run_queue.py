"""Detached queue of joint-model ablation runs: train (resume after a yield or a shutdown), score DEV, evaluate as a ranker feature; one trainer at a time.

The queue is a JSON list in <root>/queue.json: {"tag": "j1_ref_s0", "comp_minutes": 24, "args": ["--seed", "0"], "workers": 2, "note": "..."}.
A run's state is what its folder <root>/runs/<tag> holds: last.pt (training; `done` inside when the busy-time budget is used), scores_dev.parquet,
eval_dev.json. So the queue continues where it stopped after a shutdown: start it again. Every event is appended to <root>/ledger.jsonl.
The trainer is train_joint.py under gpushare (duty 0.8, game duty 0.25, 8 GB cap, yield above 9 GB held by others: exit code 75, restarted
here when the GPU is free again). Evaluations (CPU, LightGBM) run one at a time next to the next training run. A file <root>/STOP stops the
queue after the current checkpoint. Budget guard: no new run above --max_runs runs or --max_hours of GPU wall time in the GPU ledger.

usage: python train/joint/run_queue.py --root scratch/joint1 --start       (detached; does nothing when one is alive)
       python train/joint/run_queue.py --root scratch/joint1 --status
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import jpaths as P

PY = sys.executable


def jl(root, **kw):
    with open(root + '/ledger.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), **kw)) + '\n')


def gpu_hours(prefix='j1_'):
    f = P.JOINT + '/gpu_ledger.jsonl'
    return sum(json.loads(l)['gpu_wall_s'] for l in open(f) if json.loads(l)['tag'].startswith(prefix)) / 3600 if os.path.exists(f) else 0.0


def alive(root):
    import psutil
    try:
        pid = int(open(root + '/runner.pid').read()); p = psutil.Process(pid)
        return pid if 'run_queue' in ' '.join(p.cmdline()) else 0
    except Exception:
        return 0


def trainers():
    """PIDs of running train_joint.py processes (never two trainers)."""
    import psutil
    out = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        c = ' '.join(p.info['cmdline'] or [])
        if 'train_joint.py' in c and ('--mode train' in c or '--mode mine' in c):
            out.append(p.info['pid'])
    return out


def state(root, tag):
    d = f'{root}/runs/{tag}'
    if os.path.exists(d + '/eval_dev.json'):
        return 'done'
    if os.path.exists(d + '/scores_dev.parquet'):
        return 'scored'
    if os.path.exists(d + '/TRAINED'):
        return 'trained'
    if os.path.exists(d + '/FAILED'):
        return 'failed'
    return 'training' if os.path.exists(d + '/last.pt') else 'queued'


def wait_gpu(limit_gb, checks, every=30):
    import gpushare
    ok = 0
    while ok < checks:
        u = gpushare.gpu_used_gb(); ok = ok + 1 if (u is not None and u < limit_gb) else 0; time.sleep(every)


def status(root):
    q = json.load(open(root + '/queue.json')); rows = []
    for j in q:
        d = f'{root}/runs/{j["tag"]}'; r = dict(tag=j['tag'], state=state(root, j['tag']))
        if os.path.exists(d + '/history.json'):
            h = json.load(open(d + '/history.json'))
            if h:
                r.update(step=h[-1]['step'], busy_min=round(h[-1].get('busy_s', 0) / 60, 1), wall_min=round(h[-1]['wall_s'] / 60, 1), loss=round(h[-1]['loss'], 3), acc=round(h[-1]['acc'], 3))
        if os.path.exists(d + '/eval_dev.json'):
            e = json.load(open(d + '/eval_dev.json'))
            r.update(primary=round(e['primary'], 4), se=round(e['primary_se'], 4), avg=round(e['primary_avg'], 4), c2N=round(e['c2N']['gain'], 4), aJ=round(e['c2J']['alone_mrr'], 3), aD=round(e['dropJ']['alone_mrr'], 3),
                     aN=round(e['c2N']['alone_mrr'], 3))
        rows.append(r)
    for r in rows:
        print(json.dumps(r))
    print(f'GPU hours (phase 1): {gpu_hours():.2f}; runner alive: {alive(root)}; trainers: {trainers()}')


def main(a):
    import psutil
    root = a.root; os.makedirs(root + '/runs', exist_ok=True)
    if alive(root) and alive(root) != os.getpid():
        print('a runner is alive'); return
    open(root + '/runner.pid', 'w').write(str(os.getpid())); psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS); jl(root, event='runner_start', pid=os.getpid())
    ev = None; idle = 0
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    while True:
        if os.path.exists(root + '/STOP'):
            jl(root, event='runner_stop_file'); break
        q = json.load(open(root + '/queue.json'))
        if ev is not None and ev[0].poll() is not None:
            jl(root, event='eval_end', tag=ev[1], rc=ev[0].returncode, state=state(root, ev[1])); ev = None
        if ev is None:                                                          # one evaluation at a time (CPU)
            for j in q:
                if state(root, j['tag']) == 'scored' and not os.path.exists(f'{root}/runs/{j["tag"]}/EVAL_FAILED'):
                    d = f'{root}/runs/{j["tag"]}'; lf = open(d + '/eval.log', 'a')
                    ev = (subprocess.Popen([PY, HERE + '/eval_feature.py', '--scores', d + '/scores_dev.parquet', '--half', 'dev', '--out', d, '--threads', str(a.eval_threads)], stdout=lf, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, env=env), j['tag']); jl(root, event='eval_start', tag=j['tag']); break
        job = next((j for j in q if state(root, j['tag']) in ('queued', 'training', 'trained')), None)
        if job is None:
            if ev is None:
                idle += 1
                if idle > a.idle_exit * 2:
                    jl(root, event='runner_idle_exit'); break
            time.sleep(30); continue
        idle = 0; tag = job['tag']; d = f'{root}/runs/{tag}'; os.makedirs(d, exist_ok=True); st = state(root, tag)
        if st in ('queued', 'training'):
            n_started = len([x for x in os.listdir(root + '/runs') if os.path.exists(f'{root}/runs/{x}/last.pt') and not x.startswith('t_')])
            if st == 'queued' and not tag.startswith('t_') and (gpu_hours() > a.max_hours or n_started >= a.max_runs):
                jl(root, event='budget_refused', tag=tag, gpu_hours=gpu_hours(), runs=n_started); open(d + '/FAILED', 'w').write('budget'); continue
            if trainers():
                jl(root, event='foreign_trainer_running', pids=trainers()); time.sleep(60); continue
            wait_gpu(a.busy_gb, 2, 15)
            if job.get('mine_from') and not os.path.exists(d + '/mined.npy'):       # hard negatives of an earlier run's checkpoint, mined once
                with open(d + '/mine.log', 'a') as lf:
                    rc = subprocess.call([PY, HERE + '/train_joint.py', '--mode', 'mine', '--tag', tag + '_mine', '--ckpt', f'{root}/runs/{job["mine_from"]}/last.pt', '--out', d + '/mined.npy', '--workers', '0'],
                                         stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
                jl(root, event='mine_exit', tag=tag, rc=rc)
                if rc != 0:
                    open(d + '/FAILED', 'w').write(f'mine rc {rc}'); continue
            cmd = [PY, HERE + '/train_joint.py', '--mode', 'train', '--tag', tag, '--out_root', root + '/runs', '--comp_minutes', str(job['comp_minutes']), '--workers', str(job.get('workers', 2)), '--resume',
                   '--duty', '0.8', '--game_duty', '0.25', '--max_gb', '8', '--yield_gb', '9'] + [str(x) for x in job.get('args', [])] + (['--mined', d + '/mined.npy'] if job.get('mine_from') else [])
            jl(root, event='train_start', tag=tag, resume=st == 'training', cmd=' '.join(cmd[2:])); t0 = time.time()
            with open(d + '/train.log', 'a') as lf:
                rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
            jl(root, event='train_exit', tag=tag, rc=rc, wall_s=round(time.time() - t0), gpu_hours=round(gpu_hours(), 3))
            if rc == 0:
                open(d + '/TRAINED', 'w').write(time.strftime('%Y-%m-%d %H:%M:%S'))
            elif rc == 75:
                wait_gpu(8.0, 4, 30)
            elif rc == 76:
                continue
            else:
                n = int(open(d + '/CRASHES').read()) + 1 if os.path.exists(d + '/CRASHES') else 1; open(d + '/CRASHES', 'w').write(str(n))
                if n >= 3:
                    open(d + '/FAILED', 'w').write(f'rc {rc}')
                time.sleep(20)
            continue
        if st == 'trained':
            with open(d + '/score.log', 'a') as lf:
                rc = subprocess.call([PY, HERE + '/train_joint.py', '--mode', 'score', '--tag', tag + '_score', '--ckpt', d + '/last.pt', '--half', 'dev', '--workers', '0'], stdout=lf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env)
            jl(root, event='score_exit', tag=tag, rc=rc)
            if rc != 0 or not os.path.exists(d + '/scores_dev.parquet'):
                open(d + '/FAILED', 'w').write(f'score rc {rc}')
    try:
        os.remove(root + '/runner.pid')
    except OSError:
        pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--start', action='store_true'); ap.add_argument('--status', action='store_true')
    ap.add_argument('--max_runs', type=int, default=24); ap.add_argument('--max_hours', type=float, default=14.5); ap.add_argument('--eval_threads', type=int, default=3)
    ap.add_argument('--busy_gb', type=float, default=9.0); ap.add_argument('--idle_exit', type=float, default=180, help='minutes without work before the runner exits'); a = ap.parse_args()
    a.root = os.path.abspath(a.root).replace('\\', '/')
    if a.status:
        status(a.root)
    elif a.start:
        if alive(a.root):
            print('a runner is alive:', alive(a.root)); sys.exit(0)
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        lf = open(a.root + '/runner.out', 'a'); cmd = [PY, os.path.abspath(__file__), '--root', a.root, '--max_runs', str(a.max_runs), '--max_hours', str(a.max_hours), '--eval_threads', str(a.eval_threads)]
        try:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, cwd=P.REPO, env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        except OSError:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags, cwd=P.REPO, env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        print('started', p.pid)
    else:
        main(a)

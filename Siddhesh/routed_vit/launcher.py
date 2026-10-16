"""Run (variant, seed, fold) training jobs with one process per GPU (T4 x2), resumable (finished runs are skipped).

Jobs are ordered seed-major, then fold, then variant, so an interrupted session still leaves complete seeds behind.
"""
import os
import subprocess
import sys
import time

PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_name(variant, seed, fold):
    return f'{variant}_s{seed}_f{fold}'


def run_jobs(variants, seeds, folds, samples, crops, runs_dir, n_gpus=2, extra=(), log_dir=None, per_variant_extra=None):
    """folds: iterable of fold ids (-1 = official train/val split). per_variant_extra: {variant: [extra args]} appended after `extra`."""
    log_dir = log_dir or os.path.join(runs_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    jobs = [(v, s, f) for s in seeds for f in folds for v in variants]
    running = {}   # gpu -> (Popen, name)
    env_base = dict(os.environ, PYTHONPATH=PKG_PARENT + os.pathsep + os.environ.get('PYTHONPATH', ''))
    n_gpus = max(1, n_gpus)
    while jobs or running:
        for gpu in range(n_gpus):
            if gpu not in running and jobs:
                v, s, f = jobs.pop(0)
                name = run_name(v, s, f)
                out = os.path.join(runs_dir, name)
                if os.path.exists(os.path.join(out, 'metrics.json')):
                    print(f'[skip] {name}')
                    continue
                cmd = [sys.executable, '-m', 'routed_vit.train', '--variant', v, '--seed', str(s), '--fold', str(f),
                       '--samples', samples, '--crops', crops, '--out', out, *map(str, extra),
                       *map(str, (per_variant_extra or {}).get(v, []))]
                env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu))
                running[gpu] = (subprocess.Popen(cmd, stdout=open(os.path.join(log_dir, name + '.log'), 'w'),
                                                 stderr=subprocess.STDOUT, env=env, cwd=PKG_PARENT), name)
                print(f'[start] {name} on GPU {gpu}', flush=True)
        for gpu, (p, name) in list(running.items()):
            if p.poll() is not None:
                print(f'[done]  {name} exit={p.returncode}', flush=True)
                if p.returncode != 0:
                    print(open(os.path.join(log_dir, name + '.log')).read()[-3000:])
                del running[gpu]
        time.sleep(5)

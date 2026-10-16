"""Hyper-parameter optimisation of Pedestrian Graph+ with Optuna, scored by cross-validation (never by the test split).

    python -m pedgraph.hpo --samples data/samples_ctx/jaad_beh.pkl --cache data/pose_beh --out results/round2/pedgraph_beh --trials 150 --parallel 3

Stages (all resumable: the study lives in a journal file, finished stages are skipped):
  0. baselines   : the released recipe (RELEASED_CFG) and our regularised default, 5-fold CV, seed 0 -> baselines.json
  1. search      : TPE (multivariate) + median pruner on the mean fold AUC of seed 0. Each trial = 5-fold CV of one
                   configuration (architecture, input representation, optimiser, schedule, regularisation, augmentation).
                   Image branch excluded here (3-10x more expensive); it is tested in stage 3 on the best pose configuration.
  2. re-score    : the top-K trials are re-run with fresh seeds. Picking the best of ~100 noisy scores is optimistic
                   (winner's curse); the re-score is unbiased by the search. -> stage2.csv, best_cfg.json
  3. image branch: best configuration + the local-context image (gate / concat / both) with the same fresh seeds; kept only if
                   it improves the re-scored AUC by more than `--image_margin`. -> image_stage.csv, final_cfg.json
The official test split is only predicted (by final.py), never used here.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import optuna
import pandas as pd
import torch

from routed_vit.cv import make_folds

from . import cvfit
from .cvfit import DEFAULT_CFG, RELEASED_CFG, make_cfg, run_cv, score
from .fastdata import PoseData

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial):
    c = {}
    c['pose_repr'] = trial.suggest_categorical('pose_repr', ['abs', 'rel', 'abs+rel'])
    c['kp_vel'] = trial.suggest_categorical('kp_vel', [False, True])
    c['side'] = trial.suggest_categorical('side', ['none', 'ego', 'box', 'box+ego'])
    c['fusion'] = trial.suggest_categorical('fusion', ['gate', 'concat', 'both'])
    n_layers = trial.suggest_int('n_layers', 2, 3)
    widths = [trial.suggest_categorical('w1', [16, 32, 48, 64]), trial.suggest_categorical('w2', [32, 64, 96, 128])]
    if n_layers == 3:
        widths.append(trial.suggest_categorical('w3', [64, 96, 128]))
    c['widths'] = widths
    c['kernel'] = trial.suggest_categorical('kernel', [3, 5, 7])
    c['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
    c['adj_init'] = trial.suggest_categorical('adj_init', ['identity', 'skeleton'])
    c['head_hidden'] = trial.suggest_categorical('head_hidden', [0, 32, 64])
    c['epochs'] = trial.suggest_int('epochs', 15, 80, step=5)
    c['bs'] = trial.suggest_categorical('bs', [32, 64, 128])
    c['lr'] = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
    c['wd'] = trial.suggest_float('wd', 1e-4, 0.2, log=True)
    c['sched'] = trial.suggest_categorical('sched', ['onecycle', 'cosine'])
    c['label_smooth'] = trial.suggest_float('label_smooth', 0.0, 0.15)
    c['ema'] = trial.suggest_categorical('ema', [0.0, 0.95, 0.99, 0.995])
    c['jitter'] = trial.suggest_float('jitter', 0.0, 0.05)
    c['gscale'] = trial.suggest_float('gscale', 0.0, 0.15)
    c['hflip'] = trial.suggest_float('hflip', 0.0, 0.5)
    c['joint_drop'] = trial.suggest_float('joint_drop', 0.0, 0.25)
    c['time_crop'] = trial.suggest_categorical('time_crop', [0.0, 0.25, 0.5])
    c['tta'] = trial.suggest_categorical('tta', [False, True])
    return make_cfg(**c)


def cfg_to_params(cfg):
    """Inverse of suggest() for an existing config (used to enqueue known-good starting points)."""
    p = {k: cfg[k] for k in ['pose_repr', 'kp_vel', 'side', 'fusion', 'kernel', 'dropout', 'adj_init', 'head_hidden', 'epochs', 'bs',
                             'lr', 'wd', 'sched', 'label_smooth', 'ema', 'jitter', 'gscale', 'hflip', 'joint_drop', 'time_crop', 'tta']}
    p['n_layers'] = len(cfg['widths'])
    p['w1'], p['w2'] = cfg['widths'][:2]
    if len(cfg['widths']) == 3:
        p['w3'] = cfg['widths'][2]
    return p


def cfg_from_trial(trial_params):
    class _Fixed:   # replay suggest() on stored parameters
        def __init__(self, p):
            self.p = p

        def _get(self, name, *args, **kwargs):
            return self.p[name]

        suggest_categorical = suggest_int = suggest_float = _get
    return suggest(_Fixed(trial_params))


def _dump(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--trials', type=int, default=150, help='total number of search trials (over all workers)')
    ap.add_argument('--parallel', type=int, default=1, help='worker processes sharing the GPU (small kernels: 2-4 helps)')
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--rescore_seeds', type=int, default=3)
    ap.add_argument('--image_margin', type=float, default=0.005)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_epochs', type=int, default=0, help='debug only: cap every configuration at this many epochs')
    ap.add_argument('--worker', action='store_true', help='internal: helper process that only runs search trials')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cvfit.MAX_EPOCHS = a.max_epochs or None
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = PoseData(a.samples, a.cache, dev, with_images=True)
    folds = make_folds(D.pool_rec, a.n_folds, 0)
    journal = os.path.join(a.out, 'study.journal')
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
        journal, lock_obj=optuna.storages.journal.JournalFileOpenLock(journal)))   # open-lock: symlink locks need admin rights on Windows

    def study():
        return optuna.create_study(study_name='pg', storage=storage, load_if_exists=True, direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=a.seed + (os.getpid() % 1000 if a.worker else 0),
                                                                      multivariate=True, n_startup_trials=20),
                                   pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))

    def objective(trial):
        return score(run_cv(suggest(trial), D, folds, [a.seed], trial=trial))

    # ---------------------------------------------------------------- stage 0: baselines (+ seed trials for the search)
    base_path = os.path.join(a.out, 'baselines.json')
    if not a.worker and not os.path.exists(base_path):
        out = {}
        for name, cfg in [('released_recipe', RELEASED_CFG), ('regularised_default', DEFAULT_CFG)]:
            t0 = time.time()
            r = run_cv(cfg, D, folds, [a.seed])
            out[name] = {'cv_fold_auc_mean': score(r), 'fold_auc': r['fold_auc'][0], 'minutes': (time.time() - t0) / 60}
            print(f'baseline {name}: mean fold AUC {score(r):.4f}  ({out[name]["minutes"]:.1f} min)', flush=True)
        _dump(base_path, out)
    st = study()
    if not a.worker and len(st.trials) == 0:
        st.enqueue_trial(cfg_to_params(DEFAULT_CFG))
        st.enqueue_trial(cfg_to_params(dict(DEFAULT_CFG, pose_repr='abs', kp_vel=False, side='ego', adj_init='identity',
                                            jitter=0.01, gscale=0.0, hflip=0.0, joint_drop=0.0, ema=0.0, label_smooth=0.0,
                                            dropout=0.25, wd=1e-3, lr=5e-3, epochs=30, bs=32, time_crop=0.5, tta=False)))

    # ---------------------------------------------------------------- stage 1: search
    n_done = len([t for t in st.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    if a.worker:
        st.optimize(objective, n_trials=max(1, (a.trials - n_done) // a.parallel))
        return
    remaining = max(0, a.trials - n_done)
    if remaining:
        procs = []
        for _ in range(a.parallel - 1):
            cmd = [sys.executable, '-m', 'pedgraph.hpo', *(argv if argv is not None else sys.argv[1:]), '--worker']
            procs.append(subprocess.Popen(cmd, cwd=os.getcwd()))
        st.optimize(objective, n_trials=max(1, remaining // a.parallel), callbacks=[
            lambda s, t: print(f'trial {t.number:3d} {t.state.name:8s} value={t.value if t.value is not None else float("nan"):.4f} '
                               f'best={s.best_value:.4f}', flush=True)])
        for p in procs:
            p.wait()
    st = study()
    tdf = st.trials_dataframe()
    tdf.to_csv(os.path.join(a.out, 'trials.csv'), index=False)
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f'search finished: {len(done)} complete / {len(st.trials)} trials, best stage-1 score {st.best_value:.4f}')

    # ---------------------------------------------------------------- stage 2: re-score the top-K with fresh seeds
    seeds = [1000 + i for i in range(a.rescore_seeds)]
    s2_path = os.path.join(a.out, 'stage2.csv')
    if os.path.exists(s2_path):
        s2 = pd.read_csv(s2_path)
    else:
        top = sorted(done, key=lambda t: -t.value)[:a.topk]
        rows = []
        for t in top:
            r = run_cv(cfg_from_trial(t.params), D, folds, seeds)
            rows.append({'trial': t.number, 'stage1': t.value, 'rescore': score(r), 'rescore_std_over_seeds':
                         float(np.std([np.mean(f) for f in r['fold_auc']])), 'minutes': r['minutes']})
            print(f'rescore trial {t.number}: stage1 {t.value:.4f} -> {rows[-1]["rescore"]:.4f}', flush=True)
        s2 = pd.DataFrame(rows).sort_values('rescore', ascending=False)
        s2.to_csv(s2_path, index=False)
    best_trial = int(s2.iloc[0]['trial'])
    best_cfg = cfg_from_trial(st.trials[best_trial].params)
    best_score = float(s2.iloc[0]['rescore'])
    _dump(os.path.join(a.out, 'best_cfg.json'), {'trial': best_trial, 'rescore': best_score, 'cfg': best_cfg})
    print(f'best config: trial {best_trial}, re-scored mean fold AUC {best_score:.4f}')

    # ---------------------------------------------------------------- stage 3: image branch on top of the best configuration
    img_path = os.path.join(a.out, 'image_stage.csv')
    if os.path.exists(img_path):
        idf = pd.read_csv(img_path)
    else:
        rows = [{'variant': 'pose only (best)', 'use_image': False, 'fusion': best_cfg['fusion'], 'img_aug': 0, 'score': best_score}]
        for fusion in ['gate', 'concat', 'both']:
            for img_aug in [0.5, 1.0]:
                r = run_cv(dict(best_cfg, use_image=True, fusion=fusion, img_aug=img_aug), D, folds, seeds)
                rows.append({'variant': f'+image {fusion} aug{img_aug}', 'use_image': True, 'fusion': fusion, 'img_aug': img_aug,
                             'score': score(r)})
                print(rows[-1], flush=True)
        idf = pd.DataFrame(rows)
        idf.to_csv(img_path, index=False)
    top_img = idf[idf.use_image].sort_values('score', ascending=False).iloc[0]
    final = dict(best_cfg)
    if top_img['score'] > best_score + a.image_margin:
        final.update(use_image=True, fusion=top_img['fusion'], img_aug=float(top_img['img_aug']))
        print(f'image branch kept: {top_img["variant"]} ({top_img["score"]:.4f} vs {best_score:.4f})')
    else:
        print(f'image branch not kept: best {top_img["score"]:.4f} vs pose-only {best_score:.4f} (margin {a.image_margin})')
    _dump(os.path.join(a.out, 'final_cfg.json'), {'cfg': final})


if __name__ == '__main__':
    main()

"""Final cross-validated evaluation of the tuned Pedestrian Graph+ (and its baselines / ablations / config ensemble).

    python -m pedgraph.final --samples data/samples_ctx/jaad_beh.pkl --cache data/pose_beh --hpo results/round2/pedgraph_beh --out results/round2/pedgraph_beh/final \\
                             --seeds 5 --baseline_seeds 3 --ablate --ensemble_topk 3

Everything is K-fold cross-validated over the pooled train+val pedestrians (video-grouped); the official test split is predicted
by the fold models and averaged (seeds x folds). Writes preds.npz (OOF + test probabilities per model), summary.csv, ablation.csv.
"""
import argparse
import json
import os

import numpy as np
import optuna
import pandas as pd
import torch

from routed_vit.cv import make_folds

from . import cvfit
from .cvfit import DEFAULT_CFG, RELEASED_CFG, run_cv, score
from .fastdata import PoseData
from .hpo import cfg_from_trial
from .report import majority_line, table


def ablation_variants(cfg):
    """One-factor-at-a-time removals / swaps of the tuned configuration (only those that change something)."""
    v = {}
    if cfg['use_image']:
        v['no image branch'] = dict(use_image=False)
    else:
        v['+ image branch (gate)'] = dict(use_image=True, fusion='gate')
    if cfg['side'] != 'none':
        v['no side branch (box/ego)'] = dict(side='none')
    if cfg['side'] != 'ego':
        v['side = ego only'] = dict(side='ego')
    if cfg['kp_vel']:
        v['no keypoint velocity'] = dict(kp_vel=False)
    for rep in ['abs', 'rel', 'abs+rel']:
        if cfg['pose_repr'] != rep:
            v[f'pose repr = {rep}'] = dict(pose_repr=rep)
    if cfg['adj_init'] != 'identity':
        v['identity adjacency init'] = dict(adj_init='identity')
    if cfg['ema'] > 0:
        v['no EMA'] = dict(ema=0.0)
    if cfg['tta']:
        v['no flip-TTA'] = dict(tta=False)
    if cfg['label_smooth'] > 0:
        v['no label smoothing'] = dict(label_smooth=0.0)
    v['no augmentation'] = dict(jitter=0.0, gscale=0.0, hflip=0.0, joint_drop=0.0, time_crop=0.0, img_aug=0.0)
    return {k: dict(cfg, **d) for k, d in v.items()}


def top_configs(hpo_dir, k):
    """Configs of the k best re-scored trials (stage2.csv)."""
    journal = os.path.join(hpo_dir, 'study.journal')
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
        journal, lock_obj=optuna.storages.journal.JournalFileOpenLock(journal)))
    st = optuna.load_study(study_name='pg', storage=storage)
    s2 = pd.read_csv(os.path.join(hpo_dir, 'stage2.csv')).sort_values('rescore', ascending=False).head(k)
    return [cfg_from_trial(st.trials[int(t)].params) for t in s2.trial]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--hpo', required=True, help='output dir of pedgraph.hpo (final_cfg.json, stage2.csv, study.journal)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=5, help='training seeds for the tuned model (x folds models are ensembled)')
    ap.add_argument('--baseline_seeds', type=int, default=3)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--ablate', action='store_true')
    ap.add_argument('--ablation_seeds', type=int, default=3)
    ap.add_argument('--ensemble_topk', type=int, default=0, help='also ensemble the top-k re-scored configurations')
    ap.add_argument('--max_epochs', type=int, default=0, help='debug only: cap every configuration at this many epochs')
    a = ap.parse_args(argv)
    cvfit.MAX_EPOCHS = a.max_epochs or None
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = PoseData(a.samples, a.cache, dev, with_images=True)
    folds = make_folds(D.pool_rec, a.n_folds, 0)
    cfg = json.load(open(os.path.join(a.hpo, 'final_cfg.json')))['cfg']
    seeds = [2000 + i for i in range(a.seeds)]          # disjoint from the search / re-score seeds
    results = {}

    def go(name, c, n_seeds):
        print(f'== {name} ({n_seeds} seeds x {a.n_folds} folds)', flush=True)
        results[name] = run_cv(c, D, folds, seeds[:n_seeds], verbose=False)
        r = results[name]
        print(f'   mean fold AUC {score(r):.4f}  ({r["minutes"]:.1f} min)', flush=True)

    go('released recipe (CV)', RELEASED_CFG, a.baseline_seeds)
    go('regularised default', DEFAULT_CFG, a.baseline_seeds)
    go('TUNED', cfg, a.seeds)
    if a.ensemble_topk:
        tops = top_configs(a.hpo, a.ensemble_topk)
        parts = [run_cv(c, D, folds, seeds[:max(2, a.seeds // 2)]) for c in tops]
        results[f'ensemble of top-{a.ensemble_topk} configs'] = {
            'oof': np.concatenate([p['oof'] for p in parts]), 'test': np.concatenate([p['test'] for p in parts]),
            'fold_auc': sum([p['fold_auc'] for p in parts], []), 'minutes': sum(p['minutes'] for p in parts)}
    df, compact = table(results, D)
    df.to_csv(os.path.join(a.out, 'summary.csv'), index=False)
    print(majority_line(D))
    print(compact.round(4).to_string())
    np.savez_compressed(os.path.join(a.out, 'preds.npz'), y_pool=D.y_np[D.pool_rows], y_test=D.y_np[D.test_rows],
                        **{f'{k}__{part}': v[part] for k, v in results.items() for part in ('oof', 'test')})
    json.dump({'tuned_cfg': cfg, 'released_cfg': RELEASED_CFG}, open(os.path.join(a.out, 'configs.json'), 'w'), indent=1)
    if a.ablate:
        abl = {}
        for name, c in ablation_variants(cfg).items():
            abl[name] = run_cv(c, D, folds, seeds[:a.ablation_seeds])
            print(f'ablation {name}: mean fold AUC {score(abl[name]):.4f}', flush=True)
        abl['TUNED (reference)'] = results['TUNED'] if a.ablation_seeds >= a.seeds else run_cv(cfg, D, folds, seeds[:a.ablation_seeds])
        adf, acompact = table(abl, D)
        adf.to_csv(os.path.join(a.out, 'ablation.csv'), index=False)
        print(acompact.round(4).to_string())


if __name__ == '__main__':
    main()

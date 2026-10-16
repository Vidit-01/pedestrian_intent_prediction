"""Train / evaluate the GTransPDM reimplementation (gtranspdm.py) on one JAAD subset.

    python -m pedgraph.run_gtrans --samples data/samples_ctx/jaad_all.pkl --cache data/pose_all --out results/round2/gtranspdm/all
    python -m pedgraph.run_gtrans --samples data/samples_ctx/jaad_beh.pkl --cache data/pose_beh --out results/round2/gtranspdm/beh

Two protocols, both with a FIXED 32-epoch budget and the model of the last epoch (no checkpoint picking):
  paper  : the official JAAD train/val/test split, `--seeds` seeds. This is the paper's own setting (JAAD_all, default split), so
           accuracy / AUC / F1 / precision at threshold 0.5 are comparable in kind with its table (see caveats in gtranspdm.py).
  cv     : 5 video-grouped folds over train+val x `--cv_seeds` seeds, test predicted by the fold ensemble. Same protocol as the
           Pedestrian Graph+ search, so the two models are directly comparable (out-of-fold AUC and test AUC).
`--with_pg` also runs the released Pedestrian Graph+ recipe (RELEASED_CFG) under the same two protocols.
Writes summary_paper.csv / summary_cv.csv and preds_*.npz into --out.
"""
import argparse
import os

import numpy as np
import torch

from routed_vit.cv import make_folds

from .cvfit import GTRANS_CFG, RELEASED_CFG, run_cv, score
from .fastdata import PoseData
from .report import majority_line, table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=5, help='seeds for the paper protocol')
    ap.add_argument('--cv_seeds', type=int, default=3)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--with_pg', action='store_true', help='also run the released Pedestrian Graph+ recipe for comparison')
    ap.add_argument('--skip_cv', action='store_true')
    ap.add_argument('--skip_paper', action='store_true', help='resume helper: the paper-protocol part already finished')
    ap.add_argument('--max_epochs', type=int, default=0, help='debug only')
    a = ap.parse_args(argv)
    from . import cvfit
    cvfit.MAX_EPOCHS = a.max_epochs or None
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = PoseData(a.samples, a.cache, dev, with_images=a.with_pg)
    print(f'{a.samples}: pooled train+val {D.n_pool} windows (official train {D.n_train}), test {D.n_test}; {majority_line(D)}', flush=True)
    models = {'GTransPDM (ours, from paper)': GTRANS_CFG}
    if a.with_pg:
        models['Pedestrian Graph+ (released recipe)'] = RELEASED_CFG
    seeds = [3000 + i for i in range(a.seeds)]

    # ---- paper protocol: official split
    official = [(np.arange(D.n_train), np.arange(D.n_train, D.n_pool))]
    res = {}
    for name, cfg in models.items():
        if a.skip_paper:
            break
        res[name] = run_cv(cfg, D, official, seeds)
        print(f'[paper protocol] {name}: val AUC {score(res[name]):.4f} ({res[name]["minutes"]:.1f} min)', flush=True)
    if res:
        df, compact = table(res, D)
        df.to_csv(os.path.join(a.out, 'summary_paper.csv'), index=False)
        np.savez_compressed(os.path.join(a.out, 'preds_paper.npz'), y_test=D.y_np[D.test_rows],
                            **{f'{k}__test': v['test'] for k, v in res.items()})
        print(compact.round(4).to_string(), flush=True)

    # ---- CV protocol
    if not a.skip_cv:
        folds = make_folds(D.pool_rec, a.n_folds, 0)
        res = {}
        for name, cfg in models.items():
            res[name] = run_cv(cfg, D, folds, [4000 + i for i in range(a.cv_seeds)])
            print(f'[cv protocol] {name}: mean fold AUC {score(res[name]):.4f} ({res[name]["minutes"]:.1f} min)', flush=True)
        df, compact = table(res, D)
        df.to_csv(os.path.join(a.out, 'summary_cv.csv'), index=False)
        np.savez_compressed(os.path.join(a.out, 'preds_cv.npz'), y_pool=D.y_np[D.pool_rows], y_test=D.y_np[D.test_rows],
                            **{f'{k}__{p}': v[p] for k, v in res.items() for p in ('oof', 'test')})
        print(compact.round(4).to_string(), flush=True)


if __name__ == '__main__':
    main()

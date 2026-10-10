"""Control baselines (Vidit.md 3.6): majority class and a non-temporal logistic regression on the LAST observed
frame's kinematic features. CPU only. Writes metrics.json in the same format as train.py.

    python simple_baselines.py --samples .../jaad_beh.pkl --out runs/simple_beh
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasets import kinematic_features, load_samples
from evaluate import compute_metrics, select_threshold

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--kin_feats', nargs='+', default=['box', 'cwh', 'speed'])
    args = ap.parse_args()
    data = load_samples(args.samples)

    # majority class of the training split, predicted with probability = train positive rate
    prior = data['train']['label'].mean()
    pred = {s: np.full(len(data[s]['label']), prior) for s in ['val', 'test']}
    res = {'majority': {s: compute_metrics(data[s]['label'], pred[s]) for s in ['val', 'test']}}

    X = {s: kinematic_features(data[s], args.kin_feats)[:, -1] for s in ['train', 'val', 'test']}
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X['train'], data['train']['label'])
    p = {s: clf.predict_proba(X[s])[:, 1] for s in ['val', 'test']}
    thr = select_threshold(data['val']['label'], p['val'])
    res['logreg_last_frame'] = {s: compute_metrics(data[s]['label'], p[s], thr) for s in ['val', 'test']}

    for name, r in res.items():
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'metrics.json'), 'w') as f:
            json.dump(r, f, indent=1)
        t = r['test']
        print(f'{name:18s} TEST acc {t["bench_acc"]:.3f} auc(bench) {t["bench_auc"]:.3f} f1 {t["bench_f1"]:.3f} '
              f'| true auc {t["auc"]:.3f} bal_acc@val-thr {t["thr_balanced_acc"]:.3f}')

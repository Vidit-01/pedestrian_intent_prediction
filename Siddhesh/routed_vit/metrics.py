"""Metrics mirroring the repo's evaluate.py (kept separate: `evaluate` collides with the HF package on Kaggle),
plus cluster (per-pedestrian) bootstrap CIs and a paired bootstrap for AUC differences."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)


def select_threshold(y, p):
    """Threshold maximising balanced accuracy; call on VALIDATION predictions only."""
    cands = np.unique(np.concatenate([p, [0.5]]))
    return float(cands[int(np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in cands]))])


def _thr(y, p, thr, pre):
    yh = (p >= thr).astype(int)
    return {pre + 'acc': accuracy_score(y, yh), pre + 'balanced_acc': balanced_accuracy_score(y, yh),
            pre + 'f1': f1_score(y, yh, zero_division=0), pre + 'precision': precision_score(y, yh, zero_division=0),
            pre + 'recall': recall_score(y, yh, zero_division=0)}


def compute_metrics(y, p, thr=0.5):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=np.float64)
    two = len(np.unique(y)) == 2
    m = {'n': len(y), 'pos': int(y.sum()), 'auc': roc_auc_score(y, p) if two else float('nan'),
         'bench_auc': roc_auc_score(y, np.round(p)) if two else float('nan'), 'brier': brier_score_loss(y, p),
         'threshold': thr}
    m.update(_thr(y, p, 0.5, 'bench_'))
    m.update(_thr(y, p, thr, 'thr_'))
    return {k: float(v) for k, v in m.items()}


def _clusters(groups):
    _, inv = np.unique(groups, return_inverse=True)
    return [np.where(inv == g)[0] for g in range(inv.max() + 1)]


def bootstrap_metrics(y, p, groups, thr, n_boot=1000, seed=0, keys=('auc', 'bench_acc', 'bench_f1', 'thr_f1')):
    """95% percentile CIs, resampling whole pedestrians (windows of one pedestrian are highly correlated)."""
    rng = np.random.default_rng(seed)
    cl = _clusters(groups)
    pt = compute_metrics(y, p, thr)
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([cl[i] for i in rng.integers(0, len(cl), len(cl))])
        if len(np.unique(y[idx])) < 2:
            continue
        m = compute_metrics(y[idx], p[idx], thr)
        for k in keys:
            boots[k].append(m[k])
    return {k: (pt[k], float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boots.items()}


def paired_auc_diff(y, preds_a, preds_b, groups, n_boot=1000, seed=0):
    """AUC_B - AUC_A averaged over seeds (preds_*: list of [n] arrays, one per seed), same pedestrian resamples."""
    rng = np.random.default_rng(seed)
    cl = _clusters(groups)
    d0 = np.mean([roc_auc_score(y, b) for b in preds_b]) - np.mean([roc_auc_score(y, a) for a in preds_a])
    ds = []
    for _ in range(n_boot):
        idx = np.concatenate([cl[i] for i in rng.integers(0, len(cl), len(cl))])
        if len(np.unique(y[idx])) < 2:
            continue
        ds.append(np.mean([roc_auc_score(y[idx], b[idx]) for b in preds_b]) -
                  np.mean([roc_auc_score(y[idx], a[idx]) for a in preds_a]))
    return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

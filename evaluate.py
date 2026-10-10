"""Shared evaluation for every model (Baseline A, ACIT, future models).

Metrics (compute_metrics):
  auc          : true ROC-AUC on probabilities
  bench_*      : PCPA-benchmark style at threshold 0.5 -> acc, f1, precision, recall and bench_auc, which is
                 roc_auc_score on np.round(p) (that is what the benchmark and the ACIT paper's tables report)
  thr_*        : same threshold metrics at a threshold chosen on VALIDATION (max balanced accuracy)
  balanced_acc, brier

Aggregate seeds:  python evaluate.py runs/gru_box_s0 runs/gru_box_s1 runs/gru_box_s2
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)


def threshold_metrics(y, p, thr, prefix):
    yh = (p >= thr).astype(int)
    return {f'{prefix}acc': accuracy_score(y, yh), f'{prefix}balanced_acc': balanced_accuracy_score(y, yh),
            f'{prefix}f1': f1_score(y, yh, zero_division=0), f'{prefix}precision': precision_score(y, yh, zero_division=0),
            f'{prefix}recall': recall_score(y, yh, zero_division=0)}


def select_threshold(y, p):
    """Threshold maximising balanced accuracy. Call on validation predictions only."""
    cands = np.unique(np.concatenate([p, [0.5]]))
    scores = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in cands]
    return float(cands[int(np.argmax(scores))])


def compute_metrics(y, p, thr=0.5):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=np.float64)
    two = len(np.unique(y)) == 2
    m = {'n': int(len(y)), 'pos': int(y.sum()), 'auc': roc_auc_score(y, p) if two else float('nan'),
         'bench_auc': roc_auc_score(y, np.round(p)) if two else float('nan'), 'brier': brier_score_loss(y, p),
         'threshold': thr}
    m.update(threshold_metrics(y, p, 0.5, 'bench_'))
    m.update(threshold_metrics(y, p, thr, 'thr_'))
    return {k: float(v) for k, v in m.items()}


def aggregate(run_dirs, split='test'):
    rows = []
    for d in run_dirs:
        with open(os.path.join(d, 'metrics.json')) as f:
            rows.append(json.load(f)[split])
    keys = ['bench_acc', 'bench_auc', 'bench_f1', 'bench_precision', 'bench_recall', 'auc', 'thr_balanced_acc', 'thr_f1', 'brier']
    out = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in keys}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--split', default='test')
    args = ap.parse_args()
    agg = aggregate(args.runs, args.split)
    print(f'{len(args.runs)} runs, split={args.split}  (mean +- std)')
    for k, (m, s) in agg.items():
        print(f'  {k:18s} {m:.3f} +- {s:.3f}')

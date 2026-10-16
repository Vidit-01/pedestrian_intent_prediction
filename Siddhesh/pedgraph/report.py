"""Result tables for cross-validated Pedestrian Graph+ runs (same columns as routed_vit.reports.results_table)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from routed_vit.metrics import bootstrap_metrics, compute_metrics, select_threshold


def summarize(name, res, D, n_boot=500):
    """One row per training seed plus an 'ens' row (all seeds and folds averaged, bootstrap CIs over test pedestrians).
    oof_auc = pooled out-of-fold AUC on the 216 train+val pedestrians (selection signal); the threshold for acc/F1 is the
    balanced-accuracy optimum on the OOF predictions, never on test."""
    y_pool, y_te = D.y_np[D.pool_rows], D.y_np[D.test_rows]
    g_te = D.groups[D.test_rows]
    rows = []
    for si in range(res['oof'].shape[0]):
        mk = ~np.isnan(res['oof'][si])          # official-split runs only score the val rows
        thr = select_threshold(y_pool[mk], res['oof'][si][mk])
        m = compute_metrics(y_te, res['test'][si], thr)
        rows.append({'model': name, 'seed': si, 'oof_auc': roc_auc_score(y_pool[mk], res['oof'][si][mk]),
                     'fold_auc': f"{np.mean(res['fold_auc'][si]):.3f}±{np.std(res['fold_auc'][si]):.3f}",
                     'test_auc': m['auc'], 'bench_acc': m['bench_acc'], 'bench_f1': m['bench_f1'], 'bench_precision': m['bench_precision'], 'bench_recall': m['bench_recall'], 'thr': thr,
                     'thr_bacc': m['thr_balanced_acc'], 'thr_f1': m['thr_f1'], 'bench_auc': m['bench_auc'], 'brier': m['brier']})
    oof, test = np.nanmean(res['oof'], 0), res['test'].mean(0)
    mk = ~np.isnan(oof)
    thr = select_threshold(y_pool[mk], oof[mk])
    ci = bootstrap_metrics(y_te, test, g_te, thr, n_boot=n_boot, keys=('auc', 'bench_acc', 'thr_f1'))
    m = compute_metrics(y_te, test, thr)
    rows.append({'model': name, 'seed': 'ens', 'oof_auc': roc_auc_score(y_pool[mk], oof[mk]), 'fold_auc': '',
                 'test_auc': m['auc'], 'test_auc_ci': '[%.3f, %.3f]' % ci['auc'][1:], 'bench_acc': m['bench_acc'],
                 'bench_f1': m['bench_f1'], 'bench_precision': m['bench_precision'], 'bench_recall': m['bench_recall'], 'thr': thr, 'thr_bacc': m['thr_balanced_acc'], 'thr_f1': m['thr_f1'],
                 'bench_auc': m['bench_auc'], 'brier': m['brier']})
    return rows


def majority_line(D):
    y = D.y_np[D.test_rows]
    p = y.mean()
    return (f'test positive rate {p:.3f}: always predicting "crosses" gives accuracy {max(p, 1 - p):.3f} and F1 '
            f'{2 * p / (1 + p):.3f}; acc/F1 at a fixed threshold are only meaningful above that, AUC / balanced accuracy always are.')


def table(results, D, n_boot=500):
    """results: {name: run_cv output} -> DataFrame (all rows) and a compact per-model summary (ens rows)."""
    rows = []
    for name, r in results.items():
        rows += summarize(name, r, D, n_boot)
    df = pd.DataFrame(rows)
    return df, df[df.seed == 'ens'].set_index('model')[['oof_auc', 'test_auc', 'test_auc_ci', 'thr_bacc', 'thr_f1', 'bench_acc', 'bench_f1', 'bench_precision']]

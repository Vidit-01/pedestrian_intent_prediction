"""Generates the three Kaggle notebooks (round 2, cross-validated protocol); the package source is embedded with %%writefile,
so a notebook needs no repo checkout:
    kaggle/A/jaad_dense_vit_A.ipynb    Model A  - PedViViT-style dense ViT, recipe search, box-only control, Phase-A probes
    kaggle/B/jaad_routed_vit_B.ipynb   Model B  - B-soft / B-sparse / FiLM control + routing analysis + hypotheses H1-H5
    kaggle/C/jaad_pedgraph_C.ipynb     Model C  - Pedestrian Graph+ (Optuna search, final CV evaluation, ablations)
Run:  python build_notebooks.py        (rerun after editing any file of routed_vit/ or pedgraph/)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = ['__init__.py', 'model.py', 'metrics.py', 'cv.py', 'recipes.py', 'data.py', 'train.py', 'launcher.py', 'analysis.py', 'reports.py']


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(True)}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n').splitlines(True)}


def package_cells():
    cells = [md('## 1. Code\nThe `routed_vit` package (model, data, training, analysis) is embedded below with `%%writefile`, so the notebook is '
                'self-contained. Only the public JAAD tooling of the base repo (`download_jaad.py`, `build_jaad_samples.py`) is cloned.')]
    cells.append(code('''
!pip install -q -U timm
!git clone -q --depth 1 https://github.com/Vidit-01/pedestrian_intent_prediction.git /kaggle/working/repo || true
!mkdir -p /kaggle/working/code/routed_vit /kaggle/working/out
!nvidia-smi -L && df -h /kaggle/working /tmp | head -5
'''))
    for f in PKG:
        src = open(os.path.join(HERE, 'routed_vit', f), encoding='utf-8').read()
        cells.append(code(f'%%writefile /kaggle/working/code/routed_vit/{f}\n' + src))
    return cells


CONFIG = '''
import os, sys, json, glob, shutil, pickle, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
WORK = '/kaggle/working'
sys.path.insert(0, f'{WORK}/code')
REPO, JAAD, OUT = f'{WORK}/repo', f'{WORK}/JAAD', f'{WORK}/out'
RUNS = f'{OUT}/runs'
SAMPLE_TYPE = 'beh'            # 'beh' (JAAD_beh, default) or 'all' (JAAD_all, ~4x more samples/crops)
SEEDS = [42, 7, 97]            # training seeds
N_FOLDS = 5                    # video-grouped CV folds over train+val (216 pedestrians); the official test split is only predicted
FOLDS = list(range(N_FOLDS))
MICRO_BS, WORKERS = 16, 2      # per GPU process; 16 clips x 8 frames = 128 images / step (lower MICRO_BS on OOM)
BASE_EXTRA = ['--n_folds', N_FOLDS, '--micro_bs', MICRO_BS, '--workers', WORKERS]
CROPS = f'{WORK}/crops_{SAMPLE_TYPE}'
SAMPLES = f'{OUT}/jaad_{SAMPLE_TYPE}.pkl'
os.makedirs(RUNS, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('GPUs:', torch.cuda.device_count(), '| samples:', SAMPLES)
'''

DATA_CELLS = [
    md('## 3. Data (JAAD, same PCPA protocol as the base repo)\nSamples = 16 observed frames, TTE 30-60, stride 3, official video split. A clip uses 8 of the 16 frames '
       '(odd indices, last frame included) as 224x224 pedestrian-centred square crops (1.5x box). Annotated context (never a model input) is read from the JAAD annotations.'),
    code('''
# JAAD annotations + split ids + jaad_data.py; the 346 mp4s come from the attached Kaggle dataset (no download), fallback = official zip
clips = glob.glob('/kaggle/input/**/JAAD_clips', recursive=True)
cmd = f'--clips_from {clips[0]}' if clips else '--clips'
print('clips source:', clips[0] if clips else 'downloading official zip (3.1 GB)')
!python {REPO}/download_jaad.py --root {JAAD} {cmd}
'''),
    code('''
from routed_vit.data import build_samples, load_samples, context_table
# reuse samples/crops from an attached earlier notebook output if available
prev = glob.glob(f'/kaggle/input/**/out/jaad_{SAMPLE_TYPE}.pkl', recursive=True)
if os.path.exists(SAMPLES):
    data = load_samples(SAMPLES)
elif prev:
    shutil.copy(prev[0], SAMPLES); data = load_samples(SAMPLES); print('reused', prev[0])
else:
    data = build_samples(REPO, JAAD, SAMPLES, SAMPLE_TYPE)
for s in ['train', 'val', 'test']:
    r = data[s]; print(f"{s}: windows={len(r['label'])} peds={len(set(zip(r['vid'], r['pid'])))} pos={int(r['label'].sum())}")
context_table(data)
'''),
    code('''
from routed_vit.data import build_crop_cache, CropDataset
prev = glob.glob(f'/kaggle/input/**/crops_{SAMPLE_TYPE}', recursive=True)
if prev and not os.path.exists(CROPS):
    CROPS = prev[0]; print('reusing crop cache', CROPS)
else:
    build_crop_cache(data, f'{JAAD}/JAAD_clips', CROPS, workers=4)
print('crop cache:', CROPS, f'{sum(os.path.getsize(os.path.join(CROPS, f)) for f in os.listdir(CROPS)) / 1e9:.2f} GB')
'''),
    code('''
# look at what the model sees: 8 crops of 3 training clips (pos / neg)
ds = CropDataset(data['train'], CROPS)
fig, ax = plt.subplots(3, 8, figsize=(16, 6.4))
for r, i in enumerate(np.random.default_rng(0).choice(len(ds), 3, replace=False)):
    b = ds[int(i)]
    for t in range(8):
        ax[r, t].imshow(b['x'][t].permute(1, 2, 0).numpy()); ax[r, t].axis('off')
    ax[r, 0].set_title(f"label={int(b['y'])}", fontsize=8, loc='left')
plt.tight_layout(); plt.show()
'''),
]


FOLD_CELL = code('''
from routed_vit.cv import pool_trainval, make_folds, describe_folds
pool = pool_trainval(data)
N_POOL, N_TEST = len(pool['label']), len(data['test']['label'])
display(pd.DataFrame(describe_folds(pool, make_folds(pool, N_FOLDS, 0))))
''')

RES_COLS = ("[c for c in ['variant', 'seed', 'folds', 'oof_auc', 'fold_auc', 'test_auc', 'test_auc_ci', 'bench_acc', 'bench_f1', "
            "'thr', 'thr_bacc', 'thr_f1', 'bench_auc'] if c in {df}]")

NB_A = [
    md('''
# Model A — dense PedViViT-style baseline (JAAD), round 2: cross-validated, regularised, selection-free

Implements the **dense** half of `idea.md`, built on **VIT-Ped / PedViViT** (arXiv 2601.01989): a ViViT factorised encoder (per-frame ViT-S/16 spatial encoder -> 2-layer temporal transformer over per-frame CLS tokens) on the pedestrian crop, plus a small non-visual transformer on delta-encoded bounding boxes (2 layers, 4 heads), concatenation + FFN -> one sigmoid unit.
Ego speed is deliberately **not** an input, because it is one of the analysed contexts.

**What changed after round 1.** Round 1 selected the best epoch on the 22-pedestrian validation split; every run peaked at epoch 1 (train loss 1e-5 by epoch 4), so the baseline was a one-epoch model and the routed-vs-dense comparison meant little. Now:
1. **Evaluation:** train+val are pooled (216 pedestrians) and split into 5 **video-grouped folds**. Each fold model is trained on 4 folds and scored on the held-out one (out-of-fold, OOF, AUC = selection-free estimate on all 216 pedestrians); the official **test split is only predicted** and the fold models are averaged. No test-set selection anywhere.
2. **Training:** fixed budget, no early stopping / checkpoint picking; warm-up + cosine to ~0; the **EMA of the weights at the end of the schedule** is the model. Lower LR with layer-wise decay, stochastic depth, label smoothing, clip-consistent affine jitter, box noise and box-branch modality dropout.
3. **Recipe search:** four fine-tuning recipes (`routed_vit/recipes.py`) are compared by pooled OOF AUC (seed 42); the winner is trained with all seeds and written to `recipe.json`, which notebook B re-uses.
4. **Controls:** a **box-only** model (no pixels) shows how much of the signal is bounding-box dynamics.

**Contents:** data -> folds -> recipe search -> 3 seeds x 5 folds with the winner -> CV/test metrics -> box-only control -> Phase-A probes (head importance by mean ablation, per-head gradient share across context).
Outputs (`/kaggle/working/out`) are consumed by notebook B (attach this notebook's output as an input to B).

**Kaggle settings:** Accelerator **GPU T4 x2** - Internet **On** - Add Input -> Dataset `christopheronyiuke/jaad-video`. Expected run time: about 3-4 h (recipe search 4 x 5 folds, then 2 more seeds x 5 folds). Resumable: finished runs are skipped.
'''),
    *package_cells(),
    md('## 2. Config'),
    code(CONFIG + '''
SWEEP_SEED = 42            # seed used for the recipe search
'''),
    *DATA_CELLS,
    md('### Cross-validation folds (video-grouped, stratified by label; identical for every model)'),
    FOLD_CELL,
    md('## 4. Sanity checks (shapes, parameters, memory)'),
    code('''
from routed_vit.model import RoutedViT, count_params
from routed_vit.data import prep
m = RoutedViT('dense', drop_path=0.1).to(device).train()
x = prep(torch.randint(0, 255, (MICRO_BS, 8, 3, 224, 224), dtype=torch.uint8, device=device), train=True)
bx = torch.randn(MICRO_BS, 8, 4, device=device)
with torch.autocast('cuda', dtype=torch.float16):
    logit, info = m(x, box=bx)
logit.float().mean().backward()
print('params: %.2fM' % (count_params(m)[0] / 1e6), '| logit', tuple(logit.shape),
      '| peak GPU mem %.1f GB at %d clips/step' % (torch.cuda.max_memory_allocated() / 1e9, MICRO_BS))
del m, x, logit, info; torch.cuda.empty_cache()
'''),
    md('''## 5. Recipe search (dense model, seed 42, 5-fold CV)
Four fine-tuning recipes, same folds, same everything else. Selection rule: highest **pooled OOF AUC** (each window scored by a model that never saw its video). With four candidates on 216 pedestrians the choice is itself noisy; the table shows the margins. The test AUC column is printed for transparency only and is not used.
'''),
    code('''
from routed_vit.launcher import run_jobs
from routed_vit.recipes import RECIPES, DEFAULT_RECIPE
from routed_vit.reports import collect
from sklearn.metrics import roc_auc_score
SWEEP = f'{OUT}/sweep'
ngpu = torch.cuda.device_count() or 1
for name, rargs in RECIPES.items():
    run_jobs(['dense'], [SWEEP_SEED], FOLDS, SAMPLES, CROPS, f'{SWEEP}/{name}', n_gpus=ngpu, extra=BASE_EXTRA + rargs)
y_pool = np.asarray(pool['label'])
rows = []
for name in RECIPES:
    c = collect(f'{SWEEP}/{name}', 'dense', [SWEEP_SEED], FOLDS, N_POOL, N_TEST).get(SWEEP_SEED)
    if c and c['folds_done'] == N_FOLDS:
        rows.append({'recipe': name, 'oof_auc': roc_auc_score(y_pool, c['oof']), 'fold_auc_mean': np.mean(c['fold_auc']),
                     'fold_auc_std': np.std(c['fold_auc']), 'test_auc (diagnostic only)': roc_auc_score(data['test']['label'], c['test'])})
sweep_df = pd.DataFrame(rows).sort_values('oof_auc', ascending=False)
display(sweep_df.round(4))
BEST = sweep_df.iloc[0]['recipe'] if len(sweep_df) else DEFAULT_RECIPE
print('selected recipe:', BEST, RECIPES[BEST])
json.dump({'recipe': BEST, 'args': RECIPES[BEST], 'sweep': sweep_df.to_dict('records')}, open(f'{OUT}/recipe.json', 'w'), indent=1, default=float)
'''),
    md('## 6. Train Model A with the selected recipe (3 seeds x 5 folds; seed 42 is re-used from the search)\nResumable: finished runs are skipped.'),
    code('''
for f in FOLDS:   # re-use the search runs of the winning recipe
    src, dst = f'{SWEEP}/{BEST}/dense_s{SWEEP_SEED}_f{f}', f'{RUNS}/dense_s{SWEEP_SEED}_f{f}'
    if os.path.exists(src + '/metrics.json') and not os.path.exists(dst):
        shutil.copytree(src, dst)
run_jobs(['dense'], SEEDS, FOLDS, SAMPLES, CROPS, RUNS, n_gpus=ngpu, extra=BASE_EXTRA + RECIPES[BEST])
'''),
    md('## 7. Results\nOut-of-fold AUC (216 pedestrians, selection-free), per-fold spread, and the test AUC of the fold ensemble with 95% pedestrian-cluster bootstrap CIs (row `ens` = all seeds and folds). Thresholds for acc/F1 come from the OOF predictions, never from test. Compare acc/F1 with the majority-class numbers printed below the table: only AUC and balanced accuracy carry signal.'),
    code('''
from routed_vit.reports import results_table, plot_history
plot_history(RUNS, ['dense'], SEEDS, FOLDS)
res_A = results_table(RUNS, ['dense'], SEEDS, FOLDS, data)
res_A[''' + RES_COLS.format(df='res_A') + ''']
'''),
    md('''### Control: bounding boxes only (no pixels)
Same folds and seeds, 30 epochs (the box branch is tiny). If the full model is not clearly above this, the visual branch adds little on this protocol.
'''),
    code('''
RUNS_BOX = f'{OUT}/runs_boxonly'
run_jobs(['dense'], SEEDS, FOLDS, SAMPLES, CROPS, RUNS_BOX, n_gpus=ngpu,
         extra=BASE_EXTRA + ['--box_only', '--epochs', 30, '--head_lr', 1e-3, '--box_drop', 0, '--workers', 0])
res_box = results_table(RUNS_BOX, ['dense'], SEEDS, FOLDS, data)
res_box.assign(variant='box-only')[['variant', 'seed', 'folds', 'oof_auc', 'fold_auc', 'test_auc', 'test_auc_ci', 'thr_bacc', 'thr_f1']]
'''),
    md('## 8. Phase-A probes (dense model, fold 0 of each seed)\n**Head importance** $I^l_{i,h}=|u(x_i)-u(x_i;O^{l,h}\\leftarrow\\bar O^{l,h})|$ (mean ablation) on 300 random test clips, and **gradient norms** $\\|\\nabla_{\\theta^{l,h}}\\ell_i\\|$ for the blocks that B will route (7-12). Saved to `runs/dense_s*_f0/probes.npz`.'),
    code('''
from routed_vit.reports import head_probes, grad_entropy_table, importance_by_context, heatmap
from routed_vit.data import CTX_NAMES, CTX_LEVELS
ANA_FOLD = 0
probesA = {s: head_probes(f'{RUNS}/dense_s{s}_f{ANA_FOLD}', data, CROPS, device, n_samples=300) for s in SEEDS if os.path.exists(f'{RUNS}/dense_s{s}_f{ANA_FOLD}/metrics.json')}
imp = np.mean([p['importance'].mean(0) for p in probesA.values()], 0)       # [12 layers, 6 heads]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
heatmap(imp, 'mean head importance |du| (A, mean over seeds)', ax=ax[0], ylabels=range(1, 13))
heatmap(np.mean([p['grad_norms'].mean(0) for p in probesA.values()], 0), 'mean grad norm per head (blocks 7-12)', ax=ax[1], ylabels=range(7, 13))
plt.tight_layout(); plt.show()
'''),
    code('''
# does the importance of a head depend on context? (ego action, signalised, traffic light)
ctx_test = data['test']['ctx']
p0 = next(iter(probesA.values()))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for a, name in zip(ax, ['ego_act', 'signalized', 'traffic_light']):
    byc = importance_by_context(p0, ctx_test, name)
    for lv, m in byc.items():
        a.plot(m[6:].reshape(-1), marker='o', ms=3, label=lv)
    a.set_title(f'importance of routed-layer heads by {name}', fontsize=9); a.set_xlabel('layer*6 + head (blocks 7-12)'); a.legend(fontsize=6)
plt.tight_layout(); plt.show()
# H5 reference: normalised entropy of the gradient share across context bins (1 = uniform over contexts)
entA = pd.DataFrame({s: grad_entropy_table(p, ctx_test) for s, p in probesA.items()})
entA['mean'] = entA.mean(1); entA.round(3)
'''),
    md('## 9. Save'),
    code('''
summary = {'sample_type': SAMPLE_TYPE, 'seeds': SEEDS, 'n_folds': N_FOLDS, 'recipe': BEST, 'crops': os.path.basename(CROPS)}
json.dump(summary, open(f'{OUT}/summary_A.json', 'w'))
res_A.drop(columns=[c for c in res_A.columns if c.endswith('_pt')]).to_csv(f'{OUT}/results_A.csv', index=False)
res_box.to_csv(f'{OUT}/results_boxonly.csv', index=False)
entA.to_csv(f'{OUT}/grad_entropy_A.csv')
# checkpoints are large and the probes needed downstream are already saved: drop them
for f in glob.glob(f'{OUT}/**/best.pt', recursive=True): os.remove(f)
!du -sh {OUT} {CROPS}
'''),
]

NB_B = [
    md('''
# Model B — routed PedViViT (B-soft / B-sparse / FiLM control) on JAAD, round 2

Implements the **routed** half of `idea.md` on top of the PedViViT-style model of notebook A: in the spatial ViT-S encoder, blocks 1-6 are dense and blocks 7-12 carry per-sample (per-clip) **head gates** from one router per layer reading
$z=\\mathrm{LN}(\\text{frame-averaged CLS after block 6})$.

* **B-soft:** $g=H\\,\\mathrm{softmax}(s/\\tau)$ - nests A at init (zero router).  * **B-sparse:** top-3 of 6 heads, $g=H\\,\\mathrm{softmax}_{S}(s/\\tau)$, noisy selection, first 40% of steps soft (Prop. 1 nesting), balancing + z-loss.
* **FiLM:** unconstrained $\\gamma,\\beta$ on the concatenated head space (control for Prop. 2: conditioning vs. selection).

**Round 2 protocol (same as notebook A).** Video-grouped 5-fold CV over train+val, fixed-budget training with EMA weights, **the exact fine-tuning recipe selected in notebook A** (`recipe.json`), identical folds / seeds / augmentation for every variant. The official test split is only predicted. H1 compares the fold-ensembled test predictions of each variant with A's by a paired pedestrian-cluster bootstrap.
Round 1's B-vs-A comparison was uninformative because the dense baseline was a one-epoch model; B-sparse additionally collapsed to identical heads for every clip, so its training knobs changed (`routed_vit/recipes.py::SPARSE_EXTRA`: longer soft phase, lower router LR, stronger balancing; these are untested guesses) and the routing diagnostics below show whether that worked.

**Measurements:** H1 performance - H2 routing not collapsed - H3 routes track context given the label (Miller-Madow MI + permutation null) - H4 causal swap tests with a magnitude-matched control - H5 gradient-share specialization.

**Kaggle settings:** GPU T4 x2 - Internet On - Inputs: dataset `christopheronyiuke/jaad-video` and **the output of notebook A** (provides the recipe, A's runs for H1/H5 and the crop cache). Expected run time: about 4-5 h (3 variants x 3 seeds x 5 folds), resumable.
'''),
    *package_cells(),
    md('## 2. Config'),
    code(CONFIG + '''
VARIANTS = ['soft', 'sparse', 'film']
ANA_FOLDS = [0, 1]         # runs (per seed) used for the H2/H3 analyses; probes (H5) and swap tests (H4) use fold 0 only
SWAP_SEEDS = [42]          # swap tests are the expensive probe; add seeds if time allows
N_PERM = 1000              # permutation null size (idea.md 8.2)
# notebook A's output (runs + probes + recipe), if attached
A_DIRS = glob.glob('/kaggle/input/**/out', recursive=True)
A_OUT = next((d for d in A_DIRS if glob.glob(d + '/runs/dense_s*')), None)
A_RUNS = A_OUT + '/runs' if A_OUT else None
print('notebook-A output:', A_OUT)
'''),
    *DATA_CELLS,
    md('### Cross-validation folds (identical to notebook A)'),
    FOLD_CELL,
    md('### Fine-tuning recipe (selected in notebook A by pooled out-of-fold AUC)'),
    code('''
from routed_vit.recipes import RECIPES, DEFAULT_RECIPE, SPARSE_EXTRA
if A_OUT and os.path.exists(A_OUT + '/recipe.json'):
    RECIPE = json.load(open(A_OUT + '/recipe.json'))['recipe']
else:
    RECIPE = DEFAULT_RECIPE; print('recipe.json not found: falling back to the default recipe (attach notebook A output to use its selection)')
print('recipe:', RECIPE, RECIPES[RECIPE])
'''),
    md('## 4. Sanity checks: Prop. 1 (B nests A), parameter overhead, memory'),
    code('''
from routed_vit.model import RoutedViT, count_params
from routed_vit.data import prep
torch.manual_seed(0); A = RoutedViT('dense', drop_path=0.1).to(device).eval()
torch.manual_seed(0); B = RoutedViT('soft', drop_path=0.1).to(device).eval()
x = prep(torch.randint(0, 255, (4, 8, 3, 224, 224), dtype=torch.uint8, device=device)); bx = torch.randn(4, 8, 4, device=device)
with torch.no_grad():
    ua, ub = A(x, bx)[0], B(x, bx)[0]
print('Prop.1  max |u_A - u_Bsoft| at init =', float((ua - ub).abs().max()), '(must be 0)')
for v in ['dense', 'soft', 'sparse', 'film']:
    tot, r = count_params(RoutedViT(v)); print(f'{v:7s} params {tot / 1e6:.3f}M  routing/film params {r / 1e3:.1f}k')
del A, B
m = RoutedViT('sparse', drop_path=0.1).to(device).train()
xx = prep(torch.randint(0, 255, (MICRO_BS, 8, 3, 224, 224), dtype=torch.uint8, device=device), train=True)
with torch.autocast('cuda', dtype=torch.float16):
    l, info = m(xx, box=torch.randn(MICRO_BS, 8, 4, device=device))
(l.float().mean() + info['bal'] + info['zloss']).backward()
print('B-sparse peak GPU mem %.1f GB at %d clips/step' % (torch.cuda.max_memory_allocated() / 1e9, MICRO_BS))
del m, xx, l, info; torch.cuda.empty_cache()
'''),
    md('## 5. Train B variants (same recipe, folds and seeds as A)'),
    code('''
from routed_vit.launcher import run_jobs
run_jobs(VARIANTS, SEEDS, FOLDS, SAMPLES, CROPS, RUNS, n_gpus=torch.cuda.device_count() or 1,
         extra=BASE_EXTRA + RECIPES[RECIPE], per_variant_extra={'sparse': SPARSE_EXTRA})
'''),
    md('## 6. H1 — does routing help? (fold-ensembled test AUC, paired pedestrian-cluster bootstrap vs A)'),
    code('''
from routed_vit.reports import results_table, plot_history, load_outputs, ensemble_test_probs
from routed_vit.metrics import paired_auc_diff
from routed_vit.data import groups_of
plot_history(RUNS, VARIANTS, SEEDS, FOLDS)
res_B = results_table(RUNS, VARIANTS, SEEDS, FOLDS, data)
display(res_B[''' + RES_COLS.format(df='res_B') + '''])
y_te, grp_te = data['test']['label'], groups_of(data['test'])
h1 = {}
if A_RUNS:
    pA = ensemble_test_probs(A_RUNS, 'dense', SEEDS, FOLDS, N_POOL, N_TEST)
    res_A = results_table(A_RUNS, ['dense'], SEEDS, FOLDS, data)
    for v in VARIANTS:
        pB = ensemble_test_probs(RUNS, v, SEEDS, FOLDS, N_POOL, N_TEST)
        if pA and pB:
            h1[v] = paired_auc_diff(y_te, pA, pB, grp_te)
            print(f'H1  AUC({v}) - AUC(A) = {h1[v][0]:+.4f}  95% CI [{h1[v][1]:+.4f}, {h1[v][2]:+.4f}]  ->', 'supported' if h1[v][1] > 0 else 'not supported')
    display(res_A[''' + RES_COLS.format(df='res_A') + '''])
else:
    print('Attach notebook A output to compute H1 (paired comparison against the dense model).')
'''),
    md('''## 7. H2/H3 — is the routing structured, and does it follow context beyond the label?
$\\mathcal S^l=I(\\text{sample};\\text{head})$ against a decorrelated-$z$ reference (each $z$ feature permuted independently; permuting whole rows would leave $\\mathcal S$ unchanged). Routes $r$: selected head set (B-sparse) or 8-means cluster of $g$ (B-soft). $I(r;c\\mid y)$ with Miller-Madow correction vs. a permutation null that shuffles context between pedestrians within each label stratum.
**H2 pass rule (tightened):** $\\mathcal S$ above the reference in more than half of the cells **and** mean $\\mathcal S/\\log H > 0.05$. Round 1 passed the first condition with $\\mathcal S/\\log H \\approx 0.001$, which is practically collapsed.
The first cell also reports how many distinct routes (head subsets) B-sparse actually uses per layer (out of 20 possible).'''),
    code('''
from routed_vit import analysis as an
from routed_vit.reports import load_run, heatmap
from routed_vit.data import CTX_NAMES
ctx_te = data['test']['ctx']
ANA = [(v, s, f) for v in ['soft', 'sparse'] for s in SEEDS for f in ANA_FOLDS if os.path.exists(f'{RUNS}/{v}_s{s}_f{f}/metrics.json')]
H2, H3, USED = [], [], []
for v, s, f in ANA:
    d = f'{RUNS}/{v}_s{s}_f{f}'
    o = load_outputs(d); idx = o['idx']
    S = an.routing_specialization(o['pi'])
    model, _ = load_run(d, device)
    null = np.mean([an.specialization_null(model, o['z'].astype(np.float32), device, seed=k) for k in range(10)], 0)
    del model
    H2.append(pd.DataFrame({'variant': v, 'seed': s, 'fold': f, 'layer': range(7, 13), 'S': S, 'S_null': null, 'S/logH': S / np.log(6)}))
    route = an.route_ids(o['gates'], o.get('mask'))
    if v == 'sparse':
        USED.append({'variant': v, 'seed': s, 'fold': f, 'distinct_routes_per_layer': [int(len(np.unique(route[:, l]))) for l in range(route.shape[1])]})
    t = an.label_context_mi(route, data['test']['label'][idx], ctx_te[idx], CTX_NAMES, grp_te[idx], n_perm=N_PERM)
    t.insert(0, 'fold', f); t.insert(0, 'seed', s); t.insert(0, 'variant', v); H3.append(t)
H2 = pd.concat(H2); H3 = pd.concat(H3)
display(H2.groupby(['variant', 'layer'])[['S', 'S_null', 'S/logH']].mean().round(4))
display(pd.DataFrame(USED))
'''),
    code('''
# H3: per context variable, in how many (seed, fold, layer) cells is I(r;c|y) above the 95th percentile of the permutation null?
H3['above'] = H3['above_null']
tab = H3.groupby(['variant', 'context']).agg(cells=('above', 'size'), above=('above', 'sum'), I_r_y=('I(r;y)', 'mean'),
                                              I_r_c_given_y=('I(r;c|y)', 'mean'), null_p95=('null_p95', 'mean'), min_p=('p_value', 'min'))
display(tab.round(4))
# gate regression: partial R^2 of context (all variables) in g ~ c + y, per layer x head
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a, v in zip(ax, ['soft', 'sparse']):
    rs = []
    for vv, s, f in ANA:
        if vv != v: continue
        o = load_outputs(f'{RUNS}/{v}_s{s}_f{f}')
        rs.append(an.gate_regression(o['gates'], ctx_te[o['idx']], data['test']['label'][o['idx']], CTX_NAMES))
    if rs: heatmap(np.mean(rs, 0), f'{v}: partial R2 of context in gate', ax=a, ylabels=range(7, 13), vmin=0)
plt.tight_layout(); plt.show()
'''),
    md('''## 8. H4 — causal route tests (swap the gates of clip $j$ into clip $i$)
$\\Gamma=E[\\Delta\\mid y_i=y_j,c_i\\ne c_j]-E[\\Delta\\mid c_i=c_j,y_i\\ne y_j]$ with pairs matched on $\\|g_i-g_j\\|_1$; control = random gate (a real gate with heads shuffled per layer) matched to the same residual-stream change (+-10%). `frac_swap_gt_ctrl` = share of matched swaps whose effect exceeds the control; `Gamma_kept` uses only those. Fold 0 of the seeds in `SWAP_SEEDS`.'''),
    code('''
from routed_vit.data import CropDataset
H4 = []
ds_te = CropDataset(data['test'], CROPS)
for v in ['soft', 'sparse']:
    for s in SWAP_SEEDS:
        d = f'{RUNS}/{v}_s{s}_f0'
        if not os.path.exists(d + '/metrics.json'): continue
        model, _ = load_run(d, device); o = load_outputs(d)
        pairs = an.swap_experiment(model, ds_te, o['gates'][np.argsort(o['idx'])], data['test']['label'], ctx_te, CTX_NAMES, device, n_pairs=40, n_ctrl=8)
        del model; torch.cuda.empty_cache()
        t = an.swap_summary(pairs); t.insert(0, 'seed', s); t.insert(0, 'variant', v); H4.append(t)
        pairs.to_csv(f'{OUT}/swap_pairs_{v}_s{s}.csv', index=False)
H4 = pd.concat(H4) if H4 else pd.DataFrame()
display(H4.round(4))
'''),
    md('## 9. H5 — does selection drive specialization? (normalised entropy of the gradient share over context bins, lower = more specialised)'),
    code('''
from routed_vit.reports import head_probes, grad_entropy_table
ent = {}
for v in ['soft', 'sparse']:
    for s in SEEDS:
        if os.path.exists(f'{RUNS}/{v}_s{s}_f0/metrics.json'):
            ent[(v, s)] = grad_entropy_table(head_probes(f'{RUNS}/{v}_s{s}_f0', data, CROPS, device, n_samples=300, importance=False), ctx_te)
if A_RUNS:
    for s in SEEDS:
        if os.path.exists(f'{A_RUNS}/dense_s{s}_f0/probes.npz'):
            ent[('dense', s)] = grad_entropy_table(dict(np.load(f'{A_RUNS}/dense_s{s}_f0/probes.npz')), ctx_te)
E = pd.DataFrame(ent).T
E.index.names = ['variant', 'seed']
H5 = E.groupby('variant').mean(); H5['mean_over_contexts'] = H5.mean(1)
display(H5.round(3))
'''),
    md('## 10. Hypotheses summary'),
    code('''
rows = []
for v in VARIANTS:
    if v in h1: rows.append({'hyp': 'H1', 'variant': v, 'statistic': f'dAUC={h1[v][0]:+.4f} [{h1[v][1]:+.4f},{h1[v][2]:+.4f}]', 'verdict': 'supported' if h1[v][1] > 0 else 'not supported'})
for v, d in H2.groupby('variant'):
    f = float((d.S > d.S_null).mean()); r = d['S/logH'].mean()
    rows.append({'hyp': 'H2', 'variant': v, 'statistic': f'S>S_null in {f:.0%} of cells; mean S/logH={r:.3f}', 'verdict': 'supported' if (f > 0.5 and r > 0.05) else 'not supported (collapsed)'})
for v, d in H3.groupby('variant'):
    f = float(d.above_null.mean()); rows.append({'hyp': 'H3', 'variant': v, 'statistic': f'I(r;c|y) > null p95 in {f:.0%} of (seed,fold,layer,context) cells', 'verdict': 'supported' if f > 0.25 else 'not supported'})
if len(H4):
    for v, d in H4.groupby('variant'):
        ok = ((d.Gamma_lo > 0) & (d.frac_swap_gt_ctrl > 0.5)).mean(); rows.append({'hyp': 'H4', 'variant': v, 'statistic': f'Gamma CI>0 and swaps>control for {ok:.0%} of contexts', 'verdict': 'supported' if ok > 0.5 else 'not supported'})
if {'sparse', 'soft', 'dense'} <= set(H5.index):
    m = H5['mean_over_contexts']; rows.append({'hyp': 'H5', 'variant': 'sparse', 'statistic': f"H(q): sparse {m['sparse']:.3f} < soft {m['soft']:.3f} ~ dense {m['dense']:.3f}", 'verdict': 'supported' if m['sparse'] < min(m['soft'], m['dense']) - 0.005 else 'not supported'})
summary = pd.DataFrame(rows); display(summary)
summary.to_csv(f'{OUT}/hypotheses_B.csv', index=False)
res_B.drop(columns=[c for c in res_B.columns if c.endswith('_pt')]).to_csv(f'{OUT}/results_B.csv', index=False)
H2.to_csv(f'{OUT}/H2.csv', index=False); H3.to_csv(f'{OUT}/H3.csv', index=False); H4.to_csv(f'{OUT}/H4.csv', index=False); H5.to_csv(f'{OUT}/H5.csv')
json.dump({'recipe': RECIPE, 'sparse_extra': SPARSE_EXTRA, 'seeds': SEEDS, 'n_folds': N_FOLDS}, open(f'{OUT}/summary_B.json', 'w'))
for f in glob.glob(f'{RUNS}/*/best.pt'): os.remove(f)
'''),
    md('''
### Reading the outcomes (idea.md 9)
| Result | Meaning |
|---|---|
| B ~ A, S ~ 0 | router collapsed or nothing to exploit |
| B ~ A, S high, H3 fails | router learned the label; no gain |
| B > A, H3 fails | gain from extra per-sample capacity, not context (compare with FiLM) |
| B > A, H3 and H4 pass | claim supported **for reweighting**; selection vs. conditioning is settled only by B-sparse vs. FiLM |
'''),
]

# ----------------------------------------------------------------------------- notebook C: Pedestrian Graph+ (optimised)
PG_PKG = ['__init__.py', 'model.py', 'fastdata.py', 'net.py', 'gtranspdm.py', 'cvfit.py', 'report.py', 'hpo.py', 'final.py', 'run_gtrans.py']
SHARED_PKG = ['__init__.py', 'metrics.py', 'cv.py']


def pg_package_cells():
    cells = [md('## 1. Code\nThe `pedgraph` package (model, GPU data pipeline, cross-validated trainer, Optuna search, final evaluation) and the three '
                'small shared modules of `routed_vit` (metrics, CV folds) are embedded with `%%writefile`, so the notebook is self-contained.')]
    cells.append(code('''
!pip install -q optuna
!mkdir -p /kaggle/working/code/routed_vit /kaggle/working/code/pedgraph /kaggle/working/out
!nvidia-smi -L
'''))
    for f in SHARED_PKG:
        cells.append(code(f'%%writefile /kaggle/working/code/routed_vit/{f}\n' + open(os.path.join(HERE, 'routed_vit', f), encoding='utf-8').read()))
    for f in PG_PKG:
        cells.append(code(f'%%writefile /kaggle/working/code/pedgraph/{f}\n' + open(os.path.join(HERE, 'pedgraph', f), encoding='utf-8').read()))
    return cells


NB_C = [
    md('''
# Model C — Pedestrian Graph+ on JAAD, round 2: cross-validated search, tuned model, ablations

Round 1 trained the (clean-room) Pedestrian Graph+ once per seed on the official split and kept the best of 30 epochs by validation AUC on **22 pedestrians**; validation AUC jumped between 0.47 and 0.88 from one epoch to the next, so the checkpoint choice was noise (val 0.8-0.88 vs test 0.60), and the model overfit (train loss 0.05).

**Round 2.**
1. **Protocol:** train+val pooled (216 pedestrians), 5 video-grouped folds, fixed epoch budget, EMA weights, **out-of-fold AUC** as the only selection signal; the official test split is only predicted (fold models x seeds averaged).
2. **Speed:** all windows live on the GPU and augmentation is vectorised, so one 5-fold evaluation of a configuration takes minutes and an actual search is affordable.
3. **Search (Optuna, TPE + pruning):** input representation (absolute / box-relative pose, keypoint velocity), side stream (box trajectory, ego action), architecture (width, depth, kernel, skeleton-initialised adjacency, fusion), optimiser and schedule, regularisation (dropout, weight decay, label smoothing, EMA, flip-TTA), augmentation (keypoint jitter, zoom/shift, left-right flip, joint drop-out, time crop). The released recipe is cross-validated under the same protocol as a baseline.
4. **Winner's-curse control:** the top configurations are **re-scored with fresh seeds**; the image branch is then tested on top of the best pose configuration and kept only if it helps beyond a margin.
5. **Final:** the tuned model with 5 seeds x 5 folds, baselines, one-factor ablations, and an ensemble of the top configurations.

**Data:** create a Kaggle dataset from `jaad_beh.pkl` (PCPA windows with context) and the `pose_beh/` ViTPose cache (about 100 MB): run `python kaggle/C/make_dataset.py` locally and the `kaggle datasets create` command it prints, then attach it as input.
**Kaggle settings:** GPU (a T4 is plenty), Internet On (for `pip install optuna`).
'''),
    *pg_package_cells(),
    md('## 2. Config'),
    code('''
import os, sys, json, glob, subprocess, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
WORK = '/kaggle/working'; OUT = f'{WORK}/out'; CODE = f'{WORK}/code'
sys.path.insert(0, CODE)
SAMPLES = next(iter(glob.glob('/kaggle/input/**/jaad_beh.pkl', recursive=True)), None)
CACHE = next(iter(glob.glob('/kaggle/input/**/pose_beh', recursive=True)), None)
print('samples:', SAMPLES, '| pose cache:', CACHE)
assert SAMPLES and CACHE, 'attach the dataset with jaad_beh.pkl and pose_beh/ (see the first cell of the notebook)'
N_TRIALS = 150        # search trials (one trial = one 5-fold CV of a configuration); a few minutes each, pruned trials much less
PARALLEL = 2          # worker processes sharing the GPU
HPO = f'{OUT}/hpo'; FINAL = f'{OUT}/final'
ENV = dict(os.environ, PYTHONPATH=CODE)
def run(mod, args):   # run a pedgraph CLI step; resumable (finished stages and trials are skipped)
    cmd = [sys.executable, '-m', mod, *map(str, args)]
    p = subprocess.run(cmd, cwd=CODE, env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print('\\n'.join(l for l in p.stdout.splitlines() if 'Warning' not in l and 'sampler=' not in l)[-6000:])
    assert p.returncode == 0
'''),
    md('## 3. Data and folds'),
    code('''
from pedgraph.fastdata import PoseData
from routed_vit.cv import make_folds, describe_folds
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = PoseData(SAMPLES, CACHE, dev)
print(f'windows: pooled train+val {D.n_pool}, test {D.n_test}; keypoints {tuple(D.kp.shape)}, crops {tuple(D.img.shape)}')
display(pd.DataFrame(describe_folds(D.pool_rec, make_folds(D.pool_rec, 5, 0))))
print('positive rate: pooled train+val %.3f, test %.3f' % (D.y_np[D.pool_rows].mean(), D.y_np[D.test_rows].mean()))
'''),
    md('''## 4. Baselines + hyper-parameter search + re-scoring (resumable; this is the long cell)
Stage 0 cross-validates the released recipe and our regularised default. Stage 1 runs the Optuna study. Stage 2 re-scores the top configurations with fresh seeds. Stage 3 tests the image branch on top of the best pose configuration.
'''),
    code('''
run('pedgraph.hpo', ['--samples', SAMPLES, '--cache', CACHE, '--out', HPO, '--trials', N_TRIALS, '--parallel', PARALLEL])
'''),
    code('''
base = json.load(open(f'{HPO}/baselines.json'))
print({k: round(v['cv_fold_auc_mean'], 4) for k, v in base.items()}, '<- mean fold AUC (seed 0, 5 folds)')
s2 = pd.read_csv(f'{HPO}/stage2.csv'); display(s2.round(4))
display(pd.read_csv(f'{HPO}/image_stage.csv').round(4))
tr = pd.read_csv(f'{HPO}/trials.csv')
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ok = tr[tr.state == 'COMPLETE']
ax[0].scatter(ok.number, ok.value, s=10); ax[0].plot(ok.number, ok.value.cummax(), 'r-'); ax[0].set_title('stage-1 score (mean fold AUC) per trial'); ax[0].set_xlabel('trial')
ax[1].scatter(s2.stage1, s2.rescore); ax[1].plot([.5, .8], [.5, .8], 'k--', lw=.5); ax[1].set_xlabel('stage-1 score'); ax[1].set_ylabel('re-score (fresh seeds)'); ax[1].set_title("winner's curse")
plt.tight_layout(); plt.show()
'''),
    code('''
import optuna
journal = f'{HPO}/study.journal'
st = optuna.load_study(study_name='pg', storage=optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(journal, lock_obj=optuna.storages.journal.JournalFileOpenLock(journal))))
imp = optuna.importance.get_param_importances(st)
pd.Series(imp).head(15).iloc[::-1].plot.barh(figsize=(6, 4), title='parameter importance (fANOVA)'); plt.tight_layout(); plt.show()
print(json.dumps(json.load(open(f'{HPO}/final_cfg.json'))['cfg'], indent=1))
'''),
    md('## 5. Final cross-validated evaluation (tuned model, baselines, ablations, top-config ensemble)\nTuned model: 5 fresh seeds x 5 folds (25 models) averaged on the official test split. Ablations remove one component of the tuned model at a time (3 seeds each).'),
    code('''
run('pedgraph.final', ['--samples', SAMPLES, '--cache', CACHE, '--hpo', HPO, '--out', FINAL, '--seeds', 5, '--baseline_seeds', 3,
                       '--ablate', '--ablation_seeds', 3, '--ensemble_topk', 3])
'''),
    code('''
summ = pd.read_csv(f'{FINAL}/summary.csv')
cols = [c for c in ['model', 'seed', 'oof_auc', 'fold_auc', 'test_auc', 'test_auc_ci', 'bench_acc', 'bench_f1', 'thr_bacc', 'thr_f1'] if c in summ]
display(summ[summ.seed == 'ens'][cols].round(4))
from pedgraph.report import majority_line
print(majority_line(D))
summ['seed'] = summ.seed.astype(str)
display(summ[summ.seed != 'ens'].groupby('model')[['oof_auc', 'test_auc']].agg(['mean', 'std']).round(4))
'''),
    code('''
abl = pd.read_csv(f'{FINAL}/ablation.csv'); abl = abl[abl.seed.astype(str) == 'ens'].set_index('model')[['oof_auc', 'test_auc', 'thr_bacc']]
abl['d_oof_vs_tuned'] = abl.oof_auc - abl.loc['TUNED (reference)', 'oof_auc']
display(abl.sort_values('oof_auc').round(4))
'''),
    code('''
from sklearn.metrics import roc_curve, roc_auc_score
P = np.load(f'{FINAL}/preds.npz'); yp, yt = P['y_pool'], P['y_test']
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for name in ['released recipe (CV)', 'TUNED']:
    for a, (y, p, ttl) in zip(ax, [(yp, P[f'{name}__oof'].mean(0), 'out-of-fold (216 train+val pedestrians)'), (yt, P[f'{name}__test'].mean(0), 'official test')]):
        f, t, _ = roc_curve(y, p); a.plot(f, t, label=f'{name} AUC {roc_auc_score(y, p):.3f}'); a.set_title(ttl)
for a in ax: a.plot([0, 1], [0, 1], 'k--', lw=.5); a.legend(fontsize=7)
plt.tight_layout(); plt.show()
'''),
    md('## 6. Save'),
    code('''
import shutil
for f in ['final_cfg.json', 'best_cfg.json', 'baselines.json', 'stage2.csv', 'image_stage.csv', 'trials.csv']:
    shutil.copy(f'{HPO}/{f}', f'{OUT}/{f}')
!du -sh {OUT}
'''),
]


def write(name, cells, kdir):
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 4}
    for sub in (os.path.join('kaggle', kdir),):
        path = os.path.join(HERE, sub, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(nb, open(path, 'w', encoding='utf-8'), indent=1)
        print('wrote', path, len(cells), 'cells')


if __name__ == '__main__':
    write('jaad_dense_vit_A.ipynb', NB_A, 'A')
    write('jaad_routed_vit_B.ipynb', NB_B, 'B')
    write('jaad_pedgraph_C.ipynb', NB_C, 'C')

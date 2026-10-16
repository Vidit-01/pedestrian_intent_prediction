# Pedestrian crossing intention on JAAD: ViT, routed ViT, Pedestrian Graph+, GTransPDM

Four models that predict whether a pedestrian will cross the road, trained and compared on the **JAAD** dataset (`JAAD_beh` and `JAAD_all`) under one protocol.
The PCPA window protocol and the data tooling come from the base repository (`build_jaad_samples.py`, `download_jaad.py` at the repository root, one level above this folder); this folder holds the models, experiments and results.

| | Model | Paper | Link |
|---|---|---|---|
| A | Dense ViT (PedViViT-style) | VIT-Ped: Visionary Intention Transformer for Pedestrian Behavior Analysis (Elkammar, Gamaleldin, Elias, 2026) | https://arxiv.org/abs/2601.01989 |
| B | Routed ViT (B-soft, B-sparse, FiLM control) | Same backbone as A; the head-gating is our own design, specified in [docs/idea.md](docs/idea.md) | https://arxiv.org/abs/2601.01989 |
| C | Pedestrian Graph+ | Pedestrian Graph+: A Fast Pedestrian Crossing Prediction Model Based on Graph Convolutional Networks (Cadena, Qian, Wang, Yang, IEEE T-ITS 2022) | https://doi.org/10.1109/TITS.2022.3173537 , code: https://github.com/RodrigoGantier/Pedestrian_graph_plus |
| D | GTransPDM | GTransPDM: A Graph-embedded Transformer with Positional Decoupling for Pedestrian Crossing Intention Prediction (Xie, Lin, Zheng, Gong, Lopez, IEEE SPL 2025) | https://arxiv.org/abs/2409.20223 |

## The models

**A. Dense ViT** (`routed_vit/model.py`, variant `dense`). Following VIT-Ped, a video-ViT (ViViT-style) factorised encoder: an ImageNet-21k pretrained ViT-S/16 runs on 8 of the 16 observed frames (224x224 crops centred on the pedestrian), a 2-layer temporal transformer runs over the per-frame CLS tokens, and a small transformer over delta-encoded bounding boxes is concatenated before a one-logit head (24.16 M parameters). Ego speed is deliberately not an input because it is one of the analysed context variables. The paper leaves most sizes open, so the ViT-S choice, widths and fusion are ours. Trained with layer-wise learning-rate decay, optional frozen first blocks, stochastic depth, label smoothing, EMA weights and clip-consistent augmentation; four fine-tuning recipes are compared by out-of-fold AUC (`routed_vit/recipes.py`). A box-only variant (no pixels) is the control.

**B. Routed ViT** (`variant` `soft`, `sparse`, `film`). The same backbone with per-clip gates on the attention heads of blocks 7-12, produced by a linear router that reads the frame-averaged CLS token after block 6. B-soft reweights all six heads (it equals A at initialisation, checked to a max difference of 0.0); B-sparse keeps the top 3 of 6 heads with noisy selection, load-balancing and z-loss; FiLM is an unconstrained per-channel scale and shift used as a control. The question tested is whether routing heads per clip helps and whether the routes follow driving context (hypotheses H1-H5, full spec in [docs/idea.md](docs/idea.md)). The sparse-routing machinery builds on ideas from sparse mixture-of-experts (Shazeer et al. 2017; Switch Transformer, Fedus et al.; ST-MoE, Zoph et al.) and FiLM (Perez et al. 2018).

**C. Pedestrian Graph+** (`pedgraph/model.py`, `pedgraph/net.py`). A graph convolutional network over a 19-joint skeleton with learned adjacency and temporal convolutions, whose features are gated by an image branch (local-context crop) and an ego-vehicle branch (70 k parameters in the released form). Poses come from ViTPose-B on the annotated boxes. **Built from the authors' released code, not from the paper** (the paper is paywalled and no open copy was found), so the paper's best variant (3D keypoints, segmentation map, pose forecasting) is not reproduced. `PedGraphNet` is a configurable version that was tuned with Optuna by cross-validation (search, fresh-seed re-scoring, image-branch test, ablations).

**D. GTransPDM** (`pedgraph/gtranspdm.py`). A pose-graph transformer: 4 graph-convolution layers with learnable edge weights encode the 20-joint skeleton, a Positional Decoupling Module turns the box track into offsets to two reference lines plus an area-ratio depth cue, an ego-motion encoder is added, and a 4-layer, 4-head transformer (d=64) classifies the fused sequence (343,755 parameters). **No code is public; it was reimplemented from the arXiv paper** with the paper's training recipe (AdamW, lr 5e-5, batch 64, 32 epochs). Choices the paper leaves open are listed in the module docstring.

Supporting references (cited from memory, not re-verified; see [docs/papers.md](docs/papers.md)): ViT (arXiv 2010.11929), ViViT (2103.15691), ViTPose (2204.12484), sparse MoE (1701.06538), Switch Transformer (2101.03961), FiLM (1709.07871), JAAD (Rasouli et al., ICCVW 2017), PCPA benchmark protocol (Kotseruba et al., WACV 2021), Optuna (Akiba et al., KDD 2019).

## Data and protocol

* **JAAD**, official video split, PCPA windows: 16 observed frames, time-to-event 30-60 frames, stride 3, label = pedestrian crosses.
  `JAAD_beh`: 2134 / 242 / 1881 train / val / test windows (test 62.6% positive). `JAAD_all` adds non-behavioural pedestrians: 8613 / 1265 / 6732 windows (test 17.5% positive).
* **Round 1:** official split, best epoch picked on the 22-pedestrian validation set. This overfit and selected noise (see [docs/RESULTS.md](docs/RESULTS.md)).
* **Round 2 (all results below):** train+val pooled, **5 video-grouped folds**, fixed epoch budget, EMA weights, out-of-fold (OOF) AUC as the only selection signal; the test split is only predicted by the fold models and never used to select anything.
* Accuracy and F1 at threshold 0.5 are near the majority-class baseline on these test sets; AUC and balanced accuracy are the numbers to read.

## Results (test AUC; round 2 unless marked)

| Model | JAAD_beh | JAAD_all |
|---|---|---|
| A. Dense ViT | 0.674 (round 1: 0.656) | not run |
| B-soft / B-sparse / FiLM | 0.694 / 0.634 / 0.644 (round 1: 0.630 / 0.545 / 0.631) | not run |
| C. Pedestrian Graph+, released recipe (in CV) | 0.693 | search in progress, no test result yet |
| C. Pedestrian Graph+, regularised default | 0.720 | |
| C. Pedestrian Graph+, tuned (OOF AUC 0.779) | 0.671 | |
| D. GTransPDM, paper protocol / 5-fold CV | 0.713 / 0.721 | 0.899 / 0.910 |

* Routing does not help (B-soft vs A: +0.003, CI [-0.022, +0.028]); the routers collapse to near-uniform gates.
* GTransPDM on JAAD_all matches the paper's accuracy and F1 (0.860 / 0.639 vs 0.87 / 0.64).
* Tuning Pedestrian Graph+ raised the cross-validated AUC (0.607 to 0.779) but not the test AUC; the untuned regularised default is best on test. This gap is discussed in [docs/RESULTS.md](docs/RESULTS.md) and is the main caveat of the study.
* A box-only control scores chance level in cross-validation (OOF AUC 0.498).

Full tables, ablations, hypotheses H1-H5 and confidence intervals: [docs/RESULTS.md](docs/RESULTS.md). Raw outputs: `results/`.

## Repository layout

```
README.md                  this file
CLAUDE.md                  pipeline, protocol and gotchas
kaggle_commands.md         Kaggle run sheet (push, status, download results)
build_notebooks.py         generates kaggle/A|B|C notebooks (package source embedded)
docs/
  RESULTS.md               full project record: setups, tables, caveats, next steps
  papers.md                model-to-paper mapping and supporting references
  idea.md                  formal spec of the routed ViT and hypotheses H1-H5
routed_vit/                models A and B (model, data, cv folds, train, recipes, analysis, reports, launcher)
pedgraph/                  models C and D (GPU data, networks, CV trainer, Optuna search, final evaluation); has its own README
kaggle/A|B|C/              notebooks + kernel metadata (C also make_dataset.py)
scripts/resume_pg_all.sh   resume the Pedestrian Graph+ JAAD_all search
results/
  round1/                  official-split runs: vit_A, vit_B, pedgraph_IVT
  round2/                  vit_A, vit_B (Kaggle result tables), pedgraph_beh, pedgraph_all (partial), gtranspdm
data/  logs/               git-ignored: JAAD clones/clips, sample pickles, pose caches, run logs
```

## Reproduce

Data (the base-repo scripts `download_jaad.py` and `build_jaad_samples.py` sit one level up, at the repository root; run these from this folder):
```bash
python ../download_jaad.py --root data/JAAD --clips
python -c "from routed_vit.data import build_samples; build_samples('..','data/JAAD','data/samples_ctx/jaad_beh.pkl','beh')"   # and 'all' -> jaad_all.pkl
python -m pedgraph.extract --samples data/samples_ctx/jaad_beh.pkl --clips data/JAAD/JAAD_clips --out data/pose_beh                                       # and jaad_all -> pose_all
```
* **Models A and B** (Kaggle, 2x T4): `kaggle/A/jaad_dense_vit_A.ipynb`, then `kaggle/B/jaad_routed_vit_B.ipynb` (see [kaggle_commands.md](kaggle_commands.md)). Rebuild the notebooks with `python build_notebooks.py`.
* **Model C** (local GPU): `python -m pedgraph.hpo ...` then `python -m pedgraph.final ...`; **Model D**: `python -m pedgraph.run_gtrans ...`. Exact commands in [pedgraph/README.md](pedgraph/README.md).
* Requirements: PyTorch, `timm`, `scikit-learn`, `optuna`, `opencv-python`, `transformers` (ViTPose), `pandas`, `matplotlib`.

## Status

Models A, B, D and Pedestrian Graph+ on JAAD_beh are finished. Pedestrian Graph+ on JAAD_all is paused (search 18 complete trials, best mean fold AUC 0.9215); resume with `bash scripts/resume_pg_all.sh`. The reimplementations carry documented assumptions (section "Differences" in `pedgraph/README.md` and the docstring of `pedgraph/gtranspdm.py`); their numbers are not directly comparable to the papers' tables.

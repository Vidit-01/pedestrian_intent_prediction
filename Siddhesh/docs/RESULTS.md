# Siddhesh — Pedestrian crossing intent on JAAD: dense vs routed ViT, Pedestrian Graph+, GTransPDM

Written for: Siddhesh and collaborators (project record: what was built, how it was run, what came out).
State of this file: 2026-10-12 evening. Round 2 results for the ViTs (A, B), Pedestrian Graph+ on JAAD_beh and GTransPDM on both datasets are final. Pedestrian Graph+ on **JAAD_all is paused** (section 7).
All "test" numbers are on the official JAAD test split (171 pedestrians in JAAD_beh, 612 in JAAD_all). AUC is the number that carries signal; see the majority-class warning in section 3.

---

## 1. Summary

Four model families, two protocols (round 1 = single official split; round 2 = video-grouped cross-validation). Test AUC, JAAD_beh unless stated:

| Model | Round 1 (official split, best-epoch by val AUC) | Round 2 (CV, fixed budget, no selection on test) |
|---|---|---|
| **A. Dense ViT** (PedViViT-style) | 0.656 ± 0.020 | **0.674** (CI 0.600–0.751), out-of-fold (OOF) AUC 0.644 |
| **B-soft** / B-sparse / FiLM | 0.630 / 0.545 / 0.631 | **0.694** / 0.634 / 0.644; OOF 0.650 / 0.625 / 0.612 |
| **C. Pedestrian Graph+** (code-based reimplementation) | 0.603 ± 0.013 | released recipe in CV 0.693; regularised default 0.720; **tuned 0.671** (OOF 0.779) |
| **D. GTransPDM** (reimplemented from the paper) | not run | **0.713** (paper protocol), 0.721 (CV ensemble) |
| GTransPDM, **JAAD_all** | not run | **0.899** (paper protocol), 0.910 (CV ensemble) |

Headline findings
1. **Round 1 was not trained properly.** The dense ViT memorised the training set by epoch 4 (train loss 3e-5) and was a one-epoch checkpoint picked on 22 validation pedestrians; Pedestrian Graph+ validation AUC jumped between 0.47 and 0.88 from one epoch to the next. Round 2 fixes the protocol, not just the models.
2. **Routing still does not help** (H1 not supported for any variant, with a properly trained baseline). B-soft is level with A (dAUC +0.003, CI [-0.022, +0.028]); B-sparse and FiLM are about 0.04 lower (CIs include 0). Routers are practically collapsed (S/log H of 0.001-0.002).
3. **Pose models beat the image ViT** on JAAD_beh: GTransPDM 0.71-0.72 vs dense ViT 0.67. A box-only control scores OOF AUC 0.50 (chance) and test AUC 0.62.
4. **CV tuning did not transfer to the test split for Pedestrian Graph+.** The search raised the cross-validated AUC from 0.607 (released recipe) to 0.779, but test AUC went from 0.693 to 0.671, and the untuned "regularised default" (OOF 0.688) has the best test AUC (0.720). Section 5.3 discusses why; this is the most important caveat of the study.
5. **GTransPDM reproduces the paper's JAAD_all accuracy/F1** (0.860 / 0.639 vs the paper's 0.87 / 0.64), and its benchmark-style AUC (AUC of rounded predictions, 0.800) matches the paper's reported 0.78 better than our real AUC (0.899).
6. **Accuracy and F1 at 0.5 are near the majority-class baseline** on JAAD_beh (section 3), so they are not evidence of skill there.

---

## 2. Datasets and protocols

* **JAAD** (Rasouli, Kotseruba, Tsotsos, ICCVW 2017), official video split, windows from the PCPA benchmark protocol (Kotseruba et al., WACV 2021) built with the base repo's `build_jaad_samples.py`: 16 observed frames, time-to-event 30-60 frames, stride 3. Label = pedestrian crosses.
* **JAAD_beh** (behaviour-annotated pedestrians): train 2134 windows / 194 peds (1760 pos), val 242 / 22 (176 pos), test 1881 / 171 (1177 pos). Train+val is 82% positive, test 63%.
* **JAAD_all** (adds the non-behavioural pedestrians, label 0): train 8613 / 783 peds (1760 pos), val 1265 / 115 (176 pos), test 6732 / 612 (1177 pos = **17.5% positive**; always predicting "no crossing" gives accuracy 0.825).
* **Context labels** (analysis only, never model inputs): signalized, intersection, designated crossing, group size, lanes, ego action, traffic light, pedestrian-crossing sign.
* **Round 1 protocol:** train on the official train split, early stopping / best epoch by validation AUC (22 pedestrians in JAAD_beh), 3 seeds.
* **Round 2 protocol (all models):** train+val pooled (216 pedestrians in JAAD_beh), 5 folds grouped by video and stratified by label (`routed_vit/cv.py`, identical for every model and seed). Each fold model predicts its held-out fold (out-of-fold AUC = the selection signal) and the test split (fold ensemble). **Fixed epoch budget, EMA weights, no early stopping, no checkpoint selection, test never used for selection.** Thresholds for acc/F1 come from OOF predictions. 95% CIs resample whole test pedestrians.
* **Poses** for models C and D come from ViTPose-B (`usyd-community/vitpose-base-simple`) on the annotated boxes (17 COCO joints + neck + pelvis = 19 nodes; GTransPDM adds a body-centre node = 20). Caches: `data/pose_beh` (201 videos), `data/pose_all` (293 videos).

---

## 3. Warnings that apply to every number

* **Majority baseline.** JAAD_beh test is 62.6% positive: accuracy 0.626, F1 0.770 for "always crosses". No model is clearly above that at threshold 0.5. JAAD_all test: accuracy 0.825 for "never crosses". Read AUC and balanced accuracy.
* **CV vs test.** Train+val is 82% positive and the test videos differ from the training videos. CV AUCs and test AUCs are not expected to agree, and in one case (Pedestrian Graph+ tuning) they moved in opposite directions (section 5.3).
* **Small test set.** 171 pedestrians in JAAD_beh with heavily overlapping windows; AUC differences below about 0.04-0.05 are not distinguishable (see CIs).
* **Reimplementations.** Pedestrian Graph+ is built from the authors' code, not from the paper (paywalled; not accessible via IEEE, Unpaywall, Semantic Scholar, arXiv). GTransPDM has no public code and was written from the arXiv paper with documented assumptions (section 4.4).

---

## 4. Models

### 4.1 A. Dense PedViViT-style ViT (`routed_vit/model.py`, variant `dense`)
Per-frame ViT-S/16 (ImageNet-21k pretrained, 12 blocks, 6 heads) on 8 of the 16 frames (224x224 crops centred on the pedestrian, 1.5x the box), a 2-layer temporal transformer over the per-frame CLS tokens, plus a non-visual 2-layer / 4-head transformer on delta-encoded boxes; concatenation + FFN -> one logit. 24.16 M parameters. **Ego speed is not an input** (it is one of the analysed contexts). Base paper: VIT-Ped, arXiv 2601.01989 (Elkammar, Gamaleldin, Elias, Jan 2026; the abstract confirms a transformer / ViViT-style model on JAAD; the box branch and all sizes come from the project notes and our choices; the paper is not in the literature map).
Round 2 training: fixed budget, warm-up + cosine, EMA weights, layer-wise LR decay, stochastic depth, label smoothing, clip-consistent affine jitter, box noise, box-branch modality dropout, optional frozen first blocks. A 4-recipe search (`routed_vit/recipes.py`) picks the recipe by pooled OOF AUC.

### 4.2 B. Routed ViT (`routed_vit/model.py`, variants `soft`, `sparse`, `film`; spec `docs/idea.md`)
Same backbone; in blocks 7-12 each head output gets a per-clip gate from a linear router on `z = LN(frame-averaged CLS after block 6)`; gates sum to H = 6. B-soft = `6 softmax(s/tau)` (equals A at init, checked: max difference 0.0); B-sparse = top-3 of 6 heads (noisy selection, soft start, balancing + z-loss); FiLM = unconstrained per-channel scale/shift (control). The gating design is our own; related ideas: sparse mixture-of-experts (Shazeer 2017, Switch Transformer, ST-MoE), FiLM (Perez 2018). Round 2 uses the recipe selected in A, plus B-sparse settings meant to avoid collapse (soft phase 40%, router LR x5, balancing 0.05).

### 4.3 C. Pedestrian Graph+ (`pedgraph/model.py`, `pedgraph/net.py`)
Cadena, Qian, Wang, Yang, IEEE T-ITS 2022 (literature map 5.4). Adaptive graph convolution (3 learned adjacency subsets over the skeleton) + temporal convolution, gated by an image branch (local-context crop) and an ego-vehicle branch; 70 k parameters. `model.py` matches the released checkpoints' parameter names and shapes. The paper was not read; the paper's best model (3D keypoints, segmentation map, pose forecasting) is not reproduced.
Round 2 adds `PedGraphNet` (configurable pose representation, widths, kernel, adjacency init, fusion, side streams) and a GPU-resident pipeline, and an Optuna search by 5-fold CV (section 5.3).

### 4.4 D. GTransPDM (`pedgraph/gtranspdm.py`)
Xie, Lin, Zheng, Gong, Lopez, IEEE SPL 2025, arXiv 2409.20223 (literature map 5.3). **Reimplemented from the paper (no code released); 343,755 parameters.** From the paper: 20 keypoints relative to the box top-left; Positional Decoupling Module (offsets to two reference lines, area-ratio depth cue, displacement, velocity); 4 graph-convolution layers (d=64, normalised adjacency with learnable edge weights, Conv2D residuals); position and ego-motion encoders; 4-layer / 4-head Transformer (d=64, dropout 0.1); AdamW, lr 5e-5 constant, batch 64, 32 epochs, weight decay 1e-4, cross-entropy, no augmentation.
Our assumptions (not in the paper): which point on the reference lines is used (we use the point at the pedestrian's row / column), Conv2D kernel 1x1, skeleton edge list and the three averaged joints, sinusoidal positional encoding, FFN width 256, BatchNorm on inputs, ViTPose instead of AlphaPose, one logit with BCE instead of 2-class softmax, ego state fed as the scalar 0..4.

---

## 5. Results (round 2)

### 5.1 A. Dense ViT, JAAD_beh (Kaggle, 2x T4; notebook `kaggle/A`)
Recipe search (dense, seed 42, 5-fold CV), selection by pooled OOF AUC:

| Recipe | OOF AUC | Mean fold AUC (± std) | Test AUC (diagnostic, not used) |
|---|---|---|---|
| **r3_frozen6** (selected: lr 5e-5, LLRD 0.9, first 6 blocks frozen, 10 epochs) | 0.636 | 0.623 ± 0.082 | 0.662 |
| r4_strong_reg | 0.619 | 0.594 ± 0.109 | 0.669 |
| r1_lr2e-5 | 0.606 | 0.575 ± 0.089 | 0.674 |
| r2_lr1e-5_long | 0.583 | 0.559 ± 0.067 | 0.680 |

Final model (3 seeds x 5 folds with r3_frozen6):

| Seed | OOF AUC | Test AUC |
|---|---|---|
| 42 / 7 / 97 | 0.636 / 0.588 / 0.623 | 0.662 / 0.672 / 0.677 |
| **ensemble (15 models)** | 0.644 | **0.674** [0.600, 0.751]; acc@0.5 0.647, F1 0.773, balanced acc 0.650 |

Box-only control (no pixels, 30 epochs): OOF AUC 0.498, test AUC 0.616 [0.548, 0.687]. The box track alone has no cross-validated signal; its higher test AUC is more likely the train/test shift than skill.
The four recipes differ by less than the fold spread (±0.08-0.11), so the ranking is only suggestive; freezing the first 6 blocks was the best choice, consistent with overfitting being the problem.

### 5.2 B. Routed ViT, JAAD_beh (Kaggle, 2x T4; notebook `kaggle/B`, recipe r3_frozen6, 3 seeds x 5 folds)

| Variant | OOF AUC | Test AUC (ensemble) | Per-seed test AUC mean ± std | acc@0.5 / F1@0.5 |
|---|---|---|---|---|
| Dense (A) | 0.644 | 0.674 [0.600, 0.751] | 0.670 | 0.647 / 0.773 |
| B-soft | 0.650 | 0.694 [0.624, 0.767] | 0.673 ± 0.029 | 0.644 / 0.770 |
| B-sparse | 0.625 | 0.634 [0.557, 0.717] | 0.631 ± 0.036 | 0.653 / 0.774 |
| FiLM | 0.612 | 0.644 [0.563, 0.729] | 0.631 ± 0.028 | 0.622 / 0.756 |

| Hypothesis | Result |
|---|---|
| **H1** routing helps | **Not supported.** dAUC vs A: B-soft +0.003 [-0.022, +0.028]; B-sparse -0.039 [-0.097, +0.014]; FiLM -0.039 [-0.095, +0.015] |
| **H2** routing not collapsed | **Collapsed.** S above the decorrelated-z reference in 100% of cells, but S/log H = 0.002 (soft) and 0.001 (sparse); tightened pass rule (needs mean S/log H > 0.05) fails |
| **H3** routes track context given the label | Beyond the permutation null in 90% (soft) and 68% (sparse) of (seed, fold, layer, context) cells. With H2 collapsed this most likely reflects scene appearance in the router input, not use of context |
| **H4** swapping routes has context-specific effects | **Not supported** (0% of contexts for soft, 12% for sparse pass Gamma > 0 and swaps beating the magnitude-matched control) |
| **H5** selection drives specialization | Gradient-share entropy: dense 0.874, soft 0.841, sparse 0.721. Nominally supported, but sparse routing is collapsed (H2), so the lower entropy likely reflects a fixed head subset, not specialization. Do not read it as support |

Net reading (idea.md outcome table): B is not better than A and the routers are near-uniform, so the closest row is "router collapsed or nothing to exploit". The B-sparse changes (longer soft phase, lower router LR, stronger balancing) did not cure the collapse.

### 5.3 C. Pedestrian Graph+, JAAD_beh (local RTX 3050; `pedgraph/hpo.py`, `pedgraph/final.py`)
Search: Optuna (TPE + median pruning), each trial = 5-fold CV of one configuration, 120 trials (89 complete, 31 pruned, 3 stale trials from an interrupted session marked failed), then the top 8 re-scored with fresh seeds, then the image branch tested on the winner.

* Baselines under the same CV (mean fold AUC, seed 0): released recipe 0.655; regularised default 0.719.
* Best stage-1 score 0.822; **re-scored winner (trial 115) 0.825 ± 0.006** over fresh seeds (top 8 re-scores 0.783-0.825; the winner held up).
* Winner: box-relative pose coordinates + keypoint velocity, no side branch, identity adjacency, 2 graph stages of width 64 (kernel 5, head hidden 32), dropout 0.35, weight decay 0.03, label smoothing 0.03, 20 epochs one-cycle (lr 1.1e-3, batch 128), EMA 0.95, flip-TTA, flip aug 0.35, time crop 0.5.
* Image branch: best variant (gate, aug 1.0) 0.814 vs 0.825 pose-only; all six image variants were lower, so the image branch was **not kept**.

Final evaluation (5 seeds x 5 folds for the tuned model; 3 seeds for the others):

| Model | OOF AUC | Test AUC (ensemble) | Per-seed test AUC |
|---|---|---|---|
| Released recipe in our CV | 0.600 | 0.693 [0.618, 0.760] | 0.689 ± 0.004 |
| Regularised default (untuned) | 0.690 | **0.720** [0.652, 0.784] | 0.718 ± 0.004 |
| **Tuned** | **0.779** | 0.671 [0.595, 0.745] | 0.670 ± 0.006 |
| Ensemble of the top-3 configs | 0.771 | 0.666 [0.587, 0.739] | 0.664 ± 0.009 |

Ablations of the tuned model (OOF AUC with test AUC in brackets; reference 0.779 [0.670]): absolute pose instead of box-relative 0.637 [0.642]; abs+rel 0.712 [0.651]; no augmentation 0.746 [0.662]; ego-only side branch 0.763 [0.675]; with image branch (gate) 0.773 [0.642]; no flip-TTA 0.770 [0.672]; no label smoothing 0.777 [0.669]; no keypoint velocity 0.782 [0.659]; no EMA 0.783 [0.670]. Box-relative pose representation and augmentation are the components that matter in CV; EMA, label smoothing and TTA do not.

**Why tuned OOF 0.779 but test 0.671?** The CV gain is real within the pooled train+val videos (re-scored with fresh seeds, ablations consistent), but it does not carry to the official test videos: the untuned default and even the released recipe are better on test. Likely causes (not tested): (a) the test split differs in videos and base rate (63% vs 82% positive), so a representation tuned to the training videos' distribution (box-relative pose without position cues) loses signal that transfers; (b) a search over 120 configurations adapts to the 216 pooled pedestrians even with fresh-seed re-scoring (same folds); (c) test AUC itself is noisy (CI width about 0.15). The test split was never used for selection, so the tuned test number is an honest result, but it argues for selecting final models on a split that resembles the test distribution, or for a grouped CV with more videos per fold (e.g. JAAD_all, section 7).

### 5.4 D. GTransPDM (local RTX 3050; `pedgraph/run_gtrans.py`)

| Dataset | Protocol | Test AUC | Benchmark AUC (rounded preds) | acc@0.5 | F1@0.5 | Precision@0.5 | Balanced acc (OOF threshold) |
|---|---|---|---|---|---|---|---|
| JAAD_beh | paper (official split, 32 epochs, 5 seeds) | 0.713 [0.637, 0.783] (seeds 0.679-0.725) | 0.594 | 0.675 | 0.779 | 0.678 | 0.621 |
| JAAD_beh | 5-fold CV x 3 seeds (OOF AUC 0.594) | 0.721 [0.644, 0.785] | 0.599 | 0.684 | 0.787 | 0.680 | 0.635 |
| JAAD_all | paper (5 seeds) | **0.899** [0.868, 0.924] | 0.800 | 0.860 | 0.639 | 0.583 | 0.807 |
| JAAD_all | 5-fold CV x 3 seeds (OOF AUC 0.871) | **0.910** [0.882, 0.933] | 0.795 | 0.881 | 0.661 | 0.659 | 0.809 |

Comparison with the paper's JAAD_all row (accuracy 0.87, AUC 0.78, F1 0.64, precision 0.64): accuracy and F1 are close to ours (0.860 / 0.639), precision is lower (0.583), and the paper's AUC of 0.78 is nearest our rounded-prediction AUC (0.800), so the paper probably reports that variant. Majority-class accuracy on JAAD_all is 0.825, so 0.86-0.88 is a modest gain; AUC/balanced accuracy show the real signal. The OOF AUC on JAAD_all (0.87-0.94) is high because most negatives are non-behavioural pedestrians that are easy to separate.

---

## 6. Round 1 results (kept for history; official split, JAAD_beh, 3 seeds)

| Model | Test AUC | acc@0.5 | F1@0.5 |
|---|---|---|---|
| A dense ViT | 0.656 ± 0.020 | 0.632 | 0.760 |
| B-soft / B-sparse / FiLM | 0.630 / 0.545 / 0.631 | 0.627 / 0.546 / 0.624 | 0.762 / 0.601 / 0.755 |
| Pedestrian Graph+ (IVT) | 0.603 ± 0.013 | 0.633 | 0.735 |

Round 1 findings: H1 not supported for any variant (B-sparse significantly worse, -0.111); routers almost input-independent; H4 not supported; the dense ViT peaked on validation at epoch 1 in all seeds (the baseline was a one-epoch model); Pedestrian Graph+ validation AUC 0.80-0.88 vs test 0.60.

---

## 7. Paused / not done

* **Pedestrian Graph+ on JAAD_all (full settings, PAUSED):** 120-trial search with 3 workers, then re-score (top 8, 3 seeds), image-branch stage, and `pedgraph.final` (tuned 5 seeds x 5 folds, baselines, ablations, top-3 ensemble) into `results/round2/pedgraph_all/`. The search stopped with 18 complete and 5 pruned trials (best mean fold AUC 0.9215); baselines under 5-fold CV (mean fold AUC, seed 0): released recipe 0.843, regularised default 0.896. Roughly 40 more GPU hours on the RTX 3050 are needed. Resume with `bash scripts/resume_pg_all.sh` (finished trials are kept in the Optuna journal).
* No GTransPDM or Pedestrian Graph+ hyper-parameter tuning against the test split was done anywhere.
* PIE, the segmentation map / 3D keypoints variant of Pedestrian Graph+, and the paper-faithful Pedestrian Graph+ (needs the PDF) are not done.

---

## 8. Where everything is

| Item | Location |
|---|---|
| Docs | `README.md` (overview), `docs/idea.md` (routing spec), `docs/papers.md` (models and the papers behind them), `CLAUDE.md`, `kaggle_commands.md` |
| ViT package (model, data, CV, training, analysis) | `routed_vit/` (`recipes.py`: the 4 recipes and B-sparse settings; `cv.py`: folds) |
| Pedestrian Graph+ / GTransPDM package | `pedgraph/` (`fastdata.py`, `net.py`, `cvfit.py`, `hpo.py`, `final.py`, `gtranspdm.py`, `run_gtrans.py`; README inside) |
| Notebook generator | `build_notebooks.py` (writes `kaggle/A`, `B`, `C`; rerun after editing any package file) |
| Kaggle notebooks | A `kaggle.com/code/siddheshnadkarni/jaad-dense-vit-a`, B `.../jaad-routed-vit-b` (both ran on 2x T4); C (`kaggle/C`, needs the dataset from `kaggle/C/make_dataset.py`) not run on Kaggle |
| ViT results (CSV/JSON) | `results/round2/vit_A/` (`results_A.csv`, `results_boxonly.csv`, `recipe.json`, `summary_A.json`), `results/round2/vit_B/` (`results_B.csv`, `hypotheses_B.csv`, `H2-H5.csv`); full kernel outputs stay on Kaggle |
| Pedestrian Graph+ JAAD_beh | `results/round2/pedgraph_beh/` (`stage2.csv`, `image_stage.csv`, `final_cfg.json`, `trials.csv`, `final/summary.csv`, `final/ablation.csv`, `final/preds.npz`) |
| Pedestrian Graph+ JAAD_all | `results/round2/pedgraph_all/` (paused, partial) |
| GTransPDM | `results/round2/gtranspdm/beh/`, `results/round2/gtranspdm/all/` (`summary_paper.csv`, `summary_cv.csv`, `preds_*.npz`) |
| Round 1 runs / results | `results/round1/pedgraph_IVT/`, `results/round1/vit_A/`, `results/round1/vit_B/` |
| Data | `data/JAAD` (annotations, clips), `data/samples_ctx/jaad_beh.pkl`, `jaad_all.pkl`, `data/pose_beh`, `data/pose_all` |
| Logs | `logs/` (`hpo_*.log`, `final_*.log`, `gtrans_*.log`, `queue.log`) |

Reproduce locally (from this directory): `python -m pedgraph.hpo ...`, `python -m pedgraph.final ...`, `python -m pedgraph.run_gtrans ...`; commands in `pedgraph/README.md`. Kaggle ViT notebooks: `python -m kaggle kernels push -p kaggle/A --accelerator NvidiaTeslaT4`, then B (A first: B reads A's recipe and crop cache).

---

## 9. Suggested next steps

1. Wait for Pedestrian Graph+ on JAAD_all and compare it with GTransPDM's 0.90-0.91 there; check whether the CV-to-test gap seen on JAAD_beh also appears on JAAD_all (larger, 783 train pedestrians).
2. Investigate the CV-to-test gap for Pedestrian Graph+ directly: tune on a split that mimics the test distribution (e.g. hold out whole videos with the test base rate), or restrict the search to representations that include position cues (abs+rel scored 0.712 OOF / 0.651 test, so this needs testing).
3. Give the Pedestrian Graph+ authors' paper PDF to the reimplementation (needs your access) and diff it against the code-based version.
4. Routing (B): the collapse is robust to the sparse-training changes; if the idea is pursued, give the router an informative input (box size / ego state) or a stronger anti-collapse objective, then rerun H2-H4.
5. Add PIE and more seeds for statistical power; JAAD_beh test AUC CIs are about ±0.07.

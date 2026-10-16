# Pedestrian Graph+ on JAAD (literature-map Cluster 5, entry 5.4)

Clean-room PyTorch reimplementation of Cadena et al., *Pedestrian Graph+*, IEEE T-ITS 2022, built from the authors' public code
(github.com/RodrigoGantier/Pedestrian_graph_plus, MIT). The paper PDF itself was not accessible (IEEE paywall, ResearchGate 403).

## Layout

| file | role |
|---|---|
| `model.py` | the reimplemented network; parameter names/shapes are identical to the released checkpoints (`check_against_checkpoint`), 70 k params (2D, image + vehicle, binary head) |
| `extract.py` | poses (ViTPose-B on the annotated boxes, 17 COCO + neck + pelvis = 19 nodes) and the 192x64 local-context crop, cached per video |
| `data.py`, `train.py` | **round 1** (legacy): torch Dataset + single run per seed on the official split, best-epoch checkpoint by val AUC. Kept for reference |
| `fastdata.py` | **round 2**: all 4257 windows as GPU tensors; vectorised feature construction and augmentation |
| `net.py` | configurable `PedGraphNet` (+ `Released` wrapper of `model.PedGraphPlus`) |
| `cvfit.py` | fixed-budget training with EMA, K-fold video-grouped CV (`run_cv`), `DEFAULT_CFG`, `RELEASED_CFG` |
| `hpo.py` | Optuna search (stages 0-3, below) |
| `final.py` | tuned model x seeds x folds, baselines, ablations, top-config ensemble |
| `report.py` | result tables (OOF AUC, test AUC with pedestrian-bootstrap CI, thresholds from OOF) |

## Why round 2

Round 1 trained once per seed and kept the best of 30 epochs by validation AUC on **22 pedestrians**. Validation AUC moved between
0.47 and 0.88 from one epoch to the next (seed 42) and the train loss went to 0.05, so the "trained" model was a noise-selected
checkpoint of an overfit network (val 0.80-0.88, test 0.60). Round 2 changes the protocol, not just the model:

* train+val pooled (216 pedestrians), 5 video-grouped folds (`routed_vit/cv.py`, identical to the ViT notebooks);
* fixed epoch budget, EMA weights, no checkpoint picking; the selection signal is the **out-of-fold AUC**;
* the official test split is only predicted (fold models x seeds averaged) and never used for any choice.

## Run (from this `Siddhesh/` directory; fits a 6 GB GPU, pose-only trials take about 3 minutes)

```bash
# data preparation (unchanged from round 1)
python ../download_jaad.py --root data/JAAD --clips
python ../build_jaad_samples.py --jaad_root data/JAAD --out data/samples --sample_type beh
python -m pedgraph.extract --samples data/samples_ctx/jaad_beh.pkl --clips data/JAAD/JAAD_clips --out data/pose_beh

# search: baselines -> Optuna (resumable) -> re-score top-K with fresh seeds -> image-branch test
python -m pedgraph.hpo --samples data/samples_ctx/jaad_beh.pkl --cache data/pose_beh --out results/round2/pedgraph_beh --trials 150 --parallel 3

# final evaluation: tuned model (5 seeds x 5 folds), baselines, ablations, ensemble of the top-3 configurations
python -m pedgraph.final --samples data/samples_ctx/jaad_beh.pkl --cache data/pose_beh --hpo results/round2/pedgraph_beh --out results/round2/pedgraph_beh/final \
       --seeds 5 --baseline_seeds 3 --ablate --ensemble_topk 3
```
On Kaggle use `kaggle/C/jaad_pedgraph_C.ipynb` (`kaggle/C/make_dataset.py` packages the inputs). `--max_epochs 3` on either command is a
smoke-test switch.

## What the search covers (`hpo.py::suggest`)

* input: absolute / box-relative / both pose coordinates, keypoint velocity; side stream = box trajectory and/or ego action (one-hot);
* network: widths (2-3 stages), temporal kernel, dropout, adjacency initialised as identity (released) or [identity, skeleton, 2-hop],
  fusion of side/image features by gating (released), concatenation, or both, optional hidden layer in the head;
* optimisation: epochs, batch size, LR, weight decay, one-cycle vs cosine, label smoothing, EMA decay, flip-TTA;
* augmentation: keypoint jitter (proportional to person size), zoom/shift, left-right flip (joints swapped), joint drop-out, time crop.

Stage 1 scores a configuration by the mean fold AUC of one seed. Picking the best of ~100 noisy scores is optimistic, so stage 2
re-runs the top-K with fresh seeds, and stage 3 adds the image branch (gate / concat / both) to the winner and keeps it only if the
re-scored AUC improves by more than `--image_margin` (default 0.005). The image branch is excluded from stage 1 because it is
3-10x more expensive per trial; stage 3 and the ablation table then measure what it adds.

## Differences from the paper / released code (be explicit when comparing numbers)
* 2D keypoints from ViTPose; the paper's best model uses 3D keypoints + a segmentation map (S) + pose forecasting (F) from their own pre-extracted data. Not included here.
* 16 observed frames (PCPA protocol), the released code uses 32. Binary head: no 'irrelevant' class.
* `RELEASED_CFG` is the released recipe inside the CV pipeline with our GPU augmentations (the released PIL stack with posterise / invert /
  solarize is replaced; keypoint jitter is Gaussian and proportional to person size instead of +-4/+-3 px) and a 96x32 crop (`img_down=2`).
* The released stage-1 image gate is a no-op; kept as is in `model.py` (`stage1_image_gate=True` applies it). `PedGraphNet` gates after every stage.
* Crop margin (10%) for the local-context image is an assumption; the paper's crop recipe is not in the released code.

## Caveat on interpreting CV vs test
Pooled train+val is 82% positive, the official test split 63%, and the test videos differ from the training videos. Cross-validated
AUCs (all-videos-seen-at-train-time distribution) can therefore be higher than the test AUC; the CV protocol removes selection noise,
it does not remove this shift. Both numbers are reported side by side in every table.

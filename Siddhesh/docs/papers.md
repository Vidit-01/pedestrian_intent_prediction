# Models and the papers they build on

Verified online: VIT-Ped (arXiv 2601.01989), GTransPDM (arXiv 2409.20223) and the Pedestrian Graph+ DOI / GitHub code. The supporting papers (ViT, ViViT, ViTPose, MoE, FiLM, etc.) were cited from memory and not re-checked; verify them before putting them in a report.

## The three models

| | Model | Code | Main source paper |
|---|---|---|---|
| A | Dense PedViViT-style ViT | `routed_vit/model.py` (`variant='dense'`) | VIT-Ped / PedViViT, arXiv 2601.01989 |
| B | Routed ViT (B-soft, B-sparse, FiLM control) | `routed_vit/model.py` (`soft`, `sparse`, `film`), spec in `docs/idea.md` | PedViViT backbone + mixture-of-experts style routing (own design, see below) |
| D | GTransPDM | `pedgraph/gtranspdm.py` | Xie, Lin, Zheng, Gong, Lopez, *GTransPDM*, IEEE Signal Processing Letters 2025, arXiv 2409.20223 (map entry 5.3) |
| C | Pedestrian Graph+ | `pedgraph/model.py` (released form), `pedgraph/net.py` (configurable form) | Cadena, Qian, Wang, Yang, *Pedestrian Graph+*, IEEE T-ITS 2022 |

## Model A: dense PedViViT-style ViT
Per-frame ViT-S/16 spatial encoder, a 2-layer temporal transformer over the per-frame CLS tokens, and a small transformer on delta-encoded bounding boxes, fused by concatenation into one sigmoid output.

- **PedViViT / VIT-Ped**, arXiv 2601.01989: the overall architecture. The paper leaves most sizes unspecified, so the ViT-S choice, widths and fusion sizes are ours.
- **ViViT: A Video Vision Transformer**, Arnab et al., ICCV 2021: the factorised encoder (spatial transformer, then temporal transformer).
- **An Image is Worth 16x16 Words (ViT)**, Dosovitskiy et al., ICLR 2021: the spatial encoder.
- **How to train your ViT?**, Steiner et al., 2021: the AugReg ImageNet-21k pretrained ViT-S/16 weights (`vit_small_patch16_224.augreg_in21k_ft_in1k` via `timm`).

Training techniques used in round 2:
- **Stochastic depth**: Huang et al., *Deep Networks with Stochastic Depth*, ECCV 2016.
- **AdamW**: Loshchilov and Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019.
- **Layer-wise learning-rate decay**: a standard ViT fine-tuning practice popularised by ELECTRA (Clark et al., 2020) and BEiT (Bao et al., 2022).
- **Label smoothing**: Szegedy et al., *Rethinking the Inception Architecture*, CVPR 2016.
- **EMA of weights**: standard practice (Polyak averaging); no single source.

## Model B: routed ViT
Same backbone as A. In blocks 7-12 each attention head output is multiplied by a per-clip gate produced by a router.

- **Sparse top-k routing with noisy selection**: Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, ICLR 2017.
- **Load-balancing loss**: Fedus et al., *Switch Transformers*, JMLR 2022.
- **Router z-loss**: Zoph et al., *ST-MoE*, 2022.
- **FiLM control** (unconstrained per-channel scale and shift): Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018.
- The head-gating formulation, the nesting property (B-soft equals A at initialisation) and hypotheses H1-H5 are our own design, written up in `docs/idea.md`. Gating attention heads per sample is not taken from one specific paper.

## Model C: Pedestrian Graph+
Graph convolution over a 19-joint skeleton with temporal convolutions, gated by a local-context image branch and an ego-vehicle branch.

- **Pedestrian Graph+**: Cadena, Qian, Wang, Yang, IEEE Transactions on Intelligent Transportation Systems, 2022 (literature map entry 5.4; paper https://doi.org/10.1109/TITS.2022.3173537; authors' open-source code https://github.com/RodrigoGantier/Pedestrian_graph_plus, MIT licence). **The paper itself was not read** (paywalled); the model was implemented from the authors' public code (github.com/RodrigoGantier/Pedestrian_graph_plus, MIT licence), and the parameter names and shapes match their released checkpoints.
- **ST-GCN** (lineage only, not verified as cited by the Pedestrian Graph+ authors): Yan et al., *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*, AAAI 2018: skeleton graph plus temporal convolution.
- **Adaptive graph convolution (2s-AGCN)** (lineage only, same caveat): Shi et al., CVPR 2019: learnable adjacency matrices, which the released network uses.
- **ViTPose** (pose extraction): Xu et al., *ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation*, NeurIPS 2022. We use the HuggingFace checkpoint `usyd-community/vitpose-base-simple` on the annotated boxes.
- **Optuna** (search): Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019, with the TPE sampler (Bergstra et al., NeurIPS 2011).

## Model D: GTransPDM
Pose-skeleton graph encoder (4 GCN layers) plus a Positional Decoupling Module (offsets to two reference lines, area-ratio depth cue), ego-motion encoder and a 4-layer Transformer.
- **GTransPDM** (arXiv 2409.20223, v2 May 2025; confirmed on arXiv, open access): the paper was read in full (arXiv HTML) and the model reimplemented because no code is public. Choices the paper leaves open (reference-point rule, Conv2D kernel, skeleton edges, positional encoding, FFN width, input normalisation, AlphaPose replaced by ViTPose) are listed in the docstring of `pedgraph/gtranspdm.py`.
- Evaluated on JAAD_all and JAAD_beh; on JAAD_all it matches the paper's accuracy and F1 (0.860 / 0.639 vs 0.87 / 0.64).

## Dataset and protocol
- **JAAD**: Rasouli, Kotseruba, Tsotsos, *Are They Going to Cross? A Benchmark Dataset and Baseline for Pedestrian Crosswalk Behavior*, ICCV Workshops 2017.
- **PCPA benchmark protocol** (16 observed frames, time-to-event 30-60, stride 3, official video split): Kotseruba, Rasouli, Tsotsos, *Benchmark for Evaluating Pedestrian Action Prediction*, WACV 2021, as implemented in the base repo `build_jaad_samples.py`.

## Baselines in the base repo (not part of this study)
`pedestrian_intent_prediction/` also contains a GRU baseline and ACIT (`models/gru.py`, `models/acit.py`). Those belong to the base repo's own spec (`Vidit.md`).


## Round 2 improvements are not paper-derived
Cross-validation, EMA, augmentation, box-relative pose features, skeleton-initialised adjacency and the Optuna search are generic methods chosen by us, not taken from a specific pedestrian-intent paper.

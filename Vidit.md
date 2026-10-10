from pathlib import Path
import pypandoc

content = r"""# Pedestrian Intention Prediction: Baselines A and B

## Purpose

This document defines two reproducible baselines to implement before developing a new architecture that combines predictive representation learning (JEPA-inspired objectives) with established sequence and multimodal techniques:

- **Baseline A — conventional multimodal sequence model:** a compact GRU-based classifier, with PCPA as the preferred literature reference/reproduction target if its implementation and dependencies are practical.
- **Baseline B — ACIT:** an attention-centric multimodal architecture using visual appearance, optical flow, semantic maps, vehicle speed, and pedestrian bounding boxes, followed by cross-modal fusion and temporal Transformer encoding.

The immediate objective is not to reproduce every pedestrian-intention paper. It is to establish trustworthy reference points for (1) what ordinary sequential/multimodal learning can do, and (2) what explicit cross-modal attention adds. These baselines will later let us test whether future-latent prediction improves intent prediction.

## 1. Research question and comparison logic

The planned novel model is expected to use ideas such as JEPA-style latent prediction or decision-oriented learning together with established sequence modelling. The baselines must therefore isolate the contribution of those ideas.

| Baseline | What it controls for | Main question |
|---|---|---|
| A: GRU / PCPA-style multimodal model | Standard supervised temporal learning and straightforward fusion | How well can the task be solved from observed appearance and/or kinematics without a novel predictive objective? |
| B: ACIT | Explicit cross-modal attention before temporal aggregation | Does a more elaborate attention-based fusion design improve over conventional fusion? |
| Future proposed model | Predictive latent objective and/or decision-oriented objective | Does the new learning objective improve early prediction under matched conditions? |

**Important:** do not compare published numbers as if they were directly comparable unless the dataset subset, train/validation/test split, prediction horizon, observation window, preprocessing, and metrics match. A reported score from a paper is a reference, not a substitute for a local reproduced baseline.

## 2. Dataset and protocol

### 2.1 Start with JAAD

Use the official JAAD release and its annotation interface:

- Official repository: https://github.com/ykotseruba/JAAD
- Dataset paper: Kotseruba, Rasouli, and Tsotsos, *Joint Attention in Autonomous Driving (JAAD)*, 2016. Search/reference page: https://arxiv.org/abs/1609.04741

JAAD provides driving videos and annotations for pedestrians and their behaviour. It is not a turnkey package of all model inputs used in recent papers. RGB frames and annotation-derived boxes/labels are available, while optical flow, semantic maps, depth, pose, and some kinematic variables may need to be generated or obtained from a paper-specific preprocessing pipeline.

Use the official repository documentation and annotation schema as the source of truth for field names and label meanings. Do not assume every annotation field exists for every pedestrian or every frame.

### 2.2 JAAD-all and JAAD-beh

These names are commonly used for different pedestrian populations/evaluation settings in the JAAD literature. Their exact definitions can vary with the paper's preprocessing and protocol. In this project:

1. Record the exact filtering rule and sample counts used by the selected implementation.
2. Do not infer that all labels are available for every JAAD-all pedestrian.
3. Treat JAAD-beh as a behaviour-annotated subset only when that matches the cited implementation's definition.
4. Keep the selected protocol fixed for Baseline A, Baseline B, and the proposed model.

Store a manifest of video IDs, pedestrian IDs, frame ranges, labels, and split assignment. Split by video (or by the official split if provided by the reference implementation), not by individual frames. Otherwise, adjacent frames from the same clip can leak into train and test sets.

### 2.3 Observation window and prediction horizon

Choose one protocol before training and keep it fixed. For an initial reproduction, a practical target is:

- Input: a short observed sequence, e.g. 16 frames.
- Prediction target: crossing intention at a clearly specified anticipation horizon.
- Sampling rate: document the actual frame stride and resulting duration.

Do not assume that “16 frames” always means the same duration. At 30 FPS, 16 consecutive frames span roughly half a second if measured from first to last timestamp; with frame subsampling, the duration differs. Papers may also define anticipation relative to crossing onset differently. Follow the chosen paper's exact protocol when reproducing it, and use a separate matched protocol for cross-model comparisons.

Define the label formally. For example, if predicting crossing at horizon \(h\), specify whether the label means crossing begins within a future interval, the pedestrian is crossing at time \(t+h\), or a frame-level state copied from annotations. These definitions are not interchangeable.

### 2.4 Metrics

Report at minimum:

- **AUC-ROC:** ranking performance across thresholds.
- **F1:** threshold-dependent precision/recall balance.
- **Balanced accuracy:** average recall over the two classes.
- **Precision and recall:** to expose false-positive and false-negative behaviour.
- **Calibration** (e.g. Brier score or reliability curve) if predicted probabilities will be interpreted as uncertainty or used in decision-making.

Use the same threshold-selection policy across models. Select a threshold on the validation set only; do not tune it on the test set. Report per-video or per-seed variation where practical.

## 3. Baseline A — conventional multimodal sequence model

### 3.1 Recommended implementation target

Implement a **small GRU-based temporal classifier first**, then use PCPA as the literature baseline to reproduce or compare against. This keeps the first working model manageable and gives a clean reference before adding optical flow, semantic segmentation, or large visual backbones.

PCPA means *Pedestrian Crossing Prediction with Attention* and is a relevant established multimodal pedestrian-intention architecture. The paper and official implementation should be checked for their exact input modalities, attention design, and evaluation protocol before claiming an exact reproduction.

Sources:
- PCPA paper search / publication record: https://scholar.google.com/scholar?q=Pedestrian+Crossing+Prediction+with+Attention+PCPA
- JAAD official dataset repository: https://github.com/ykotseruba/JAAD
- PIE dataset and benchmark (useful additional benchmark/reference): https://github.com/arasouli/PIE

If an exact PCPA implementation is available and its dependencies work, reproduce it using its published configuration. If not, label the GRU implementation **“GRU multimodal baseline inspired by conventional pedestrian-intention models”**, not “PCPA reproduction.”

### 3.2 Inputs

Build the first version from annotation-derived features so that it can run before the visual preprocessing pipeline is complete.

**Minimal kinematic input per frame:**
- Pedestrian bounding box: \([x_{\min}, y_{\min}, x_{\max}, y_{\max}]\), normalized by frame width and height.
- Optional box-derived features: normalized centre \((c_x,c_y)\), width, height, and frame-to-frame displacement.
- Ego-vehicle speed, if available and reliably aligned to the frame.
- Optional annotation fields such as occlusion, only if available at inference time and their use is documented.

Do not accidentally include future-derived information (for example, a box trajectory computed using frames after the observation cutoff). Any velocity or displacement feature must use only observed frames.

**Visual extension (second step, not required for the first smoke test):**
- Crop the pedestrian bounding box from each observed RGB frame.
- Encode each crop using a pretrained CNN or ViT.
- Optionally encode the full scene separately.
- Fuse visual features with the kinematic sequence.

This staged implementation helps determine whether data parsing, labels, temporal alignment, and split logic are correct before adding an expensive image backbone.

### 3.3 Reference architecture

For input sequence \(X \in \mathbb{R}^{B \times T \times F}\):

1. Linear projection: \(F \rightarrow d\), e.g. \(d=128\).
2. One- or two-layer GRU with hidden size 128.
3. Use the final valid hidden state (or a masked temporal pooling operation).
4. MLP classification head, e.g. \(128 \rightarrow 64 \rightarrow 1\).
5. Sigmoid probability for binary crossing classification.

For a multimodal version, use separate encoders and fuse their outputs:

- Kinematics: GRU over box/speed features.
- Pedestrian appearance: CNN/ViT over pedestrian crops, followed by a GRU over frame embeddings.
- Optional scene appearance: a separate image encoder and temporal aggregation.
- Fusion: concatenate modality vectors and apply an MLP.

Keep the initial architecture small. Avoid adding attention, Mamba, depth, optical flow, and a JEPA loss simultaneously; doing so makes the baseline hard to interpret.

### 3.4 Training

- Loss: binary cross-entropy with logits.
- Optimizer: AdamW.
- Use a validation set for early stopping and threshold selection.
- If classes are imbalanced, report the unweighted result first, then evaluate class weighting or a sampler as a separate ablation.
- Fix random seeds and log the exact data split.
- Save model configuration, preprocessing configuration, checkpoint, validation metrics, and test predictions.

A sensible initial training setup is a learning rate around \(10^{-3}\) for a small GRU, batch size chosen to fit memory, and early stopping on validation AUC. These are starting hyperparameters, not claims about the original PCPA configuration. Tune only on validation data.

### 3.5 Required ablations

Run the following in order:

1. Bounding-box sequence only.
2. Bounding boxes + ego speed.
3. Add box-derived displacement/velocity computed from observed frames.
4. Add pedestrian RGB crop features.
5. Add full-scene features, if available.

These ablations quantify how much each input source contributes. Report parameter count and inference latency alongside predictive metrics.

### 3.6 Baseline A acceptance criteria

Do not proceed to architecture comparisons until:
- Sample generation is deterministic.
- No video or pedestrian identity crosses splits in a way that leaks near-duplicate frames.
- Labels and anticipation horizon have been manually checked on sample clips.
- A majority-class and simple non-temporal baseline have been measured.
- Validation performance is stable across at least three seeds, if compute permits.
- Test predictions are saved once and not repeatedly used for model selection.

## 4. Baseline B — ACIT

### 4.1 Paper

**ACIT: Attention-Guided Cross-Modal Interaction Transformer for Pedestrian Crossing Intention Prediction.**

Paper: https://arxiv.org/abs/2511.20020  
PDF: https://arxiv.org/pdf/2511.20020

Use the paper itself as the source of truth for architecture and evaluation details. The following is a working architecture summary; verify exact dimensions and implementation choices against the PDF and any official code before treating it as an exact reproduction.

### 4.2 Inputs and preprocessing

ACIT uses six input modalities, grouped into visual and kinematic inputs:

**Visual**
1. Local RGB pedestrian crop.
2. Local optical flow.
3. Global semantic segmentation map.
4. Global optical flow.

**Kinematic**
5. Ego-vehicle speed.
6. Pedestrian bounding box.

The paper describes SegFormer for semantic segmentation and RAFT for optical flow, with Swin Transformer V2-Base visual encoding. This preprocessing is more expensive than Baseline A, so cache generated features/maps to avoid recomputing them during every training epoch.

Important implementation questions to settle from the paper/code:
- Exact video frame stride and observation duration.
- Whether the four visual branches share all or part of their Swin weights.
- Optical-flow normalization and channel representation.
- Semantic class palette and conversion to model input.
- Whether the speed signal is available for the chosen dataset and how missing values are handled.
- Exact label horizon and dataset split.

Do not substitute estimated speed, depth, or segmentation labels without documenting the substitution.

### 4.3 Visual encoding

The paper's visual backbone is Swin Transformer V2-Base. The described pipeline resizes inputs to \(256 \times 256\), produces a spatial feature map, and reduces channels to 256 with a \(1\times1\) convolution. The resulting feature map is described as \(8\times8\times256\), i.e. 64 spatial tokens per frame.

Two visual pairs are processed:
- Local RGB with local optical flow.
- Global semantic map with global optical flow.

### 4.4 AVMI — visual cross-modal interaction

The Attention-based Visual Modality Interaction module fuses each visual pair.

At a high level:
- An appearance/semantic feature supplies queries for self-attention.
- In a second path, appearance/semantic features query optical-flow keys and values.
- The paths are combined with a residual connection and a learnable contribution weight.
- Spatial tokens are pooled to obtain a 256-dimensional local or global representation per frame.

Conceptually:
\[
F_L = \operatorname{AVMI}(F_{\text{local RGB}}, F_{\text{local flow}})
\]
\[
F_G = \operatorname{AVMI}(F_{\text{global semantic}}, F_{\text{global flow}})
\]

The exact attention equations, head dimensions, normalization and learnable scalar should be copied from the paper or official implementation rather than reconstructed from this high-level description.

### 4.5 AMMI — kinematic cross-modal interaction

Speed and pedestrian bounding-box features are separately projected to 256 dimensions. Bidirectional cross-attention lets the speed representation attend to box features and vice versa. The outputs are residual-connected, normalized, and passed through a feed-forward network to produce a 256-dimensional motion representation.

This is a learned interaction, not simple concatenation: the model can represent relationships between vehicle motion and pedestrian location.

### 4.6 MMFF — modality fusion

At each time step, the local visual, global visual, and motion representations interact through cross-modal attention. The three 256-dimensional representations are concatenated to produce a 768-dimensional vector per frame.

For a 16-frame input, the temporal input is therefore:
\[
X_{\text{fused}} \in \mathbb{R}^{B \times 16 \times 768}.
\]

Confirm the exact tensor shape at each implementation stage; dimensions above are based on the paper's described configuration.

### 4.7 Temporal Transformer and classifier

The fused sequence receives positional encoding and a CLS token, then passes through a two-layer Transformer encoder with six attention heads. The CLS representation is fed to an MLP to produce the crossing probability.

The paper reports ablations in which replacing temporal attention with pooling, replacing MMFF with addition, or simplifying the visual interaction reduces performance. Reproduce those ablations only after the main implementation is correct; otherwise they add debugging work without establishing a trustworthy baseline.

### 4.8 Implementation plan for ACIT

Implement in this order:

1. **Data loader:** return synchronized RGB crop, full-frame RGB/semantic input, flow, box, speed, label, video ID, pedestrian ID, and timestamps.
2. **Preprocessing validation:** visualize RGB crops, semantic maps, and optical flow side by side for randomly selected clips.
3. **Cached visual features:** run the visual backbone and store outputs to disk if training compute is limited.
4. **AVMI:** test shapes and gradients on a single batch.
5. **AMMI:** verify the speed and box projections and bidirectional attention.
6. **MMFF:** check that three 256-D streams yield 768-D features per frame.
7. **Temporal Transformer + head:** overfit a tiny subset to check the end-to-end training path.
8. **Full training:** use the paper's protocol and hyperparameters where available.
9. **Ablations:** only after reproducing the main result.

Use a small synthetic batch to verify each module before running a full dataset. Confirm that gradients reach all intended trainable components, that no target-frame information enters the input, and that padding/missing modalities are handled consistently.

### 4.9 Compute and engineering considerations

ACIT has significantly more preprocessing and visual computation than the kinematics-only baseline:
- RAFT flow generation can be expensive.
- Semantic segmentation must be run for the relevant frames.
- Swin-B visual encoding is relatively heavy.
- Four visual modalities and temporal batching increase storage requirements.

Practical mitigations:
- Generate each modality once and cache it.
- Store arrays in an efficient format with a manifest linking each record to video, pedestrian, and timestamp.
- Start with a small number of videos to validate the pipeline.
- Do not change input resolution or backbone without marking the experiment as a modified ACIT baseline.
- Keep the full paper-faithful configuration separate from any resource-constrained variant.

## 5. Fair comparison protocol

Use a single shared evaluation script for Baseline A, ACIT, and the proposed model.

Record these fields for every experiment:

| Field | Required record |
|---|---|
| Dataset/version | Exact download/version and annotation configuration |
| Subset | JAAD-all, JAAD-beh, or another precisely defined subset |
| Split | Video IDs in train/validation/test |
| Input window | Frame count, stride, duration |
| Prediction horizon | Exact label definition and horizon |
| Modalities | Every input and its source |
| Preprocessing | Model/version and input resolution |
| Architecture | Layer counts, dimensions, heads, parameter count |
| Training | Loss, optimizer, learning rate, batch size, epochs, seeds |
| Evaluation | AUC, F1, balanced accuracy, precision, recall |
| Efficiency | Inference latency, hardware, batch size |
| Reproducibility | Commit hash, config file, checkpoint identifier |

Use at least three random seeds where practical. Report mean and standard deviation, not just the best run. Keep the final test set untouched until architecture and hyperparameter selection are complete.

### Essential controls

- **Majority-class baseline:** shows the accuracy attainable without useful prediction.
- **Box-only baseline:** establishes how much the model can infer from pedestrian location and motion.
- **Visual-only baseline:** useful for separating appearance/context from kinematics.
- **Ablation without cross-modal interaction:** tests whether ACIT's interaction modules help beyond simple fusion.
- **Ablation without latent prediction:** later, this will be the key control for the proposed JEPA-inspired method.

## 6. How these baselines support the novel model

The later architecture should be evaluated as a controlled change, not a wholesale replacement of every component.

A useful experimental progression is:

1. Baseline A: supervised GRU with conventional features.
2. Baseline B: ACIT with its multimodal attention pipeline.
3. Baseline A + future-latent prediction objective, holding the classifier/encoder as stable as possible.
4. The best conventional architecture + latent prediction.
5. Ablations that remove latent prediction, remove future targets, or replace the predictive objective with an ordinary auxiliary supervised objective.

The key claim to test is whether future latent prediction adds value beyond ordinary supervised temporal learning and beyond ACIT's cross-modal attention. If the proposed model changes backbone, fusion, loss, and preprocessing simultaneously, any performance difference will be difficult to attribute to JEPA.

## 7. Primary sources

1. **JAAD official repository and annotation interface:** https://github.com/ykotseruba/JAAD
2. **JAAD paper:** https://arxiv.org/abs/1609.04741
3. **ACIT paper:** https://arxiv.org/abs/2511.20020
4. **PIE dataset and benchmark repository:** https://github.com/arasouli/PIE
5. **PCPA paper search:** https://scholar.google.com/scholar?q=Pedestrian+Crossing+Prediction+with+Attention+PCPA

Before claiming exact reproduction, consult each paper's full methods, supplementary material, official repository, and split files. If an official implementation differs from the paper text, record which version was followed.

## Final recommendation

Implement **Baseline A first** and make the data protocol reliable. Then implement **ACIT as Baseline B** if the cross-modal-attention comparison is central to the paper. Do not start with a JEPA objective until both the label construction and ordinary supervised baselines are trustworthy. The resulting experiment will answer a meaningful question: whether predictive latent learning contributes beyond established temporal modelling and multimodal attention.
"""

output_path = Path("/mnt/data/pedestrian_intent_baselines_A_B.md")
pypandoc.convert_text(content, "md", format="md", outputfile=str(output_path), extra_args=["--standalone"])
print(f"Created: {output_path}")

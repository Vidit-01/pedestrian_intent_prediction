# Dense vs. Routed ViT for Pedestrian Intent: Formal Specification

Two models, identical except for how per-sample gates act on the attention heads of the later layers.

- **Variant A (dense):** every head contributes equally for every sample.
- **Variant B (routed):** a router reads an early representation and assigns each sample a gate over heads.
  - **B-soft** (all heads active, reweighted per sample): exactly what you described.
  - **B-sparse** (top-$k$ heads per sample): the only regime in which samples are literally *routed to different heads*. B-soft reweights; it does not route. Prop. 2 shows why the distinction matters.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $x_i$ | Input sample (pedestrian clip, $T$ frames of a pedestrian-centred crop) |
| $y_i\in\{0,1\}$ | Crossing label |
| $c_i$ | Annotated context (signal, crosswalk, ego speed bin, group size, …), never fed to the model |
| $L$, $H$, $d$, $d_k=d/H$ | Depth, heads per layer, width, head width |
| $N$ | Tokens per attention call |
| $L_0$ | Last dense layer; layers $\mathcal{R}=\{L_0+1,\dots,L\}$ are routed in B |
| $\theta^{l,h} = (W_Q^{l,h},W_K^{l,h},W_V^{l,h},W_O^{l,(h)})$ | Parameters of head $h$ in layer $l$ |

## 2. Shared backbone

**Tokens.** $X^0 = [\,x_{\text{cls}};\ E(x)\,] + P \in \mathbb{R}^{(N+1)\times d}$.

**Block $l$ (pre-LN):**
$$
U^l = X^{l-1} + \mathrm{MHA}_l\big(\mathrm{LN}(X^{l-1})\big),\qquad
X^l = U^l + \mathrm{MLP}_l\big(\mathrm{LN}(U^l)\big).
$$

**Head $h$:**
$$
O^{l,h}(X) = \mathrm{softmax}\!\Big(\tfrac{(XW_Q^{l,h})(XW_K^{l,h})^\top}{\sqrt{d_k}}\Big)\,XW_V^{l,h}\ \in\mathbb{R}^{(N+1)\times d_k}.
$$

**Decomposition identity.** Split $W_O^l\in\mathbb{R}^{d\times d}$ into row blocks $W_O^{l,(h)}\in\mathbb{R}^{d_k\times d}$. Then
$$
\mathrm{Concat}(O^{l,1},\dots,O^{l,H})\,W_O^l = \sum_{h=1}^{H} O^{l,h}\,W_O^{l,(h)}.
$$
So MHA is a **sum of per-head contributions**, and a gate is simply a scalar on each term.

**Readout.** $\hat p_i = \sigma\big(w^\top \mathrm{LN}(X^L_{\text{cls}}) + b\big)$.

---

## 3. Variant A — Dense

For all $l$:
$$
\mathrm{MHA}^{A}_l(X)=\sum_{h=1}^{H} O^{l,h}(X)\,W_O^{l,(h)} .
$$

## 4. Variant B — Routed

Layers $l\le L_0$ are identical to A.

### 4.1 Routing state

One state per sample (pedestrian), computed once:
$$
z_i = \mathrm{LN}\big(X^{L_0}_{i,\text{cls}}\big)\in\mathbb{R}^d .
$$
For a frame-wise backbone, use $z_i=\mathrm{LN}\big(\tfrac1T\sum_t X^{L_0}_{i,t,\text{cls}}\big)$, which gives one route per clip.

### 4.2 Router

One router per routed layer, all reading the same $z$. Head indices are not aligned across layers, so each layer needs its own map.
$$
s^l_i = W_r^l z_i + b_r^l \in \mathbb{R}^H,\qquad \pi^l_i = \mathrm{softmax}(s^l_i/\tau).
$$

### 4.3 Gates

Both regimes are normalized so that $\sum_h g^l_{i,h} = H$. This keeps the residual-stream scale comparable to A.

**B-soft:**
$$
g^l_{i,h} = H\,\pi^l_{i,h}.
$$

**B-sparse** (selected set $S^l_i = \mathrm{TopK}(s^l_i + \varepsilon^l_i,\,k)$):
$$
g^l_{i,h} = H\,\mathbb{1}[h\in S^l_i]\;\frac{e^{s^l_{i,h}/\tau}}{\sum_{j\in S^l_i}e^{s^l_{i,j}/\tau}},\qquad
\varepsilon^l_i\sim\mathcal{N}\big(0,\ \mathrm{softplus}(W_n^l z_i)^2\big)\ \text{(training only)}.
$$

### 4.4 Routed attention

For $l\in\mathcal{R}$:
$$
\mathrm{MHA}^{B}_l(X_i; z_i)=\sum_{h=1}^{H} g^l_{i,h}\;O^{l,h}(X_i)\,W_O^{l,(h)} .
$$
In B-sparse, heads with $g=0$ are **not computed**, which is where the compute saving comes from.

---

## 5. Properties

**Proposition 1 (B nests A).** If $W_r^l=0$ and $b_r^l=0$, then in B-soft $g^l_{i,h}=1$ for all $i,h$, and $\mathrm{MHA}^B_l=\mathrm{MHA}^A_l$.

*Proof.* $\pi = \mathbf{1}/H$, so $g = H\pi = \mathbf{1}$. Substitute into §4.4. ∎

*Consequence.* Initialize B from the same weights as A with a zero router. The two are the same function at step 0, so any divergence is attributable to routing. With B-sparse and $k<H$, exact equivalence at initialization is impossible. Either start from B-soft and anneal to top-$k$, or accept a step-0 gap and report it.

**Proposition 2 (B-soft is a restricted FiLM).** Write $\tilde O = \mathrm{Concat}_h O^{l,h} \in\mathbb{R}^{(N+1)\times d}$. Then
$$
\mathrm{MHA}^B_l = \big(\tilde O \odot \gamma(z)\big)W_O^l,\qquad \gamma(z)= g^l(z)\otimes\mathbf{1}_{d_k}\in\mathbb{R}^{d},
$$
which is FiLM on the concatenated head space with $\beta=0$ and $\gamma$ constrained to be block-constant with $\sum$ fixed.

*Consequence.* A > B-soft or B-soft > A tells you whether **per-sample head reweighting** helps. It cannot tell you whether *selection* beats *conditioning*. That needs B-sparse, and ideally an unconstrained FiLM baseline.

**Proposition 3 (gradient partitioning).** Let $\delta^l_i = \partial\ell_i/\partial\,\mathrm{MHA}^B_l(X_i)$. For head parameters,
$$
\nabla_{\theta^{l,h}}\mathcal{L} = \frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}} g^l_{i,h}\;\Big(\tfrac{\partial\,(O^{l,h}W_O^{l,(h)})}{\partial\theta^{l,h}}\Big)^{\!\top}\delta^l_i .
$$
- **B-soft:** $g_{i,h}>0$ for all $i$, so every head is trained on every sample. Specialization can only arise through gate magnitude.
- **B-sparse:** the gradient for head $h$ depends only on $\mathcal{D}_h=\{i: h\in S_i\}$. Heads are trained on disjoint-ish data subsets, and this is the mechanism by which selection could produce specialization that A and B-soft cannot.

**Proposition 4 (router gradient).**
- B-soft: fully differentiable.
- B-sparse: $\partial g_{i,h}/\partial s_{i,j}$ is non-zero only for $h,j\in S_i$. Unselected logits receive no task gradient. They move only through noise and the balancing loss, which is why both are needed.

---

## 6. Objective

$$
\mathcal{L}
= \underbrace{\frac{1}{|\mathcal{B}|}\sum_i w_{y_i}\,\ell_{\text{BCE}}(\hat p_i,y_i)}_{\text{task (class-weighted)}}
+\lambda_{\text{bal}}\sum_{l\in\mathcal{R}}\mathcal{L}^l_{\text{bal}}
+\lambda_{z}\sum_{l\in\mathcal{R}}\mathcal{L}^l_{z}.
$$

**Load balancing.** With $f^l_h=\frac{1}{k|\mathcal{B}|}\sum_i\mathbb{1}[h\in S^l_i]$ and $P^l_h=\frac{1}{|\mathcal{B}|}\sum_i\pi^l_{i,h}$:
- B-sparse: $\mathcal{L}^l_{\text{bal}} = H\sum_h f^l_h P^l_h$.
- B-soft: $\mathcal{L}^l_{\text{bal}}=\mathrm{CV}^2\big(\sum_i g^l_{i,\cdot}\big)$.

**Router z-loss:**
$$
\mathcal{L}^l_{z}=\frac{1}{|\mathcal{B}|}\sum_i\Big(\log\sum_h e^{s^l_{i,h}}\Big)^2.
$$

**Temperature.** Anneal $\tau$ from 2 to 1 over the first 20% of steps.

**Defaults.** $\lambda_{\text{bal}}=10^{-2}$, $\lambda_z=10^{-3}$, and $w_y$ set to inverse class frequency.

---

## 7. Cost accounting (per routed layer, per sample)

| | Attention FLOPs | Extra params |
|---|---|---|
| A | $H\,(4Nd\,d_k + 2N^2 d_k)$ | — |
| B-soft | same as A $+\ O(dH)$ | $dH+H$ |
| B-sparse | $k\,(4Nd\,d_k + 2N^2 d_k)+O(dH)$ | $dH+H$ (+ $dH$ noise net) |

With $d=384$ and $H=6$, the router adds about 2.3k parameters per layer. **A and B are effectively parameter-matched.** B-sparse executes a fraction $k/H$ of the routed-layer attention compute.

---

## 8. Measurement

All quantities are computed on the test set for each of 3 seeds, with 95% bootstrap CIs.

### 8.1 Performance

AUC (primary), accuracy, F1, under the PCPA protocol (PIE, JAAD).

### 8.2 Routing structure

For each routed layer $l$, let $\bar\pi^l = \frac1n\sum_i \pi^l_i$ be the average gate.

**Routing specialization** is the mutual information between sample and head:
$$
\mathcal{S}^l = \mathbb{H}(\bar\pi^l) - \frac1n\sum_i \mathbb{H}(\pi^l_i) = I(i;\,h)\ \text{under the gate distribution}.
$$
- $\mathcal{S}^l\approx 0$: collapse, or input-independent gates.
- $\mathcal{S}^l\approx\log H$: every sample uses a deterministic head and all heads are used equally.

**Label vs. context dependence.** Let $r_i$ be the discretized route: the set $S_i$ for B-sparse, or a $k$-means cluster of $g_i$ for B-soft. Estimate
$$
I(r;y)\quad\text{and}\quad I(r;c\mid y)=\sum_{y}p(y)\,I(r;c\mid Y=y)
$$
with plug-in estimators plus the Miller–Madow correction.
- **Null distribution:** permute $c$ within each label stratum, 1,000 times.
- **Label shortcut:** if $I(r;y)$ is large and $I(r;c\mid y)$ is not above the null, the router is acting as a classifier.

**Per-head regression.** Regress $g^l_{\cdot,h} \sim c + y$ and report the partial $R^2$ of $c$.

### 8.3 Causal route tests (B only)

Define the swap effect on the output logit $u$:
$$
\Delta_{ij}=\big|\,u(x_i;\,g(z_j)) - u(x_i;\,g(z_i))\,\big|.
$$

**Contrast**, with pairs matched on route distance $\|g(z_i)-g(z_j)\|_1$:
$$
\Gamma=\mathbb{E}\big[\Delta_{ij}\mid y_i=y_j,\ c_i\neq c_j\big]-\mathbb{E}\big[\Delta_{ij}\mid c_i=c_j,\ y_i\neq y_j\big].
$$

**Magnitude-matched control.** For each swap, draw a random gate $g'$ with the same residual-stream change, i.e.
$$
\|X^L(x_i;g')-X^L(x_i;g_i)\|_2 \approx \|X^L(x_i;g_j)-X^L(x_i;g_i)\|_2 \quad (\pm10\%).
$$
Denote the resulting effect $\Delta'_{ij}$. Keep the real swap's effect only where $\Delta_{ij}>\Delta'_{ij}$.

### 8.4 Head importance (A, for Phase A)

$$
I_{i,h}^l=\big|\,u(x_i)-u(x_i;\ O^{l,h}\leftarrow \bar O^{l,h})\,\big|,
$$
where $\bar O^{l,h}$ is the dataset-mean output of the head (mean-ablation).

### 8.5 Specialization (tests Prop. 3)

Gradient share of head $h$ across context bins:
$$
q^l_h(c)=\frac{\sum_{i:\,c_i=c}\|\nabla_{\theta^{l,h}}\ell_i\|}{\sum_i\|\nabla_{\theta^{l,h}}\ell_i\|}.
$$
Report the per-head entropy $\mathbb{H}(q^l_h)$. It should be lower in B-sparse than in B-soft and A if gradient partitioning drives specialization.

---

## 9. Hypotheses

| | Statement | Test | Requires |
|---|---|---|---|
| H1 | Routing helps | $\mathrm{AUC}_B-\mathrm{AUC}_A>0$, CI excludes 0 | A, B |
| H2 | Routing is not collapsed | $\mathcal{S}^l$ above the value under shuffled $z$ | B |
| H3 | Routes track context, not just the label | $I(r;c\mid y)$ > permutation null (95th percentile) | B |
| H4 | Routes are functionally context-specific | $\Gamma>0$ and real swaps exceed magnitude-matched swaps | B |
| H5 | Selection drives specialization | $\mathbb{H}(q_h)$: B-sparse < B-soft ≈ A | A, B-soft, B-sparse |

### Interpreting the 2-variant outcomes

| Result | Meaning |
|---|---|
| B ≈ A, $\mathcal{S}\approx0$ | Router collapsed or there is nothing to exploit |
| B ≈ A, $\mathcal{S}$ high, H3 fails | Router learned the label; no gain |
| B > A, H3 fails | Gain is from extra per-sample capacity, not context |
| B > A, H3 and H4 pass | Claim supported **for reweighting**; selection vs. conditioning still open unless B-sparse and FiLM are run |

---

## 10. Training protocol

- **Same everything:** pretrained init, data splits, augmentation, optimizer, schedule, step count, and seeds {42, 7, 97}.
- **Initialization of B:** start from the same pretrained weights as A, with $W_r^l=0$, $b_r^l=0$. By Prop. 1, B-soft equals A at step 0.
- **Learning rate:** router LR = 10× backbone LR. Optionally freeze the router for the first 5% of steps.
- **Checkpoint selection:** by validation AUC, identical rule for both variants.

### Default config (T4-sized)

- **Backbone:** ViT-S/16, ImageNet-pretrained ($d=384$, $H=6$, $L=12$, $d_k=64$).
- **Input:** frame-wise spatial attention on 224² pedestrian crops ($N=197$ per frame), $T=8$ frames.
- **Temporal mixing:** a 2-layer temporal transformer over the per-frame CLS tokens.
- **Routing:**
  - $L_0=6$; spatial blocks 7–12 are routed.
  - $z$ is the frame-averaged CLS token at block 6, so there is one route per clip.
  - B-sparse uses $k=3$, giving $\binom{6}{3}=20$ possible routes per layer. That is small enough to analyse exhaustively.
- **Training:** AMP, batch 16 with gradient accumulation as memory allows, 3 seeds.
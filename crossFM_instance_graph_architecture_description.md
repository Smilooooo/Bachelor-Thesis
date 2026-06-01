# Cross-FM Instance Graph (CFIG)

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $F$ | Number of foundation models ($F = 3$) |
| $d$ | Feature dimension per FM ($d = 512$) |
| $C$ | Number of classes |
| $N$ | Number of support images per class (shots) |
| $T$ | Number of augmented passes (sample epochs, $T = 10$) |
| $B$ | Batch size (query images) |
| $\phi_f(\cdot)$ | Image encoder of foundation model $f$ |
| $\psi_f(\cdot)$ | Text encoder of foundation model $f$ |
| $\mathbf{V}^{(f)} \in \mathbb{R}^{CNT \times d}$ | Multi-view visual support features from FM $f$ |
| $\mathbf{T}^{(f)} \in \mathbb{R}^{C \times d}$ | Text prototypes from FM $f$ |
| $\mathbf{Q}^{(f)} \in \mathbb{R}^{B \times d}$ | Query features from FM $f$ |
| $\mathbf{W}\_{\text{VV}}, \mathbf{W}\_{\text{TT}}, \ldots$ | Graph weight matrices $\in \mathbb{R}^{d \times d}$ |
| $\alpha\_v, \alpha\_t$ | Learnable residual blend scalars (visual, text) |
| $\gamma$ | Learnable visual/text prototype blend scalar |
| $\tau$ | Learnable logit scale (temperature) |

All operations are applied per FM $f \in \{1, \ldots, F\}$ in parallel. Each FM's features stay in their native $d$-dimensional space (no cross-FM concatenation or projection). We stack across FMs for compact notation: $\mathbf{V} \in \mathbb{R}^{F \times CNT \times d}$, $\mathbf{T} \in \mathbb{R}^{F \times C \times d}$, $\mathbf{Q} \in \mathbb{R}^{F \times B \times d}$.

---

## 2. Prototype Extraction

Done once before training begins. All FM encoders remain frozen throughout.

### 2.1 Multi-View Visual Support Features

Unlike MLP GraphAdapter which averages augmented passes into a single prototype per class, the Cross-FM Instance Graph retains every augmented view as a separate graph node. For each FM $f$, class $c$, shot $k$, and augmentation pass $t$:

$$
\mathbf{v}^{(f)}_{c,k,t} = \frac{\phi_f(\mathbf{x}^{(k,t)}_c)}{\|\phi_f(\mathbf{x}^{(k,t)}_c)\|_2} \in \mathbb{R}^d
$$

where $\mathbf{x}^{(k,t)}_c$ is the $k$-th support image of class $c$ with random augmentation at pass $t$. These are sorted by $(c, k)$ and concatenated over all passes to form:

$$
\mathbf{V}^{(f)} \in \mathbb{R}^{CNT \times d}
$$

At 4-shot with 7 classes and $T = 10$: $\mathbf{V}^{(f)} \in \mathbb{R}^{280 \times 512}$ per FM.

### 2.2 Text Prototypes

Identical to MLP GraphAdapter. For each FM $f$ and class $c$, a dataset-specific prompt is encoded:

$$
\mathbf{T}^{(f)}_c = \frac{\psi_f(\text{prompt}(c))}{\|\psi_f(\text{prompt}(c))\|_2} \in \mathbb{R}^d
$$

Both $\mathbf{V}^{(f)}$ and $\mathbf{T}^{(f)}$ are registered as frozen buffers on the model.

---

## 3. Transductive Query Injection

At each forward pass, query images are encoded by all $F$ frozen FMs and L2-normalised to produce $\mathbf{Q}^{(f)} \in \mathbb{R}^{B \times d}$. These are then appended to the visual support nodes so that the query participates in graph message passing (transductive inference):

$$
\mathbf{V}_{\text{all}}^{(f)} = \begin{bmatrix} \mathbf{V}^{(f)} \\ \mathbf{Q}^{(f)} \end{bmatrix} \in \mathbb{R}^{(CNT + B) \times d}
$$

After graph refinement, the result is split back:

$$
\mathbf{V}_{\text{refined}}^{(f)} = \mathbf{V}_{\text{all,refined}}^{(f)}[{:CNT}], \quad
\mathbf{Q}_{\text{refined}}^{(f)} = \mathbf{V}_{\text{all,refined}}^{(f)}[{CNT:}]
$$

---

## 4. Heterogeneous Message Passing

One round of heterogeneous message passing over two node types: visual instance nodes $\mathbf{V}_{\text{all}}^{(f)} \in \mathbb{R}^{M \times d}$ (where $M = CNT + B$) and text class nodes $\mathbf{T}^{(f)} \in \mathbb{R}^{C \times d}$.

All operations below are computed independently per FM $f$. The FM index is omitted for readability.

### 4.1 Intra-FM Square Adjacency (V→V and T→T)

For a set of $n$ same-type nodes $\mathbf{X} \in \mathbb{R}^{n \times d}$ (either $\mathbf{V}_{\text{all}}$ or $\mathbf{T}$), the pairwise cosine similarity is:

$$
S_{ij} = \max\!\left(0,\; \frac{\mathbf{x}_i^\top \mathbf{x}_j}{\|\mathbf{x}_i\|_2 \|\mathbf{x}_j\|_2}\right), \quad S_{ii} = 0
$$

Similarities are computed on detached features (no gradient through the topology).

**Top-k sparsification.** Each node retains edges only to its $k$ most similar neighbours, then the result is symmetrised:

$$
\mathcal{N}_k(i) = \text{argtop-k}_j S_{ij}
$$

$$
\tilde{S}_{ij} = \begin{cases}
S_{ij} & \text{if } j \in \mathcal{N}_k(i) \text{ or } i \in \mathcal{N}_k(j) \\
0 & \text{otherwise}
\end{cases}
$$

**Symmetric normalisation with self-loops**:

$$
\tilde{\mathbf{A}} = \tilde{\mathbf{S}} + \mathbf{I}_n
$$

$$
\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}}   \tilde{\mathbf{D}}^{-\frac{1}{2}}
$$

where $\tilde{\mathbf{D}}\_{ii} = \sum\_j \tilde{A}\_{ij}$. This produces the adjacencies $\hat{\mathbf{A}}\_{\text{VV}} \in \mathbb{R}^{M \times M}$ and $\hat{\mathbf{A}}\_{\text{TT}} \in \mathbb{R}^{C \times C}$.

### 4.2 Intra-FM Rectangular Affinity (T↔V cross-modal)

Since visual and text node sets have different sizes ($M$ vs $C$), a rectangular affinity matrix is constructed:

$$
S_{ic} = \max\!\left(0,\; \frac{\mathbf{v}_i^\top \mathbf{t}_c}{\|\mathbf{v}_i\|_2 \|\mathbf{t}_c\|_2}\right), \quad \mathbf{S} \in \mathbb{R}^{M \times C}
$$

After optional top-k sparsification (each visual node keeps its $k$ most similar text classes), two normalised versions are derived for the two message directions:

**Text → Visual** (row-normalised):

$$
\mathbf{S}^{\text{TV}}_{i,:} = \frac{\mathbf{S}_{i,:}}{\sum_{c'} S_{ic'}}, \quad \mathbf{S}^{\text{TV}} \in \mathbb{R}^{M \times C}
$$

**Visual → Text** (column-normalised, transposed): each text node receives a weighted combination of visual instance features:

$$
\mathbf{S}^{\text{VT}}_{c,:} = \frac{\mathbf{S}_{:,c}^\top}{\sum_{i'} S_{i'c}}, \quad \mathbf{S}^{\text{VT}} \in \mathbb{R}^{C \times M}
$$

### 4.3 Cross-FM Same-Instance Messages

Nodes at the same index across FMs correspond to the same image (features are extracted in a fixed order). The cross-FM message for node $i$ in FM $f$ is the mean of the corresponding node from all other FMs:

$$
\mathbf{m}^{\text{xV},(f)}_i = \frac{1}{F - 1} \sum_{f' \neq f} \mathbf{v}^{(f')}_i
$$

$$
\mathbf{m}^{\text{xT},(f)}_c = \frac{1}{F - 1} \sum_{f' \neq f} \mathbf{t}^{(f')}_c
$$

No adjacency matrix is needed — the connection is a fixed one-to-one mapping by index.

### 4.4 Message Computation

Six messages are computed per FM using shared (across FMs) linear transformations $\mathbf{W}_{\cdot} \in \mathbb{R}^{d \times d}$ (no bias):

| Message | Formula | Shape |
|---------|---------|-------|
| $\mathbf{m}\_{\text{VV}}$ | $\hat{\mathbf{A}}\_{\text{VV}} \mathbf{V}\_{\text{all}} \mathbf{W}\_{\text{VV}}$ | $M \times d$ |
| $\mathbf{m}\_{\text{TT}}$ | $\hat{\mathbf{A}}\_{\text{TT}} \mathbf{T} \mathbf{W}\_{\text{TT}}$ | $C \times d$ |
| $\mathbf{m}\_{\text{TV}}$ | $\mathbf{S}^{\text{TV}} \mathbf{T} \mathbf{W}\_{\text{TV}}$ | $M \times d$ |
| $\mathbf{m}\_{\text{VT}}$ | $\mathbf{S}^{\text{VT}} \mathbf{V}\_{\text{all}} \mathbf{W}\_{\text{VT}}$ | $C \times d$ |
| $\mathbf{m}\_{\text{xV}}$ | $\mathbf{M}^{\text{xV}} \mathbf{W}\_{\text{xV}}$ | $M \times d$ |
| $\mathbf{m}\_{\text{xT}}$ | $\mathbf{M}^{\text{xT}} \mathbf{W}\_{\text{xT}}$ | $C \times d$ |

### 4.5 Gated Aggregation and Update

The messages are combined with learnable gate scalars $g\_{\text{cv}}, g\_{\text{ct}}, g\_{\text{xv}}, g\_{\text{xt}} \in \mathbb{R}$ (initialised to 0, so $\sigma(g) = 0.5$ at the start):

$$
\mathbf{a}_V = \mathbf{m}_{\text{VV}} + \sigma(g_{\text{cv}}) \cdot \mathbf{m}_{\text{TV}} + \sigma(g_{\text{xv}}) \cdot \mathbf{m}_{\text{xV}}
$$

$$
\mathbf{a}_T = \mathbf{m}_{\text{TT}} + \sigma(g_{\text{ct}}) \cdot \mathbf{m}_{\text{VT}} + \sigma(g_{\text{xt}}) \cdot \mathbf{m}_{\text{xT}}
$$

The update uses a residual connection, tanh activation, dropout, and LayerNorm:

$$
\mathbf{V}'_{\text{all}} = \text{LayerNorm}\!\left(\mathbf{V}_{\text{all}} + \text{Dropout}\!\left(\tanh(\mathbf{a}_V)\right)\right)
$$

$$
\mathbf{T}' = \text{LayerNorm}\!\left(\mathbf{T} + \text{Dropout}\!\left(\tanh(\mathbf{a}_T)\right)\right)
$$

The intra-FM messages ($\mathbf{m}\_{\text{VV}}, \mathbf{m}\_{\text{TT}}$) are always active. The cross-modal and cross-FM messages are scaled by learnable gate scalars.

---

## 5. Cosine Classification Head

After graph refinement, the refined visual support nodes are split from the refined query (Section 3) and pooled to class-level prototypes. All operations below run per FM $f$ independently; the FM index is omitted for readability.

### 5.1 Per-Class Prototype Pooling

The refined support nodes are reshaped and averaged per class:

$$
\mathbf{P}^v_c = \frac{1}{NT} \sum_{k=1}^{N} \sum_{t=1}^{T} \mathbf{v}'_{c,k,t} \in \mathbb{R}^d
$$

A raw (pre-graph) mean prototype is also computed for the residual blend:

$$
\bar{\mathbf{P}}^v_c = \frac{1}{NT} \sum_{k=1}^{N} \sum_{t=1}^{T} \mathbf{v}_{c,k,t} \in \mathbb{R}^d
$$

### 5.2 Residual Blends

The graph refinement is blended with the raw features using two learnable scalars $\alpha\_v, \alpha\_t \in \mathbb{R}$ (initialised via $\sigma^{-1}(0.5) = 0$):

$$
\mathbf{P}^{v,\text{final}}_c = \sigma(\alpha_v) \cdot \mathbf{P}^v_c + (1 - \sigma(\alpha_v)) \cdot \bar{\mathbf{P}}^v_c
$$

$$
\mathbf{T}^{\text{final}}_c = \sigma(\alpha_t) \cdot \mathbf{T}'_c + (1 - \sigma(\alpha_t)) \cdot \mathbf{T}_c
$$

### 5.3 Visual-Text Prototype Blending

A learnable gate $\gamma \in \mathbb{R}$ (initialised to 0, so $\sigma(\gamma) = 0.5$) controls the balance between visual and text prototypes:

$$
\mathbf{p}^{(f)}_c = \sigma(\gamma) \cdot \mathbf{P}^{v,\text{final},(f)}_c + (1 - \sigma(\gamma)) \cdot \mathbf{T}^{\text{final},(f)}_c \in \mathbb{R}^d
$$

### 5.4 Scaled Cosine Logits with Late Fusion

For each FM $f$, the query and blended prototype are L2-normalised, and logits are computed as scaled cosine similarities:

$$
\ell^{(f)}_c = \exp(\tau) \cdot \frac{\mathbf{q}^{(f)\top}_{\text{refined}} \mathbf{p}^{(f)}_c}{\|\mathbf{q}^{(f)}_{\text{refined}}\|_2   \|\mathbf{p}^{(f)}_c\|_2}
$$

where $\tau$ is a learnable log-temperature initialised to 2.3 ($\exp(2.3) \approx 10$, the standard CLIP temperature), clamped so that $\exp(\tau) \leq 100$.

The final logits are the mean across all FMs (late fusion):

$$
\ell_c = \frac{1}{F} \sum_{f=1}^{F} \ell^{(f)}_c
$$

---

## 6. Training Objective

Training minimises label-smoothed cross-entropy:

$$
\mathcal{L} = -\sum_{c=1}^{C} \tilde{y}_c \log \text{softmax}(\boldsymbol{\ell})_c, \qquad \tilde{y}_c = (1 - \varepsilon) \mathbf{1}[c = y] + \frac{\varepsilon}{C}
$$

with smoothing coefficient $\varepsilon = 0.1$.

---

## 7. Trainable Parameters

**1,574,920** 


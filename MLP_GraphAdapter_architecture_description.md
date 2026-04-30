# MLP GraphAdapter

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $F$ | Number of foundation models ($F = 3$) |
| $d$ | Feature dimension per FM ($d = 512$) |
| $D$ | Common embedding dimension ($D = 512$) |
| $C$ | Number of classes |
| $K$ | Number of support images per class (shots) |
| $B$ | Batch size |
| $\phi_f(\cdot)$ | Image encoder of foundation model $f$ |
| $\psi_f(\cdot)$ | Text encoder of foundation model $f$ |
| $\mathbf{z}^{(f)}$ | Image feature from FM $f$, $\mathbf{z}^{(f)} \in \mathbb{R}^d$ |
| $\mathbf{P}^{v,(f)}_c$ | Visual prototype of class $c$ from FM $f$ |
| $\mathbf{P}^{t,(f)}_c$ | Text prototype of class $c$ from FM $f$ |
| $g_\theta(\cdot)$ | JointAdapter MLP with parameters $\theta$ |
| $\hat{\mathbf{P}}^{v}_c$ | Adapted visual prototype of class $c$ |
| $\hat{\mathbf{P}}^{t}_c$ | Adapted text prototype of class $c$ |
| $\hat{\mathbf{z}}$ | Adapted query feature |
| $\tilde{\mathbf{P}}^{t}_c$ | Graph-refined text prototype of class $c$ |
| $\hat{\mathbf{A}}$ | Symmetrically normalised adjacency with self-loops |
| $\alpha, \beta$ | Learnable blend scalars |
| $\tau$ | Learnable logit scale (temperature) |

---

## 2. Prototype Extraction

Done once before training begins, with all FM encoders frozen.

**Visual prototypes:** For each FM $f$ and class $c$, 10 augmented passes are made over the support set (This is kept from how the original GraphAdapter paper handles the input data). The per-class visual prototype is the mean of L2-normalised features across all passes and shots:

$$
\mathbf{P}^{v,(f)}_c = \frac{1}{TK} \sum_{t=1}^{T} \sum_{k=1}^{K} \frac{\phi_f(\mathbf{x}^{(k)}_c)}{\|\phi_f(\mathbf{x}^{(k)}_c)\|_2} \in \mathbb{R}^d
$$

**Text prototypes:** One forward pass through the frozen text encoder using a dataset-specific prompt (e.g. *"a histological image of a {class name} cell in pulmonary pathology."*):

$$
\mathbf{P}^{t,(f)}_c = \frac{\psi_f(\text{prompt}(c))}{\|\psi_f(\text{prompt}(c))\|_2} \in \mathbb{R}^d
$$

Both prototype sets are stored as frozen buffers and re-projected through the JointAdapter at every forward pass so that gradients can flow into the adapter.

---

## 3. Stage 1: Early Fusion and Joint Adaptation

The three FM encoders (BiomedCLIP, PLIP, CONCH) are frozen. For a query image, per-FM L2-normalised features are concatenated into a single vector and projected to a $D$-dimensional shared space via the **JointAdapter** $g_\theta$:

$$
\mathbf{z}^{\text{cat}} = \left[\mathbf{z}^{(1)} \,\|\, \mathbf{z}^{(2)} \,\|\, \mathbf{z}^{(3)}\right] \in \mathbb{R}^{Fd}
$$

$$
g_\theta(\mathbf{u}) = \text{LayerNorm}\!\left(\mathbf{W}_2\,\text{Dropout}\!\left(\text{GELU}\!\left(\mathbf{W}_1 \mathbf{u} + \mathbf{b}_1\right)\right) + \mathbf{b}_2\right)
$$

with $\mathbf{W}_1 \in \mathbb{R}^{H \times Fd}$, $\mathbf{W}_2 \in \mathbb{R}^{D \times H}$, and hidden width $H = D = 512$.

The same adapter is applied identically to the query, visual prototypes, and text prototypes, so all three representations reside in the same metric space:

$$
\hat{\mathbf{z}} = g_\theta(\mathbf{z}^{\text{cat}}), \qquad
\hat{\mathbf{P}}^{v}_c = g_\theta\!\left(\left[\mathbf{P}^{v,(1)}_c \| \cdots \| \mathbf{P}^{v,(F)}_c\right]\right), \qquad
\hat{\mathbf{P}}^{t}_c = g_\theta\!\left(\left[\mathbf{P}^{t,(1)}_c \| \cdots \| \mathbf{P}^{t,(F)}_c\right]\right)
$$

---

## 4. Stage 2: Text Prototype Refinement via GraphAdapter

Only the text prototypes are refined by the graph. The query $\hat{\mathbf{z}}$ is never injected into any graph and remains unchanged from Stage 1.

Both GCN streams use the same single-layer graph convolutional operation:

$$
\mathbf{H}' = \tanh\!\left(\hat{\mathbf{A}}\, \mathbf{H}\, \mathbf{W} + \mathbf{b}\right)
$$

where $\mathbf{W} \in \mathbb{R}^{D \times D}$ is a learnable weight matrix, $\mathbf{b} \in \mathbb{R}^{N \times D}$ is a learnable per-node bias, and $\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}}$ is the symmetrically normalised adjacency with self-loops. The adjacency entries are non-negative pairwise cosine similarities of the (detached) node features.

For each class $c$, text prototype $c$ is placed at node 0 (the node being refined) and $C$ context nodes are appended. After one GCN pass, only node 0 is extracted as the refined prototype. This loop runs $C$ times.

### GCN\_tt — Text-to-Text Graph

Context nodes 1 through $C$ are the adapted text prototypes of all classes:

$$
\mathbf{H}^{tt}_c = \begin{bmatrix} \hat{\mathbf{P}}^t_c \\ \hat{\mathbf{P}}^t_1 \\ \vdots \\ \hat{\mathbf{P}}^t_C \end{bmatrix} \in \mathbb{R}^{(1+C) \times D}, \qquad
\mathbf{h}^{tt\prime}_{c} = \left[\tanh\!\left(\hat{\mathbf{A}}^{tt}_c\, \mathbf{H}^{tt}_c\, \mathbf{W}_{tt} + \mathbf{b}_{tt}\right)\right]_{0,:}
$$

### GCN\_it — Image-Prototype-to-Text Graph

Context nodes 1 through $C$ are the adapted visual prototypes of all classes:

$$
\mathbf{H}^{it}_c = \begin{bmatrix} \hat{\mathbf{P}}^t_c \\ \hat{\mathbf{P}}^v_1 \\ \vdots \\ \hat{\mathbf{P}}^v_C \end{bmatrix} \in \mathbb{R}^{(1+C) \times D}, \qquad
\mathbf{h}^{it\prime}_{c} = \left[\tanh\!\left(\hat{\mathbf{A}}^{it}_c\, \mathbf{H}^{it}_c\, \mathbf{W}_{it} + \mathbf{b}_{it}\right)\right]_{0,:}
$$

### Blending and Residual

The two stream outputs are blended with a learnable scalar $\alpha$, then a residual connection to the original adapted text prototypes is added with a second learnable scalar $\beta$:

$$
\mathbf{G}_c = \sigma(\alpha_{\text{raw}}) \cdot \mathbf{h}^{tt\prime}_{c} + \big(1 - \sigma(\alpha_{\text{raw}})\big) \cdot \mathbf{h}^{it\prime}_{c}
$$

$$
\tilde{\mathbf{P}}^t_c = \sigma(\beta_{\text{raw}}) \cdot \mathbf{G}_c + \big(1 - \sigma(\beta_{\text{raw}})\big) \cdot \hat{\mathbf{P}}^t_c \in \mathbb{R}^D
$$

Both $\alpha_{\text{raw}}$ and $\beta_{\text{raw}}$ are made learnable rather than fixed, since the optimal blend values from the original single-FM setting may not be optimal in this multi-FM setup. They are initialised so that $\sigma(\alpha_{\text{raw}}) = \sigma(\beta_{\text{raw}}) = 0.7$, close to the values reported in the paper.

---

## 5. Stage 3: Cosine Classification

The adapted query and the refined text prototypes are L2-normalised, and class logits are computed as scaled dot products:

$$
\ell_c = \exp(\tau) \cdot \hat{\mathbf{z}}_{\text{norm}}^\top \tilde{\mathbf{P}}^{t,\text{norm}}_c
$$

where $\tau$ is a learnable log-temperature initialised to 2.3 (so $\exp(2.3) \approx 10$). Training minimises label-smoothed cross-entropy with smoothing coefficient $\varepsilon = 0.1$.


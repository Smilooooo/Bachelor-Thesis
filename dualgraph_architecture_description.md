# DualGraph

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $F$ | Number of foundation models ($F = 3$) |
| $d$ | Feature dimension per FM ($d = 512$) |
| $D$ | Common embedding dimension ($D = 512$) |
| $C$ | Number of classes |
| $K$ | Number of support images per class (shots) |
| $B$ | Batch size |
| $\phi_f(\cdot)$ | Frozen image encoder of foundation model $f$ |
| $\psi_f(\cdot)$ | Frozen text encoder of foundation model $f$ |
| $\mathbf{z}^{(f)}$ | L2-normalised image feature from FM $f$, $\mathbf{z}^{(f)} \in \mathbb{R}^d$ |
| $\mathbf{P}^{v,(f)}_c$ | Raw visual prototype of class $c$ from FM $f$ (frozen mean) |
| $\mathbf{P}^{t,(f)}_c$ | Raw text prototype of class $c$ from FM $f$ |
| $A_f(\cdot)$ | FMAdapter MLP for FM $f$ (trainable) |
| $\hat{\mathbf{z}}^{(f)}$ | Adapted query feature from FM $f$ |
| $\hat{\mathbf{P}}^{v,(f)}_c$ | Adapted visual prototype of class $c$ from FM $f$ |
| $\hat{\mathbf{P}}^{t,(f)}_c$ | Adapted text prototype of class $c$ from FM $f$ |
| $\mathbf{M}$ | Fixed binary adjacency mask, $\mathbf{M} \in \{0,1\}^{(1+2C)\times(1+2C)}$ |
| $\mathbf{h}^{(f)}_i$ | Node $i$ embedding after GAT for FM $f$ |
| $\alpha^{(f)}_i$ | Attention readout weight for node $i$, FM $f$ |
| $\mathbf{r}^{(f)}$ | Attention readout vector for FM $f$, $\mathbf{r}^{(f)} \in \mathbb{R}^D$ |

---


## 2. Prototype Extraction

Done once before training begins, with all FM encoders frozen.

**Visual prototypes:** For each FM $f$ and class $c$, $T = 10$ augmented passes are made over the $K$-shot support set. The per-class visual prototype is the mean of L2-normalised features:

$$
\mathbf{P}^{v,(f)}_c = \frac{1}{TK} \sum_{t=1}^{T} \sum_{k=1}^{K} \frac{\phi_f(\mathbf{x}^{(k)}_c)}{\|\phi_f(\mathbf{x}^{(k)}_c)\|_2} \in \mathbb{R}^d
$$

**Text prototypes:** One forward pass through the frozen text encoder with a dataset-specific prompt template:

$$
\mathbf{P}^{t,(f)}_c = \frac{\psi_f(\text{prompt}(c))}{\|\psi_f(\text{prompt}(c))\|_2} \in \mathbb{R}^d
$$


---

## 3. Stage 1: Per-FM Adaptation

Each FM has its own adapter $A_f$, a two-layer MLP with LayerNorm:

$$
A_f(\mathbf{u}) = \text{LayerNorm}\!\left(\mathbf{W}^{(f)}_2\,\text{Dropout}\!\left(\text{GELU}\!\left(\mathbf{W}^{(f)}_1 \mathbf{u} + \mathbf{b}^{(f)}_1\right)\right) + \mathbf{b}^{(f)}_2\right)
$$

with $\mathbf{W}^{(f)}_1, \mathbf{W}^{(f)}_2 \in \mathbb{R}^{D \times d}$ (here $d = D = 512$).

The adapter is applied identically to queries, visual prototypes, and text prototypes so that all three reside in the same metric space for this FM:

$$
\hat{\mathbf{z}}^{(f)} = A_f\!\left(\mathbf{z}^{(f)}\right), \qquad
\hat{\mathbf{P}}^{v,(f)}_c = A_f\!\left(\mathbf{P}^{v,(f)}_c\right), \qquad
\hat{\mathbf{P}}^{t,(f)}_c = A_f\!\left(\mathbf{P}^{t,(f)}_c\right)
$$

---

## 4. Stage 2: Graph Construction

For each FM $f$ and each query $\mathbf{x}$ in the batch, a $(1+2C)$-node graph is assembled:

$$
\text{Nodes}^{(f)} = \left[\hat{\mathbf{z}}^{(f)} \;\big|\; \hat{\mathbf{P}}^{v,(f)}_0,\ldots,\hat{\mathbf{P}}^{v,(f)}_{C-1} \;\big|\; \hat{\mathbf{P}}^{t,(f)}_0,\ldots,\hat{\mathbf{P}}^{t,(f)}_{C-1}\right] \in \mathbb{R}^{(1+2C)\times D}
$$

The fixed binary adjacency mask $\mathbf{M} \in \{0,1\}^{(1+2C)\times(1+2C)}$ is precomputed once at model initialisation:

$$
M_{ij} = \begin{cases}
1 & i = 0,\; j > 0 \quad\text{(query → all prototypes)} \\
1 & j = 0,\; i > 0 \quad\text{(all prototypes → query)} \\
1 & 1 \le i,j \le C,\; i \ne j \quad\text{(visual-visual)} \\
1 & C{+}1 \le i,j \le 2C,\; i \ne j \quad\text{(text-text)} \\
0 & \text{otherwise} \quad\text{(visual-text: not connected)}
\end{cases}
$$

The visual and text subgraphs are connected only through the query node.

---

## 5. Stage 3: Shared GAT

A single `GATBlock` (with $L$ layers and $H$ attention heads) is shared across all three FM graphs.

For each GAT layer, attention from node $j$ to node $i$ is computed (only for $(i,j) \in \mathcal{E}$):

$$
e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}_{\text{src}} \cdot \mathbf{W}\mathbf{h}_i + \mathbf{a}_{\text{dst}} \cdot \mathbf{W}\mathbf{h}_j\right)
$$

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}
$$

$$
\mathbf{h}'_i = \text{ELU}\!\left(\bigoplus_{h=1}^{H} \sum_{j \in \mathcal{N}(i)} \alpha^h_{ij}\,\mathbf{W}^h\mathbf{h}_j\right)
$$

where $\oplus$ denotes head concatenation and $\mathcal{N}(i)$ includes a self-loop. A residual connection is applied after each layer: $\mathbf{h}_i \leftarrow \mathbf{h}_i + \mathbf{h}'_i$.

The GAT is applied once per FM per batch, giving updated node embeddings $\mathbf{H}^{(f)} \in \mathbb{R}^{B \times (1+2C) \times D}$.

---

## 6. Stage 4: Attention Readout

Each FM has its own readout gate (a single linear layer), which assigns a scalar logit to each node. Softmax over the node axis produces normalised attention weights:

$$
g^{(f)}_i = \mathbf{w}^{(f)\top}_{\text{gate}}\, \mathbf{h}^{(f)}_i + b^{(f)}_{\text{gate}} \in \mathbb{R}
$$

$$
\alpha^{(f)}_i = \frac{\exp\!\left(g^{(f)}_i\right)}{\sum_{k=0}^{2C} \exp\!\left(g^{(f)}_k\right)}
$$

$$
\mathbf{r}^{(f)} = \sum_{i=0}^{2C} \alpha^{(f)}_i\, \mathbf{h}^{(f)}_i \in \mathbb{R}^D
$$

---

## 7. Stage 5: MLP Classification

The three FM readout vectors are concatenated and passed through a two-layer MLP classifier:

$$
\mathbf{r} = \left[\mathbf{r}^{(1)} \;\|\; \mathbf{r}^{(2)} \;\|\; \mathbf{r}^{(3)}\right] \in \mathbb{R}^{3D}
$$

$$
\text{logits} = \mathbf{W}_2\,\text{Dropout}\!\left(\text{GELU}\!\left(\mathbf{W}_1 \mathbf{r} + \mathbf{b}_1\right)\right) + \mathbf{b}_2 \in \mathbb{R}^C
$$

Training minimises label-smoothed cross-entropy with smoothing coefficient $\varepsilon = 0.1$. No auxiliary losses are used.

---

## 8. Shape Flow Summary

```
Input images:           [B, 3, 448, 448]
                        ↓ extract_features (per-FM resizing, frozen encoders)
Per-FM features:        list[3] of [B, 512]

For each FM f:
  Query features:       [B, 512]
  Visual prototypes:    [C, 512]   (stored as frozen buffer)
  Text prototypes:      [C, 512]   (stored as frozen buffer)
                        ↓ FMAdapter_f
  Adapted:              [B, 512], [C, 512], [C, 512]
                        ↓ stack to graph nodes
  Node matrix:          [B, 1+2C, 512]
                        ↓ shared GATBlock + binary adj mask [B, 1+2C, 1+2C]
  Post-GAT nodes:       [B, 1+2C, 512]
                        ↓ AttentionReadout_f
  FM readout:           [B, 512]

Concat 3 FMs:           [B, 1536]
                        ↓ MLP classifier
Logits:                 [B, C]
```

---

## 9. Hyperparameter Settings

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| **FMAdapter hidden dim** | 512 | Matches FM output dim  |
| **Common dim** $D$ | 512 | Same as above, all three FMs project to the same 512-dim space |
| **GAT layers** $L$ | 1 |  |
| **GAT heads** $H$ | 4 | head dim = $D/H = 128$ |
| **GAT dropout** | 0.1 | Applied to attention coefficients during training |
| **Adapter dropout** | 0.1 | Between the two linear layers of each FMAdapter |
| **Classifier hidden** | 512 | Two-layer MLP from 1536 → 512 → $C$ |
| **Classifier dropout** | 0.1 | Between the two linear layers |
| **Label smoothing** $\varepsilon$ | 0.1 | Reduces overconfidence when only 4–16 support images are available |
| **Prototype passes** $T$ | 10 | Augmented passes over support set for visual prototype extraction |
| **Optimiser** | AdamW | Weight decay = 0.01 |
| **Learning rate** | 0.0005 | With cosine annealing (matching the GraphAdapter setup) |
| **Max epochs** | 100 | Final checkpoint used for evaluation (not best-val) |
| **Batch size** | 32 | Both train and test |
| **Seeds** | 11111, 22222, 33333, 44444, 55555 | 5-seed average reported |
| **Shots** | 4, 8, 16 | |

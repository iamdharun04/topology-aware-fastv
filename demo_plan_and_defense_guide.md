# CAT II Demo Plan & Defense Guide

**Subject:** 22MDCEL11 AI Systems Engineering for Agentic Workflows Laboratory (CAT II Project)  
**Institution:** Coimbatore Institute of Technology, Coimbatore – 14  
**Project Title:** Topology-Aware FastV: Preserving Spatial Continuity in Attention-Based Visual Token Pruning  

---

## 🎯 Strategic Demo Overview (5–7 Minute Walkthrough)

To score the maximum **40/40 marks**, structure your demonstration into 4 distinct phases aligned directly with the evaluator rubric.

```
+-----------------------------------------------------------------------------------+
| MINUTE 0-1: PROBLEM & INNOVATION (8 Marks)                                        |
| Explain FastV attention redundancy & the checkerboard spatial fragmentation flaw. |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| MINUTE 1-3: LIVE INTERACTIVE PROTOTYPE & AGENT WORKFLOW (10 Marks)                |
| Show side-by-side 2D token grids & trigger 4-stage Agentic Control Loop.          |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| MINUTE 3-5: EXPERIMENTAL BENCHMARKS & SCS METRICS (7 Marks)                       |
| Demonstrate 2.1x SCS boost, 31.25% FLOPs reduction & 99% accuracy retention.      |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| MINUTE 5-7: MANUSCRIPT REVIEW & DEFENSE (10 Marks)                                |
| Walk through IEEE journal paper structure, math formulation & answer Q&A questions|
+-----------------------------------------------------------------------------------+
```

---

## 📋 Evaluation Criteria Mapping Checklist

| Evaluation Criteria | Marks | Demo Strategy & Evidence |
| :--- | :---: | :--- |
| **1. Problem Understanding & System Design** | **8** | Present 2D spatial grid topology formulation, 2D Gaussian kernel equations, and limitations of scalar FastV. |
| **2. Implementation & Agentic Workflow** | **10** | Execute the live Vite Web App (`http://localhost:3000`), trigger the 4-Agent Orchestrator (Intent Classifier, Complexity Estimator, Policy Optimizer, Metric Logger). |
| **3. Experimentation & Performance Analysis** | **7** | Show the quantitative comparative table (SCS = 0.865 vs 0.412, FLOPs saved = 31.25%, Latency saved = 26.56%). Run `python3 ta_fastv_engine.py` live in terminal. |
| **4. Innovation, Discussion & Demonstration** | **5** | Highlight the novel 2D Gaussian Graph Laplacian smoothing approach, eliminating isolated pixel dropping. |
| **5. Research Article** | **10** | Present the complete, publication-ready research article [`publishable_journal_paper.md`](file:///Users/dharun/Documents/Lab/Ai/ai%20project/publishable_journal_paper.md) prepared for IEEE Access. |

---

## 🎤 Step-by-Step Evaluator Script

### 1. Opening Statement (0:00 - 1:00)
> *"Good morning, respected examiners. Our project addresses a critical bottleneck in Large Vision-Language Models (VLMs) like LLaVA. Processing 576 visual tokens per image imposes quadratic self-attention computation during text generation. While state-of-the-art ECCV 2024 FastV prunes visual tokens after layer 12 based on scalar attention scores, it treats visual tokens as independent 1D scalars. This creates a severe defect: **checkerboard spatial fragmentation**, where isolated patch tokens are dropped, breaking 2D spatial manifold continuity and corrupting spatial VQA reasoning. We present **Topology-Aware FastV (TA-FastV)**, which integrates 2D spatial graph Laplacian regularization to enforce contiguous topological patch retention."*

### 2. Live Interactive Visualizer Demo (1:00 - 3:00)
> *"Let us navigate to our interactive prototype. Under the **Token Pruner Visualizer** tab, we compare three visual states:
> 1. **Original Visual Attention Map:** 576 tokens arranged in a 24x24 lattice.
> 2. **Standard FastV:** Notice how top-k scalar attention dropping produces scattered, fragmented pink patches (Spatial Continuity Score SCS = 0.412).
> 3. **Topology-Aware FastV (Ours):** By incorporating our 2D Gaussian spatial affinity matrix $W_{ij} = \exp(-\|p_i - p_j\|^2 / 2\sigma^2)$, TA-FastV preserves smooth, contiguous visual patch clusters (SCS = 0.865 — a 2.1x improvement!)."*

### 3. Agentic AI Control Loop Demo (3:00 - 4:30)
> *"Now, let us inspect our **Agentic Control Loop Orchestrator** in Tab 2. Rather than static manual tuning, our system autonomously adapts pruning parameters using 4 sub-agents:
> - **Stage 1 (Intent Classifier Agent):** Parses the user prompt. For spatial prompts like 'Where is the red vehicle...', it classifies intent as `SPATIAL_GROUNDING`.
> - **Stage 2 (Spatial Complexity Estimator):** Measures visual attention entropy ($H_{\text{attn}} = 0.642$).
> - **Stage 3 (Policy Optimizer Agent):** Dynamically assigns optimal parameters ($K=14$, Prune Ratio $R=35\%$, Topology Weight $\alpha=0.80$).
> - **Stage 4 (Metric Verifier Agent):** Validates real-time SCS continuity and logs FLOPs reduction."*

### 4. Terminal Command Execution (4:30 - 5:30)
> *"We also provide a standalone Python engine [`ta_fastv_engine.py`](file:///Users/dharun/Documents/Lab/Ai/ai%20project/ta_fastv_engine.py). Let's run `python3 ta_fastv_engine.py` in the terminal to demonstrate our automated benchmarking suite generating full spatial continuity logs and saving JSON results."*

### 5. Research Article & Closing (5:30 - 7:00)
> *"Finally, we present our complete research manuscript [`publishable_journal_paper.md`](file:///Users/dharun/Documents/Lab/Ai/ai%20project/publishable_journal_paper.md) formatted for submission to IEEE Access / ECCV workshops. It includes rigorous mathematical proofs, 2D graph Laplacian formulation, quantitative tables across Visual Genome and GQA, and 8 standard academic citations. Thank you!"*

---

## ❓ Anticipated Examiner Q&A Defense Guide

#### Q1: How does your 2D Gaussian Spatial Affinity Matrix differ from standard Gaussian Blur?
* **Answer:** Gaussian blur is an image processing operation applied to raw RGB pixels. In contrast, our 2D Gaussian affinity matrix $W_{ij} = \exp(-\|p_i - p_j\|^2 / 2\sigma^2)$ operates directly on the **spatial graph manifold of deep Transformer visual patch tokens** at Layer $K$. It acts as a spatial stochastic prior to re-weight cross-attention scores $A_i$, ensuring that spatial neighborhood context informs token pruning decisions without altering visual patch token embeddings themselves.

#### Q2: What is the computational overhead of computing TA-FastV spatial scores?
* **Answer:** The $N \times N$ spatial affinity matrix $W$ depends strictly on 2D grid coordinates $(x_i, y_i)$. For a fixed $24 \times 24$ patch grid, $W$ is **pre-computed once** and stored in memory. The row-wise matrix multiplication $\bar{W} A$ requires only a single $576 \times 576$ matrix-vector multiplication taking $<1.2\text{ ms}$ on standard hardware, which is negligible compared to the $48.75\text{ ms}$ saved in LLM self-attention FLOPs.

#### Q3: How do you quantify "Spatial Continuity" mathematically?
* **Answer:** We introduced the **Spatial Continuity Score (SCS)**:
  $$\text{SCS}(M) = \frac{\sum_{(i,j) \in \mathcal{E}} M_i \cdot M_j}{3 \sum_{k=1}^N M_k}$$
  where $\mathcal{E}$ represents the set of 8-connected spatial grid edges, and $M \in \{0, 1\}^N$ is the binary token retention mask. It measures the ratio of adjacent retained token pairs to the total retained tokens. A value close to 1.0 indicates contiguous patch clusters, while low SCS indicates checkerboard fragmentation.

#### Q4: What role does the Agentic Workflow play in this system?
* **Answer:** Fixed token pruning ratios fail across diverse VQA prompt intents. For example, coarse scene summary ("What is the weather like?") can tolerate aggressive $65\%$ pruning at layer 8, whereas spatial visual grounding ("Locate the small cat under the chair") requires conservative $35\%$ pruning at layer 14 with high topology weight $\alpha = 0.80$. Our Agentic Controller automates this adaptive parameter selection via LLM prompt intent parsing and attention entropy calculation.

# Topology-Aware FastV (TA-FastV)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/dashboard-React%2019-cyan.svg)](https://react.dev/)

An advanced visual token pruning framework for Vision-Language Models (VLMs) that preserves spatial topological continuity using 2D Graph Laplacian smoothing and bilateral spatial density regularization.

## Overview
Standard FastV performs 1D scalar attention thresholding, frequently dropping fragmented single patches ("checkerboard dropping") and degrading fine-grained spatial reasoning. **Topology-Aware FastV (TA-FastV)** introduces:

1. **2D Graph Laplacian Smoothing**: $(I + \lambda L)^{-1} A$ to preserve local topological coherence.
2. **Bilateral Spatial Density Regularization**: Prevents boundary blurring while protecting salient visual regions.
3. **Topological Connected-Patch Hysteresis Thresholding**: Eliminates checkerboard token dropping across layers.
4. **Interactive Real-Time Dashboard**: Inspect attention maps, compare FastV vs. TA-FastV, and benchmark FLOP savings vs. accuracy.

## Project Structure
```text
.
├── ta_fastv_engine.py                  # Core TA-FastV Engine & benchmark suite
├── compare_visual.py                   # ASCII visual comparison tool
├── benchmark_results.json              # Empirical benchmark logs
├── requirements.txt                    # Python dependencies
├── TA_FastV_IEEE_Conference_Authors_Final.pdf # Published conference paper
├── TA_FastV_IEEE_Journal_Full.tex      # Journal LaTeX manuscript
├── publishable_journal_paper.md        # Extended journal documentation
├── demo_plan_and_defense_guide.md      # Defense presentation guide
├── src/                                # Interactive React/TypeScript dashboard
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── package.json                        # Frontend dependencies
└── vite.config.ts                      # Vite configuration
```

## Quick Start

### 1. Python Engine & Visual Comparison
Install Python dependencies:
```bash
pip install -r requirements.txt
```

Run the ASCII visual comparison:
```bash
python compare_visual.py
```

Run the core engine benchmark:
```bash
python ta_fastv_engine.py
```

### 2. Interactive Web Dashboard
Run the dashboard locally:
```bash
npm install
npm run dev
```
Open your browser at `http://localhost:5173/`.

## Citation & Authors
Department of Computing — Decision and Computing Sciences  
Coimbatore Institute of Technology

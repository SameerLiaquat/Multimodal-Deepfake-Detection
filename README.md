# Cross-Attention Fusion of Blend-Boundary and Semantic Detectors for Deepfakes

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DL_framework-orange)
![Status](https://img.shields.io/badge/status-active_research-brightgreen)

> Investigates whether fusing a spatial blend-boundary detector (Face X-Ray) with a semantic foundation-model detector (GenD, DINOv3-based) via a trainable cross-attention mechanism improves deepfake detection beyond either model alone — with both pretrained backbones kept fully frozen.

**Headline result:** the fusion model beat GenD alone in **5/5 independent training runs** on FaceForensics++ (97.241% ± 0.040% vs. 96.648% AUC), and the same margin held up on an **independent cross-dataset benchmark** (Celeb-DF v2, 92.835% vs. 92.275% video-level AUC) with no retraining — evidence the improvement generalizes rather than being an artifact of one dataset.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Technical Challenges & Fixes](#key-technical-challenges--fixes)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

Existing deepfake detectors tend to specialise in one of two directions: spatial-artifact detectors (e.g. Face X-Ray) that find the blend boundary left by face-swap manipulation, or semantic foundation-model detectors (e.g. GenD) that form a holistic judgement from a large pretrained vision transformer. This project fuses both, and validates the result both in-distribution and cross-dataset.

**Aim:** to investigate whether fusing Face X-Ray with GenD via a trainable cross-attention mechanism, with both pretrained backbones kept frozen, improves deepfake detection accuracy beyond either model alone.

---

## Architecture

```
Face X-Ray patches (4096 × 18)         GenD CLS token (1 × 1024)
        │                                       │
        ▼                                       ▼
 Patch projection                       Query projection
 Linear 18→512, L2 norm                 Linear 1024→512, L2 norm
        │                                       │
        └───────────────┬───────────────────────┘
                         ▼
              Cross-attention (8 heads)
        Query = GenD  ·  Key/Value = Face X-Ray patches
                         │
                         ▼
              Residual + LayerNorm
              (512-dim fused representation)
                         │
                         ▼
                    Classifier
        Linear 512→128 → ReLU → Dropout → Linear 128→2
                         │
                         ▼
                Real / Fake prediction
```

Both backbones (HRNet-W18 for Face X-Ray, DINOv3 ViT-L for GenD) are kept fully frozen throughout — only the fusion head (~1.65M parameters) is trained, isolating the fusion mechanism's contribution from any backbone fine-tuning effect.

---

## Key Technical Challenges & Fixes

Two significant preprocessing bugs were found and fixed through systematic diagnostic testing before any architectural conclusions were drawn — together they accounted for far more of the initial performance gap than any modelling choice.

| Bug | Root cause | Fix | Impact |
|---|---|---|---|
| **Face X-Ray backbone never loading** | A checkpoint key-prefix mismatch silently discarded ~1954 of ~1958 pretrained weight tensors on load | Removed the erroneous prefix-stripping step; verified via missing/unexpected key counts | Standalone AUC: 52.1% → 87% |
| **Missing face-crop preprocessing** | Both models were receiving full, uncropped video frames instead of the tightly-cropped faces they were trained on | Added a face-detection + crop step (1.3× margin) before each model's own preprocessing | GenD standalone AUC: 78.5% → 96.1% (frame-level) |

---

## Results

### Final validated comparison (FaceForensics++, in-distribution)

| Configuration | Mechanism | AUC |
|---|---|---|
| Face X-Ray alone | real mask + classifier pipeline | ~87% |
| GenD alone | real trained classifier, exact-match val set | 96.648% |
| Fusion — gated cross-attention | trainable GenD-only fallback | 97.074% |
| Fusion — gated, real GenD fallback | frozen real classifier, gate bias −2.0 | 97.025% |
| Fusion — gated, real GenD fallback | frozen real classifier, gate bias 0.0 | 96.944% |
| **Fusion — no gate (winning architecture)** | **always 100% fused branch** | **97.221%** (single run) |

### Multi-seed robustness validation

Same architecture, 5 independent training runs from different random seeds, identical data:

| Seed | AUC | Beats GenD alone (96.648%)? |
|---|---|---|
| 0 | 97.241% | Yes |
| 1 | 97.187% | Yes |
| 2 | 97.215% | Yes |
| 3 | 97.256% | Yes |
| 4 | 97.305% | Yes |

**Mean: 97.241% ± 0.040%** — 5/5 seeds beat GenD alone, by a margin roughly 15× larger than the run-to-run variation of the method itself.

### Cross-dataset generalization (Celeb-DF v2, official 518-video test split, no retraining)

| Model | Frame-level AUC | Video-level AUC |
|---|---|---|
| Face X-Ray alone | 66.897% | 76.664% |
| GenD alone | 83.482% | 92.275% |
| **Fusion (5-seed ensemble)** | **84.408%** | **92.835%** |

The re-measured GenD-alone figure (92.275%) matches GenD's own published Celeb-DF v2 result (92.2%, DINO variant) to within 0.08 points — independent confirmation the extraction pipeline is faithful to the standard protocol.

---

## Repository Structure

### Final pipeline
| File | Purpose |
|---|---|
| `fusion_crossattn_gated_v3.py` | Fully corrected pipeline, gated architecture, feature caching |
| `fusion_no_gate_pure.py` | Winning architecture — no-gate cross-attention fusion |
| `fusion_multiseed_validation.py` | Trains the winning architecture across 5 seeds |
| `evaluate_celebdf.py` | Cross-dataset evaluation on Celeb-DF v2 |

### Diagnostics
| File | Purpose |
|---|---|
| `inspect_facexray_checkpoint.py` | Raw checkpoint inspection |
| `verify_facexray_real_pipeline.py` | A/B tests for loading, normalization, face-cropping |
| `inspect_and_verify_gend.py` | GenD model structure inspection |
| `verify_gend_facecrop.py` | A/B test confirming GenD required face-cropped input |
| `compare_gend_vs_fusion_exact_match.py` | Exact-match GenD-alone baseline |

### Architecture experiments / ablations
| File | Purpose |
|---|---|
| `fusion_crossattn_gated.py` | First gated fusion implementation (pre-fix) |
| `fusion_crossattn_gated_v2.py` | Gated fusion with the Face X-Ray fix applied |
| `fusion_experiments_v3.py` | Regularization and residual-correction fusion comparison |
| `fusion_real_gend_fallback.py` | Tests replacing the trained-from-scratch fallback with GenD's real classifier |
| `fusion_improvements.py` | Reusable fusion-variant building blocks |

---

## Setup & Installation

```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]

conda create -n fusion python=3.12
conda activate fusion
pip install torch torchvision opencv-python pillow scikit-learn tqdm numpy transformers

hf auth login
```

Pretrained weights required:
- Face X-Ray: HRNet-W18 checkpoint (`best_model.pth.tar`)
- GenD: `yermandy/GenD_DINOv3_L` (auto-downloaded from Hugging Face)

Datasets: [FaceForensics++](https://github.com/ondyari/FaceForensics) (c23), [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) (official request form required)

---

## Usage

```bash
# Fully corrected pipeline: extracts + caches features, trains, evaluates
python fusion_crossattn_gated_v3.py --epochs 30

# Winning architecture across 5 seeds (reuses cached features)
python fusion_multiseed_validation.py

# Cross-dataset evaluation, no retraining
python evaluate_celebdf.py
```

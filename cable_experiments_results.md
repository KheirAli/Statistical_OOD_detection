# Experiments — Cable

## Overview

This document summarizes anomaly detection experiments on the **Cable** dataset (11 images) using ResNet features combined with PCA-based representations. The baseline AUC is **0.964**. Results are reported without spatial smoothing (SP AUC) and with spatial smoothing (P AUC).

The experiments explore:

- **Number of PCA basis components** (5, 6, 7)
- **Bin sizes** for RGB and PCA histograms
- ResNet architecture depth (18, 50, 101, 152)

---

## Number of PCA = 5

| Configuration | SP AUC (No Smoothing) | P AUC (Smoothing) |
|---|---|---|
| ResNet-18, RGB bin=64, PCA bin=16 | 0.929092 | 0.969670 |
| ResNet-50, RGB bin=64, PCA bin=16 | 0.933123 | 0.964438 |
| ResNet-101, RGB bin=64, PCA bin=16 | 0.952901 | 0.976476 |
| ResNet-152, RGB bin=64, PCA bin=16 | 0.936909 | 0.968014 |
| ResNet-18, RGB bin=32, PCA bin=8 | 0.913860 | 0.962917 |
| ResNet-50, RGB bin=32, PCA bin=8 | 0.925561 | 0.963925 |
| ResNet-101, RGB bin=32, PCA bin=8 | 0.922242 | 0.965878 |
| ResNet-152, RGB bin=32, PCA bin=8 | 0.927082 | 0.965930 |

---

## Number of PCA = 6

| Configuration | SP AUC (No Smoothing) | P AUC (Smoothing) |
|---|---|---|
| ResNet-18, RGB bin=32, PCA bin=8 | 0.917299 | 0.964270 |
| ResNet-50, RGB bin=32, PCA bin=8 | 0.932938 | 0.968015 |
| ResNet-101, RGB bin=32, PCA bin=8 | 0.931938 | 0.966765 |
| ResNet-152, RGB bin=32, PCA bin=8 | 0.931235 | 0.965930 |

---

## Number of PCA = 7

| Configuration | SP AUC (No Smoothing) | P AUC (Smoothing) |
|---|---|---|
| ResNet-18, RGB bin=32, PCA bin=8 | 0.921544 | 0.965164 |
| ResNet-50, RGB bin=32, PCA bin=8 | 0.938900 | 0.970233 |
| ResNet-101, RGB bin=32, PCA bin=8 | 0.933080 | 0.969702 |
| ResNet-152, RGB bin=32, PCA bin=8 | 0.933438 | 0.967900 |

---

## Key Observations

- **ResNet-101 with larger bins achieves the best result**: SP AUC of 0.9529 and P AUC of **0.9765** at PCA=5 with RGB bin=64 and PCA bin=16, surpassing the baseline.
- **Spatial smoothing consistently improves performance**, bringing most configurations above or near the 0.964 baseline.
- **Deeper ResNets tend to perform better** on Cable, unlike the Faces dataset where ResNet-18 dominated — ResNet-50 and ResNet-101 often lead here.
- **Larger bin sizes** (RGB=64, PCA=16) outperform smaller bins (RGB=32, PCA=8) at PCA=5.
- **Increasing PCA components** from 5 to 7 provides marginal gains for mid-depth architectures (ResNet-50 reaches P AUC 0.9702 at PCA=7).

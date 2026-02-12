# 🚨 FFP Video Anomaly Detection — Temporal + Bezier (B3)

Future Frame Prediction (FFP) based **video anomaly detection** with **Temporal Consistency** and **Bezier Trajectory** regularization (**B3 ablation**).

- ✅ **Anomaly Score:** PSNR (prediction error)
- ✅ **Metric:** ROC-AUC
- ✅ **Backbone:** Generator + Discriminator + **FlowNet2-SD (teacher / frozen)**

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Baseline Pipeline](#-baseline-pipeline)
- [What’s New in B3](#-whats-new-in-b3)
  - [Temporal Consistency Loss](#1-️-temporal-consistency-loss-flow-warp)
  - [Bezier Trajectory Loss](#2--bezier-trajectory-loss-flow-trajectory)
- [Gradient Flow Proof (Bezier → Generator)](#-gradient-flow-proof-bezier--generator)
- [Why FlowNet2-SD is Frozen](#-why-flownet2-sd-is-frozen)
- [Curriculum Ramp Schedule](#-curriculum-ramp-schedule)
- [Model Components](#-model-components)
- [Environment](#-environment)
- [Repository Structure](#-repository-structure)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results & Artifacts](#-results--artifacts)
- [Reproducibility Notes](#-reproducibility-notes)

---

## ✨ Overview

This repository is the **B3 (Temporal + Bezier)** extension of a clean FFP baseline.

**Main idea:**  
FFP predicts the next frame \((t+1)\). Abnormal events produce higher prediction error → higher anomaly score.

B3 improves **prediction stability** and **motion coherence** by adding:
- **Temporal Consistency Loss** (flow-warp based)
- **Bezier Trajectory Loss** (smooth flow-trajectory regularization)

---

## 🧠 Baseline Pipeline

**Input:** 4 consecutive frames  
\[
(x_{t-3}, x_{t-2}, x_{t-1}, x_t) \rightarrow \hat{x}_{t+1}
\]

**Model:**
- **Generator (UNet):** predicts future frame
- **Discriminator (PixelDiscriminator):** GAN regularization
- **FlowNet2-SD:** teacher optical flow estimator (**frozen**)

**Anomaly Score:**
- `PSNR(predicted_frame, ground_truth_frame)`

---

## 🧩 What’s New in B3?

### 1) ⏱️ Temporal Consistency Loss (Flow-warp)

Enforces that the predicted future frame is **motion-consistent** with the last input frame.

**Steps:**
1. compute optical flow between \(x_t\) and \(x_{t+1}\) (teacher)
2. warp using flow
3. penalize inconsistencies

✅ Activated only when `lam_t > 0`

---

### 2) 🌀 Bezier Trajectory Loss (Flow-trajectory)

Encourages **smooth motion evolution** across consecutive flows:

- `flow_12`: \(t-3 \rightarrow t-2\)
- `flow_23`: \(t-2 \rightarrow t-1\)
- `flow_34`: \(t-1 \rightarrow t\)
- `flow_45`: \(t \rightarrow t+1\)

Bezier loss regularizes the **trajectory** across these flows.

✅ Activated only when `lam_b > 0`

---

## ✅ Gradient Flow Proof (Bezier → Generator)

### Does Bezier really affect the Generator?
✅ **Yes.** Bezier is truly active and sends gradients into generator parameters.

In B3 implementation:
- `flow_12 / flow_23 / flow_34` come from GT frames → can be `detach()`
- BUT **`flow_45` is computed from `(frame_4, pred)` without detach**
- therefore **Bezier loss depends on `pred`**
- and gradients flow back to generator parameters

### 🔍 Proof from debug logs
When Bezier is OFF:
- `lam_b = 0.0`
- `bez_l = 0.0`
- `bez_l.requires_grad = False`

When Bezier becomes ON:
- `lam_b > 0`
- `bez_l.requires_grad = True`
- `[CHECK] bez_raw reqgrad: True`
- generator gradients remain non-zero (`gen grad mean abs: ...`)

✅ This confirms Bezier is **not only printed**, it is **optimizing the generator**.

---

## 🧊 Why FlowNet2-SD is Frozen?

FlowNet2-SD is used as a **teacher / measurement** network:
- `requires_grad = False`
- `eval()` mode

✅ Even if FlowNet is frozen, gradient can still flow **through its operations to `pred`**, because the loss is a function of `pred`.

---

## 📈 Curriculum Ramp Schedule

To avoid hurting AUC early (false positives), losses are activated gradually:

### Temporal Ramp
- `temp_ramp_start = 16000`
- `temp_ramp_end   = 20000`
- `lambda_temporal_max = 0.01`

### Bezier Ramp
- `bez_ramp_start = 24000`
- `bez_ramp_end   = 29000`
- `lambda_bezier_max = 0.002`

---

## 🏗️ Model Components

| Component | Role |
|---|---|
| **UNet Generator** | Predict next frame (12 → 3 channels) |
| **PixelDiscriminator** | GAN training stabilization |
| **FlowNet2-SD (Teacher)** | Optical flow estimation (**frozen**) |

---

## ⚙️ Environment

- **Python:** 3.8  
- **PyTorch:** 1.13.1  
- **CUDA:** 11.7  
- **Conda env:** `ffp_env`

---

## 📂 Repository Structure

```bash
.
├── data/                     # datasets (not included)
├── model/
│   ├── generator.py
│   └── discriminator.py
├── flownet/
│   └── pretrained/FlowNet2-SD.pth   # not included
├── training/
│   ├── train_func.py
│   ├── losses.py
│   ├── train_pre_func.py
│   └── train_ing_func.py
├── train.py
├── eval.py
├── utils.py
└── config.py


## ▶️ Training 
python train.py --dataset=ped2 --iters=29000 --batch_size=4 --val_interval=1000 --save_interval=1000 --manualseed=50

Outputs
- weights/latest_ped2.pth — periodic checkpoint
- weights/best_model_ped2.pth — best AUC checkpoint


<div align="center">

# DTC-TRUS

### Distilling Temporal Coherence into 2D Networks for TRUS Prostate Video Segmentation

[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-2F80ED)](#citation)
[![Project Page](https://img.shields.io/badge/Project-Page-111827)](https://dydevelop.github.io/DTC-TRUS/)
[![Dataset](https://img.shields.io/badge/Dataset-TRUS--V%20%40%20KHDP-10B981)](https://khdp.net/database/data-search-detail/TRUS-V)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C)](#installation)

Official implementation of **DTC-TRUS**, a temporally consistent learning framework for real-time prostate segmentation in transrectal ultrasound videos. :heart_eyes::heart_eyes: **Accepted on *MICCAI 2026*!!**

</div>

---

## Overview

DTC-TRUS addresses a practical limitation of frame-by-frame 2D segmentation in TRUS videos: adjacent frames can produce unstable, flickering prostate masks. Instead of deploying a computationally heavy 3D or recurrent model, DTC-TRUS uses video dynamics only during training and distills temporal coherence into a standard 2D student network.

At inference, the deployed model remains a single-frame 2D segmenter: **no optical flow, no teacher model, and no temporal module are required**.

![DTC-TRUS teaser](./figures/Figure1.png)

### Highlights

- **Temporal coherence with 2D inference**: video consistency is learned during training, while test-time inference remains lightweight.
- **Confidence-weighted temporal consistency**: optical-flow supervision is down-weighted in acoustically unstable or occluded regions.
- **Dual-scale prototype alignment**: local foreground prototypes improve boundary consistency, while global background prototypes stabilize scene semantics.
- **TRUS-V benchmark**: 2,679 annotated TRUS video frames from axial and sagittal views are released through KHDP.

---

## Links

| Resource | Status | Link |
|---|---:|---|
| Project page | Available | [dydevelop.github.io/DTC-TRUS](https://dydevelop.github.io/DTC-TRUS/) |
| Code | Available | [github.com/DYDevelop/DTC-TRUS](https://github.com/DYDevelop/DTC-TRUS) |
| TRUS-V dataset | Available | [KHDP dataset page](https://khdp.net/database/data-search-detail/TRUS-V) |
| Paper | Pending | Official paper URL will be added after publication |

---

## Method

The framework consists of four training signals that are applied to a 2D student network.

![Framework overview](./figures/Figure2.png)

### 1. Self-supervised equivariance and knowledge distillation

Flip-based pseudo-labels encourage transformation-consistent predictions without dense per-frame video labels. A frozen teacher trained on static images regularizes the student and helps preserve anatomical priors during video adaptation.

### 2. Confidence-weighted temporal consistency

Given adjacent frames, optical flow warps the previous prediction into the current frame. A non-occlusion confidence map suppresses unreliable gradients from occlusion, acoustic shadows, and unstable background regions.

```text
L_con = 1 - (1 / N) * sum_p M_noc(p) * (1 - |P_t(p) - P_t-1->t(p)|)
```

### 3. Dual-scale prototype alignment

Local and global prototypes are extracted from decoder and bottleneck features. The local foreground prototype stabilizes boundaries, while the global background prototype improves scene-level temporal consistency.

```text
L_proto = w_cb * (1 - cos(v_loc_cf,t-1, v_loc_cf,t))
        + w_cf * (1 - cos(v_glob_cb,t-1, v_glob_cb,t))
```

### 4. Total objective

```text
L_total = lambda_seg * L_seg
        + lambda_KD  * L_KD
        + lambda_con * L_con
        + lambda_proto * L_proto
```

---

## TRUS-V Benchmark

**TRUS-V** is a multi-view prostate ultrasound video segmentation benchmark designed to evaluate both segmentation accuracy and temporal stability.

| Property | Details |
|---|---|
| Total frames | 2,679 densely annotated frames |
| Patients | 10 patients |
| Views | Axial and sagittal |
| Sequences | 20 continuous video sequences |
| Split | 2,400 training frames / 279 testing frames, patient-level |
| Annotation | Semi-automated 5-fold U-Net ensemble + radiologist refinement |
| Ensemble DSC | 0.95 axial / 0.93 sagittal |
| Access | [TRUS-V on KHDP](https://khdp.net/database/data-search-detail/TRUS-V) |

To access the dataset, sign in to the Korea Health Data Platform (KHDP) and follow the instructions on the TRUS-V dataset page.

---

## Results

### SUN-SEG-Easy unseen split

| Method | Type | S<sub>α</sub> ↑ | E<sup>mn</sup><sub>ϕ</sub> ↑ | F<sup>w</sup><sub>β</sub> ↑ | Dice ↑ | Sen ↑ |
|---|---|---:|---:|---:|---:|---:|
| U-Net | 2D | 0.669 | 0.677 | 0.459 | 0.530 | 0.420 |
| ACSNet | 2D | 0.782 | 0.779 | 0.642 | 0.713 | 0.601 |
| MSRF-Net | 2D | 0.794 | 0.780 | 0.652 | 0.701 | 0.600 |
| SSTAN | Video | 0.805 | 0.838 | 0.691 | 0.726 | 0.662 |
| DALA | Video | 0.837 | 0.854 | 0.722 | 0.768 | 0.721 |
| **DTC-TRUS** | **2D** | **0.816** | **0.882** | **0.738** | **0.746** | **0.719** |

Reported inference speed: **89.95 FPS** with ACSNet.

### TRUS-V benchmark

| Method | Type | S<sub>α</sub> ↑ | E<sup>mn</sup><sub>ϕ</sub> ↑ | F<sup>w</sup><sub>β</sub> ↑ | Dice ↑ | Sen ↑ |
|---|---|---:|---:|---:|---:|---:|
| U-Net | 2D | 0.964 | 0.985 | 0.808 | 0.817 | 0.829 |
| U-Net++ | 2D | 0.967 | 0.988 | 0.821 | 0.820 | 0.827 |
| MSRF-Net | 2D | 0.965 | 0.987 | 0.821 | 0.821 | 0.838 |
| SSTAN | Video | 0.963 | 0.980 | 0.816 | 0.819 | 0.837 |
| DALA | Video | 0.908 | 0.931 | 0.539 | 0.729 | 0.746 |
| **DTC-TRUS** | **2D** | **0.967** | **0.988** | **0.830** | **0.829** | **0.839** |

Reported inference speed: **127.97 FPS** with U-Net++.

### Ablation study on SUN-SEG-Easy unseen

| Configuration | S<sub>α</sub> ↑ | Dice ↑ |
|---|---:|---:|
| Baseline, `L_seg` only | 0.274 | 0.171 |
| + Knowledge distillation | 0.793 | 0.701 |
| + KD + prototype alignment | 0.813 | 0.735 |
| + KD + temporal consistency | 0.810 | 0.722 |
| + KD + single-scale prototype + consistency | 0.808 | 0.721-0.722 |
| **Full DTC-TRUS** | **0.816** | **0.746** |

---

## Installation

```bash
git clone https://github.com/DYDevelop/DTC-TRUS.git
cd DTC-TRUS

conda create -n seg_bench python=3.10 -y
conda activate seg_bench

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

pip install einops yacs matplotlib opencv-python timm \
    ml-collections pydicom pandas scikit-learn kornia

pip install albumentations==1.3.1
```

---

## Repository Structure

```text
project_root/
├── src/
│   ├── network/
│   │   ├── conv_based/              # U-Net, U-Net++, U-Net 3+
│   │   ├── transformer_based/       # TransNetR, ColonSegNet, AttU-Net, ACSNet
│   │   └── hybrid_based/            # PraNet, MedNeXt, CMUNet, CMUNeXt, UNeXt, MSRF-Net
│   ├── dataloader/
│   │   ├── dataset.py               # Static image dataloader
│   │   └── temporal_dataset_ddp.py  # Video sequence dataloader, DDP-compatible
│   ├── utils/
│   │   ├── losses.py                # BCE-Dice loss, structure loss
│   │   ├── metrics.py               # IoU, DSC, SE, PC, F1, ACC
│   │   ├── metrics_SUN.py           # S-measure, E-measure, weighted F-measure
│   │   └── util.py                  # Optical flow, warping, AverageMeter
│   └── load_model.py
├── scripts/
│   ├── img_main.py                  # Stage 1: static-image teacher training
│   ├── vid_main_flip_kd_proto_ddp.py # Stage 2: temporally consistent student training
│   ├── vid_infer_single_patient.py  # Per-patient video inference and visualization
│   ├── compute_flow_warp.py         # Optical-flow utilities
│   └── 3D_recon.py                  # 3D mesh reconstruction from NIfTI masks
├── image_checkpoint/                # Pretrained teacher checkpoints
├── video_results/                   # Saved NIfTI segmentation outputs
├── single_patient_results/          # Per-patient inference videos
└── etc_dataset/
    └── list/
        └── etc_dataset.csv
```

---

## Usage

### Stage 1: train the static-image teacher

```bash
python scripts/img_main.py \
    --model U_Net \
    --dataset_csv Prostate_whole.csv \
    --base_lr 0.01 \
    --batch_size 4 \
    --epoch 50 \
    --img_size 256 448 \
    --mode train
```

Evaluate a trained teacher checkpoint:

```bash
python scripts/img_main.py \
    --model U_Net \
    --mode test \
    --checkpoint /path/to/checkpoint.pth
```

### Stage 2: train the temporally consistent student

The student is trained on video sequences using pseudo-labeling, teacher distillation, confidence-weighted temporal consistency, and prototype alignment.

Single-GPU example:

```bash
python scripts/vid_main_flip_kd_proto_ddp.py \
    --model U_Net \
    --use_prototype \
    --checkpoint image_checkpoint/checkpoint_axi/U_Net_0_model.pth \
    --dataset SUN \
    --base_lr 1e-3 \
    --batch_size 8 \
    --epoch 50 \
    --sequence_length 3 \
    --flow_method farneback \
    --seg_lam 3.0 \
    --kd_lam 1.0 \
    --con_lam 2.0 \
    --proto_lam 0.1 \
    --temperature 4.0 \
    --gpu_id 0
```

Multi-GPU example with DDP and AMP:

```bash
python scripts/vid_main_flip_kd_proto_ddp.py \
    --model U_Net \
    --use_prototype \
    --checkpoint image_checkpoint/checkpoint_axi/U_Net_0_model.pth \
    --multi_gpu \
    --use_amp \
    --batch_size 4 \
    --gradient_accumulation_steps 2
```

Target specific GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/vid_main_flip_kd_proto_ddp.py \
    --multi_gpu --use_amp
```

<details>
<summary>Key training arguments</summary>

| Argument | Default | Description |
|---|---:|---|
| `--model` | `U_Net` | Backbone architecture |
| `--use_prototype` | `False` | Enable dual-scale prototype alignment |
| `--checkpoint` | - | Path to pretrained teacher checkpoint |
| `--sequence_length` | `3` | Number of frames per temporal sequence |
| `--frame_gap` | `1` | Temporal stride between frames |
| `--flow_method` | `farneback` | Optical flow method: `farneback`, `lucas_kanade`, or `ncc` |
| `--seg_lam` | `3.0` | Segmentation loss weight |
| `--kd_lam` | `1.0` | Knowledge distillation loss weight |
| `--con_lam` | `2.0` | Temporal consistency loss weight |
| `--proto_lam` | `0.1` | Prototype alignment loss weight |
| `--temperature` | `4.0` | KD temperature |
| `--multi_gpu` | `False` | Enable DDP multi-GPU training |
| `--use_amp` | `False` | Enable automatic mixed precision |
| `--patience` | `10` | Early stopping patience |

</details>

### Inference on a single patient video

```bash
python scripts/vid_infer_single_patient.py \
    --patient_id 15213598 \
    --model U_Net \
    --checkpoint /path/to/student_checkpoint.pth \
    --video_dir /path/to/video_test \
    --output_dir /path/to/single_patient_results \
    --view axi \
    --fps 30
```

Output videos are saved under:

```text
<output_dir>/<checkpoint_stem>/<patient_id>_<checkpoint_info>.mp4
```

### Optional: 3D reconstruction

```bash
python scripts/3D_recon.py
```

Configure the patient ID, mode, and view plane in the script header. Outputs are saved under `3D_recon/<patient_id>/<datetime>/`.

---

## Supported Architectures

| Category | Models |
|---|---|
| Conv-based | U-Net, U-Net++, U-Net 3+, ColonSegNet |
| Transformer-based | AttU-Net, ACSNet, PraNet, MedNeXt, TransNetR |
| Hybrid | MSRF-Net, CMUNet, CMUNeXt, UNeXt |

---

## Datasets

| Dataset | Modality | Size | Purpose |
|---|---|---:|---|
| **TRUS-V** | Prostate TRUS video | 2,679 frames | Video training and evaluation |
| Static TRUS | Prostate TRUS images | 2,140 axial + 2,260 sagittal images | Teacher pretraining |
| **SUN-SEG** | Colonoscopy video | 158,690 frames | Generalization evaluation |

---

## Citation

```bibtex
@inproceedings{kim2026dtctrus,
  title     = {Distilling Temporal Coherence into 2D Networks for Transrectal Ultrasound Prostate Video Segmentation},
  author    = {Kim, Dong Yeong and Lee, JunGyu and Choi, Jaewon and Seo, June Young and Kim, Myeongseop and Choi, Jinwook and Kim, Taek Min and Kim, Young-Gon},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2026},
  note      = {Accepted}
}
```

Proceedings details, DOI, and the official paper URL will be added after publication.

---

## License and Data Terms

This repository is released for research use. Please refer to `LICENSE` for code usage terms.

TRUS-V is distributed through KHDP. Users must follow the access requirements and data-use terms specified on the [TRUS-V dataset page](https://khdp.net/database/data-search-detail/TRUS-V).

---

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) and the Korea Health Technology R&D Project through KHIDI, funded by the Ministry of Health & Welfare, Republic of Korea.

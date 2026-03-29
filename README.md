# Self-Supervised Learning Under Compute Constraints

> **Systematic evaluation of Self-Supervised Learning (SSL) frameworks — contrastive, invariance-based, and regularization-based — under strict computational constraints (100M parameter limit, 96×96 input resolution, random initialization).**

## 🎯 Problem

Self-supervised learning has shown impressive results on large-scale vision benchmarks, but most published results use significant compute budgets, large model architectures, and high-resolution inputs. **What happens when you strip those advantages away?**

This project investigates which SSL families remain effective under realistic resource constraints — the kind of constraints practitioners face when they don't have access to large GPU clusters.

## 🏗️ Approach

We evaluated five SSL frameworks across three families:

| Family | Method | Core Idea |
|--------|--------|-----------|
| **Contrastive** | MoCo v2, MoCo v3 | Learn by contrasting positive/negative pairs |
| **Invariance-based** | EMP-SSL | Learn representations invariant to augmentations |
| **Regularization-based** | VICReg, Barlow Twins | Prevent representation collapse through regularization |

### Constraints Applied
- **≤100M parameters** (no large ViTs or ResNet-152+)
- **96×96 input resolution** (not the standard 224×224)
- **Random initialization** (no ImageNet-pretrained backbones)

### Key Methodology
- Exploratory data analysis revealed significant **distribution mismatches** — most notably the edge density gap in fine-grained tasks — between training and evaluation data
- To address this, we strategically expanded the training corpus from **500K to 1.7M images** using curated datasets (OpenImages V7, Places365, iNaturalist)
- Transitioned from generalist models to an **ensemble of specialized architectures**: ConvNeXtV2-Tiny with Barlow Twins + Vision Transformer-Tiny with MoCo v3
- Shifted from k-NN evaluation to **optimized linear probing**, yielding a ~15% performance increase across all benchmarks

## 📊 Key Results

| Benchmark | Dataset | Accuracy |
|-----------|---------|----------|
| Fine-grained classification | CUB-200 | **39.12%** |
| General classification | MiniImageNet | **74.34%** |
| Scene understanding | SUN397 | **46.33%** |

- Achieved ~15% performance increase across all benchmarks by switching from k-NN to optimized linear probing
- Regularization-based methods (Barlow Twins) showed stronger robustness to resolution reduction
- Distribution-aware data expansion disproportionately helped contrastive approaches (MoCo v3)
- The specialist ensemble strategy outperformed any single generalist model

## 🔗 Model Checkpoints

Pretrained model checkpoints (during and after competition): [Google Drive](https://drive.google.com/drive/folders/1bal5xLJM3U5cUwNQzEsZMuqjy0O3deE5?usp=drive_link)

## 🛠️ Tech Stack

- **Frameworks:** PyTorch, torchvision
- **Architectures:** ConvNeXtV2-Tiny, ViT-Tiny
- **SSL Implementations:** MoCo v2/v3, EMP-SSL, VICReg, Barlow Twins
- **Datasets:** OpenImages V7, Places365, iNaturalist, CUB-200, MiniImageNet, SUN397
- **Analysis:** Python, Matplotlib, custom evaluation pipelines

## 🔗 Context

Group project for **Deep Learning**, NYU Center for Data Science, Fall 2025. Team project focused on practical SSL under real-world compute constraints.

## 👥 Contributors

Team project — see repository contributors for full list.

# Real-Time AI-Assited pN Stage Classification using Deep Learning and Digital Pathology

An end-to-end computational pathology system for automated breast cancer pathological lymph node (pN) staging using whole-slide images (WSIs) from the CAMELYON17 challenge. The system combines deep learning ensemble modeling (Branch 1) with spatial graph-based histopathological modeling (Branch 2) following the methodology of Tauqeer et al. (Scientific Reports, 2025).

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Dataset Specification](#dataset-specification)
3. [WSI Preprocessing Pipeline](#wsi-preprocessing-pipeline)
4. [Branch 1: Deep Learning Ensemble](#branch-1-deep-learning-ensemble)
5. [Branch 2: Spatial Graph Modeling & Selective Neighborhood Attention](#branch-2-spatial-graph-modeling--selective-neighborhood-attention)
6. [Automated pN-Staging Logic](#automated-pn-staging-logic)
7. [Installation & Requirements](#installation--requirements)
8. [Training and Evaluation Workflow](#training-and-evaluation-workflow)
9. [Diagnostic Web Dashboard](#diagnostic-web-dashboard)
10. [Repository Structure](#repository-structure)

---

## System Architecture

The following flowchart outlines the end-to-end data processing and model inference pipeline:

```
                          Whole Slide Images (WSIs)
                          [5 Lymph Nodes per Patient]
                                     |
                                     v
                        [WSI Preprocessing Module]
                        - Multi-resolution TIFF reader (tifffile + zarr)
                        - HSV + Otsu tissue foreground masking
                        - Non-overlapping patch extraction (256x256 px)
                        - Macenko optical density stain normalization
                                     |
           +-------------------------+-------------------------+
           |                                                   |
           v                                                   v
   [Branch 1: DL Ensemble]                             [Branch 2: SNA Graph Model]
   - Normalized patches (.h5)                          - 2048-D features + coords (.pt)
   - ResNet-50 patch classifier                        - Nuclei & Tissue Feature Fusion
   - DenseNet-121 patch classifier                     - 8-neighbor spatial graph builder
   - Slide-level max-mean pooling                      - Selective Neighborhood Attention (Top-4)
   - Patient 5-class distribution                      - Slide-level attention pooling
     (pN0, pN0(i+), pN1mi, pN1, pN2)                   - 2D tumor probability map reconstruction
           |                                           - Connected-component lesion quantification
           |                                             (ITC, Micro, Macro)
           |                                           - AJCC 8th Edition rule-based pN staging
           |                                                   |
           +-------------------------+-------------------------+
                                     |
                                     v
                       [Interactive Web Dashboard]
                       - Real-time slide tumor heatmap inspection
                       - Branch 1 probability distributions
                       - Branch 2 lesion breakdown and staging audit
                       - Multi-patient cohort evaluation matrix
```

---

## Dataset Specification

The pipeline operates on the CAMELYON17 challenge benchmark:
- **Slide Modality**: Whole Slide Images of sentinel lymph node sections stained with Hematoxylin and Eosin (H&E).
- **Patient Series**: Each patient examination consists of exactly 5 lymph node slides (`node_0.tif` through `node_4.tif`).
- **Ground Truth**: Defined in `dataset/stage_labels.csv` at two granularities:
  - **Slide-Level Labels**: `negative`, `itc`, `micro`, `macro`.
  - **Patient-Level Stages**: `pN0`, `pN0(i+)`, `pN1mi`, `pN1`, `pN2`.

---

## WSI Preprocessing Pipeline

Implemented in `src/preprocessing/` and executed via `run_preprocessing.py`.

### 1. Pyramidal TIFF Reader (`src/utils/wsi_reader.py`)
- Reads gigapixel WSIs at Level 0 ($~0.24\,\mu\text{m/pixel}$ at $40\times$ magnification) with native random-access chunking via `tifffile` and `zarr`.
- Eliminates the need to load complete multi-gigabyte images into system memory.

### 2. Tissue Masking & Artifact Rejection (`src/preprocessing/tissue_mask.py`)
- Extracts a downsampled thumbnail of the slide ($16\times$ to $32\times$ downsample).
- Converts RGB to HSV color space and computes an adaptive Otsu saturation threshold combined with luminance filtering to segment tissue from background glass.
- Applies morphological closing and hole-filling to eliminate tissue folds, dust particles, and pen markers.

### 3. Patch Extraction (`src/preprocessing/patch_extractor.py`)
- Generates a non-overlapping Level 0 coordinate grid ($256 \times 256$ pixels).
- Retains only patches containing $\ge 50\%$ tissue area based on the thumbnail mask.

### 4. Macenko Stain Normalization (`src/preprocessing/stain_normalizer.py`)
- Converts RGB patch intensities to Optical Density (OD) space:
  $$\text{OD} = -\log_{10}\left(\frac{I + 1}{255}\right)$$
- Projects pixels onto the two principal eigenvectors corresponding to Hematoxylin and Eosin stain vectors.
- Normalizes stain saturation to reference vectors to remove staining variability across laboratory preparation sites.

---

## Branch 1: Deep Learning Ensemble

Implemented in `src/branch1/` and trained via `train_branch1.py`.

### 1. Patch Classifiers (`resnet_model.py`, `densenet_model.py`)
- **ResNet-50**: Deep residual convolutional network with mixed-precision (FP16) feature projection.
- **DenseNet-121**: Densely connected convolutional network capturing fine-grained cellular patterns.
- Both models output patch-level malignancy logits and calibrated probabilities.

### 2. Slide-to-Patient Aggregation (`patient_aggregator.py`, `stage_predictor.py`)
- For each slide, extracts high-percentile tumor scores (mean of top-50 patches, 90th percentile, and maximum activation).
- Constructs a 5-slide feature representation for the patient.
- Multi-layer perceptron predicts a calibrated 5-class probability vector over the target pN stages:
  $$\mathbf{p} = [p(\text{pN0}), p(\text{pN0(i+)}), p(\text{pN1mi}), p(\text{pN1}), p(\text{pN2})]$$

### 3. Ensemble Fusion (`ensemble_module.py`)
- Blends normalized posteriors using weighted averaging:
  $$P_{\text{ensemble}}(\text{Stage}) = w \cdot P_{\text{ResNet}}(\text{Stage}) + (1 - w) \cdot P_{\text{DenseNet}}(\text{Stage})$$

---

## Branch 2: Spatial Graph Modeling & Selective Neighborhood Attention

Implemented in `src/branch2/` following Tauqeer et al. (Scientific Reports, 2025) and trained via `train_branch2.py`.

### 1. Feature Representation & Fusion (`feature_fusion.py`)
- **Nuclei Branch**: Computes cellular morphology and spatial density features.
- **Tissue Branch**: Computes structural histopathology patterns.
- Both streams are concatenated and projected to a unified 1024-dimensional feature representation per patch.

### 2. Spatial Adjacency Graph (`neighborhood_builder.py`)
- For every patch $i$ at coordinate $(x_i, y_i)$, identifies its 8 immediate spatial neighbors within an adjacent Chebyshev/Euclidean distance ($r \le 100\,\mu\text{m}$).
- Generates an adjacency index tensor $\mathbf{E} \in \mathbb{R}^{N \times 8}$ along with a boundary validity mask $\mathbf{M} \in \{0, 1\}^{N \times 8}$.

### 3. Selective Neighborhood Attention (SNA) (`selective_attention.py`)
- Computes attention coefficients between the center patch and each valid neighbor:
  $$e_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}$$
- Selects only the **top-4 highest-attention neighbors** to filter out background noise while preserving strong boundary transitions.
- Evaluates multi-head self-attention across selected neighbors to form a context-aware patch representation.
- Patch classification head outputs calibrated binary tumor probabilities ($p_i \in [0, 1]$).

### 4. Slide-Level Attention Pooling (`classifier_heads.py`)
- Computes attention weights $\beta_i$ for all patches in the slide:
  $$\beta_i = \frac{\exp(\mathbf{w}^\top \tanh(\mathbf{V} \mathbf{h}_i))}{\sum_{k=1}^N \exp(\mathbf{w}^\top \tanh(\mathbf{V} \mathbf{h}_k))}$$
- Aggregates patch embeddings into a single slide vector $\mathbf{s} = \sum_{i=1}^N \beta_i \mathbf{h}_i$, which is classified to predict slide malignancy.

### 5. Hierarchical Loss Function (`loss.py`)
- Optimizes a multi-task objective balancing instance-level, bag-level, and attention regularization:
  $$\mathcal{L}_{\text{total}} = 0.3 \mathcal{L}_{\text{instance}} + 0.2 \mathcal{L}_{\text{attention}} + 0.5 \mathcal{L}_{\text{bag}}$$

---

## Automated pN-Staging Logic

Implemented in `src/branch2/staging_evaluator.py`.

### 1. 2D Tumor Map Reconstruction
- Patch coordinates and tumor probabilities are projected onto a downsampled 2D spatial grid.
- Thresholding with cutoff $T$ (default $T = 0.5$) yields a binary tumor mask $\mathbf{B}_{\text{slide}}$.

### 2. Connected-Component Lesion Quantification
- Extracts connected tumor components from $\mathbf{B}_{\text{slide}}$ via 8-connectivity.
- Calculates physical lesion area:
  $$A = n_{\text{pixels}} \times (\text{pixelsize} \times \text{downsample\_factor})^2 \quad [\mu\text{m}^2]$$
- Categorizes each lesion according to clinical criteria:
  - **Isolated Tumor Cells (ITC)**: $A \le 200\,\mu\text{m}^2$ (single cells or clusters $\le 0.2\,\text{mm}$)
  - **Micro-metastasis**: $200\,\mu\text{m}^2 < A \le 2,000,000\,\mu\text{m}^2$ ($0.2\,\text{mm} < d \le 2.0\,\text{mm}$)
  - **Macro-metastasis**: $A > 2,000,000\,\mu\text{m}^2$ ($> 2.0\,\text{mm}$)

### 3. Patient Staging Decision Rule (AJCC 8th Edition)
Across the 5 lymph node examinations for a given patient:
1. If $\ge 4$ nodes contain **macro-metastases** $\rightarrow$ **`pN2`**
2. Else if $1 \text{ to } 3$ nodes contain **macro-metastases** $\rightarrow$ **`pN1`**
3. Else if $\ge 1$ node contains **micro-metastases** (and 0 macro) $\rightarrow$ **`pN1mi`**
4. Else if $\ge 1$ node contains **ITC** (and 0 micro, 0 macro) $\rightarrow$ **`pN0(i+)`**
5. Else $\rightarrow$ **`pN0`** (all nodes negative)

---

## Installation & Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4050 6GB Laptop GPU).
- **RAM**: Minimum 16 GB recommended.
- **Disk**: High-speed SSD storage for WSI caching.

### Environment Setup
```powershell
# Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install tifffile zarr shapely opencv-python pandas numpy scikit-learn streamlit plotly tqdm pyyaml
```

---

## Training and Evaluation Workflow

### 1. Batch Preprocessing
Extracts patches, masks, and features across raw WSIs:
```powershell
# Preprocess a single slide
python run_preprocessing.py --slide patient_000_node_0.tif

# Preprocess all available slides in batch
python run_preprocessing.py --batch
```

### 2. Train Branch 1 (Deep Learning Ensemble)
```powershell
python train_branch1.py --epochs 3 --lr 0.0001 --batch-size 32
```
Outputs predictions to: `output/branch1_predictions/branch1_patient_predictions.json`

### 3. Train Branch 2 (Selective Neighborhood Attention)
```powershell
python train_branch2.py --epochs 5 --lr 0.0005 --tumor-threshold 0.5
```
Outputs:
- Trained model weights: `output/branch2_weights/branch2_model.pt`
- Patient staging predictions: `output/branch2_predictions/branch2_patient_predictions.json`

### 4. Precompute Slide Heatmaps for Visualizer
```powershell
python scripts/generate_slide_visualizations.py
```
Generates continuous probability heatmaps, binary detection masks, and blended overlays for all slides in `output/slide_visualizations/`.

---

## Diagnostic Web Dashboard

The web dashboard provides an interactive evaluation environment for reviewing model predictions against CAMELYON17 ground-truth data.

### Launching the Application
```powershell
python -m streamlit run app.py --server.port 8501
```
Navigate to: `http://localhost:8501`

### Dashboard Capabilities
- **Summary Metrics**: High-level patient status showing Ground Truth, Branch 1 prediction, Branch 2 prediction, positive lymph node count ($k/5$), and total detected lesion count with concordance indicators.
- **Slide Inspection (Tab 1)**: Interactive whole-slide viewer supporting multiple display modes:
  - Alpha-blended tumor overlay (histology + heatmap)
  - Continuous Jet probability heatmap
  - Tissue foreground mask
  - Binary lesion detection map
  - Side-by-side comparison
- **Branch 1 Analysis (Tab 2)**: 5-class posterior probability bar chart and decomposition table comparing ResNet-50 and DenseNet-121 contributions, with dynamic weight adjustment.
- **Branch 2 Analysis (Tab 3)**: Complete 5-slide nodal series table, lesion size category histogram, and explicit staging rule trace.
- **Cohort Summary (Tab 4)**: Comprehensive evaluation matrix comparing all cohort patients, displaying individual branch accuracy and mutual agreement rates.

---

## Repository Structure

```
pN_Stage_Classification/
├── app.py                             # Streamlit diagnostic web dashboard
├── train_branch1.py                   # Branch 1 training and evaluation pipeline
├── train_branch2.py                   # Branch 2 training and evaluation pipeline
├── run_preprocessing.py               # Batch WSI preprocessing script
├── configs/
│   └── config.yaml                    # System hyperparameters and paths
├── scripts/
│   └── generate_slide_visualizations.py # Slide heatmap & overlay precomputation
├── src/
│   ├── utils/
│   │   ├── wsi_reader.py              # Pyramidal TIFF/WSI reader (tifffile + zarr)
│   │   └── annotation_parser.py       # ASAP XML ground truth annotation parser
│   ├── preprocessing/
│   │   ├── tissue_mask.py             # HSV + Otsu tissue segmentation
│   │   ├── stain_normalizer.py        # Macenko stain optical density normalizer
│   │   └── patch_extractor.py         # Patch coordinate generator & filtering
│   ├── branch1/
│   │   ├── patch_dataset.py           # HDF5 patch dataset loader
│   │   ├── resnet_model.py            # ResNet-50 patch classifier
│   │   ├── densenet_model.py          # DenseNet-121 patch classifier
│   │   ├── patient_aggregator.py      # Slide-level score feature aggregator
│   │   ├── stage_predictor.py         # 5-class patient pN stage predictor
│   │   └── ensemble_module.py         # ResNet + DenseNet late fusion ensemble
│   └── branch2/
│       ├── nuclei_extractor.py        # Cellular morphology feature extractor
│       ├── tissue_extractor.py        # Tissue structural feature extractor
│       ├── feature_fusion.py          # 1024-D feature fusion module
│       ├── neighborhood_builder.py    # 8-neighbor spatial adjacency graph builder
│       ├── selective_attention.py     # Selective Neighborhood Attention (SNA)
│       ├── classifier_heads.py        # Patch head & slide-level attention pooling
│       ├── loss.py                    # Hierarchical 3-part loss function
│       └── staging_evaluator.py       # 2D map builder, lesion quantification & staging
├── tests/
│   ├── test_wsi_pipeline.py           # Preprocessing unit tests
│   ├── test_branch1_modules.py        # Branch 1 modular tests
│   └── test_branch2_modules.py        # Branch 2 modular tests
└── output/
    ├── branch1_patches/               # Preprocessed .h5 patch files
    ├── branch1_predictions/           # Branch 1 output JSON predictions
    ├── branch2_features/              # Preprocessed .pt feature tensors
    ├── branch2_predictions/           # Branch 2 output JSON predictions
    ├── branch2_weights/               # Trained model checkpoint (.pt)
    ├── tissue_masks/                  # Extracted slide tissue mask PNGs
    └── slide_visualizations/          # Heatmaps, overlays, and slide metadata
```

---

## References

1. **CAMELYON17 Challenge**: *Validation of Deep Learning Algorithms for Detection of Lymph Node Metastases in Patients with Breast Cancer*. JAMA, 2018.
2. **Tauqeer et al.**: *Automated breast cancer staging from histopathology whole slide images using selective neighborhood attention*. Scientific Reports, 2025.
3. **AJCC Cancer Staging Manual (8th Edition)**: Breast Cancer Staging Protocol. American Joint Committee on Cancer, 2017.

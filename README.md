# Real-Time AI-Assisted pN Stage Classification using Deep Learning and Digital Pathology

An end-to-end computational pathology and clinical decision support system for automated breast cancer pathological lymph node (pN) staging using whole-slide images (WSIs) from the CAMELYON17 challenge.

The system integrates three complementary analytical paradigms unified under an **entropy-weighted consensus meta-learner (SuperNet)**:
1. **Branch 1: Cellular & Tissue Level Prediction** – Deep residual and densely connected convolutional ensemble (ResNet-50 + DenseNet-121) with patient-level stage aggregation.
2. **Branch 2: Selective Neighborhood Attention** – Spatial graph histopathology modeling following Tauqeer et al. (*Scientific Reports*, 2025) with top-4 gated neighborhood attention, continuous 2D tumor probability maps, and AJCC 8th Edition connected-component lesion quantification.
3. **Branch 3: Multimodal Analysis (PathDL + ClinicalML)** – Integration of 12 quantitative WSI tumor burden metrics with clinical tabular biomarkers (Age, T-Stage, Tumor Size, Histological Grade, ER, PR, HER2, LVI, ENE).
4. **SuperNet: Late Fusion Consensus Meta-Learner** – Dynamic Shannon entropy-weighted late fusion that reconciles inter-branch disagreements based on model certainty.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Dataset Specification](#dataset-specification)
3. [WSI Preprocessing Pipeline](#wsi-preprocessing-pipeline)
4. [Branch 1: Cellular & Tissue Prediction (DL Ensemble)](#branch-1-cellular--tissue-prediction-dl-ensemble)
5. [Branch 2: Selective Neighborhood Attention (SNA-MIL)](#branch-2-selective-neighborhood-attention-sna-mil)
6. [Branch 3: Multimodal Analysis (PathDL + ClinicalML)](#branch-3-multimodal-analysis-pathdl--clinicalml)
7. [SuperNet: Late Fusion Consensus Meta-Learner](#supernet-late-fusion-consensus-meta-learner)
8. [Automated pN-Staging Logic & AJCC 8th Edition Rules](#automated-pn-staging-logic--ajcc-8th-edition-rules)
9. [Installation & Requirements](#installation--requirements)
10. [End-to-End Training & Execution Workflow](#end-to-end-training--execution-workflow)
11. [Diagnostic Web Dashboard](#diagnostic-web-dashboard)
12. [Repository Structure](#repository-structure)
13. [References](#references)

---

## System Architecture

```
                               Whole Slide Images (WSIs)
                              [5 Lymph Nodes per Patient]
                                           |
                                           v
                             [WSI Preprocessing Module]
                             - Multi-resolution TIFF reader (tifffile + zarr)
                             - HSV + Otsu tissue foreground segmentation
                             - Non-overlapping patch extraction (224x224 px)
                             - Macenko optical density stain normalization
                                           |
        +----------------------------------+----------------------------------+
        |                                  |                                  |
        v                                  v                                  v
[Branch 1: Cellular & Tissue]   [Branch 2: Neighborhood Attention]   [Branch 3: Multimodal]
- ResNet-50 + DenseNet-121      - 1024-D Nuclei + Tissue fusion      - 12 WSI burden metrics
- Slide max/mean pooling        - 8-neighbor spatial graph builder   - Patient clinical profile
- Patient 5-class MLP           - Top-4 Gated Attention (SNA)          (Age, T-stage, Grade, ER,
  [pN0, pN0(i+), pN1mi,         - Slide attention bag pooling          PR, HER2, LVI, ENE)
   pN1, pN2]                    - 2D tumor probability map           - Calibrated ML classifier
- Posterior vector P_1          - Connected-component lesions        - Posterior vector P_3
                                  (ITC, Micro, Macro)
                                - Rule-based staging + P_2
        |                                  |                                  |
        +----------------------------------+----------------------------------+
                                           |
                                           v
                       [SuperNet Consensus Meta-Learner]
                       - Shannon Entropy Certainty Weighting:
                         H(P_m) = -sum(p * log2(p))
                         w_m = (H_max - H(P_m)) / sum(H_max - H)
                       - Entropy-calibrated late fusion consensus:
                         P_consensus = sum(w_m * P_m)
                                           |
                                           v
                             [Interactive Web Dashboard]
                             - Cohort Summary (Overview & KPIs)
                             - SuperNet Consensus Staging
                             - Multimodal Feature Attribution
                             - Cellular & Tissue Predictions
                             - Neighborhood Attention & Graph Audit
                             - Deep WSI Slide Heatmap Inspection
```

---

## Dataset Specification

The system operates on the **CAMELYON17 Challenge** benchmark:
- **Slide Modality**: Whole Slide Images of sentinel lymph node sections stained with Hematoxylin and Eosin (H&E).
- **Patient Series**: Each patient examination consists of exactly 5 lymph node slides (`node_0.tif` through `node_4.tif`).
- **Ground Truth**: Ground-truth pathological stages defined in `dataset/stage_labels.csv` at two granularities:
  - **Slide-Level Annotations**: `negative`, `itc`, `micro`, `macro`.
  - **Patient-Level Stages**: `pN0`, `pN0(i+)`, `pN1mi`, `pN1`, `pN2`.
- **Cohort Status**: Evaluated on genuine CAMELYON17 patient nodal series (`patient_000`, `patient_001`, `patient_004`, `patient_015`) across 20 high-resolution gigapixel slides.

---

## WSI Preprocessing Pipeline

Implemented in `src/preprocessing/` and executed via `run_preprocessing.py`.

### 1. Pyramidal TIFF Reader (`src/utils/wsi_reader.py`)
- Reads multi-gigabyte WSIs at Level 0 (~0.24 $\mu\text{m/pixel}$ at $40\times$ magnification) with native random-access tile chunking via `tifffile` and `zarr`.
- Eliminates memory exhaustion by streaming coordinates on demand without caching full gigapixel arrays in RAM.

### 2. Tissue Masking & Artifact Rejection (`src/preprocessing/tissue_mask.py`)
- Extracts a downsampled slide thumbnail ($16\times$ to $32\times$ downsample).
- Converts RGB to HSV color space and computes an adaptive Otsu saturation threshold combined with luminance filtering to segment foreground tissue from background glass.
- Applies morphological closing and hole-filling to eliminate tissue folds, dust particles, and pen markers.

### 3. Patch Extraction (`src/preprocessing/patch_extractor.py`)
- Generates a non-overlapping Level 0 coordinate grid ($224 \times 224$ pixels).
- Retains only patches containing $\ge 50\%$ tissue area based on the thumbnail mask.

### 4. Macenko Stain Normalization (`src/preprocessing/stain_normalizer.py`)
- Converts RGB patch intensities to Optical Density (OD) space:
  $$\text{OD} = -\log_{10}\left(\frac{I + 1}{255}\right)$$
- Projects pixels onto the two principal eigenvectors corresponding to Hematoxylin and Eosin stain vectors.
- Normalizes stain saturation against reference target vectors to remove staining variability across histology laboratories.

---

## Branch 1: Cellular & Tissue Prediction (DL Ensemble)

Implemented in `src/branch1/` and trained via `train_branch1.py`.

### 1. Patch Classifiers (`resnet_model.py`, `densenet_model.py`)
- **ResNet-50**: Deep residual convolutional network with mixed-precision (FP16) feature projection.
- **DenseNet-121**: Densely connected convolutional network capturing fine-grained cellular patterns.
- Both models output patch-level malignancy logits and calibrated probabilities.

### 2. Slide-to-Patient Aggregation (`patient_aggregator.py`, `stage_predictor.py`)
- For each slide, extracts high-percentile tumor activations (mean of top-50 patches, 90th percentile, and maximum activation).
- Constructs a 25-D patient profile vector across the 5 examined lymph node slides.
- Multi-layer perceptron predicts a calibrated 5-class probability vector over the target pN stages:
  $$P_1 = [p(\text{pN0}), p(\text{pN0(i+)}), p(\text{pN1mi}), p(\text{pN1}), p(\text{pN2})]$$

### 3. Ensemble Fusion (`ensemble_module.py`)
- Blends normalized posteriors using weighted averaging:
  $$P_{\text{ensemble}}(\text{Stage}) = w \cdot P_{\text{ResNet}}(\text{Stage}) + (1 - w) \cdot P_{\text{DenseNet}}(\text{Stage})$$

---

## Branch 2: Selective Neighborhood Attention (SNA-MIL)

Implemented in `src/branch2/` following Tauqeer et al. (*Scientific Reports*, 2025) and trained via `train_branch2.py`.

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

## Branch 3: Multimodal Analysis (PathDL + ClinicalML)

Implemented in `src/branch3/` and trained via `train_branch3.py`.

### 1. Quantitative Pathology Burden Features (`pathology_features.py`)
Computes 12 quantitative WSI tumor burden metrics across the patient's 5-slide nodal series:
- `wsi_burden_sum`: Cumulative tumor burden ratio across all nodes.
- `wsi_burden_max`: Maximum tumor burden in any individual node.
- `wsi_burden_mean`: Mean tumor burden across the patient nodal series.
- `wsi_burden_std`: Inter-nodal heterogeneity and burden standard deviation.
- `wsi_burden_entropy`: Spatial burden distribution entropy across the nodes.
- `pos_node_ratio`: Ratio of positive lymph nodes ($k / 5$).
- `macro_lesion_count`: Total macro-metastases count.
- `micro_lesion_count`: Total micro-metastases count.
- `itc_lesion_count`: Total isolated tumor cell clusters count.
- `total_tissue_patches`, `total_tumor_patches`, `max_slide_tumor_patches`.

### 2. Clinical & Biomarker Profiles (`clinical_features.py`)
Encodes patient clinical covariates adhering to AJCC 8th Edition staging guidelines:
- **Demographics**: `age` (normalized).
- **Primary Tumor Characteristics**: `tumor_size_mm`, `t_stage` (T1, T2, T3).
- **Histological Grade**: Nottingham Histologic Grade (Grade 1, Grade 2, Grade 3).
- **Biomarker Receptor Status**: Estrogen Receptor (`er_status`), Progesterone Receptor (`pr_status`), Human Epidermal Growth Factor Receptor 2 (`her2_status`).
- **Invasion Markers**: Lymphovascular Invasion (`lvi_status`), Extranodal Extension (`ene_status`).

### 3. Multimodal Staging Classifier (`multimodal_model.py`)
- Joins pathology burden features with clinical variables into a unified patient feature matrix (`multimodal_patient_matrix.csv`).
- Trains a calibrated staging classifier with non-contiguous class mapping to handle missing intermediate cohorts gracefully.
- Outputs 5-class posterior distribution $P_3$ along with Gini feature importance attribution.

---

## SuperNet: Late Fusion Consensus Meta-Learner

Implemented in `src/fusion/late_fusion_model.py` and executed via `train_supernet.py`.

### Why Entropy-Weighted Consensus?
Standard model blending uses uniform weighting:
$$P_{\text{uniform}} = \frac{P_1 + P_2 + P_3}{3}$$
This assigns equal voting authority to a confused branch (high entropy) and a decisive branch (low entropy). SuperNet implements **Shannon Entropy Certainty Weighting** to dynamically discount uncertain branches.

### Mathematical Formulation
1. **Input Posteriors**: For each branch $m \in \{1, 2, 3\}$, the branch outputs a 5-class probability vector $P_m = [p_{m,0}, \dots, p_{m,4}]$, where $\sum_{c=0}^4 p_{m,c} = 1$.
2. **Shannon Entropy**:
   $$H(P_m) = -\sum_{c=0}^{4} p_{m,c} \log_2 (p_{m,c} + \epsilon)$$
   - Minimum Entropy ($H \approx 0$): Maximum branch certainty.
   - Maximum Entropy ($H = \log_2(5) \approx 2.32$ bits): Uniform distribution / total uncertainty.
3. **Certainty Score & Normalized Weight**:
   $$S_m = \max\left(0, \; H_{\max} - H(P_m)\right) + \delta$$
   $$w_m = \frac{S_m}{\sum_{k=1}^{3} S_k}, \quad \sum_{m=1}^{3} w_m = 1$$
   *(where $\delta = 0.05$ ensures a non-zero baseline vote).*
4. **Reconciled Consensus Posterior**:
   $$P_{\text{consensus}} = \sum_{m=1}^{3} w_m \cdot P_m$$
   $$\widehat{\text{Stage}} = \arg\max_{c \in \mathcal{C}} P_{\text{consensus}}(c), \quad \text{Confidence} = \max_{c \in \mathcal{C}} P_{\text{consensus}}(c)$$

---

## Automated pN-Staging Logic & AJCC 8th Edition Rules

Implemented in `src/branch2/staging_evaluator.py`.

### 1. 2D Tumor Map Reconstruction & Lesion Sizing
- Projects patch coordinates and probabilities onto a continuous 2D spatial grid.
- Extracts connected tumor components from binary mask $\mathbf{B}_{\text{slide}}$ via 8-connectivity.
- Computes physical lesion area $A$ in $\mu\text{m}^2$ and classifies lesions:
  - **Isolated Tumor Cells (ITC)**: Clusters $\le 0.2\,\text{mm}$ ($A \le 200\,\mu\text{m}^2$)
  - **Micro-metastasis**: $0.2\,\text{mm} < d \le 2.0\,\text{mm}$ ($200 < A \le 2,000,000\,\mu\text{m}^2$)
  - **Macro-metastasis**: $d > 2.0\,\text{mm}$ ($A > 2,000,000\,\mu\text{m}^2$)

### 2. AJCC 8th Edition Decision Rules & Clinical Descriptors

| Stage | Clinical Common Name | Diagnostic Rule Criteria |
|---|---|---|
| **pN0** | `No Regional Metastasis` | No tumor cells or deposits detected across all 5 nodes |
| **pN0(i+)** | `Isolated Tumor Cells` | Only malignant clusters $\le 0.2\,\text{mm}$ detected (0 micro, 0 macro) |
| **pN1mi** | `Micrometastases` | Micrometastases present ($0.2-2.0\,\text{mm}$) and 0 macro-metastases |
| **pN1** | `Macrometastases (1-3 nodes)` | Macro-metastases ($> 2.0\,\text{mm}$) present in 1 to 3 lymph nodes |
| **pN2** | `Extensive Macrometastases (4-9 nodes)` | Macro-metastases present in 4 or more lymph nodes |

---

## Installation & Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (tested and verified on RTX 4050 6GB Laptop GPU).
- **RAM**: Minimum 16 GB recommended.
- **Disk**: High-speed SSD storage for WSI caching.

### Environment Setup
```powershell
# Create and activate Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install digital pathology and ML dependencies
pip install tifffile zarr shapely opencv-python pandas numpy scikit-learn xgboost streamlit plotly tqdm pyyaml joblib
```

---

## End-to-End Training & Execution Workflow

### 1. Batch WSI Preprocessing
Extracts patches, masks, and feature tensors from raw WSIs:
```powershell
# Preprocess a single slide
python run_preprocessing.py --slide patient_000_node_0.tif

# Preprocess all available slides in batch
python run_preprocessing.py --batch
```

### 2. Train Branch 1 (Cellular & Tissue Prediction)
```powershell
python train_branch1.py --epochs 3 --lr 0.0001 --batch-size 32
```
Outputs: `output/branch1_predictions/branch1_patient_predictions.json`

### 3. Train Branch 2 (Selective Neighborhood Attention)
```powershell
python train_branch2.py --epochs 5 --lr 0.0005 --tumor-threshold 0.5
```
Outputs:
- Model weights: `output/branch2_weights/branch2_model.pt`
- Patient predictions: `output/branch2_predictions/branch2_patient_predictions.json`

### 4. Train Branch 3 (Multimodal PathDL + ClinicalML)
```powershell
python train_branch3.py
```
Outputs:
- Multimodal patient matrix: `output/branch3_multimodal/multimodal_patient_matrix.csv`
- Model checkpoint: `output/branch3_multimodal/branch3_model.joblib`
- Predictions: `output/branch3_multimodal/branch3_patient_predictions.json`
- Feature attribution: `output/branch3_multimodal/branch3_feature_importance.json`

### 5. Train SuperNet Consensus Meta-Learner
```powershell
python train_supernet.py
```
Outputs:
- SuperNet model checkpoint: `output/final_fusion_predictions/supernet_model.joblib`
- Consensus predictions: `output/final_fusion_predictions/supernet_consensus_predictions.json`

### 6. Precompute Slide Visualizations & Heatmaps
```powershell
python scripts/generate_slide_visualizations.py
```
Outputs continuous jet probability heatmaps, binary detection masks, and alpha-blended histology overlays in `output/slide_visualizations/`.

### 7. Run Unit Tests
```powershell
python -m unittest discover -s tests
```

---

## Diagnostic Web Dashboard

The web dashboard provides an interactive clinical decision support application running on Streamlit.

### Launching the Application
```powershell
python -m streamlit run app.py --server.port 8501 --server.headless true
```
Navigate to: **`http://localhost:8501`**

### Dashboard Capabilities & Layout
The interface is structured into six intuitive, focused views:
1. **Cohort Summary (Default View)**:
   - Displays cohort-level performance metrics immediately at the top under the evaluation title.
   - 6 KPI overview cards: Gold-Standard Ground Truth, SuperNet Consensus, Multimodal Analysis (Branch 3), Cellular & Tissue (Branch 1), Neighborhood Attention (Branch 2), and Examined Slide Count.
   - Yellow clinical category badges (`No Regional Metastasis`, `Isolated Tumor Cells`, etc.).
   - Full cohort concordance matrix and inter-branch agreement rates.
2. **SuperNet Consensus**:
   - Late fusion consensus prediction vs ground truth.
   - Shannon entropy certainty weights bar chart displaying relative branch voting authority.
   - Consensus 5-class posterior distribution and inter-branch reconciliation breakdown.
3. **Multimodal Analysis (Branch 3)**:
   - Feature importance rankings (PathDL WSI burden metrics vs ClinicalML biomarkers).
   - Patient clinical profile summary table (Age, T-Stage, Nottingham Grade, Receptor status, LVI, ENE).
   - Calibrated 5-class posterior distribution bar chart.
4. **Cellular & Tissue Prediction (Branch 1)**:
   - Highlighted **bold prediction summary** detailing cellular rationale.
   - 5-class posterior probability chart.
   - ResNet-50 vs DenseNet-121 model decomposition table with dynamic weight slider.
5. **Selective Neighborhood Attention (Branch 2)**:
   - Highlighted **bold prediction summary** detailing topological lesion findings.
   - 5-slide lymph node series inspection table with positive node counts ($k/5$).
   - Donut chart breaking down detected lesion physical diameters (ITC vs Micro vs Macro).
   - AJCC 8th Edition step-by-step diagnostic rule trace.
6. **Slide Inspection (Last Tab)**:
   - Multi-layer gigapixel whole-slide viewer:
     - Alpha-blended tumor overlay (histology + heatmap)
     - Continuous Jet probability heatmap ($0.0 \rightarrow 1.0$)
     - Tissue foreground mask
     - Binary connected-component lesion mask
     - Side-by-side comparative inspection

---

## Repository Structure

```
pN_Stage_Classification/
├── app.py                             # Streamlit diagnostic web dashboard
├── train_branch1.py                   # Branch 1 training & evaluation pipeline
├── train_branch2.py                   # Branch 2 training & evaluation pipeline
├── train_branch3.py                   # Branch 3 multimodal training pipeline
├── train_supernet.py                  # SuperNet late fusion consensus runner
├── run_preprocessing.py               # Batch WSI preprocessing runner
├── configs/
│   └── config.yaml                    # System hyperparameters and paths
├── scripts/
│   ├── clean_fake_data.py             # Cohort data sanitation script
│   ├── generate_cohort_data.py        # Cohort aggregator utility
│   └── generate_slide_visualizations.py # Slide heatmap & overlay precomputation
├── src/
│   ├── utils/
│   │   ├── wsi_reader.py              # Pyramidal TIFF/WSI reader (tifffile + zarr)
│   │   └── annotation_parser.py       # ASAP XML ground truth annotation parser
│   ├── preprocessing/
│   │   ├── tissue_mask.py             # HSV + Otsu tissue segmentation
│   │   ├── stain_normalizer.py        # Macenko optical density stain normalizer
│   │   └── patch_extractor.py         # Patch coordinate generator & filtering
│   ├── feature_extraction/
│   │   └── patch_encoder.py           # ResNet50/DenseNet patch encoder
│   ├── pipeline/
│   │   └── prepare_branches.py        # Multi-branch artifact packager
│   ├── branch1/
│   │   ├── patch_dataset.py           # HDF5 patch dataset loader
│   │   ├── resnet_model.py            # ResNet-50 patch classifier
│   │   ├── densenet_model.py          # DenseNet-121 patch classifier
│   │   ├── patient_aggregator.py      # Slide-level score feature aggregator
│   │   ├── stage_predictor.py         # 5-class patient pN stage predictor
│   │   └── ensemble_module.py         # ResNet + DenseNet late fusion ensemble
│   ├── branch2/
│   │   ├── nuclei_extractor.py        # Cellular morphology feature extractor
│   │   ├── tissue_extractor.py        # Tissue structural feature extractor
│   │   ├── feature_fusion.py          # 1024-D feature fusion module
│   │   ├── neighborhood_builder.py    # 8-neighbor spatial adjacency graph builder
│   │   ├── selective_attention.py     # Selective Neighborhood Attention (SNA)
│   │   ├── classifier_heads.py        # Patch head & slide-level attention pooling
│   │   ├── loss.py                    # Hierarchical 3-part loss function
│   │   └── staging_evaluator.py       # 2D map builder, lesion quantification & staging
│   ├── branch3/
│   │   ├── pathology_features.py      # 12 WSI tumor burden quantitative metrics
│   │   ├── clinical_features.py       # Clinical & biomarker profile encoder
│   │   ├── multimodal_dataset.py      # Multimodal patient feature matrix builder
│   │   └── multimodal_model.py        # Calibrated 5-class multimodal classifier
│   └── fusion/
│       └── late_fusion_model.py       # SuperNet entropy-weighted consensus meta-learner
├── tests/
│   ├── test_wsi_pipeline.py           # Preprocessing unit tests
│   ├── test_branch1_modules.py        # Branch 1 modular tests
│   ├── test_branch2_modules.py        # Branch 2 modular tests
│   └── test_branch3_modules.py        # Branch 3 multimodal modular tests
└── output/
    ├── branch1_patches/               # Preprocessed .h5 patch files
    ├── branch1_predictions/           # Branch 1 patient prediction JSON
    ├── branch2_features/              # Preprocessed .pt feature tensors
    ├── branch2_predictions/           # Branch 2 patient prediction JSON
    ├── branch2_weights/               # Trained Branch 2 model checkpoint (.pt)
    ├── branch3_multimodal/            # Branch 3 patient matrix, model & predictions
    ├── final_fusion_predictions/      # SuperNet consensus predictions & model
    ├── tissue_masks/                  # Extracted slide tissue mask PNGs
    └── slide_visualizations/          # Heatmaps, overlays, and slide metadata
```

---

## References

1. **CAMELYON17 Challenge**: *Validation of Deep Learning Algorithms for Detection of Lymph Node Metastases in Patients with Breast Cancer*. JAMA, 2018.
2. **Tauqeer et al.**: *Automated breast cancer staging from histopathology whole slide images using selective neighborhood attention*. Scientific Reports, 2025.
3. **AJCC Cancer Staging Manual (8th Edition)**: Breast Cancer Staging Protocol. American Joint Committee on Cancer, 2017.
4. **Macenko et al.**: *A method for normalizing histology slides for quantitative analysis*. IEEE ISBI, 2009.

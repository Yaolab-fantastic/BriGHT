# BriGHT: Transcriptome-Regularized Multimodal Neuroimaging for Brain Disorder Prediction

## Overview

BriGHT is a biologically grounded multimodal learning framework for brain disorder prediction that integrates transcriptomic information with neuroimaging data. Unlike conventional graph or hypergraph-based approaches that rely solely on imaging-derived structures, BriGHT introduces a transcriptome-derived structural reference to regularize region-of-interest (ROI) representations during learning.

Specifically, we construct transcriptomic co-expression modules from GWAS-guided gene sets and the Allen Human Brain Atlas (AHBA), and transform them into an ROI-level hypergraph. This hypergraph serves as a biological prior that softly anchors ROI representations across subjects and modalities, improving cross-subject consistency while preserving disease-related variability.

In addition, BriGHT incorporates a reliability-aware multimodal fusion strategy that adaptively integrates different imaging modalities based on prediction confidence, cross-modal agreement, and decision certainty. This allows the model to dynamically adjust modality contributions under heterogeneous data quality and subject-specific variability.

Extensive experiments on ADNI, ADHD-200, and REST-meta-MDD demonstrate that BriGHT consistently outperforms state-of-the-art graph and hypergraph learning methods across multiple brain disorder prediction tasks.

---

## Code and Data Availability

The implementation of BriGHT is publicly available to facilitate reproducibility and further research.

The repository includes:

- Preprocessed sample datasets (`Example_data/`)
- Pre-trained models and checkpoints
- Scripts for transcriptomic hypergraph construction and multimodal learning, including:
  - `1.sample_weight_calculating.py`
  - `2.edge_score_calculating.py`
  - `3.calculating_mean_std_zscore.py`
  - `Build_wgcna_hypergraph.py`
  - `Model/model.py`
  - `train.py`
  - `utils.py`
- Supplementary materials (e.g., `SuppInformation_BriGHT.pdf`)

These resources support end-to-end reproduction of the proposed framework, including:
1. Transcriptome-derived hypergraph construction  
2. Subject-specific ROI network generation  
3. Model training and evaluation  

For questions, technical support, or collaboration inquiries, please contact:
**zhoujie.fan@hrbeu.edu.cn**

---

## Requirements

- Python 3.8  
- PyTorch 1.10.2  
- PyTorch Geometric  
- scikit-learn  
- numpy  

---

## Data Preparation

The datasets used in this study are publicly available from:

- ADNI  
- ADHD-200  
- REST-meta-MDD  

All data are preprocessed following standard pipelines described in the paper. See **Neuroimaging data acquisition and preprocessing** for details.

---

## Disclaimer

This software is provided for research purposes only and is not approved for clinical use.

---

## Acknowledgments

This work was developed in the Yao Lab. We sincerely thank all collaborators and contributors for their support.

---

## Citation

If you use this code, please cite
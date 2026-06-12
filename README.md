# BriGHT: transcriptome-regularized multimodal neuroimaging for brain disorder prediction

## Supplementary Information
Supplementary information for this manuscript is available in the file SuppInformation_BriGHT.pdf.

## Code and Data Availability

The source code associated with this study is publicly available to support transparency, reproducibility, and further research. We provide preprocessed sample datasets, pre-trained models (including checkpoints), and implementations for both single-sample and transcriptome-driven hypergraph construction. The repository includes key Python scripts such as 1.sample_weight_calculating.py, 2.edge_score_calculating.py, 3.calculating_mean_std_zscore.py, Build_wgcna_hypergraph.py, train.py, and utils.py, along with example datasets (Example_data), trained model files, and supplementary materials (e.g., SuppInformation_BriGHT.pdf).

These resources are intended to facilitate understanding, validation, and reproduction of the proposed framework. For any questions, technical support, or additional information, please contact us at zhoujie.fan@hrbeu.edu.cn.

## Requirements

- Python 3.8
- PyTorch 1.10.2
- PyTorch Geometric
- scikit-learn
- numpy

## Data Preparation
The data used can be obtained from ADNI and ADHD-200 and REST-meta-MDD. See Data acquisition and preprocessing

## Disclaimer
This tool is for research purposes and not approved for clinical use.

## Acknowledgments
This tool is developed in Yao Lab. We thank all the contributors and collaborators for their support.

## Citation
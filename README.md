# MODE: multimodal deep autoencoder for high-resolution tissue dissociation

This is a novel multimodal autoencoder framework with parallel decoders - to jointly purify CTS multi-omic profiles and estimate the multimodal cellular compositions. MODE allows the input of cell markers detectable in scRNA-seq reference and accounts for the between-tissue heterogeneity in target bulk data, without the requirement of feature annotation mapping across molecular modalities. MODE is trained on the large-scale pseudo bulk multiomes constructed by probabilistic data generation with an external scRNA-seq reference and individualized non-RNA reference panel recovered from the target bulk profile.
For more details, please refer to the [paper](https://www.biorxiv.org/content/10.1101/2025.01.02.631152v1).

## Model Overview
<p align="center">
  <img width="80%" src="https://github.com/jsuncompubio/MODE/blob/main/images/MODE_overview.png">
</p>

## Setup

### Dependencies
The MODE framework is implemented in python, the following environment is suggested:

[![python >3.10](https://img.shields.io/badge/python-3.10-brightgreen)](https://www.python.org/) 
[![torch >1.8.0](https://img.shields.io/badge/torch-1.8.0-orange)](https://github.com/pytorch/pytorch) 
[![numpy-1.22.4](https://img.shields.io/badge/numpy-1.22.4-red)](https://github.com/numpy/numpy) 
[![pandas-2.0.0](https://img.shields.io/badge/pandas-2.0.0-lightgrey)](https://github.com/pandas-dev/pandas) 
[![scikit__learn-1.2.2](https://img.shields.io/badge/scikit__learn-1.2.2-green)](https://github.com/scikit-learn/scikit-learn)


### Environment setup
1. Create a virtual environment
    ```
    python -m venv <myenv>
    ```
    To activate the virtual environment:
    ```
    source <myenv>/bin/activate
    ```

2. Install PyTorch
    ```
    pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```
    MODE is developed using PyTorch. While PyTorch is included automatically when installing MODE, it is recommended that users manually install [PyTorch](https://pytorch.org/get-started/locally/) to offer more flexibility and better compatibility with the compute platform.

### MODE installation

Install from PyPI:
  ```
  pip install MODE_omics
  ```  
    
Now you have created the environment for running MODE. To test the installation, try to import MODE in Python.

  ```python
  import MODE
  ```

## Usage

```python
from MODE import Deconvolution

SignatureMatrix1, CellFractionPrediction1, SignatureMatrix2, CellFractionPrediction2 = \
        Deconvolution(sc_rna='sim_sc_ref.txt', real_bulk1='sim_bulk_rna.txt', real_bulk2='sim_bulk_prot.txt',
                      omics1='RNAseq', omics2='Protein',
                      d_prior=[4.226208, 8.187766, 1.592641, 16.311203],
                      cell_type=['Astro', 'EN_Neuron', 'Microglia', 'Oligo'],
                      subj_var=0.1, step_p=1e-5, step_s=1e-5, eps=1e-4, max_iter=500,
                      sparse=True, sparse_prob=0.5, sep='\t',
                      datatype='counts', genelenfile=None,
                      batch_size=128, epochs=300)
```

### Input data
1. Single cell RNAseq reference: cell by gene matrix in txt format
2. Bulk RNA data: sample by gene matrix in txt format
3. Bulk non-RNA data: sample by feature matrix in txt format

### Parameters
#### Input
- `sc_rna`: file name of scRNAseq reference data (.txt format)
- `real_bulk1`: file name of target bulk RNA data (.txt format)
- `real_bulk2`: file name of target bulk non-RNA data (.txt format), non-RNA omic can be proteomic, DNA methylation, and ATACseq
- `omics1`: 'RNAseq', omics1 is always RNAseq
- `omics2`: 'Protein', 'DNAm', or 'ATACseq'. Users need to specify the type of non-RNA bulk data 
- `genelenfile`: optional, used when target bulk RNA is in TPM or FPKM

#### JNMF initialization
- `d_prior`: a prior Dirichlet distribution estimated from multi-subject single cell data
- `cell_type`: a list of query cell type names, need to be the same as unique cell types in the scRNAseq reference and follow alphabet order
- `subj_var`: between subject variance added to the prior Dirichlet distribution
- `step_p`: step size in projected gradient descent for cell count fraction parameter
- `step_s`: step size in projected gradient descent for cell size parameter
- `eps`: convergence criteria for projected gradient descent
- `max_iter`: maximum iteration for projected gradient descent

#### multimodal autoencoder
- `sparse`: True or False, users can indicate whether to add random sparsity to the simulated ground truth cell proportions or not
- `sparse_prob`: float number in (0, 1) to indicate the extent of sparsity
- `variance_threshold`: a feature filtering criteria for non DNAm omics, float number in (0, 1) indicating the proportion of features to keep according to variance from high to low 
- `scaler`: 'mms' or 'ss', used to preprocess datasets, 'mms': min-max scaler, 'ss': standard scaler
- `datatype`: datatype of bulk RNA target, use 'TPM', 'FPKM' or 'counts'
- `batch_size`: int, batch size in training
- `epochs`: int, epochs in training
- `seed`: set seed for reproducibility

## Citation

If you find MODE useful in your research and applications, please cite us:
```bibtex
@article{sun2025high,
  title={High-resolution digital dissociation of brain tumors with deep multimodal autoencoder},
  author={Sun, Jiao and Pan, Yue and Lin, Tong and Smith, Kyle and Onar-Thomas, Arzu and Robinson, Giles W and Zhang, Wei and Northcott, Paul A and Li, Qian},
  journal={bioRxiv},
  pages={2025--01},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

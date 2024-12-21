# MODE: multimodal deep autoencoder for high-resolution tissue dissociation

This is a novel multimodal autoencoder framework with parallel decoders - to jointly purify CTS multi-omic profiles and estimate the multimodal cellular compositions. MODE allows the input of cell markers detectable in scRNA-seq reference and accounts for the between-tissue heterogeneity in target bulk data, without the requirement of feature annotation mapping across molecular modalities. MODE is trained on the large-scale pseudo bulk multiomes constructed by probabilistic data generation with an external scRNA-seq reference and individualized non-RNA reference panel recovered from the target bulk profile.

## Model Overview
<p align="center">
  <img width="60%" src="https://github.com/jsuncompubio/MODE/blob/main/images/MODE_overview.png">
</p>

## Setup

1. Dependencies
```
git clone https://github.com/jsuncompubio/MODE
```
2. Setup environment
```
conda env create -f environment.yaml
conda activate mode
cd MODE
```

## Usage

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
```

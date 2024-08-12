# MODE: Deep autoencoder for multi-omics tissue recovery deconvolution

This is a multimodal deep neural autoencoder framework that allows the input of more cell markers detectable in scRNA-seq reference and accounts for the between-tissue heterogeneity in target bulk data, without the requirement of feature annotation mapping across molecular modalities.

## Model Overview

## Install

1. Download our repository
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
4. Gene length info: used to transform pseudo bulk RNA expression values (count to TPM/FPKM), see example in

### Parameters

 

## Citation

If you find MODE useful in your research and applications, please cite us:
```bibtex
```

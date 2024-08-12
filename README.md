# MODE: Deep autoencoder for multi-omics tissue recovery deconvolution

This is a multimodal deep neural autoencoder framework that allows the input of more cell markers detectable in scRNA-seq reference and accounts for the between-tissue heterogeneity in target bulk data, without the requirement of feature annotation mapping across molecular modalities.

## Model Overview
<p align="center">
  <img width="60%" src="[https://github.com/jsuncompubio/MODE/images/MODE_overview.png]">
</p>

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
#### Input
- `sc_rna`:
- `real_bulk1`:
- `real_bulk2`:
- `omics1`:
- `omics2`:
- `genelenfile`:

#### JNMF initialization
- `d_prior`:
- `cell_type`:
- `subj_var`:
- `step_p`:
- `step_s`:
- `eps`:
- `max_iter`:

#### multimodal autoencoder
- `sparse`:
- `sparse_prob`:
- `variance_threshold`: for non DNAm omics
- `scaler`:
- `datatype`: datatype of bulk RNA target
- `batch_size`:
- `epochs`:
- `seed`:

## Citation

If you find MODE useful in your research and applications, please cite us:
```bibtex
```

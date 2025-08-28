# KP-GRN predict

![abstract](readme_fig/Workflow.png)

## Overview 🔍
This repository contains the analysis pipeline and KP isolate GRN prediction model.

A deep learning-based framework for predicting transcription factor (TF) binding sites and regulatory interactions in *K. pneumoniae*. The framework integrates multiple data types and uses advanced neural network architectures to achieve high prediction accuracy.

## Model Architecture 🏗️
### Input requirement
- DNA sequence features (one-hot encoding)
- Genomic context information
- Genomic annotation files (GFF format)

### Neural Network Structure
1. **DNA sequences encoding**
   - Extract promoter region of each gene
   - Use one-hot to encode the sequences

2. **Protein embedding extraction**
   - Extract embedding from pre-trained model

3. **Feature Integration**
   - Multi-head attention for feature fusion
   - Residual connections
   - Final dense layers for prediction

### Dependencies 📦

Mandatory software:

| Name         | Version  | Source|
|:-------------|:---------|:--------|
| **samtools** | v1.17    |conda|
| **minimap2** | v2.17    |conda|
| **python**   | \>=3.9.2 |conda|
| **nanoCEM**  | 0.0.5.8  |Pypi|
| **h5py**  | 3.8.0  |Pypi|
| **pod5**  | 0.2.4  |Pypi|
| **Samtools**  | 1.17  |conda|
| **SeqKit**  | 2.6.1  |conda|
| **ont-fast5-api**  | 4.1.1  |Pypi|
| **slow5tools**  | 1.2.0  |Pypi|
| **memes**  | 1.8.0  |R|
| **Giraffe**  | 0.1.0.14  |Pypi|

### Usage examples 💻
```python
# Load model and make predictions
from kp_dl_framework import TFPredictor

# Initialize predictor
predictor = TFPredictor(model_path='models/best_model.h5')

# Predict binding sites
sequences = load_sequences('input.fasta')
predictions = predictor.predict_binding(sequences)

# Network inference
network = predictor.infer_network(
    tf_list=tf_list,
    target_genes=target_genes,
    expression_data=expr_data
)
```

# Kleb-TF

This GitHub repository contains all the custom scripts and shell commands used in our paper,

**Topology and Evolution of Global Transcriptional Regulatory Networks Across the Klebsiella pneumoniae Pan-Genome**.


## Graphic abstract
![abstract](./abstract/abstract.png)

## Data & Models Availability

All raw sequence files, processed data, and pre-trained models are publicly available:

* **Raw Sequence Files:** Uploaded to NCBI GEO under accession [GSE306886](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE306886).
* **Processed Data:** Available on [Zenodo](https://doi.org/10.5281/zenodo.21786623).
* **GRN-predictor Model:** Published on Hugging Face at [Dylan-LE/KP-GRM-predictor](https://huggingface.co/Dylan-LE/KP-GRM-predictor/).

### Setup & Model Download

To use the GRN-predictor, you will also need to download the dependency models from ESM-DBP.

1. Navigate to the ESM-DBP Hugging Face repository: [zengwenwu/ESM-DBP](https://huggingface.co/zengwenwu/ESM-DBP/tree/main).
2. Download the required weights and place them directly into your local `GRN-predictor` directory:

```bash
# Example structure
└── your-project-repo/
    └── GRN-predictor/
        └── [Downloaded ESM-DBP model files here]
```

## Code available

System requirement: Ubuntu 22.04; R version 4.5.1; CUDA version ≥ 12.1;

**Notes:** We highly recommend creating a separate conda environment.

### Install


        git clone https://github.com/LuBeifang/Kleb-TF/
        cd Kleb-TF/GRN-predictor
        conda env create -f requirements.yml
        conda activate KPTF
        wget -c https://huggingface.co/Dylan-LE/KP-GRM-predictor/resolve/main/96TF/best_model.pt
        wget -c https://huggingface.co/zengwenwu/ESM-DBP/resolve/main/ESM-DBP.model

        
### KP-GRN predict
        usage: predict.py [-h] [-tf INPUT_TF_FASTA] [-g GENOME] [-gff GFF] [-model MODEL] [-t THRESHOLD] [-o OUTPUT]

        Predict TF and DNA binding

        options:
        -h, --help            show this help message and exit
        -tf, --input_tf_fasta INPUT_TF_FASTA
                                Directory path of input TF fasta file
        -g, --genome GENOME   Directory path of input KP isolates genome fasta file
        -gff GFF              Directory path of input KP isolates gene annotation gff file
        -model MODEL          Path to the model file
        -t, --threshold THRESHOLD
                                Prediction threshold
        -o, --output OUTPUT   Output directory path
### Example 

        python predict.py -tf example/VmrR.fa -g example/SH12.fna -gff example/SH12.gff -model best_model.pt -t 0.95 -o result_file
        
## Cite us
This project is licensed under the MIT License. See the LICENSE file for details.

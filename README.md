# Kleb-TF

This GitHub repository contains all the custom scripts and shell commands used in our paper,

**Topology and Evolution of Global Transcriptional Regulatory Networks Across the Klebsiella pneumoniae Pan-Genome**.


## Graphic abstract
![abstract](./abstract/abstract.png)

## Data available
All raw sequence files are uploaded to NCBI (GEO:[GSE306886](https:)).

Processed files are uploaded to [Zenodo](10.5281/zenodo.16993559). 

Trained model for GRN-predictor is published to [Hugging Face](https://huggingface.co/Dylan-LE/KP-GRM-predictor/).

Download the models of ESM-DBP at [Hugging Face](https://huggingface.co/zengwenwu/ESM-DBP/tree/main).

If you have any further requests, please contact xindeng@cityu.edu.hk or beifanglu2-c@my.cityu.edu.hk. We are pleased to help!

## Code available

System requirement: Ubuntu 22.04; R version: 4.5.1.

**Notes:** We highly recommend creating a separate conda environment to manage the following software tools.

### Install


        git clone https://github.com/LuBeifang/Kleb-TF/
        cd Kleb-TF/GRN-predictor
        conda env create -f environment.yml
        wget -c https://huggingface.co/LucileLu77/KP-GRM-predictor/resolve/main/best_model.pt
        
### KP-GRN predict
        usage: predict_pro_permu.py [-h] [-tf INPUT_TF_FASTA] [-g GENOME] [-gff GFF] [-model MODEL] [-t THRESHOLD] [-o OUTPUT]

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

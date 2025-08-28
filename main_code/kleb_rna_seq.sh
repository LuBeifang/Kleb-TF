#!/usr/bin/env bash
#RNA-seq raw results in dir raw/
#conda activate dap
mkdir ../trim
mkdir rawqc
#QC
fastqc *gz -o rawqc/
#make sure the files name format in <ID-repeat_Ln_1.fq.gz>
for i in $(ls *.fq.gz | cut -d '_' -f 1-2 | sed 's/_/_/g' | sort | uniq); do echo ${i}; done
#trim
for i in $(ls *.fq.gz | cut -d '_' -f 1-2 | sed 's/_/_/g' | sort | uniq); do trim_galore -q 20 --paired --phred33 --stringency 3 ${i}_1.fq.gz ${i}_2.fq.gz --gzip -o ../trim ; echo ${i}; done
echo "trim_galore finished"
cd ../trim
#map
#remember to revise the reference
for i in $(ls *.fq.gz | cut -d '_' -f 1-2 | sed 's/_/_/g' | sort | uniq); do bowtie2 -p 20 -x /media/lu/lu2023/ref/hvkp4_genome -1 ${i}_1_val_1.fq.gz -2 ${i}_2_val_2.fq.gz -S ${i}.sam; samtools view -@ 8 -bS ${i}.sam > ${i}.bam; samtools sort -@ 8 ${i}.bam > ${i}.sorted.bam; rm ${i}.sam ${i}.bam; done
echo "bam produced and sorted"
featureCounts -T 10 -t gene -g ID -a /media/lu/lu2023/ref/hvkp4.gff  -o ../mu_counts.txt *bam
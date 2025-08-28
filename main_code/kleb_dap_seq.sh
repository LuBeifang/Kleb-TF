#!/usr/bin/env bash
#DAP-seq raw results in dir raw/
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
bedtools intersect -a EV-1_L1.sorted.bam -b EV-2_L1.sorted.bam > EV.sorted.bam
for i in $(ls *.fq.gz | cut -d '_' -f 1-2 | sed 's/_/_/g' | sort | uniq); do echo ${i}; macs2 callpeak -g 6.2e6 -B -p 0.05 --nomodel --extsize 75 -t ${i}.sorted.bam -c EV.sorted.bam -n ${i}; done
echo "macs2 finished!"
for i in $(ls *narrowPeak | cut -d '-' -f 1 | sed 's/_/_/g' | sort | uniq); do echo ${i}; bedtools intersect -r  -a ${i}-1_L1_peaks.narrowPeak -b ${i}-2_L1_peaks.narrowPeak > ${i}.bed ; done
echo "Cheers!"
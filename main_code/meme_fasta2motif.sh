#!/bin/bash

# set input dir and output dir
input_dir="/media/lu/lu2024/methyl_dap/fasta2/"
output_dir="/media/lu/lu2024/methyl_dap/fasta2/meme"

# mkdir -p "$output_dir"

for fasta_file in "$input_dir"/*.fasta; do

    filename=$(basename -- "$fasta_file")
    filename="${filename%.*}"

    file_output_dir="$output_dir/$filename"

    meme "$fasta_file" -o "$file_output_dir" -dna -nostatus -time 14400 -mod zoops -nmotifs 3 -minw 6 -maxw 50 -objfun classic -revcomp -markov_order 0

    echo "Finished processing $fasta_file"
done

for i in *; do echo ${i}; cd ${i}; mv meme.html ../${i}.html; cd ..;  done
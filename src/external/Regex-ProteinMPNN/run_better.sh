#!/bin/bash
source activate PROTEINMPNN_ENV

input_folder="/path/to/input/pdbs"
input_pattern="target_.*"

output_folder="/path/to/output/folder"

temp_folder="/path/to/temp/folder"

# Use regex to specify fixed amino acids
fix_pattern="AAAAAAAAA.*AAAAAAAAAAA"

num_outputs="5"
sampling_temperature="0.05"
design_chains="A"

proteinmpnn_path="/path/to/ProteinMPNN/"

python /path/to/repo/better_mpnn.py $input_folder $output_folder $temp_folder \
    --input_pattern $input_pattern \
    --design_chains $design_chains \
    --fix_pattern $fix_pattern \
    --num_outputs $num_outputs \
    --sampling_temperature $sampling_temperature \
    --mpnn_path $proteinmpnn_path

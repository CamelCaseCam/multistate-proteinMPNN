####################################################################################################
# This is a modified UI for proteinMPNN that adds quality of life features (especially for batch 
# processing)
# Created by: Cameron Kroll
####################################################################################################

'''
This script takes the following inputs:

IO:
 - Input directory: the directory containing the input files
 - Input pattern (optional): the pattern to match for input files
 - Output directory: the directory to save the output files
 - Temp directory: the directory to save temporary files
Design:
 - Design chains (optional): the chains to design
 - Fix pattern (optional): the pattern to match for fixed residues
 - Fix ranges (optional): ranges of residues to fix in the format "chain:res1-res2,chain:res3-res4"
 - Num outputs (optional): the number of outputs per input file to generate
Inference:
 - Sampling temperature (optional): the "predictability" of the model
 - mpnn_path (optional): the path to the proteinMPNN directory
Development:
 - Test mode (optional): Not a parameter, implicitly set if no input or output directory is passed
'''

import os
import sys
import argparse
import shutil
import subprocess
import re
import random

from Bio.PDB import PDBParser

# Data structure for chain/residue indices (only needs to implement __eq__)
class ChainResidue:
    def __init__(self, chain, residue):
        self.chain = chain
        self.residue = residue

    def __eq__(self, other):
        return self.chain == other.chain and self.residue == other.residue
    
    def __hash__(self):
        return hash((self.chain, self.residue))
    
    def __repr__(self):
        return f"{self.chain}:{self.residue}"
    
# Because biopython is annoying
three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}
    
def chain_residues_to_indices(residues : set[ChainResidue], design_chains : list[str]) -> str:
    '''
    Converts the chain/residue indices to the format used by proteinMPNN using the following steps:
    1. Sort chain/residue indices into a dictionary with chains as keys
    2. Sort the residues for each chain that will be designed (ascending order)
    3. In order of design chains, generate a string for each chain (space-separated indices if there are any, otherwise empty string)
    4. Concatenate the strings for each chain with a comma separator
    '''
    # Sort the residues by chain
    residues_dict = {}
    for residue in residues:
        if residue.chain not in residues_dict:
            residues_dict[residue.chain] = []
        residues_dict[residue.chain].append(residue.residue)
    
    # Sort the residues for each chain that will be designed
    design_residues = []
    for chain in design_chains:
        design_residues.append((chain, sorted(residues_dict[chain]) if chain in residues_dict else []))
    
    # Generate the string for each chain
    chain_strings = []
    for chain, residues in design_residues:
        if len(residues) == 0:
            chain_strings.append('')
        else:
            chain_strings.append(' '.join(map(str, residues)))
    
    # Concatenate the strings for each chain
    return ', '.join(chain_strings)

def get_input_files(args):
    input_files = []
    if args.input_pattern is None:
        input_pattern = r'.*\.pdb'
    else:
        input_pattern = args.input_pattern
    input_pattern = re.compile(input_pattern)
    for file in os.listdir(args.input_dir):
        if input_pattern.match(file):
            input_files.append(file)
    if len(input_files) == 0:
        print('No input files found')
        return None
    
    return input_files

def handle_ranges(ranges : str, fixed_positions : set, seqs):
    # Split the ranges
    ranges = ranges.split(',')
    for r in ranges:
        # Split the chain and residue range
        chain, res_range = r.split(':')
        res_range = res_range.split('-')
        res1, res2 = int(res_range[0]), int(res_range[1])
        # Check if range is valid
        if chain not in seqs:
            raise ValueError(f"Chain {chain} not found in input file")
        if res1 < 1 or res2 > len(seqs[chain]):
            raise ValueError(f"Residue range {res1}-{res2} is out of bounds for chain {chain}")
        assert res1 <= res2, "Residue range is invalid"

        # Add the residues to the fixed positions
        for i in range(res1, res2+1):
            fixed_positions.add(ChainResidue(chain, i))

def handle_fix_pattern(fix_pattern : re.Pattern, fixed_positions : set, seqs):
    for chain in seqs:
        # Find all matches in the sequence
        for match in fix_pattern.finditer(seqs[chain]):
            # Add the residues to the fixed positions
            for i in range(match.start(), match.end()):
                fixed_positions.add(ChainResidue(chain, i))

def run_inference(input_path, args):
    '''
    The inference pipeline is as follows:
    1. Copy the input file to `temp_dir/inputs/...`
    2. Use `parse_multiple_chains.py` to parse the input file to `temp_dir/parsed_pdbs.jsonl`
    3. Use `assign_fixed_chains.py` to assign fixed chains to `temp_dir/assigned_pdbs.jsonl`
    4. Use fixed info to generate fixed positions 
    5. Use `make_fixed_positions_dict.py` to assign fixed positions to `temp_dir/fixed_pdbs.jsonl`
    6. Use `protein_mpnn_run.py` to generate outputs
    7. Delete temporary files
    '''

    # 1. Copy the input file to `temp_dir/inputs/...`
    input_filename = os.path.basename(input_path)
    input_dir = os.path.join(args.temp_dir, 'inputs')
    os.makedirs(input_dir, exist_ok=True)
    tmpinput_path = os.path.join(input_dir, input_filename)
    shutil.copy(input_path, tmpinput_path)

    # 2. Use `parse_multiple_chains.py` to parse the input file to `temp_dir/parsed_pdbs.jsonl`
    cmd = ['python', os.path.join(args.mpnn_path, 'helper_scripts/parse_multiple_chains.py'), 
           f"--input_path={input_dir}", f"--output_path={os.path.join(args.temp_dir, 'parsed_pdbs.jsonl')}"]
    output = subprocess.run(cmd)
    assert output.returncode == 0, "Error parsing input file"
    # Print the output of the script
    print("Parsed input file")

    # 3. Use `assign_fixed_chains.py` to assign fixed chains to `temp_dir/fixed_pdbs.jsonl`
    cmd = ['python', os.path.join(args.mpnn_path, 'helper_scripts/assign_fixed_chains.py'),
            f"--input_path={os.path.join(args.temp_dir, 'parsed_pdbs.jsonl')}", 
            f"--output_path={os.path.join(args.temp_dir, 'assigned_pdbs.jsonl')}",
            "--chain_list", f"{args.design_chains}"]
    assert subprocess.run(cmd).returncode == 0, f"Error assigning fixed chains for fixed chains {args.design_chains}"
    print("Assigned fixed chains")

    # 4. Use fixed info to generate fixed positions
    fixed_positions = set()
    # Get sequences from the orginal input file (fixed pdb has way more stuff)
    seqs = []
    parser = PDBParser()
    structure = parser.get_structure('input', input_path)
    for model in structure:
        model_seqs = {}
        for chain in model:
            chain_seq = ''
            for residue in chain:
                chain_seq += three_to_one[residue.resname]
            model_seqs[chain.get_id()] = chain_seq
        seqs.append(model_seqs)
    
    assert len(seqs) == 1, "Only one model per input file is supported"
    seqs = seqs[0]
    # Get fixed residues
    if args.fix_ranges is not None:
        handle_ranges(args.fix_ranges, fixed_positions, seqs)

    if args.fix_pattern is not None:
        fix_pattern = re.compile(args.fix_pattern)
        handle_fix_pattern(fix_pattern, fixed_positions, seqs)

    # Generate the fixed positions string
    fixed_positions_str = chain_residues_to_indices(fixed_positions, args.design_chains.split(' '))
    
    # 5. Use `make_fixed_positions_dict.py` to assign fixed positions to `temp_dir/fixed_pdbs.jsonl`
    cmd = ['python', os.path.join(args.mpnn_path, 'helper_scripts/make_fixed_positions_dict.py'),
            f"--input_path={os.path.join(args.temp_dir, 'parsed_pdbs.jsonl')}",
            f"--output_path={os.path.join(args.temp_dir, 'fixed_pdbs.jsonl')}",
            "--chain_list", f"{args.design_chains}",
            "--position_list", f"{fixed_positions_str}"]
    assert subprocess.run(cmd).returncode == 0, f"Error assigning fixed positions for fixed positions \"{fixed_positions_str}\""

    # 6. Use `protein_mpnn_run.py` to generate outputs
    cmd = ['python', os.path.join(args.mpnn_path, 'protein_mpnn_run.py'),
           "--jsonl_path", str(os.path.join(args.temp_dir, 'parsed_pdbs.jsonl')),
           "--chain_id_jsonl", str(os.path.join(args.temp_dir, 'assigned_pdbs.jsonl')),
           "--fixed_positions_jsonl", str(os.path.join(args.temp_dir, 'fixed_pdbs.jsonl')),
           "--out_folder", args.output_dir,
           "--num_seq_per_target", (str(args.num_outputs) if args.num_outputs is not None else '1'),
           "--sampling_temp", (str(args.sampling_temperature) if args.sampling_temperature is not None else '0.05'),
           "--seed", str(random.randint(0, 1000000)),
           "--batch_size", "1"]
    assert subprocess.run(cmd).returncode == 0, "Error running proteinMPNN"

    # 7. Delete temporary files (maintain anything we didn't create in that directory)
    os.remove(os.path.join(args.temp_dir, 'parsed_pdbs.jsonl'))
    os.remove(os.path.join(args.temp_dir, 'assigned_pdbs.jsonl'))
    os.remove(os.path.join(args.temp_dir, 'fixed_pdbs.jsonl'))
    os.remove(tmpinput_path)

    print(f"Finished processing {input_filename}")

def test():
    print("Testing residue range handling")
    fixed_positions = set()
    range_input = 'A:1-10,B:1-10'
    seqs = {'A': 'A'*10, 'B': 'B'*10}
    handle_ranges(range_input, fixed_positions, seqs)

    assert len(fixed_positions) == 20, "Incorrect number of fixed positions"
    print("Residue range handling passed")

    print("Testing residue pattern handling")
    fixed_positions = set()
    pattern = "NEVERGNNAGIVE"
    seqs = {'A': "G" + pattern + "G", 'B': "AAA" + pattern + "AAA" + pattern}
    fix_pattern = re.compile(pattern)
    handle_fix_pattern(fix_pattern, fixed_positions, seqs)

    assert len(fixed_positions) == 3 * len(pattern), "Incorrect number of fixed positions"
    for i in range(1, 1 + len(pattern)):
        assert ChainResidue('A', i) in fixed_positions, "Fixed position not found"
    for i in range(3, 3 + len(pattern)):
        assert ChainResidue('B', i) in fixed_positions, "Fixed position not found"
    for i in range(6 +len(pattern), 6 + 2*len(pattern)):
        assert ChainResidue('B', i) in fixed_positions, "Fixed position not found"
    print("Residue pattern handling passed")

    # Test chain_residues_to_indices
    indices = set()
    indices.add(ChainResidue('A', 1))
    indices.add(ChainResidue('A', 2))
    indices.add(ChainResidue('B', 1))
    indices.add(ChainResidue('B', 2))
    indices.add(ChainResidue('D', 3))
    indices.add(ChainResidue('D', 1))
    indices.add(ChainResidue('D', 2))
    indices.add(ChainResidue('F', 1))

    design_chains = ['A', 'B', 'C', 'D']
    ind = chain_residues_to_indices(indices, design_chains)
    assert ind == "1 2, 1 2, , 1 2 3", "Incorrect chain/residue indices"



def main():
    # Check if in test mode
    if len(sys.argv) == 1:
        test()
        return

    parser = argparse.ArgumentParser(description='Run proteinMPNN on a set of input files')
    parser.add_argument('input_dir', help='The directory containing the input files')
    parser.add_argument('output_dir', help='The directory to save the output files')
    parser.add_argument('temp_dir', help='The directory to save temporary files')
    parser.add_argument('--input_pattern', help='The pattern to match for input files')
    parser.add_argument('--design_chains', help='The chains to design')
    parser.add_argument('--fix_pattern', help='The pattern to match for fixed residues')
    parser.add_argument('--fix_ranges', help='Ranges of residues to fix in the format "chain:res1-res2,chain:res3-res4"')
    parser.add_argument('--num_outputs', help='The number of outputs per input file to generate')
    parser.add_argument('--sampling_temperature', help='The "predictability" of the model')
    parser.add_argument('--mpnn_path', help='The path to the proteinMPNN directory')
    args = parser.parse_args()

    if args.mpnn_path is None:
        # CHANGE THIS TO CUSTOMIZE
        print("Warning: using developer's default path to proteinMPNN. You probably want to change this or specify --mpnn_path.")
        args.mpnn_path = '/media/cameronk/Linuxdata/ProteinMPNN/'

    # First, find all the input files
    input_files = get_input_files(args)
    if input_files is None:
        print("Exiting")
        return

    # Run proteinMPNN on each input file
    for inputs in input_files:
        try:
            run_inference(os.path.join(args.input_dir, inputs), args)
        except:
            print(f"Error processing {inputs}")
            print(f"Regex fix pattern: {args.fix_pattern}")
            return

if __name__ == '__main__':
    main()
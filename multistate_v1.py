# NOTE: must activate "pmpnn" conda environment before running this script

from src.program import parse_program

import argparse
import torch
import os
import copy
from tqdm import tqdm

from src.external.ProteinMPNN.protein_mpnn_utils import ProteinMPNN, tied_featurize
from src.external.better_mpnn import ChainResidue


hidden_dim = 128
num_layers = 3 

WEIGHT_PATH = "src/external/ProteinMPNN/vanilla_model_weights/v_48_030.pt"
BATCH_SIZE = 1
MAX_LENGTH = 200000
BACKBONE_NOISE = 0.0
PSSM_THRESHOLD = 0.0

initial_temperature = 0.01
final_temperature = 0.1
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
alphabet_dict = dict(zip(alphabet, range(21)))   
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def main():
    # TODO: move all this into the program definition
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input file pattern")
    parser.add_argument("output_path", help="Output path (must exist)")
    parser.add_argument("--generations", type=int, default=100, help="Maximum number of generations to run")
    parser.add_argument("--seqs", type=int, default=10, help="Number of sequences to generate")
    args = parser.parse_args()
    inference(args.input_file, args.output_path, args.generations, args.seqs)

def test():
    inference("tests/3CWQ/3CWQ.mst", "tests/3CWQ/out", 100, 10, save_traj=True)

def inference(input_file, output_path, generations=100, seqs=10, save_traj=True):
    seq, aminos, database, fixed_positions, fixed_residues, bias, mapping, chainlengths, output = parse_program(input_file)
    # Make assigned chains
    assigned_obj = make_assigned_chains(database, mapping)
    # Make fixed positions
    fixed_pos_obj = make_fixed_positions(database, fixed_residues, mapping)
    
    # Load model
    checkpoint = torch.load(WEIGHT_PATH, map_location=device)
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim,
                        hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
                        augment_eps=BACKBONE_NOISE, k_neighbors=checkpoint['num_edges'])
    
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Model loaded")

    # Make sure output path exists
    if not output_path[-1] == '/':
        output_path += '/'
    if not os.path.exists(output_path + "seqs"):
        os.makedirs(output_path + "seqs")

    if save_traj and not os.path.exists(output_path + "traj"):
        os.makedirs(output_path + "traj")
    
    # Run generations
    start = copy.deepcopy(output)
    fixed_pos_obj_start = copy.deepcopy(fixed_pos_obj)
    fixed_residues_start = copy.deepcopy(fixed_residues)
    for i in range(seqs):
        if save_traj:
            with open(output_path + f"traj/traj_{i}.fa", "w") as f:
                f.write(f">seq_{i}_base\n")
                for chain in output.keys():
                    f.write("".join(output[chain]) + ":")
                f.write("\n")
        for j in tqdm(range(generations - 1)):
            if save_traj:
                with open(output_path + f"traj/traj_{i}.fa", "a") as f:
                    f.write(f">seq_{i}_gen_{j}\n")
                    for chain in output.keys():
                        f.write("".join(output[chain]) + ":")
                    f.write("\n")
            gen_temp = initial_temperature + (final_temperature - initial_temperature) * (j / generations)
            run_generation(model, database, assigned_obj, fixed_pos_obj, fixed_residues, bias, mapping, chainlengths, output,
                            gen_temp)
        # Run final generation
        run_generation(model, database, assigned_obj, fixed_pos_obj, fixed_residues, bias, mapping, chainlengths, output, 
                       temperature=1.0, allow_no_change=False)
        if save_traj:
            with open(output_path + f"traj/traj_{i}.fa", "a") as f:
                f.write(f">seq_{i}_final\n")
                for chain in output.keys():
                    f.write("".join(output[chain]) + ":")
                f.write("\n")
        # Save sequence
        with open(output_path + f"seqs/seqs.fa", "a") as f:
            f.write(f">seq_{i}\n")
            for chain in output.keys():
                f.write("".join(output[chain]) + ":")
            f.write("\n")
        # Reset output
        output = copy.deepcopy(start)
        fixed_pos_obj = copy.deepcopy(fixed_pos_obj_start)
        fixed_residues = copy.deepcopy(fixed_residues_start)
    
    
def run_generation(model, database, assigned_obj, fixed_pos_obj, fixed_residues, bias, mapping, chainlengths, output, temperature, allow_no_change=True):
    log_probs = get_log_probs(model, database, assigned_obj, fixed_pos_obj, mapping, chainlengths, bias)
    # log probs are a dictionary of log probabilities for each amino acid in each chain
    apply_probs(log_probs, output, fixed_pos_obj, fixed_residues, mapping, database, temperature, allow_no_change)


def apply_probs(log_probs, output, fixed_positions, fixed_residues, mapping, database, temperature, allow_no_change=True):
    '''
    We have a dictionary of log probabilities that (likely) do not sum to 1 when exponentiated. We need to add an extra
    term representing no change in the amino acid. We then convert the probabilities to logits and apply our temperature. 

    Once we have the adjusted logits, sample from the distribution to get the new amino acid for each non-fixed position.

    If a new amino acid is chosen, update the fixed positions dictionary to reflect the change.
    '''
    reverse_mapping = {name : {v: k for k, v in mapping[name].items()} for name in mapping.keys()}
    for chain in log_probs.keys():
        # Convert to probabilities
        probs = torch.exp(log_probs[chain])
        # Add the no-change term
        if allow_no_change:
            no_change = torch.ones(probs.shape[0], 1, device=device) - torch.sum(probs, dim=-1, keepdim=True)
            assert torch.allclose(torch.sum(probs, dim=-1) + no_change.squeeze(), torch.ones(probs.shape[0], device=device))
            probs = torch.cat([probs, no_change], dim=-1)
        #print(probs.cpu().numpy()[0]) # -> confirms that these are log probabilities and **not** logits
        
        # Convert to logits
        # Make sure it's the right datatype
        probs = probs.to(torch.float64)
        logits = torch.log(probs/(-probs + torch.ones_like(probs)))
        
        # Apply temperature
        logits /= temperature
        # Sample from the distribution
        new_amino_acids = torch.multinomial(torch.exp(logits), 1)
        
        # Update output
        for i, amino in enumerate(new_amino_acids):
            # If this position is fixed, do not change it
            if ChainResidue(chain, i) in fixed_residues:
                continue
            aminoletter = (alphabet + "-")[amino]
            old_amino = output[chain][i]
            if aminoletter != old_amino:
                output[chain][i] = aminoletter
                # Update fixed residues
                fixed_residues.append(ChainResidue(chain, i))
                # Update fixed positions and edit the database
                for structure in fixed_positions.keys():
                    # Reverse the mapping
                    structure_reverse_mapping = reverse_mapping[structure]
                    if chain in structure_reverse_mapping:
                        structure_chain = structure_reverse_mapping[chain]
                        if i+1 not in fixed_positions[structure][structure_chain]:
                            fixed_positions[structure][structure_chain].append(i+1)
                            
                # Update the database
                for structure in database:
                    # Get the structure name
                    name = structure["name"]
                    # Reverse the mapping
                    structure_reverse_mapping = reverse_mapping[name]
                    if chain in structure_reverse_mapping:
                        structure_chain = structure_reverse_mapping[chain]
                        structure["seq_chain_" + structure_chain] = structure["seq_chain_" + structure_chain][0:i] + aminoletter + structure["seq_chain_" + structure_chain][i+1:]
                        
def get_log_probs(model, database, assigned_obj, fixed_pos_obj, mapping, chainlengths, bias):
    probs = []
    output = {} # Map from output chain to probabilities
    with torch.no_grad():
        for structure in database:
            S_sample_list = []
            batch_clones = [copy.deepcopy(structure) for _ in range(BATCH_SIZE)]
            tfoutput = tied_featurize(batch_clones, device, assigned_obj, 
                                        fixed_pos_obj, None, None, None, None, ca_only=False)
            (X, S, mask, _, chain_M, chain_encoding_all, _, _, _, 
            _, chain_M_pos, _, residue_idx, _, 
            _, _, _, _, _, 
            _) = tfoutput
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            probs.append(log_probs)
        # Add the probabilities and convert to a usable format
        for i, structure in enumerate(assigned_obj.keys()):
            structure_chains = assigned_obj[structure]
            design_chains = structure_chains[0]
            start = 0
            for chain in design_chains:
                output_chain = mapping[structure][chain]
                # Get end of chain
                end = start + chainlengths[output_chain]
                # Check if the chain is in the output
                if output_chain in output:
                    output[output_chain] += probs[i][0, start:end, :]
                else:
                    output[output_chain] = probs[i][0, start:end, :]
                start = end
        # Add bias
        # First, we have to calculate average amino probabilities for each index in each chain
        avg_amino_probs = {}
        for chain in output.keys():
            avg_amino_probs[chain] = torch.mean(output[chain], dim=-1)
        for biasdict in bias:
            # biasdict is a dictionary from (output) chain -> list of shape (num_residues, 21)
            for chain in biasdict.keys():
                vec = torch.tensor(biasdict[chain], device=device)
                # Multiply by the average amino acid probabilities
                output[chain] += vec * avg_amino_probs[chain]
    return output

def make_fixed_positions(database, fixed_residues, mapping):
    fixed_pos_obj = {}
    for i, structure in enumerate(database):
        fixed_positions = {}
        name = structure['name']
        # Get chains
        chains = set()
        for key in structure.keys():
            if '_chain_' in key:
                chains.add(key.split('_')[-1])
        
        # For all chains in the output, add fixed positions from `fixed_residues` (adding 1 because ChainResidue is 0-indexed)
        for chain in chains:
            if chain in mapping[name]:
                output_chain = mapping[name][chain]
                ch_fixed = []
                for res in fixed_residues:
                    if res.chain == output_chain:
                        ch_fixed.append(res.residue + 1)
                fixed_positions[chain] = ch_fixed
            else:
                fixed_positions[chain] = []
        fixed_pos_obj[name] = fixed_positions
    return fixed_pos_obj


def make_assigned_chains(database, mapping):
    assigned_obj = {}
    for structure in database:
        # Get the structure name
        name = structure["name"]
        structure_mapping = mapping[name]
        # Get the chains
        chains = set()
        for key in structure.keys():
            if '_chain_' in key:
                chains.add(key.split('_')[-1])
        # Assign chains from mapping, adding other chains to `unassigned`
        assigned_chains = []
        unassigned_chains = []
        for chain in chains:
            if chain in structure_mapping:
                assigned_chains.append(chain)
            else:
                unassigned_chains.append(chain)
        # Add to assigned_obj
        assigned_obj[name] = [assigned_chains, unassigned_chains]
    return assigned_obj

if __name__ == "__main__":
    main()
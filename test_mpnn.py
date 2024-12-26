# NOTE: must activate "pmpnn" conda environment before running this script

import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess
import importlib

from src.external.ProteinMPNN.protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
from src.external.ProteinMPNN.protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

import src.external.better_mpnn as better_mpnn

hidden_dim = 128
num_layers = 3 

WEIGHT_PATH = "external/ProteinMPNN/vanilla_model_weights/v_48_030.pt"
BATCH_SIZE = 1
MAX_LENGTH = 200000
BACKBONE_NOISE = 0.0
PSSM_THRESHOLD = 0.0

temperature = 0.1
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
alphabet_dict = dict(zip(alphabet, range(21)))   
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def make_assigned_chains(dataset, assigned : list[str]):
    assigned_obj = {}
    for protein in dataset:
        # Get chains from the protein
        chains = set()
        for key in protein.keys():
            if '_chain_' in key:
                chains.add(key.split('_')[-1])
        # Assign chains from `assigned` list, adding other chains to `unassigned`
        assigned_chains = []
        unassigned_chains = []
        for chain in assigned:
            # Check chain is in the protein
            if chain in chains:
                assigned_chains.append(chain)
            else:
                raise ValueError(f"Chain {chain} not found in protein")
            
            # Remove chain from `chains`
            chains.remove(chain)
        # Add unassigned chains
        for chain in chains:
            unassigned_chains.append(chain)
        # Add to assigned_obj
        assigned_obj[protein['name']] = [assigned_chains, unassigned_chains]
    return assigned_obj

def make_fixed_positions(dataset, assigned, fixed : list[list[int]]):
    fixed_pos_obj = {}
    for i, protein in enumerate(dataset):
        fixed_positions = {}
        name = protein['name']
        # Add fixed positions for each designed chain
        for j, chain in enumerate(assigned[name][0]):
            fixed_positions[chain] = fixed[j]
        # Add fixed positions for each unassigned chain (empty array)
        for chain in assigned[name][1]:
            fixed_positions[chain] = []
        fixed_pos_obj[name] = fixed_positions

    return fixed_pos_obj


def run_mpnn(parsed_path : str, assigned : list[str], fixed : list[list[int]]):
    # Load parsed chains
    dataset = StructureDataset(parsed_path, truncate=None, max_length=MAX_LENGTH)
    
    # Make assigned chains
    assigned_obj = make_assigned_chains(dataset, assigned)
    
    # Make fixed positions
    fixed_positions = make_fixed_positions(dataset, assigned_obj, fixed)

    # Load model
    checkpoint = torch.load(WEIGHT_PATH, map_location=device)
    noise_level = checkpoint['noise_level']
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim,
                        hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
                        augment_eps=BACKBONE_NOISE, k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Model loaded")

    output_path = "test_mpnn_output/"
    if not os.path.exists(output_path + "seqs"):
        os.makedirs(output_path + "seqs")
    
    total_residues = 0
    protein_list = []
    total_step = 0

    # Inference
    with torch.no_grad():
        # We only care about doing the first protein
        protein = dataset[0]
        score_list = []
        global_score_list = []
        all_probs_list = []
        all_log_probs_list = []
        S_sample_list = []
        batch_clones = [copy.deepcopy(protein) for _ in range(BATCH_SIZE)]
        # chain_id_dict = assigned_obj
        # fixed_positions_dict = fixed_positions
        tfoutput = tied_featurize(batch_clones, device, assigned_obj, 
                                        fixed_positions, None, None, None, None, ca_only=False)
        (X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, 
            masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, 
            tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, 
            tied_beta) = tfoutput
        pssm_log_odds_mask = (pssm_log_odds_all > PSSM_THRESHOLD).float() #1.0 for true, 0.0 for false
        protein_name = protein['name']
        randn_1 = torch.randn(chain_M.shape, device=X.device)
        log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
        mask_for_loss = mask*chain_M*chain_M_pos
        scores = _scores(S, log_probs, mask_for_loss) #score only the redesigned part
        native_score = scores.cpu().data.numpy()
        ali_file = output_path + 'seqs/' + batch_clones[0]['name'] + '.fa'

        # aaaaaaaa whyyyyyyyyy
        omit_AAs_np = np.array([AA in "X" for AA in alphabet]).astype(np.float32)
        bias_AAs_np = np.zeros(len(alphabet))
        
        with open(ali_file, 'w') as f:
            randn_2 = torch.randn(chain_M.shape, device=X.device)
            sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, 
                                       temperature=temperature, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, 
                                       chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, 
                                       pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=False, 
                                       pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=False, bias_by_res=bias_by_res_all)
            S_sample = sample_dict["S"]
            log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
            mask_for_loss = mask*chain_M*chain_M_pos
            scores = _scores(S_sample, log_probs, mask_for_loss)
            scores = scores.cpu().data.numpy()
            S_sample_list.append(S_sample.cpu().data.numpy())


            '''for b_ix in range(BATCH_SIZE):
                masked_chain_length_list = masked_chain_length_list_list[b_ix]
                masked_list = masked_list_list[b_ix]
                seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                start = 0
                end = 0
                list_of_AAs = []
                for mask_l in masked_chain_length_list:
                    end += mask_l
                    list_of_AAs.append(native_seq[start:end])
                    start = end
                native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                print(native_seq)'''
    

def test_mpnn():
    pdb_folder = "external/ProteinMPNN/inputs/PDB_complexes/pdbs/"
    output_dir = "test_mpnn_output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    path_for_parsed_chains = output_dir + "parsed_pdbs.jsonl"
    path_for_assigned_chains = output_dir + "assigned_pdbs.jsonl"
    path_for_fixed_positions = output_dir + "fixed_pdbs.jsonl"

    fixed_positions = "1 2 3 4 5 6 7 8 23 25, 10 11 12 13 14 15 16 17 18 19 20 40"

    os.remove(path_for_assigned_chains) if os.path.exists(path_for_assigned_chains) else None
    os.remove(path_for_parsed_chains) if os.path.exists(path_for_parsed_chains) else None
    os.remove(path_for_fixed_positions) if os.path.exists(path_for_fixed_positions) else None

    subprocess.run(["python", "external/ProteinMPNN/helper_scripts/parse_multiple_chains.py", 
                    f"--input_path={pdb_folder}", f"--output_path={path_for_parsed_chains}"])
    
    subprocess.run(["python", "external/ProteinMPNN/helper_scripts/assign_fixed_chains.py",
                    f"--input_path={path_for_parsed_chains}", f"--output_path={path_for_assigned_chains}",
                    f"--chain_list", "A C"])
    
    subprocess.run(["python", "external/ProteinMPNN/helper_scripts/make_fixed_positions_dict.py",
                    f"--input_path={path_for_parsed_chains}", f"--output_path={path_for_fixed_positions}",
                    f"--chain_list", "A C", f"--position_list", fixed_positions])
    
    run_mpnn(path_for_parsed_chains, ["A", "C"], 
             [[int(i) for i in subst.split()] for subst in fixed_positions.split(',')])
    
if __name__ == "__main__":
    test_mpnn()
'''
Parses program files into a usable format. See the README for more information on program files.
'''

if __name__ == "__main__": exit()   # Don't run this file

from .external.better_mpnn import ChainResidue, chain_residues_to_indices, handle_ranges, handle_fix_pattern
from .external.ProteinMPNN.protein_mpnn_utils import StructureDatasetPDB, parse_PDB

import re
import os

class DebugParams:
    def __init__(self):
        self.stopat = "PDB_Load"
        self.program_file = None

def parse_program(program_file_path, debugparams : DebugParams | None = None):
    #seq : dict[str, list[ChainResidue]], aminos = dict[str, dict[str, str]], alphabet = "ACDEFGHIKLMNPQRSTVWYX", program_file = None
    if debugparams is None or debugparams.program_file is None:
        with open(program_file_path, 'r') as f:
            program = f.read()
    else:
        program = debugparams.program_file
    program = program.split('\n')
    # Remove empty lines
    program = [line for line in program if line != '']
    # Remove comments
    program = [line for line in program if not line.startswith('#')]
    # Remove leading/trailing whitespace
    program = [line.strip() for line in program]
    
    # start all/start none
    if program[0] == 'start all':
        start_all = True
    elif program[0] == 'start none':
        start_all = False
    else:
        raise ValueError(f"First line of program must be `start all` or `start none`")

    # Get output pattern
    output_pattern = program[1].split()
    if output_pattern[0] != 'output':
        raise ValueError(f"Second line of program must be `output`")
    output_pattern = output_pattern[1:]
    assert len(output_pattern) >= 1, "Output pattern must have at least one element"
    chainlengths = {}
    for i, pattern in enumerate(output_pattern):
        # Patterns take the form `chain:length`
        chain, length = pattern.split(':')
        chainlengths[chain] = int(length)
    
    bias = []

    # Load pdbs
    program = program[2:]
    startidx = 0
    
    seq : dict[str, list[ChainResidue]] = {}
    aminos : dict[str, dict[str, str]] = {}
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    database = []
    # Maps PDBs to a mapping of chain -> output chain
    mapping : dict[str, dict[str, str]] = {}

    for line in program:
        tokens = line.split()
        if tokens[0] != 'structure':
            break
        startidx += 1
        structstring = ' '.join(tokens[1:])
        load, pattern = structstring.split(">")
        pdb, location = load.split(":")
        location = location.strip()
        pdb = pdb.strip()
        pattern = pattern.strip()

        # Load PDB
        # Set pdb location to be relative to program file
        location = os.path.join(os.path.dirname(program_file_path), location)
        assert os.path.exists(location), f"File {location} does not exist"
        aminodict = load_pdb(pdb, location, seq, aminos, database, alphabet)

        # Now, create PDB mappings
        # Get keys sorted alphabetically from aminodict
        keys = sorted(aminodict.keys())
        pattern = pattern.split()
        patternidx = 0
        thismapping = {}
        for key in keys:
            # Check if the key's chain is a protein chain. We do this by checking if at least one amino acid is in the alphabet
            if not any([amino in alphabet for amino in aminodict[key]]):
                continue
            # Get the mapping
            dest = pattern[patternidx]
            patternidx += 1
            if dest == "_":
                continue
            # Check to make sure mapping is valid
            assert dest in chainlengths, f"Chain {dest} not found in output pattern"
            assert len(aminodict[key]) == chainlengths[dest], f"Chain {key} has length {len(aminodict[key])} but output chain {dest} has length {chainlengths[dest]}"
            thismapping[key] = dest
        mapping[pdb] = thismapping

    if debugparams and debugparams.stopat == "PDB_Load":
        return
    
    # Parse the rest of the program
    program = program[startidx:]

    if start_all:
        # Add chainresidues to fixed_positions
        fixed_positions : dict[str, set[ChainResidue]] = {}
        for i, protein in enumerate(seq):
            fixed_positions[protein] = set(seq[protein])
    else:
        fixed_positions : dict[str, set[ChainResidue]] = {}
        for i, protein in enumerate(seq):
            fixed_positions[protein] = set()
    
    # Parse program
    for line in program:
        # Split line into tokens
        tokens = line.split()

        # include mode is first line
        mode = tokens[0] # fix/unfix/bias
        # type is second line
        type = tokens[1]

        # Get remainder of line
        remainder = ' '.join(tokens[2:])

        # Split by colon and get pdb and chain
        tokens = remainder.split(':')
        pdb = tokens[0]
        chain = tokens[1]

        # Get extra
        extra = ':'.join(tokens[2:])
        parse_line(mode, type, pdb, chain, extra, seq, fixed_positions, bias, aminos, mapping, alphabet)
    
    # Error checking! Make sure the same output residues are fixed in all proteins
    # Get residues from the first protein
    fixed_residues : list[ChainResidue] = []
    output : dict[str, str] = {k : list("-" * v) for k,v in chainlengths.items()}    # Output sequences
    # Convert using mapping
    for res in fixed_positions[list(fixed_positions.keys())[0]]:
        # If the chain isn't in the mapping, ignore it
        if res.chain not in mapping[list(fixed_positions.keys())[0]]:
            continue
        output_chain = mapping[list(fixed_positions.keys())[0]][res.chain]
        fixed_residues.append(ChainResidue(output_chain, res.residue))
        output[output_chain][res.residue] = aminos[list(fixed_positions.keys())[0]][res.chain][res.residue]

    # Now check that all proteins have the same fixed residues    
    for protein in fixed_positions:
        # If it's not mapped to an output, it must be fixed
        if protein not in mapping:
            for res in seq[protein]:
                assert res in fixed_residues, f"Residue {res} in protein {protein} is not found in all proteins, so it must be fixed"
        else:
            # Get the output chain
            output_mapping = mapping[protein]
            # Make sure all residues that are fixed in the output chain are fixed in the input chain, and all that aren't aren't
            for res in seq[protein]:
                if res in fixed_positions[protein]:
                    output_chain = output_mapping[res.chain]
                    assert ChainResidue(output_chain, res.residue) in fixed_residues, f"Residue {res} in protein {protein} is fixed in the output chain {output_chain} but not in all proteins"
                    # Make sure it's the same residue
                    assert aminos[protein][res.chain][res.residue] == output[output_chain][res.residue], f"Residue {res} in protein {protein} is fixed in the output chain {output_chain} but is a different amino acid"
                else:
                    assert ChainResidue(output_chain, res.residue) not in fixed_residues, f"Residue {res} in protein {protein} is not fixed in the output chain {output_chain} but is fixed in other proteins"

    # All done!
    return seq, aminos, database, fixed_positions, fixed_residues, bias, mapping, chainlengths, output
        
def load_pdb(pdb : str, location : str, seq : dict[str, list[ChainResidue]], aminos : dict[str, dict[str, str]], 
             database : list[dict], alphabet = "ACDEFGHIKLMNPQRSTVWYX"):
    # Load PDB
    pdb_dict_list = parse_PDB(location, ca_only=False)
    dataset = StructureDatasetPDB(pdb_dict_list, max_length=20000, alphabet=alphabet)

    # Add to database
    assert len(dataset) == 1, "Only one protein should be loaded at a time"
    database.append(dataset[0])

    # Get chains
    chains = set()
    for key in dataset[0].keys():
        if '_chain_' in key:
            chains.add(key.split('_')[-1])
    # Construct sequence dictionary
    aminodict = {}
    for chain in chains:
        aminodict[chain] = dataset[0][f'seq_chain_{chain}']
    aminos[pdb] = aminodict
    
    # Now construct seq list
    seqs = []
    for chain in chains:
        for i in range(len(aminodict[chain])):
            seqs.append(ChainResidue(chain, i))
    seq[pdb] = seqs
    return aminodict

def parse_line(mode : str, type : str, pdb : str, chain: str, extra : str, 
               seq : dict[str, list[ChainResidue]], fixed_positions : dict[str, set[ChainResidue]], bias : list[dict[str, list[list[int]]]],
               aminos : dict[str, dict[str, str]], mapping : dict[str, dict[str, str]], chainlengths : dict[str, int], 
               alphabet = 'ACDEFGHIKLMNPQRSTVWYX'):
    if mode == 'fix':
        # Get chainresidues that match the type, pdb, chain, and extra
        matching_chainresidues = match(type, pdb, chain, extra, seq, aminos, alphabet)
        # Add chainresidues to fixed_positions
        for protein in matching_chainresidues:
            fixed_positions[protein].update(matching_chainresidues[protein])
    elif mode == 'unfix':
        # Get chainresidues that match the type, pdb, chain, and extra
        matching_chainresidues = match(type, pdb, chain, extra, seq, aminos, alphabet)
        # Remove chainresidues from fixed_positions
        for protein in matching_chainresidues:
            fixed_positions[protein] = fixed_positions[protein] - matching_chainresidues[protein]
    elif mode == 'bias':
        # Exclude the residues and bias from the `extra` string
        biastokens = extra.split()
        biasval = float(biastokens[-1])
        biasaminos = biastokens[-2]
        biasaminos = aminos.split(',')
        extra = ' '.join(biastokens[:-2])
        # Get chainresidues that match the type, pdb, chain, and extra
        matching_chainresidues = match(type, pdb, chain, extra, seq, aminos, alphabet)

        # Bias should have shape (num_residues, num_aminos)
        # Create a 1-hot bias vector for each residue
        biasdict = {}
        for chain in chainlengths:
            biasdict[chain] = [[biasval for amino in alphabet] for _ in range(chainlengths[protein])]
        # Now set matching residues to the bias value for the specified amino acids
        for protein in matching_chainresidues:
            for res in matching_chainresidues[protein]:
                # Get the output chain
                output_chain = mapping[protein][res.chain]
                for amino in biasaminos:
                    biasdict[output_chain][res.residue][alphabet.index(amino)] = 0
        bias.append(biasdict)
    else:
        raise ValueError(f"Mode {mode} not recognized")
    
def match(type : str, pdb : str, chain : str, extra : str, seq : dict[str, list[ChainResidue]], 
          aminos : dict[str, dict[str, str]], alphabet = "ACDEFGHIKLMNPQRSTVWYX") -> dict[str, set[ChainResidue]]:
    # aaaaa I should probably use a dictionary for this but not yet
    # NOTE: ChainResidues are 0-indexed
    output = {}
    for protein in seq:
        output[protein] = set()
    if type == "res":
        if pdb == "":
            # Apply to all chains
            resids = extra.split(',')
            for protein in seq:
                chainresidues = []
                # Get chainresidues that match the resids
                for res in seq[protein]:
                    if chain == "" or chain == res.chain:
                        # Resids obtained from program files are 1-indexed so we have to convert
                        if str(res.residue + 1) in resids:
                            chainresidues.append(res)
                output[protein] = set(chainresidues)
        else:
            # Apply to specific protein
            resids = extra.split(',')
            chainresidues = []
            # Get chainresidues that match the resids
            for res in seq[pdb]:
                if chain == "" or chain == res.chain:
                    if res.residue in resids:
                        chainresidues.append(res)
            output[pdb] = set(chainresidues)
    elif type == "range":
        # Get start and end
        start, end = extra.split('-')
        start = int(start)
        end = int(end)
        if pdb == "":
            # Apply to all chains
            for protein in seq:
                chainresidues = []
                # Get chainresidues that match the range
                for res in seq[protein]:
                    if chain == "" or chain == res.chain:
                        # Have to add 1 to resid because it's 0-indexed
                        if res.residue + 1 >= start and res.residue + 1 <= end:
                            chainresidues.append(res)
                output[protein] = set(chainresidues)
        else:
            # Apply to specific protein
            chainresidues = []
            # Get chainresidues that match the range
            for res in seq[pdb]:
                if chain == "" or chain == res.chain:
                    if res.residue >= start and res.residue <= end:
                        chainresidues.append(res - 1)
            output[pdb] = set(chainresidues)
    elif type == "pattern":
        # Get pattern
        pattern = extra
        pattern = re.compile(pattern)
        if pdb == "":
            # Apply to all proteins
            for protein in seq:
                # Get chainresidues that match the pattern
                if chain == "":
                    # Check all chains
                    chainresidues = []
                    for pchain in aminos[protein]:
                        # Check matching residues
                        matches = re.finditer(pattern, aminos[protein][pchain])
                        for match in matches:
                            start = match.start()
                            end = match.end()
                            for res in seq[protein]:
                                if res.chain == pchain and res.residue >= start and res.residue < end:
                                    chainresidues.append(ChainResidue(res.chain, res.residue))
                    output[protein] = set(chainresidues)
                elif chain in aminos[protein]:
                    chainresidues = []
                    # Check matching residues
                    matches = re.findall(pattern, aminos[protein][chain])
                    for match in matches:
                        start = match.start()
                        end = match.end()
                        for res in seq[protein]:
                            if res.chain == chain and res.residue >= start and res.residue < end:
                                chainresidues.append(res)
                    output[protein] = set(chainresidues)
                else:
                    raise ValueError(f"Chain {chain} not found in protein")
        else:
            if chain == "":
                # Check all chains
                for chain in aminos[pdb]:
                    chainresidues = []
                    # Check matching residues
                    matches = re.finditer(pattern, aminos[pdb][chain])
                    for match in matches:
                        start = match.start()
                        end = match.end()
                        for res in seq[pdb]:
                            if res.chain == chain and res.residue >= start and res.residue < end:
                                chainresidues.append(res)
                    output[pdb] = set(chainresidues)
            elif chain in aminos[pdb]:
                chainresidues = []
                # Check matching residues
                matches = re.findall(pattern, aminos[pdb][chain])
                for match in matches:
                    start = match.start()
                    end = match.end()
                    for res in seq[pdb]:
                        if res.chain == chain and res.residue >= start and res.residue < end:
                            chainresidues.append(res)
                output[pdb] = set(chainresidues)
            else:
                raise ValueError(f"Chain {chain} not found in protein")
    else:
        raise ValueError(f"Type {type} not recognized")
    return output

####################################################################################################
# Testing
####################################################################################################

def test():
    # Test on two sequences GAAGG and GAGG
    output = match("pattern", "", "", r"A{2,}", 
                   {"CAM": [ChainResidue("A", 0), ChainResidue("A", 1), ChainResidue("A", 2), ChainResidue("A", 3), ChainResidue("A", 4),
                            ChainResidue("B", 0), ChainResidue("B", 1), ChainResidue("B", 2), ChainResidue("B", 3)]},
                     {"CAM": {"A": "GAAGG", "B": "GAGG"}}, 20)
    assert output.keys() == {"CAM"}
    assert len(output["CAM"]) == 2
    assert ChainResidue("A", 1) in output["CAM"]
    assert ChainResidue("A", 2) in output["CAM"]

    # Test on the same sequences, but split across CAM1 and CAM2 with only CAM2 allowed as PDB
    output = match("pattern", "CAM2", "", r"A{1,}", 
                   {"CAM1": [ChainResidue("A", 0), ChainResidue("A", 1), ChainResidue("A", 2), ChainResidue("A", 3), ChainResidue("A", 4)],
                    "CAM2": [ChainResidue("B", 0), ChainResidue("B", 1), ChainResidue("B", 2), ChainResidue("B", 3)]},
                     {"CAM1": {"A": "GAAGG"}, "CAM2": {"B": "GAGG"}}, 20)
    assert output.keys() == {"CAM1", "CAM2"}
    assert len(output["CAM2"]) == 1
    assert len(output["CAM1"]) == 0
    assert ChainResidue("B", 1) in output["CAM2"]
    print("All match tests passed")


    # Test res
    output = match("res", "", "", "1,3", 
                   {"CAM": [ChainResidue("A", 0), ChainResidue("A", 1), ChainResidue("A", 2), ChainResidue("A", 3), ChainResidue("A", 4),
                            ChainResidue("B", 0), ChainResidue("B", 1), ChainResidue("B", 2), ChainResidue("B", 3)]},
                     {"CAM": {"A": "GAAGG", "B": "GAGG"}}, 20)
    assert output.keys() == {"CAM"}
    assert len(output["CAM"]) == 4
    assert ChainResidue("A", 0) in output["CAM"]
    assert ChainResidue("A", 2) in output["CAM"]
    assert ChainResidue("B", 0) in output["CAM"]
    assert ChainResidue("B", 2) in output["CAM"]
    print("All res tests passed")

    # Test range
    output = match("range", "", "", "1-3", 
                   {"CAM": [ChainResidue("A", 0), ChainResidue("A", 1), ChainResidue("A", 2), ChainResidue("A", 3), ChainResidue("A", 4),
                            ChainResidue("B", 0), ChainResidue("B", 1), ChainResidue("B", 2), ChainResidue("B", 3)]},
                     {"CAM": {"A": "GAAGG", "B": "GAGG"}}, 20)
    assert output.keys() == {"CAM"}
    assert len(output["CAM"]) == 6
    assert ChainResidue("A", 0) in output["CAM"]
    assert ChainResidue("A", 1) in output["CAM"]
    assert ChainResidue("A", 2) in output["CAM"]
    assert ChainResidue("B", 0) in output["CAM"]
    assert ChainResidue("B", 1) in output["CAM"]
    assert ChainResidue("B", 2) in output["CAM"]
    print("All range tests passed")

    # Try running program on 3CWQ ParA test
    seq, aminos, database, fixed_positions, fixed_residues, bias, mapping, chainlengths = parse_program("tests/3CWQ/3CWQ.mst")
    assert len(seq) == 2
    assert len(aminos) == 2
    assert "ParA_ADP" in aminos and "ParA_ATP" in aminos
    assert len(database) == 2

    print("Program parsing tests passed")


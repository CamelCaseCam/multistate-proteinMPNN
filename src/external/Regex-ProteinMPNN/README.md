# Regex ProteinMPNN
A wrapper around proteinMPNN to automate sequence generation and allow specifying fixed residues using regex (and ranges). See the included shell scripts for examples of how to use it. 

## Installation and Usage
1. Clone [proteinMPNN](https://github.com/dauparas/ProteinMPNN) and follow their installation instructions. 
2. Clone this repository.
3. Activate your conda environment associated with proteinMPNN. 
4. Run `better_mpnn.py` with the desired parameters.

The wrapper has the following usage:

```
usage: better_mpnn.py [-h] [--input_pattern INPUT_PATTERN] [--design_chains DESIGN_CHAINS] [--fix_pattern FIX_PATTERN] [--fix_ranges FIX_RANGES] [--num_outputs NUM_OUTPUTS]
                      [--sampling_temperature SAMPLING_TEMPERATURE] [--mpnn_path MPNN_PATH]
                      input_dir output_dir temp_dir

Run proteinMPNN on a set of input files

positional arguments:
  input_dir             The directory containing the input files
  output_dir            The directory to save the output files
  temp_dir              The directory to save temporary files

options:
  -h, --help            show this help message and exit
  --input_pattern INPUT_PATTERN
                        The pattern to match for input files
  --design_chains DESIGN_CHAINS
                        The chains to design
  --fix_pattern FIX_PATTERN
                        The pattern to match for fixed residues
  --fix_ranges FIX_RANGES
                        Ranges of residues to fix in the format "chain:res1-res2,chain:res3-res4"
  --num_outputs NUM_OUTPUTS
                        The number of outputs per input file to generate
  --sampling_temperature SAMPLING_TEMPERATURE
                        The "predictability" of the model
  --mpnn_path MPNN_PATH
                        The path to the proteinMPNN directory
```

Running the script without any arguments will test some of the code, which may be useful if you want to extend this wrapper. 

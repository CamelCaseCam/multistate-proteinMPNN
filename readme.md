# Multistate: ProteinMPNN structure generation for proteins with multiple states
## How it works
1. You submit a "pattern" to determine the generated residues in each file and the files loaded. 
2. ProteinMPNN gradually converges to a set of sequences that *should* fold into all structures. 
3. Generated sequences are returned. 

## The patterns
See [patterns.md](patterns.md)

## Inference
### V1
Version one takes in the following arguments:
```
usage: multistate_v1.py [-h] [--generations GENERATIONS] [--seqs SEQS] input_file output_path

positional arguments:
  input_file            Input file pattern
  output_path           Output path (must exist)

options:
  -h, --help            show this help message and exit
  --generations GENERATIONS
                        Maximum number of generations to run
  --seqs SEQS           Number of sequences to generate
```

The default number of generations is 100, and the default number of sequences is 10. The program uses ProteinMPNN to generate log probabilities of amino acids for each structure, and then adds them to combine probabilities. These log probabilities are then converted to regular probabilities and an additional probability is added (adding to 1.0) to act as the probability of not choosing an amino acid for that generation. At each generation, these probabilities are sampled until all amino acids are chosen or the maximum number of generations is reached. 

## Installation

First, execute the following commands:
```bash
conda create --name pmpnn
conda activate pmpnn
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Then, write a pattern file for your desired output. Follow the format shown in [3CWQ.mst](tests/3CWQ/3CWQ.mst). Pass the pattern file and desired output location to `multistate_v1.py`. 

## Caveats
I made this in like three days. What you see is what you get. I've tested everything in [my notes](testing_notes.md) but the bias functionality is completely untested and there are definitely bugs I'm missing here. If you want to help, the following would be helpful:

 - Adding tests (especially with the bias functionality)
 - Making contributions to improve the core algorithm, since it tends to get "stuck" before the final generation and I suspect this leads to suboptimal proteins being generated

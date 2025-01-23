Patterns are defined using a small domain-specific language. The domain-specific language can be defined as follows:

1. Lines that consist of only whitespace or start with "#" are ignored
2. After this, the first line is either `start all` or `start none`. `start all` will start with all amino acids in residues fixed, and `start none` starts with all amino acids unfixed. 
3. The next line defines the output. This consists of `output` followed by a set of `chain:length` pairs. The `chain` is traditionally a single character in alphabetical order (ex. `output A:10 B:15`). This defines the length of all chains in the final output structure. Note that this length **includes** fixed residues from the original sequence
4. After the `output` line, the `schedule` line defines the temperature and no_change probability for each generation. The format of this line is `schedule [python statement]`. Note that the schedule will be set to the results of `[python statement]` so patterns should be treated as code and should be inspected before being used. 
5. After the `schedule` line, the `tie` line defines tied residues. The line takes the form `tie ch1.start-stop:ch2.start-stop:ch3.start-stop [ch1.start-stop:ch2.start-stop:ch3.start-stop ...]`
6. The next few lines define the structures that the output sequence should take on. These lines take the form `structure name:path > chains`. The `path` should be relative to the pattern file. The `chains` determine how chains within the PDB file are mapped to chains in the output sequence. All non-protein chains are ignored in this mapping, so if PDB XYZ had ions in chain A, a protein in chain B, a membrane in chain C, and another protein in chain D, the line `structure XYZ:XYZ.pdb > A _` would load chain B as chain A in the output, and would not map chain D to any output (which requires that all of chain D is fixed)
7. After the final `structure` line, there are three types of lines that are used to define fixed residues and bias towards specific amino acids. Lines beginning with `fix` will fix residues, and lines beginning with `unfix` unfix residues. Lines beginning with `bias` define a bias for particlular amino acids at certain positions **in the input PDBs**. Bias positions are then mapped to output positions and combined across all structures, so only one input structure needs to have biases specified
8. `fix/unfix` lines take the form `fix/unfix match`. `bias` lines take the form `bias match A,R,N 0.5`. The second-last word is the list of amino acids and the last word is the bias scale. When the model is run, the bias will be included as another prediction of `bias scale * mean(predictions at index)` for amino acids not included in the bias list, and `mean(predictions at index) / bias scale` for amino acids included in the list. 
9. A `match` can match residues by index (`res`), range (`range`), or RegEx pattern (`pattern`). A `match` has the syntax `type structure:chain:[resid/range/pattern]`. `structure` or `chain` can be unspecified. Match ranges are inclusive at both ends. 

The following are examples of `fix` lines using different matches:
```
fix res ::1,2,3
fix range XYZ::3-9
fix pattern :A:G{2,}.N
```

And here is an example of a file for a real binder design task that I was working on:
```
start all
output A:233 B:233 C:233 D:233 E:233 F:233 G:40

schedule [(1.5, 0.1)] + [(0.1, 0.1)] * 1000

structure gen2_input:gen2_input.pdb > A B C D E F G

unfix res :G:3,5,27,29,31,32,34,36,38
```

Note that you might find that very long sequences take a long time to generate. This is because probabilities are regenerated each time an amino acid is selected by default. You can speed up generation at the cost of sequence quality by changing `use_coupling` to false on line 52 of `multistate_v1.py`
# QAlg_exciton
Code to map simple Quantum Algorithms into molecular circuits. The general methodology is outlined in 
*Designing excitonic circuits for the Deutsch–Jozsa algorithm: mitigating fidelity loss by merging gate operations*. **Phys. Chem. Chem. Phys.**, 2021,**23**, 15196-15208

The repository contains the following files:
- ga.py is an example code. It runs a Genetic algorithm code to find a spatial arrangement of Cy3-Cy5 4-dye circuit more closely resembling an specific Hamiltonian.
The code included work for mapping H_1 in Table 1 from the reference paper. 
- ga_utils.py is the Genetic Algorithm code. Different Selection, Crossover and Mutation algorithms are implemented (details in the code file). Takes a fidelity function as input,
which can be changed for general optimization problems.
- geom.py contains functions for processing the dimer geometries from calculated spatial parameters (from the GA optimization). 
The Geom instance takes as input the molecule data in the *Q_data* folder. To use the code on another set of molecules, the molecular data must be provided. 

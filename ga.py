#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:37:39 2020

@author: mariacm
"""

# %%
import numpy as np
import math
from ga_utils import GA
from geom import Geom

rho_init = np.diag([0,1,0,0])


# Cy3 and Cy5 files
# **Make sure the code and the file folders are in the same directory**
filepdb1 = "pdbs/RESP_Cy3.pdb"
filepdb2 = "pdbs/RESP_Cy5.pdb"

cm1toHartree = 4.5563e-6
Cy3E = np.genfromtxt('Qdata/Cy3_Tenergy_RESP.txt')/cm1toHartree
Cy5E = np.genfromtxt('Qdata/Cy5_Tenergy_RESP.txt')/cm1toHartree
Cy3TDM = np.genfromtxt('Qdata/Cy3_tdm_RESP.txt')[0]
Cy5TDM = np.genfromtxt('Qdata/Cy5_tdm_RESP.txt')[0]

chrgAl = np.genfromtxt('Qdata/Cy3_Pcharges.txt')
chrgBl = np.genfromtxt('Qdata/Cy5_Pcharges.txt')

# Geom instance
geom = Geom(filepdb1, filepdb2,
             pcharge1=chrgAl, pcharge2=chrgBl,
             TEnergy1=Cy3E, TEnergy2=Cy5E,
             tdm1=Cy3TDM, tdm2=Cy5TDM)

# =============================================================================
# ##Defining run parameters
# =============================================================================
# To find the "intersecting interval", d: cma+rmin = xb, cmb-rmin=xa & d=xb-xa
# [where: cma=min(cm0,cm1)&cmb=max(cm0,cm1)]
cm0, cm1 = 0.5, 0.5 
d = geom.max_rij() * 1/3

num_gen = 10 # number of generations
chroms = 20  # Number of chromosomes (size of the population)
num_parents = 6  # Number of solutions to be selected as parents
gene_types = 2  # number of different types of genes
num_genes1 = 9  # genes of first group
num_genes2 = 9  # genes of second group
keep_parents = 2  # Number of parents to keep in the next population.

# gene-pool interval
# There's 2 types of genes: Coordinates and miu angles
var_min = [d / 10, -math.pi]
var_max = [d, math.pi] 

# mutation parameters
mutation_type = "random_reset"  # types of mutation: "random_reset","swap","scramble,"inversion"
mutation_ratio = 0.5  # Percentage of genes to mutate.
# Variable ranges can be shrinked with subsequent runs for more precision
mut_min = np.array([-d / 18, -math.pi / 8])  # np.array([-d/18,-math.pi/40])
mut_max = np.array([d / 18, math.pi / 8])  # np.array([ d/18, math.pi/40])
mutation_dic = {"mutation_ratio": mutation_ratio, 'min_mutation': mut_min, 'max_mutation': mut_max}

selection_type = "tournament"  # Types: "roulette_wheel". "steady_state", "stochastic", "tournament"
crossover_type = "uniform"  # "single_point","two_point","uniform"

which = "HH"  # Which circuit to optimize

#reinitializing from prev solution
prev_solution = None #np.genfromtxt('/path/to/previous/solution/best_sol.txt') 

def fitness(pop):
    """
     Fitness function

     Parameters
     ----------
     pop : ndarray
         Population

     Returns
     -------
     fitness : float
         Calculated fitness.

     """
    ham_sys, fit, tau = geom.ham_fitness(pop, which)
    
    if  tau == 99999:
        #true when the molecules are too close to each other
        fitness_i = 0.0001
        #print('bad', pop)
    else:
        fitness_i = fit
    return fitness_i



# =============================================================================
# Running the Genetic Algorithm
# =============================================================================

# GA instance
Gen_alg = GA(fitness, num_generations=num_gen,
                      num_chromosomes=chroms,
                      gene_groups=gene_types,
                      num_genes=[num_genes1,num_genes2],
                      gene_min=var_min,
                      gene_max=var_max,
                      selection_type=selection_type,
                      num_parents=num_parents,
                      keep_parents=keep_parents,
                      crossover_type=crossover_type,
                      mutation_type=mutation_type,
                      mutation_ratio=mutation_ratio,
                      mutation_range=[mut_min,mut_max],
                      prev_result=prev_solution,
                      alt_sign=[False,True],
                      k_tournament=6,
                      filename="ga")

# Running the GA to optimize the parameters of the function.
Gen_alg.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
Gen_alg.plot_result(param_idx=1)

# Returning the details of the best solution.
best_solution, best_solution_fitness = Gen_alg.get_result()
print("Parameters of the best solution :", best_solution)
print("Fitness value of the best solution :", best_solution_fitness, "\n")

# Saving & loading the final GA.]
Gen_alg.save()
np.savetxt("best_sol.txt", best_solution) # Saving best solution for reinitialization if needed
loaded_Gen_alg = Gen_alg.load()
best_load = loaded_Gen_alg.get_result()
loaded_Gen_alg.plot_result()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:37:39 2020

@author: mariacm
"""

# So numpy doesn't annoy me
'''
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
'''
import os
os.chdir('/Volumes/Extreme_SSD/PhD_project/Code backup/DJ_paper/')
# %%
import numpy as np
import math
import gates
import prop_utils as pu
import ga_utils
import geom

rho_init = np.diag([0,1,0,0])

# =============================================================================
# ##Defining run parameters
# =============================================================================
# To find the "intersecting interval", d: cma+rmin = xb, cmb-rmin=xa & d=xb-xa
# [where: cma=min(cm0,cm1)&cmb=max(cm0,cm1)]
cm0, cm1 = 0.5, 0.5 #geom.cmass_A, geom.cmass_B
d = geom.max_rij(gates.cy3.TDM[0],gates.cy5.TDM[0]) * 1/3
# 3/4 #np.min(np.array([cm0[0],cm1[0]])) - np.max(np.array([cm0[0],cm1[0]])) + 2*geom.rmax

num_gen = 5 # number of generations
chroms = 10  # Number of chromosomes (size of the population)
num_parents = 4  # Number of solutions to be selected as parents
gene_types = 2  # number of different types of genes
num_genes1 = 9  # genes of first group
num_genes2 = 9  # genes of second group
keep_parents = 2  # Number of parents to keep in the next population.

# gene-pool interval
# There's 2 types of genes: Coordinates and miu angles
var_min = [d / 10, -math.pi]
var_max = [d, math.pi]  # np.array([d/2,math.pi])

# mutation parameters
mutation_type = "random_reset"  # types of mutation: "random_reset","swap","scramble,"inversion"
mutation_ratio = 0.5  # Percentage of genes to mutate.
mut_min = np.array([-d / 18, -math.pi / 8])  # np.array([-d/18,-math.pi/40])
mut_max = np.array([d / 18, math.pi / 8])  # np.array([ d/18, math.pi/40])
mutation_dic = {"mutation_ratio": mutation_ratio, 'min_mutation': mut_min, 'max_mutation': mut_max}

selection_type = "tournament"  # Types: "roulette_wheel". "steady_state", "stochastic", "tournament"
crossover_type = "uniform"  # "single_point","two_point","uniform"

fixed_var = (gates.cy5.TDM, gates.cy5.TEnergy, cm0, cm1)  # Second dye tdm&Tenergy, center of masses for first 2 dyes
which = "HH"  # Which circuit to optimize


prev_solution = np.genfromtxt('/Users/mariacm/Desktop/dna_orig_Q_comp/attempt_2/Q_algorithm/Open_system/GA_sol_HH_ham.txt') #reinitializing from prev solution

# =============================================================================
# Running the Genetic Algorithm
# =============================================================================
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
    ham_sys, fit, tau = pu.ham_fitness(pop, which)
    
    if  tau == 99999:
        #true when the molecules are too close to each other
        fitness_i = 0.0001
        #print('bad', pop)
    else:
        fitness_i = fit
    return fitness_i


# %%
# GA instance

Gen_alg = ga_utils.GA(fitness, num_generations=num_gen,
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
                      prev_result=None,
                      alt_sign=[False,True],
                      k_tournament=6)

# Running the GA to optimize the parameters of the function.
Gen_alg.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
Gen_alg.plot_result(param_idx=1)

# Returning the details of the best solution.
best_solution, best_solution_fitness = Gen_alg.get_result()
print("Parameters of the best solution :", best_solution)
print("Fitness value of the best solution :", best_solution_fitness, "\n")

#%%
# Saving the GA instance.
filename = 'genetic'  # The filename to which the instance is saved. The name is without extension.
Gen_alg.save(filename=filename)

# saving the optimized parameters
#np.savetxt("/Users/mariacm/Desktop/dna_orig_Q_comp/attempt_2/Q_algorithm/Open_system/GA_sol_HH_ham.txt", best_solution)

# Loading the saved GA instance.
# loaded_Gen_alg = GA.pygad.load(filename=filename)

# loaded_Gen_alg.plot_result()
# best_load = loaded_Gen_alg.best_solution()
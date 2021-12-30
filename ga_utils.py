#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 01:06:40 2020

@author: mariacm

Genetic algorithm code
Based on: 
    [**Ahmed Fawzy Gad 'Practical Computer Vision Applications Using Deep Learning with CNNs'. Dec. 2018, Apress, 978-1-4842-4167-7**]
    (https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665).


"""

import numpy as np
import random
import math
import pickle
import pandas as pd
import string


class GA:
    def __init__(self, fitness_fn,
                 num_generations=1,
                 num_chromosomes=1,
                 gene_groups=1,
                 num_genes=1,
                 gene_min=-1,
                 gene_max=1,
                 selection_type='',
                 num_parents=4,
                 keep_parents=-1,
                 crossover_type='',
                 mutation_type='',
                 mutation_ratio=0.5,
                 mutation_range=[],
                 prev_result=None,
                 alt_sign=False,
                 k_tournament=None):

        self.fitness_func = fitness_fn
        self.num_generations = num_generations
        self.gene_groups = gene_groups
        self.num_chroms = num_chromosomes

        # Initialize populations
        self.gene_groups = gene_groups
        self.num_genes = num_genes
        
        self.prev_result = prev_result
        
        # Making sure variables are input correctly
        if gene_groups > 1:
            if not isinstance(num_genes,list):
                raise ValueError(("If one more Gene Group is desired, the size for each group must be specified as a list of size gene_groups."+
                                 "\nA list wasn't provided for num_genes."))
            elif len(num_genes) != gene_groups:
                raise ValueError(("If one more Gene Group is desired, the size for each group must be specified as a list of size gene_groups."+
                                 "\nThe size of num_genes doesn't correspond to gene_groups."))
            if not isinstance(alt_sign,list):
                alt_sign = [alt_sign] * gene_groups
                
            if not isinstance(gene_min,list) or not isinstance(gene_max,list):
                raise ValueError(("If one more Gene Group is desired, the min and max limits should be provided as lists of size gene_groups."+
                                 "\nEither limits aren't of type list."))
            elif len(gene_min) != gene_groups or len(gene_max) != gene_groups:
                raise ValueError("If one more Gene Group is desired, the min and max limits should be provided as lists of size gene_groups."+
                                 "\nOne of the lists (or both) isn't of length gene_groups.")                

        self.cols = list(string.ascii_uppercase)[:self.gene_groups]
        
        if self.prev_result is None:
            pop = self.initialize_population(gene_min, gene_max, alt_sign)
        else:
            pop = self.reinitialize_population(gene_min, gene_max, alt_sign)

        self.populations = pop
        
        # Initialize selection
        self.sel_type = selection_type
        self.num_parents = num_parents
        self.k_tournament = k_tournament
        self.Sel = Selection(self.num_parents, self.k_tournament)

        # Initialize crossover
        self.cross_type = crossover_type
        self.Cross = Crossover()

        # Initialize mutation
        self.mut_type = mutation_type
        self.mut_ratio = mutation_ratio
        self.mut_min, self.mut_max = mutation_range[0], mutation_range[1]
        self.Mut = Mutation(self.mut_ratio)

        self.keep_parents = keep_parents
        if self.num_chroms < self.keep_parents:
            raise ValueError("THe number of parents to keep (keep_parents) cannot be larger than the number of chromosomes.") 

        # From original code
        if (self.keep_parents == -1):  # Keep all parents in the next population.
            self.num_offspring = self.num_chroms - self.num_parents
            # self.num_offspring = np.array([self.num_chroms] * self.gene_groups) - self.num_parents
        elif (self.keep_parents == 0) or (self.num_chroms == self.keep_parents):  # Keep no parents in the next population.
            # self.num_offspring = np.array([self.num_chroms] * self.gene_groups)
            self.num_offspring = self.num_chroms
        elif (self.keep_parents > 0):  # Keep the specified number of parents in the next population.
            # self.num_offspring = np.array([self.num_chroms] * self.gene_groups) - self.keep_parents
            self.num_offspring = self.num_chroms - self.keep_parents

        self.best_solution = []
        self.conv = False
        ###


    def initialize_population(self, low, high, alt_sign):
        """
        Creates an initial population randomly as a NumPy array.
        """
        pop = np.empty((self.num_chroms,0))
        col = []
        if self.gene_groups > 1:
            for i in range(self.gene_groups):
                size_i = (self.num_chroms, self.num_genes[i])
                pop_i = np.random.uniform(low=low[i], high=high[i], size=size_i)

                # to randomly switch the sign of some individuals in the population
                # (useful for angles to increase variation w/o including the entire range)     
                if alt_sign[i]: 
                    pop_i *= np.random.choice([-1,1], size_i, p=[0.5,0.5])   
                pop = np.hstack((pop, pop_i))
                col += (list(self.cols[i]*self.num_genes[i]))
                
        else:
            pop_size = (self.num_chroms, self.num_genes)
            pop = np.random.uniform(low=low, high=high, size=pop_size)
            if alt_sign: 
                pop *= np.random.choice([-1,1], size_i, p=[0.5,0.5]) 
        pops = pd.DataFrame(pop,columns=col)
        return pops

    def reinitialize_population(self, low, high, alt_sign):
        """
        Initializes the population from a result from a previous run. 
        Chromosome 1 corresponds to stored result, the rest are initialize randomly.
        """
        pop_size = []
        pop = []
        
        # separation previous result (as 1D array) by gene type
        if self.gene_groups > 1:
            prev = [0]*self.gene_groups
            idx_0 = 0

            for i in range(self.gene_groups):
                # get prev slice corresponding to current gene group
                idx_i = self.num_genes[i]
                prev[i] = self.prev_result[idx_0 : idx_i+idx_0].reshape(1, idx_i)
                idx_0 += idx_i
                
                # fill remaining chromosomes with random values
                size_rdm = (self.num_chroms - 1, self.num_genes[i])
                rdm_chrom = np.random.uniform(low=low[i], high=high[i], size=size_rdm)

                if alt_sign[i]: 
                    rdm_chrom *= np.random.choice([-1,1], size_rdm, p=[0.5,0.5])
                    
                pop_i = np.concatenate((prev[i], rdm_chrom), axis=0)
                pop.append(pop_i)
                pop_size.append((self.num_chroms, self.num_genes[i]))
                
        else:
            size_rdm = (self.num_chroms - 1, self.num_genes[i])
            rdm_chrom = np.random.uniform(low=low[i], high=high[i], size=size_rdm)
            if alt_sign: 
                rdm_chrom *= np.random.choice([-1,1], size_rdm, p=[0.5,0.5])
            pop = np.concatenate((prev.reshape(1, len(prev)), rdm_chrom), axis=0)
            pop_size = (self.num_chroms, self.num_genes)

        return pop, pop_size

    def run(self):
        """
        Running the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """
        # 1) Initialize population (in __init__)

        self.best_fitness = []
        self.best_chrom = []
        for generation in range(self.num_generations):
            print('\nGEN:', generation)

            # loop over gene types
            parents = []
            offspring_mutation = []

            # 2) Measuring fitness of population.
            
            fitness = self.evaluate_fitness()
            best_match_idx = np.argmax(fitness)
            # best_chrom = np.array([])
            # for j in range(self.gene_groups):
            #     best_chrom = np.append(best_chrom, self.populations[j][best_match_idx, :])
            best_chrom = self.populations.iloc[best_match_idx]
            max_fitness = fitness[best_match_idx]
            
            self.best_fitness.append(max_fitness)
            self.best_chrom.append(best_chrom)

            # 3) Selection of fittest individuals
            try:
                sel_method = getattr(self.Sel, self.sel_type, fitness)
            except AttributeError:
                raise NotImplementedError("The sel type was mistyped: Method is not implemented.")
                
            parents = sequential_dataframe(self.populations, sel_method, fitness, self.num_genes, self.gene_groups)

            # 4) Crossover of each pair of parents
            try:
                cross_method = getattr(self.Cross, self.cross_type)
            except AttributeError:
                raise NotImplementedError("The cross type was mistyped: Method is not implemented.")

            offspring_crossover = sequential_dataframe(parents, cross_method, self.num_offspring, self.num_genes, self.gene_groups)

            all_offs = []
            idx_ac = 0
            
            for g, gen in enumerate(self.num_genes):
                
                # 5) Random mutations
                try:
                    mut_method = getattr(self.Mut, self.mut_type)
                except AttributeError:
                    raise NotImplementedError("The mut type was mistyped: Method is not implemented.")

                offspring_mutation = mut_method(offspring_crossover.iloc[:,idx_ac:idx_ac+gen], 
                                                self.mut_min[g], self.mut_max[g])
                idx_ac += gen
                all_offs.append(offspring_mutation)

                # 6) Creating the new population

            offspring_mutation = pd.concat(all_offs,axis=1)
           
            if (self.keep_parents == 0):
                self.populations = offspring_mutation

            elif (self.keep_parents == -1):
                # Creating the new population based on the parents and offspring.
                self.population[:parents.shape[0]] = parents
                self.population[parents.shape[0]:] = offspring_mutation

            elif (self.keep_parents > 0):
                # Creating new population with a given number of parents
                Sel2 = Selection(self.keep_parents) # new selection 
                sel_method2 = getattr(Sel2, "steady_state", fitness)
                parents_to_keep = sequential_dataframe(self.populations, sel_method2, fitness, self.num_genes, self.gene_groups)
        
                # parents_to_keep = Sel2.steady_state(fitness, self.populations[g])
                self.populations.loc[:parents_to_keep.shape[0]-1] = parents_to_keep
                self.populations.loc[parents_to_keep.shape[0]:] = offspring_mutation.to_numpy()

            # 7) Check for convergence
            # self.evaluate_convergence()
            if self.conv:
                self.run_completed = True
                print("The algorithm has converged with Fitness = {fit:.4f}".format(fit=max_fitness))
                break

        # After the run() method completes, the run_completed flag is changed from False to True.
        self.run_completed = True

    def evaluate_fitness(self):
        """
        Calculating the fitness values of all solutions in the current population. 
        It returns:
            -fitness: An array of the calculated fitness values.
        """
        pop_fitness = []
        # pops = []
        # for i in range(self.num_chroms):
        #     all_chroms = np.array([])
        #     for j in range(self.gene_groups):
        #         all_chroms = np.append(all_chroms, self.populations[j][i])
        #     pops.append(all_chroms)
        # pops = np.array(pops)
        # Calculating the fitness value of each solution in the current population.
        # for sol in pops:
        # for index, sol in self.populations.iterrows():
        #     fitness = self.fitness_func(sol)
        #     pop_fitness.append(fitness)
        
        df_fit = self.populations.apply(self.fitness_func, axis=1)

        pop_fitness = df_fit.to_numpy()

        print("BEST!!!!: ", np.max(pop_fitness))

        return pop_fitness

    def evaluate_convergence(self):

        current_fitness = np.max(self.evaluate_fitness())
        if abs(current_fitness - self.best_fitness[-1]) < 1e-5:
            self.conv = True

    def get_result(self):
        """
        Calculates the fitness values for the current population. 
        If the run() method is not called, then it returns 2 empty lists. Otherwise, it returns the following:
            -best_solution: Best solution in the current population.
            -best_solution_fitness: Fitness value of the best solution.
        """
        if self.run_completed == False:
            raise ValueError("\nThe result has yet to be produced!\n")

        # Getting the best solution after finishing all generations.
        fitness = self.evaluate_fitness()
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = np.argmax(fitness)
        best_chrom = self.populations.iloc[best_match_idx].to_numpy()
        # for j in range(self.gene_groups):
        #     best_chrom = np.append(best_chrom, self.populations[j][best_match_idx, :])

        best_fitness = fitness[best_match_idx]
        self.best_solution = best_chrom

        return best_chrom, best_fitness

    def plot_result(self, param_idx=0):
        '''
        Creating 2 plots that summarizes how the solutions evolved.
        The first between the it number and the function output based on the current parameters for the best solution.
        The second is between the it and the fitness value of the best solution.
        Parameters
        ----------
        param_idx : TYPE, optional
            DESCRIPTION. The default is 0.


        Returns
        -------
        None.

        '''

        import matplotlib.pyplot as plt

        if self.run_completed == False:
            raise ValueError("\nThe result has yet to be produced!\n")

        best_chrom = np.array(self.best_chrom)
        plt.plot(best_chrom[:,param_idx])
        plt.xlabel("Generation")
        plt.ylabel("Param"+str(param_idx))
        plt.show()

        plt.plot(self.best_fitness)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()

    def save(self, path):
        '''
        Saving the genetic algorithm instance

        Parameters
        ----------
        filename : str
            Name of the file w/o extension

        Returns
        -------
        None.

        '''
        with open(path + "ga.pkl", 'wb') as file:
            pickle.dump(self, file)

    
    def load(path):
        '''
        Reading a saved instance of the genetic algorithm    
        
        Parameters
        ----------
        path : str
            path of file.
    
        Returns
        -------
        ga_saved : GA
           GA instance.
    
        '''
    
        try:
            with open(path + "ga.pkl", 'rb') as file:
                ga_saved = pickle.load(file)
        except (FileNotFoundError):
            print("File doesn't exist")
        return ga_saved
    
def parallelize_dataframe(df, func, second, num_genes, gene_groups):
    '''
    When the number of gene groups is small, this function will parallelize the 
    run for each group insteaad of looping through each.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    func : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    from multiprocessing import  Pool
    from functools import partial
    from itertools import repeat
    
    sp_pattern = np.add.accumulate(num_genes)
    df_t = df.T
    df_split = np.array_split(df_t, sp_pattern)[:-1]

    pool = Pool(gene_groups)
    # df = pd.concat(pool.map(func, df_split))
    # df = pd.concat(pool.map(partial(func, b=second), df_split))
    df = pd.concat(pool.starmap(func, zip(df_split, repeat(second))), axis=1)
    pool.close()
    pool.join()
    return df

def sequential_dataframe(df, func, second, num_genes, gene_groups):
    '''
    When the number of gene groups is small, this function will 
    run each group by looping through each.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    func : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''

    sp_pattern = np.add.accumulate(num_genes)
    seq  = np.append(0,sp_pattern)
    
    dfs = []
    
    sum_pop = 0

    for g in range(gene_groups):
        df_gene = df.iloc[:,seq[g]:seq[g+1]]
        df_i = func(df_gene.T, second)
        sum_pop += len(df_i.columns)
        dfs.append(df_i)
        
    dfs = pd.concat(dfs, axis=1)
    
    return dfs
    

class Selection:
    def __init__(self, num_parents, tournament_sel=None):
        if num_parents % 2 == 0:
            self.num_parents = num_parents
        else:
            self.num_parents = 0
            raise ValueError("The number of selected parents must be an even number")
        self.tournament_sel = tournament_sel

    def roulette_wheel(self, population, fitness):
        """ Roulette wheel or Fitness proportionate selection
        Parameters:
            - fitness: NumPy array of fitness function for all the individuals in the chromosome/chromosome group
            - population: Numpy array with population entries
        Returns: NumPy array with selected parents
        """
        population = population.T
        fit_sum = np.sum(fitness) + 0.000000001
    
        p_choose = fitness / fit_sum

        # Building selection probabilities
        sel_probs = np.add.accumulate(np.append([0.0],p_choose))
            
        # Randomly selecting individuals
        parents = []
        for i_parent in range(self.num_parents):
            p_random = np.random.rand()
            for p_i in range(1, len(sel_probs)):
                # Test whether p_random should return this parent
                if sel_probs[p_i - 1] <= p_random < sel_probs[p_i]:
                    parents.append(population.iloc[p_i - 1])
        n_genes = population.shape[1]
        parents = np.array(parents).reshape((self.num_parents, n_genes))
        
        return pd.DataFrame(parents)

    def stochastic(self, population, fitness):
        """ Implements Stochastic Universal Sampling:
            SUS uses a single random value to sample all of the solutions by choosing them at evenly spaced intervals.
         Parameters:
            - fitness: NumPy array of fitness function for all the genes in the chromosome/chromosome group
            - population: Numpy array with population entries
        Returns: NumPy array with selected parents
        """
        population = population.T
    
        # Roulette wheel probabilities
        fit_sum = np.sum(fitness) + 0.0000001
        p_choose = fitness / fit_sum
        
        # Building selection probabilities
        sel_probs = np.add.accumulate(np.append([0.0],p_choose))
        
        # Build pointer array
        pointer_dist = 1.0 / self.num_parents
        start_pointer = np.random.uniform(0, pointer_dist)  # sample random value selected only once
        pointers = [start_pointer + i * pointer_dist for i in range(self.num_parents)]

        # Randomly selecting individuals
        parents = []
        for i_parent in range(self.num_parents):
            for p_i in range(1, len(sel_probs)):
                # Test whether the pointer should return this parent
                if sel_probs[p_i - 1] <= pointers[i_parent] < sel_probs[p_i]:
                    parents.append(population.iloc[p_i - 1])
        n_genes = population.shape[1]
        parents = np.array(parents).reshape((self.num_parents, n_genes))
        
        return pd.DataFrame(parents)


    def truncation(self):
        ''' Implementation pending

        '''
        return None

    def tournament(self, population, fitness):
        """ Implements the Tournament Selection algorithm:
            Random groups are sampled from the population and a winner is selected each time
        Parameters
        ----------
            - fitness: NumPy array of fitness function for all the genes in the chromosome/chromosome group
            - population: Numpy array with population entries
        Returns
        -------
        NumPy array with selected parents
        
        """
        population = population.T
        parents = []
        for i_parent in range(self.num_parents):
            k_idx = np.random.randint(0, len(fitness), self.tournament_sel)
            fit_sel = fitness[k_idx]
            t_max = np.argmax(fit_sel)
            parents.append(population.iloc[k_idx[t_max]])

        n_genes = population.shape[1]
        parents = np.array(parents).reshape((self.num_parents, n_genes))
        return pd.DataFrame(parents)


    def steady_state(self, population, fitness):
        """ Implements the steady state selection algorithm
        Parameters
        ----------
            - fitness: NumPy array of fitness function for all the genes in the chromosome/chromosome group
            - population: Numpy array with population entries
        Returns
        -------
        NumPy array with selected parents
        """
        population = population.T
        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        ##for each type of population

        parents = []
        for i_parent in range(self.num_parents):
            parents.append(population.iloc[fitness_sorted[i_parent]])

        n_genes = population.shape[1]
        parents = np.array(parents).reshape((self.num_parents, n_genes))

        return pd.DataFrame(parents)


class Crossover:
    def __init__(self):
        self.check = None

    def single_point(self, parents, num_offspring):
        """Implements a single-point crossover:
            A random point is picked where parents genes ara combined to form an offspring.

        Parameters
        ----------
        parents : NumPy array
            Array with the parents selected in the previous step.
        num_offspring : double
            The number of offspring to create from parents. 

        Returns
        -------
        NumPy array with the offspring.

        """
        parents = parents.T
        nparents = parents.shape[0]
        
        ngenes = parents.shape[1]
        crossover_point = np.random.randint(0, ngenes)
        offspring = np.zeros((num_offspring, ngenes))

        for offsp in range(num_offspring):
            parent1_idx = offsp % nparents
            parent2_idx = (offsp + 1) % nparents
            # First part from the first parent
            offspring[offsp, :crossover_point] = parents.iloc[parent1_idx, :crossover_point]
            # Second part from the second parent
            offspring[offsp, crossover_point:] = parents.iloc[parent2_idx, crossover_point:]

        return pd.DataFrame(offspring)

    def two_point(self, parents, num_offspring):
        """Implements a two-point crossover:
            Two random points are picked where parents genes are combined to form an offspring.

        Parameters
        ----------
        parents : NumPy array
            Array with the parents selected in the previous step.
        num_offspring : double
            The number of offspring to create from parents. 

        Returns
        -------
        NumPy array with the offspring.

        """
        parents = parents.T
        nparents = parents.shape[0]
        ngenes = parents.shape[1]

        offspring = np.zeros((num_offspring, ngenes))

        if (ngenes == 1):  # Only one gene
            crossover_point1 = 0
        else:
            crossover_point1 = np.random.randint(0, np.ceil(ngenes / 2 + 1))

        crossover_point2 = crossover_point1 + int(ngenes / 2)

        for offsp in range(num_offspring):
            parent1_idx = offsp % nparents
            parent2_idx = (offsp + 1) % nparents
            # First part from the first parent
            offspring[offsp, :crossover_point1] = parents.iloc[parent1_idx, :crossover_point1]
            # Second part from the second parent
            offspring[offsp, crossover_point1:crossover_point2] = parents.iloc[parent2_idx,
                                                                               crossover_point1:crossover_point2]
            # Third part from the first parent
            offspring[offsp, crossover_point2:] = parents.iloc[parent1_idx, crossover_point2:]

        return pd.DataFrame(offspring)

    def uniform(self, parents, num_offspring):
        """Implements an uniform crossover:
            Each gene is selected randomly from one of the corresponding genes of the parent chromosomes.

        Parameters
        ----------
        parents : NumPy array
            Array with the parents selected in the previous step.
        num_offspring : double
            The number of offspring to create from parents. 

        Returns
        -------
        NumPy array with the offspring.
        """
        parents = parents.T
        nparents = parents.shape[0]
        ngenes = parents.shape[1]
        offspring = np.zeros((num_offspring, ngenes))

        for offsp in range(num_offspring):
            parent1_idx = offsp % nparents
            parent2_idx = (offsp + 1) % nparents
            # Random gene is selected from a random parent
            p_chosen = np.random.choice([parent1_idx, parent2_idx], ngenes)

            # offspring[offsp] = parents.iloc[p_chosen]
            for g_i in range(ngenes):
                chosen = p_chosen[g_i]
                offspring[offsp, g_i] = parents.iloc[chosen, g_i]

        return pd.DataFrame(offspring)


class Mutation:
    def __init__(self, mutation_ratio=0.5, min_mutation=0, max_mutation=1):
        self.mut_ratio = mutation_ratio

    def random_reset(self, offspring, mut_min=0, mut_max=1):
        """Implements a random mutation in a randomly chosen gene, by chossing 
           from a range [min_mutation, max_mutation]
        
        Parameters
        ----------
        offspring : NumPy array
            Offspring from, the crossover step.

        Returns
        -------
        NumPy array with the offspring after mutation.

        """
        ngenes = offspring.shape[1]
        noffsp = offspring.shape[0]
        n_mut = int(ngenes * self.mut_ratio)  # how many genes to mutate
        random_pick = np.arange(ngenes)
        np.random.shuffle(random_pick)
        random_picks = random_pick[:n_mut]
        random_lett = offspring.keys()[random_picks]
        for offsp in range(noffsp):
            for g_i in random_lett:
                random_mut = np.random.uniform(mut_min, mut_max)
                offspring.loc[offsp,g_i] = offspring.loc[offsp,g_i] + random_mut
                
        return offspring

    def swap(self, offspring):
        '''Implements a swap mutation by selecting two genes at random, and interchanging the values
        
        Parameters
        ----------
        offspring : NumPy array
            Offspring from, the crossover step.

        Returns
        -------
        NumPy array with the offspring after mutation.

        '''
        ngenes = offspring.shape[1]
        noffsp = offspring.shape[0]

        random_pick = np.arange(ngenes)
        np.random.shuffle(random_pick)
        random_picks = random_pick[:2]

        for offsp in range(noffsp):
            random1 = offspring[offsp, random_pick[0]]
            offspring.loc[offsp, random_pick[0]] = offspring.loc[offsp, random_pick[1]]
            offspring.loc[offsp, random_pick[1]] = random1

        return offspring

    def scramble(self, offspring):
        """ Implements a scramble mutation where a subset of genes is chosen 
            and their values are shuffled randomly


        Parameters
        ----------
        offspring : NumPy array
            Offspring from, the crossover step.

        Returns
        -------
        NumPy array with the offspring after mutation.

        """
        ngenes = offspring.shape[1]
        noffsp = offspring.shape[0]
        n_mut = int(ngenes * self.mut_ratio)  # how many genes to mutate

        for offsp in range(noffsp):
            start = np.random.randint(0, ngenes - n_mut)
            g_shuffled = np.arange(start, start + n_mut)
            g_subset = np.copy(g_shuffled)
            np.random.shuffle(g_shuffled)
            offspring.loc[offsp, g_subset] = offspring.loc[offsp, g_shuffled]

        return offspring

    def inversion(self, offspring):
        """ Implements an inversion mutation where a subset of genes is chosen 
            randomly and their values are inverted

        Parameters
        ----------
        offspring : NumPy array
            Offspring from, the crossover step.

        Returns
        -------
        NumPy array with the offspring after mutation.

        """
        ngenes = offspring.shape[1]
        noffsp = offspring.shape[0]
        n_mut = int(ngenes * self.mut_ratio)  # how many genes to mutate

        for offsp in range(noffsp):
            start = np.random.randint(0, ngenes - n_mut)
            g_subset = np.arange(start, start + n_mut)
            offspring.loc[offsp, g_subset] = offspring.loc[offsp, g_subset[::-1]]

        return offspring





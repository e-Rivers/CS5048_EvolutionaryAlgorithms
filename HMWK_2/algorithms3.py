import numpy as np
import random
import math
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import csv


# Father class: GeneticAlgorithm
class GeneticAlgorithm(ABC):

    def __init__(self, lowerBound, upperBound, varNum, func, popu_size, Pc=0.9, Pm = None):
        self._x_max = upperBound
        self._x_min = lowerBound
        self._func = func
        self._population = []
        self._vars = varNum 
        self._popu_size = popu_size
        self._Pc = Pc           # Probability of crossover
        self._Pm = None          # Probability of mutation

    def run(self, num_generations):
        """
        Function to run the GA 

        Input:
        num_generations: Number of generations

        Output:
        best_solutions: a list of the best solution found in each generation

        """
        # Initialize population
        self.initialize_population()
        self._Pm = 1/len(self._population[0])
        best_solutions = []
        best_individuals = []

        for generation in range(1, num_generations+1):
            
            # Evaluation
            fit_list = self._evaluation(self._getPopulation())

            # Selection
            selected_individuals = self._selection(fit_list)

            # Crossover
            children = self._crossover(selected_individuals)

            # Mutation
            mutated_population = self._mutation(children)

            # Update population
            self._population = list(mutated_population)

            # print best fitness in the generation
            fit_list = self._evaluation(self._getPopulation())
            min_index = fit_list.index(min(fit_list))

            # Store the best fitness and its corresponding decoded point
            best_fitness = fit_list[min_index]
            best_individual = self._population[min_index]
            #print(f"Este es el mejor fitness {best_fitness}, generación {generation}")
            best_solutions.append(best_fitness)
            best_individuals.append(best_individual)


            # Stopping Criterion (if the standard deviation of the last 5 generations is less than threshold 5)
            if generation % 10 == 0:
                if np.std(np.array(best_solutions[generation-10:])) < 0.05:
                    return best_solutions, best_individuals

        return best_solutions, best_individuals

    def _evaluation(self, population):
        """
        function to evaluate the population into the objective function

        input:

        population: list of the real values for each individual (decoded in the case of binary)

        ----------

        output:
        fit_list: list of the fitness of each individual
        """

        fit_list = []
        for i in population:
            current_value = self._func(i)
            fit_list.append(current_value)
            
        return fit_list
    
    # ABSTRACT METHODS

    @abstractmethod
    def _getPopulation(self):
        raise NotImplementedError

    @abstractmethod
    def initialize_population(self):
        raise NotImplementedError

    @abstractmethod
    def _crossover(self, parent1, parent2):
        raise NotImplementedError

    @abstractmethod
    def _mutation(self, individual):
        raise NotImplementedError
    
    @abstractmethod
    def _selection(self):
        raise NotImplementedError


#############################################################
############# Child class for Real Encoding #################
#############################################################

class RealGA(GeneticAlgorithm):

    def __init__(self, lowerBound, upperBound, varNum, func, popu_size, nc=20, nm=20, Pc= 0.9, Pm=0.1):
        super().__init__(lowerBound, upperBound, varNum, func, popu_size, Pc, Pm)
        self._nc = nc
        self._eta_m = nm

    def _getPopulation(self):
        return self._population

    def initialize_population(self):
        """"
        Function to randomly initialize population
        """
        for _ in range(self._popu_size):
            chromosome = [np.random.uniform(self._x_min, self._x_max) for _ in range(self._vars)]
            self._population.append(chromosome)

    def _limitIndividual(self, individual):
        """
        Function to make sure that the individuals are inside the range

        Input:
        inividual : a real individual

        Output:
        individual : an individual inside the range
        """
        for gene in range(self._vars):
            if individual[gene] > self._x_max:
                individual[gene] = self._x_max
            elif individual[gene] < self._x_min:
                individual[gene] = self._x_min

        return individual 


    # Binary tournament selection
    def _selection(self, fit_list):
        """
        Function to select individuals using the binary tournament selection method

        Input:
        fit_list: list with the fitness for each individual

        Outpu:
        parents: list of selected individuals
        """

        evaluatedIndividuals = list(zip(self._population.copy(), fit_list))

        parents = []
        for _ in self._population:

            # Step 1. Shuffle individuals
            shuffledPop = list(evaluatedIndividuals)
            np.random.shuffle(shuffledPop)

            # Step 2. Get two random individuals
            randChoice1 = np.random.randint(0, len(shuffledPop))
            randChoice2 = np.random.randint(0, len(shuffledPop))
            candidate1 = shuffledPop[randChoice1]
            candidate2 = shuffledPop[randChoice2]

            # Step 3. Make them compete based on their fitness
            fitnessCand1 = candidate1[1]
            fitnessCand2 = candidate2[1]

            # Step 4. Select the fittest individual (the one that minimizes)
            parents.append(candidate1[0] if fitnessCand1 < fitnessCand2 else candidate2[0])

        return parents

    # Simulated Binary Crossover (SBX)
    def _crossover(self, parents):

        """
        Function to perform crossover

        Input:
        parents: list of the population

        Output:
        newPopulation : list of the new population
        """

        newPopulation = []

        for parent1, parent2 in list(zip(parents[::2], parents[1::2])):
            
            # Compute the probability of crossover for the current couple
            crossProb = np.random.random()

            if crossProb <= self._Pc:

                # Step 1. Compute a random number u between 0 and 1
                u = np.random.uniform()

                # Step 2. Compute beta_m
                if u <= 0.5:
                    beta_m = (2*u)**(1 / (self._nc + 1))
                else:
                    beta_m = (1 / (2*(1 - u)))**(1 / (self._nc + 1))

                # Step 3. Produce children
                H1 = 0.5 * ((leftSide := (np.array(parent1) + np.array(parent2))) - (rightSide := beta_m*np.abs(np.array(parent2) - np.array(parent1))))
                H2 = 0.5 * (leftSide + rightSide)

                newPopulation.extend([self._limitIndividual(H1), self._limitIndividual(H2)])
            else:
                newPopulation.extend([parent1, parent2])

        return newPopulation


    # Parameter-based mutation
    def _mutation(self, population):
        """
        Function to perform parameter-based mutation

        input:
        population: list of individuals

        output: 
        newPopulation: list of new population (including mutated individuals)
        """

        newPopulation = []
        for individual in population:

            # Compute the probability of mutation for the current individual 
            mutProb = np.random.random()

            if mutProb <= self._Pm:

                # Step 1. Randomly select the gene to be mutated
                i = np.random.randint(0, len(individual))
                gene = individual[i]

                # Step 2. Compute a random number u between 0 and 1
                u = np.random.random()

                # Step 3. Compute delta sub q
                eta_m = self._eta_m # This actually is calculated as 100 + t, where t = generation num. But for the hwmk, it was required as 20
                delta = min((gene - self._x_min), (self._x_max - gene)) / (self._x_max - self._x_min)
                if u <= 0.5:
                    delta_q = (2*u + (1-2*u)*(1-delta)**(eta_m+1))**(1 / (eta_m+1)) - 1
                else:
                    delta_q = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta)**(eta_m+1))**(1 / (eta_m+1))

                # Step 4. Perform the mutation 
                deltaMax = self._x_max - self._x_min
                mutatedGene = gene + delta_q*deltaMax
                individual[i] = mutatedGene

            newPopulation.append(self._limitIndividual(individual))

        return newPopulation 


###############################################################
############## Class for the Hybrid GA and DE #################
###############################################################

class HybridDE(GeneticAlgorithm):

    def __init__(self, lowerBound, upperBound, varNum, func, popu_size):
        super().__init__(lowerBound, upperBound, varNum, func, popu_size)

    def _getPopulation(self):
        return self._population
    
    def run(self, num_generations):
        """
        Function to run the DE algorithm

        Input:
        num_generations: Number of generations

        Output:
        best_solutions: a list of the best solution found in each generation
        """
        # Initialize population
        self.initialize_population()
        self._Pm = 1/len(self._population[0])
        best_solutions = []
        best_individuals = []

        for generation in range(1, num_generations+1):
            # Evaluate current population
            fit_list = self._evaluation(self._population)
            
            new_population = []
            for i, each_ind in enumerate(self._population):
                # Mutation step (create donor vectors)
                donor_vector = self._mutation(np.array(self._population))

                # Crossover step (binomial crossover between target vector and donor vector)
                crossover_vector = self._crossover([each_ind], [donor_vector[i]])

                # Selection step: Evaluate child and select better individual
                new_individual = crossover_vector[0]
                new_fitness = self._func(new_individual)

                # If the new individual is better, replace the target individual
                if new_fitness < fit_list[i]:
                    new_population.append(new_individual)
                else:
                    new_population.append(each_ind)

            # Update population with new individuals
            self._population = new_population

            # Evaluate new population and get the best individual
            fit_list = self._evaluation(self._population)
            min_index = fit_list.index(min(fit_list))

            # Store the best fitness and its corresponding individual
            best_fitness = fit_list[min_index]
            best_individual = self._population[min_index]

            best_solutions.append(best_fitness)
            best_individuals.append(best_individual)

            # Print progress
            print(f"Generation {generation} - Best Fitness: {best_fitness}")

            # Stopping Criterion: If the standard deviation of the last 5 generations is less than threshold
            if generation >= 10 and np.std(best_solutions[-10:]) < 0.05:
                print(f"Stopping early at generation {generation} due to convergence.")
                return best_solutions, best_individuals

        return best_solutions, best_individuals

    def _evaluation(self, population):
        """
        function to evaluate the population into the objective function

        input:

        population: list of the real values for each individual (decoded in the case of binary)

        ----------

        output:
        fit_list: list of the fitness of each individual
        """

        fit_list = []
        for i in population:
            current_value = self._func(i)
            fit_list.append(current_value)
            
        return fit_list

    def initialize_population(self):
        """"
        Function to randomly initialize population
        """
        for _ in range(self._popu_size):
            chromosome = [np.random.uniform(self._x_min, self._x_max) for _ in range(self._vars)]
            self._population.append(chromosome)

    def _limitIndividual(self, individual):
        """
        Function to make sure that the individuals are inside the range

        Input:
        inividual : a real individual

        Output:
        individual : an individual inside the range
        """
        for gene in range(self._vars):
            if individual[gene] > self._x_max:
                individual[gene] = self._x_max
            elif individual[gene] < self._x_min:
                individual[gene] = self._x_min

        return individual 


    # Binomial Crossover
    def _crossover(self, parents, donor_vectors, CR=0.5):
    
        """
        Function to perform binomial crossover

        Input:
        parents: list of the population (target vectors)
        donor_vectors: list of donor vectors (mutated vectors)
        CR: crossover probability, typically between 0 and 1

        Output:
        new_population : list of the new population
        """
        
        new_population = []
        
        # Iterate through the parents and perform crossover with donor vectors
        for parent, donor in zip(parents, donor_vectors):

            parent = np.array(parent)
            donor = np.array(donor)
            
            #  decide crossover points (random)
            crossover_mask = np.random.rand(len(parent)) < CR
            j_rand = np.random.randint(0, len(parent))
            crossover_mask[j_rand] = True
            
            # Create a new child by mixing the parent and donor vectors
            child = np.where(crossover_mask, donor, parent)
            
            # Add it
            new_population.append(child)
        
        return new_population


    # Mutation
    def _mutation(self, population, F=0.5):
        newPopulation = np.empty_like(population)
        numIndividuals, numFeatures = population.shape
    
        for i in range(numIndividuals):
            indices = np.random.choice(np.delete(np.arange(numIndividuals), i), 3, replace=False)
            x1, x2, x3 = population[indices[0]], population[indices[1]], population[indices[2]]

            # Calculate the mutant vector
            mutantVector = x1 + F * (x2 - x3)

            newPopulation[i] = mutantVector
    
        return newPopulation


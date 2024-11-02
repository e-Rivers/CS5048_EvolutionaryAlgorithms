import numpy as np
import random
import math
from problems_Part1 import PROBLEMS
import matplotlib.pyplot as plt
import csv
import os

class RealGA():
    def __init__(self, problem, popu_size, nc=20, nm=20, Pc=0.9, Pm=0.1):
        self._x_max = [upper for _, upper in problem["Bounds"]]
        self._x_min = [lower for lower, _ in problem["Bounds"]]
        self._problem = problem
        self._population = []
        self._vars = len(problem["Bounds"])
        self._popu_size = popu_size
        self._Pc = Pc           # Probability of crossover
        self._Pm = None          # Probability of mutation
        self._nc = nc
        self._eta_m = nm

    def _evaluation(self, population):
        fit_list = [self._func(ind) for ind in population]
        return fit_list

    # Function to evaluate fitness including constraint handling technique transformation
    def _evaluation(self, population):
        c1 = 3250
        c2 = 2000
        ß = 2
        Y = 2

        fitnessList = []

        for individual in population:

            # Compute the overall penalty for all inequality constraints G(x)
            inequalityPenalty = 0
            if len(self._problem["Constraints"]["Inequality"]) == 0:
                for constraint in self._problem["Constraints"]["Inequality"]:
                    inequalityPenalty += max(0, constraint(individual))**ß

            # Compute the overall penalty for all equality constraints H(x)
            equalityPenalty = 0
            if len(self._problem["Constraints"]["Equality"]) == 0:
                for constraint in self._problem["Constraints"]["Equality"]:
                    equalityPenalty += abs(constraint(individual))**Y

            fitness = self._problem["Equation"](individual) + (c1*inequalityPenalty + c2*equalityPenalty)

            fitnessList.append(fitness)

        return fitnessList


    def run(self, num_generations):
        self.initialize_population()
        self._Pm = 1/len(self._population[0])
        best_solutions = []
        best_individuals = []

        for generation in range(1, num_generations+1):
            fit_list = self._evaluation(self._getPopulation())
            selected_individuals = self._selection(fit_list)
            children = self._crossover(selected_individuals)
            mutated_population = self._mutation(children)
            self._population = list(mutated_population)

            fit_list = self._evaluation(self._getPopulation())
            min_index = fit_list.index(min(fit_list))
            best_fitness = fit_list[min_index]
            best_individual = self._population[min_index]
            best_solutions.append(best_fitness)
            best_individuals.append(best_individual)

            if generation % 10 == 0:
                if np.std(np.array(best_solutions[generation-10:])) < 0.05:
                    return best_solutions, best_individuals

        return best_solutions, best_individuals

    def _getPopulation(self):
        return self._population

    def initialize_population(self):
        for _ in range(self._popu_size):
            chromosome = [np.random.uniform(self._x_min[i], self._x_max[i]) for i in range(self._vars)]
            self._population.append(chromosome)

    def _limitIndividual(self, individual):
        for gene in range(self._vars):
            if individual[gene] > self._x_max[gene]:
                individual[gene] = self._x_max[gene]
            elif individual[gene] < self._x_min[gene]:
                individual[gene] = self._x_min[gene]

        return individual 


    # Binary tournament selection
    def _selection(self, fit_list):
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
                delta = min((gene - self._x_min[i]), (self._x_max[i] - gene)) / (self._x_max[i] - self._x_min[i])
                if u <= 0.5:
                    delta_q = (2*u + (1-2*u)*(1-delta)**(eta_m+1))**(1 / (eta_m+1)) - 1
                else:
                    delta_q = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta)**(eta_m+1))**(1 / (eta_m+1))

                # Step 4. Perform the mutation 
                deltaMax = self._x_max[i] - self._x_min[i]
                mutatedGene = gene + delta_q*deltaMax
                individual[i] = mutatedGene

            newPopulation.append(self._limitIndividual(individual))

        return newPopulation 

GA = RealGA(PROBLEMS["G1"], 100)
evaluations, solutions = GA.run(100)

for ans, sol, in zip(evaluations, solutions):
    print(f"f({",   ".join([str(s) for s in sol])}) = {ans}")
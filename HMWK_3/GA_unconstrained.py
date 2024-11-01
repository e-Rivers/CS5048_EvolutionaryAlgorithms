import numpy as np
import random
import math
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import csv
import os


def layeb05(x):
    n = len(x)
    result = 0
    for i in range(n - 1):
        up = np.log(abs(np.sin(x[i] - np.pi / 2) + np.cos(x[i + 1] - np.pi)) + 0.001)
        down = abs(np.cos(2 * x[i] - x[i + 1] + np.pi / 2)) + 1
        result += up / down
    return result

def layeb10(x):
    n = len(x)
    result = 0
    for i in range(n - 1):
        result += (np.log(x[i]**2 + x[i + 1]**2 + 0.5))**2 + abs(100 * np.sin(x[i] - x[i + 1]))
    return result

def layeb15(x):
    n = len(x)
    result = 0
    for i in range(n - 1):
        sqrt_term = np.sqrt(np.abs(np.tanh(2 * abs(x[i]) - x[i + 1]**2 - 1)))  # Use np.abs to ensure non-negativity
        result += 10 * sqrt_term + abs(np.exp(x[i] * x[i + 1]) - 1)
    return result


def layeb18(x):
    n = len(x)
    result = 0
    for i in range(n - 1):
        result += np.log(np.abs(np.cos(2 * x[i] * x[i+1] / np.pi)) + 0.001) / (np.abs(np.sin(x[i] + x[i+1]) * np.cos(x[i])) + 1)
    return result

class GeneticAlgorithm(ABC):
    def __init__(self, lowerBound, upperBound, varNum, func, popu_size, Pc=0.9, Pm=None):
        self._x_max = upperBound
        self._x_min = lowerBound
        self._func = func
        self._population = []
        self._vars = varNum 
        self._popu_size = popu_size
        self._Pc = Pc           # Probability of crossover
        self._Pm = None          # Probability of mutation

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

    def _evaluation(self, population):
        fit_list = [self._func(ind) for ind in population]
        return fit_list

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


class RealGA(GeneticAlgorithm):
    def __init__(self, lowerBound, upperBound, varNum, func, popu_size, nc=20, nm=20, Pc=0.9, Pm=0.1):
        super().__init__(lowerBound, upperBound, varNum, func, popu_size, Pc, Pm)
        self._nc = nc
        self._eta_m = nm

    def _getPopulation(self):
        return self._population

    def initialize_population(self):
        self._population = [[np.random.uniform(self._x_min, self._x_max) for _ in range(self._vars)] for _ in range(self._popu_size)]

    def _limitIndividual(self, individual):
        return [min(max(gene, self._x_min), self._x_max) for gene in individual]

    def _selection(self, fit_list):
        evaluatedIndividuals = list(zip(self._population.copy(), fit_list))
        parents = []
        for _ in self._population:
            np.random.shuffle(evaluatedIndividuals)
            candidate1, candidate2 = random.sample(evaluatedIndividuals, 2)
            parents.append(candidate1[0] if candidate1[1] < candidate2[1] else candidate2[0])
        return parents

    def _crossover(self, parents):
        newPopulation = []
        for parent1, parent2 in zip(parents[::2], parents[1::2]):
            crossProb = np.random.random()
            if crossProb <= self._Pc:
                u = np.random.uniform()
                beta_m = (2 * u)**(1 / (self._nc + 1)) if u <= 0.5 else (1 / (2 * (1 - u)))**(1 / (self._nc + 1))
                H1 = 0.5 * ((leftSide := np.array(parent1) + np.array(parent2)) - (rightSide := beta_m * np.abs(np.array(parent2) - np.array(parent1))))
                H2 = 0.5 * (leftSide + rightSide)
                newPopulation.extend([self._limitIndividual(H1), self._limitIndividual(H2)])
            else:
                newPopulation.extend([parent1, parent2])
        return newPopulation

    def _mutation(self, population):
        newPopulation = []
        for individual in population:
            mutProb = np.random.random()
            if mutProb <= self._Pm:
                i = np.random.randint(0, len(individual))
                u = np.random.random()
                delta = min((individual[i] - self._x_min), (self._x_max - individual[i])) / (self._x_max - self._x_min)
                delta_q = (2 * u + (1 - 2 * u) * (1 - delta)**(self._eta_m + 1))**(1 / (self._eta_m + 1)) - 1 if u <= 0.5 else 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta)**(self._eta_m + 1))**(1 / (self._eta_m + 1))
                individual[i] += delta_q * (self._x_max - self._x_min)
            newPopulation.append(self._limitIndividual(individual))
        return newPopulation 

import pandas as pd
import matplotlib.pyplot as plt


class Report:
    def __init__(self, executions, generations):
        self.generations = generations
        self.executions = executions
        self.best_individuals = []

    def add_best_individual_at_generation(self, generation, execution, fitness_value):
        # Append generation, execution, and individual's best fitness value
        self.best_individuals.append((generation, execution, fitness_value))

    def save_to_csv(self, output_file):
        # Convert best individuals' data to DataFrame and save as CSV
        df = pd.DataFrame(self.best_individuals, columns=["Generation", "Execution", "Best Fitness"])
        df.to_csv(output_file, index=False)

    def plot_convergence(self, output_folder, func_name):
        # Define global minima for specific functions (add more as needed)
        global_minima = {
            "Layeb05 (n=2)": -6.907,
            "Layeb10 (n=2)": 0,
            "Layeb15 (n=2)": 0,
            "Layeb18 (n=2)": -6.907
        }
        
        # Plot the global minimum if defined for the function
        if func_name in global_minima:
            plt.axhline(global_minima[func_name], color='r', linestyle='--', label='Global Minimum')

        # Extract and plot each execution's best fitness over generations
        for execution in range(self.executions):
            execution_data = [record[2] for record in self.best_individuals if record[1] == execution]
            plt.plot(execution_data, label=f'Execution {execution + 1}')
        
        # Plot settings
        plt.xscale('log')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title(f'Convergence Plot for {func_name}')
        #plt.legend(loc="upper right", fontsize='small')
        plt.tight_layout()

        # Save and clear the plot for each function
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(f"{output_folder}/{func_name.replace(' ', '_')}_convergence.png")
        plt.clf()

problems = [
        (
            layeb05,
            np.array([-10, 10]).astype(float),
            "Layeb05 (n=2)",
            2
        ),
        (
            layeb10,
            np.array([-100, 100]).astype(float),
            "Layeb10 (n=2)",
            2
        ),
        (
            layeb15,
            np.array([-100, 100]).astype(float),
            "Layeb15 (n=2)",
            2
        ),
        (
            layeb18,
            np.array([-10, 10]).astype(float),
            "Layeb18 (n=2)",
            2
        )
    ]

executions = 30
num_generations = 100

for func, bounds, problem_name, dimensions in problems:
    print(f"Running Genetic Algorithm on {problem_name} for {executions} executions...")
    
    report = Report(executions, num_generations)
    
    for execution in range(executions):
        print(f"Execution {execution + 1}/{executions}")
        ga = RealGA(lowerBound=bounds[0], upperBound=bounds[1], varNum=dimensions, func=func, popu_size=50, Pc=0.9, Pm=0.1)
        best_solutions, _ = ga.run(num_generations=num_generations)
        
        for generation, best_solution in enumerate(best_solutions):
            report.add_best_individual_at_generation(generation, execution, best_solution)

    output_folder = "experiment_results_ga"
    csv_output_path = f"{output_folder}/{problem_name.replace(' ', '_')}_results.csv"
    report.save_to_csv(csv_output_path)
    print(f"Results saved to {csv_output_path}")

    report.plot_convergence(output_folder, problem_name)
    print(f"Convergence plot saved for {problem_name}")
    print("------------------------------------------------")


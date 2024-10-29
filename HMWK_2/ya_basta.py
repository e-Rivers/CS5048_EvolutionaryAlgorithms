import numpy as np
import random
import math
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import csv

class HybridDE():

    def __init__(self, lowerBound, upperBound, varNum, func, popu_size):
        self._x_min = lowerBound
        self._x_max = upperBound
        self._vars = varNum
        self._func = func
        self._popu_size = popu_size
        self._population = []

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

            ## Stopping Criterion: If the standard deviation of the last 5 generations is less than threshold
            #if generation >= 10 and np.std(best_solutions[-10:]) < 0.05:
            #    print(f"Stopping early at generation {generation} due to convergence.")
            #    return best_solutions, best_individuals

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

if __name__ == "__main__":

    # Definición de problemas (ejemplo)
    problems = [
        ( 
            lambda x: 10 * len(x) + sum([(x_i**2 - 10 * np.cos(2 * np.pi * x_i)) for x_i in x]),
            [-5.12, 5.12],
            "Rastrigin (n=2)",
            2
        ),
        (
            lambda x: sum([100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1)]),
            [-2.5, 2.5],
            "Rosenbrock (n=2)",
            2
        )
    ]

    # Run DE on each problem
    num_generations = 100
    population_size = 20

    for func, bounds, name, var_num in problems:
        print(f"\nProblem: {name}")
        de = HybridDE(bounds[0], bounds[1], var_num, func, population_size)
        best_solutions, best_individuals = de.run(num_generations)
        print(f"Best Solution: {best_solutions[-1]}")
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def run_experiments_hybrid_de(hybrid_de_class, problems, pop_size, num_generations, num_runs):
        """
        Function to run a hybrid differential evolution algorithm for a set of given problems

        input: 
        hybrid_de_class: The HybridDE class to be used
        problems: A list of problems, each with a fitness function, bounds, name, and number of decision variables.
        pop_size: Population size for the hybrid differential evolution algorithm
        num_generations: Number of generations
        num_runs: Number of independent runs

        output:
        results: List of best fitness values for each run.
        fitness_history: History of fitness values across generations for each run.
        individual_history: History of individuals across generations for each run.
        """
        results = []
        fitness_history = []
        individual_history = []

        # Loop through each problem
        for func, bounds, _, decision_vars in problems:
            for _ in range(num_runs):
                # Initialize HybridDE with the bounds and function
                lower_bound, upper_bound = bounds  # Extract min and max bounds
                hybrid_de = hybrid_de_class(lower_bound, upper_bound, decision_vars, func, pop_size)
                
                # Run the hybrid DE
                best_fitness, best_individuals = hybrid_de.run(num_generations=num_generations) 
                
                # Get the best fitness value in this run
                best_index = np.argmin(best_fitness)
                results.append((best_fitness[best_index], best_individuals[best_index]))
                fitness_history.append(best_fitness)
                individual_history.append(best_individuals)
        
        return results, fitness_history, individual_history


    # Initialize problems
    problems = [
        (
            lambda x: 10 * len(x) + sum([(x_i ** 2 - 10 * np.cos(2 * np.pi * x_i)) for x_i in x]),
            np.array([-5.12, 5.12]).astype(float),
            "Rastrigin (n=2)",
            2
        ),
        (
            lambda x: sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)]),
            np.array([-2.5, 2.5]).astype(float),
            "Rosenbrock (n=2)",
            2
        )
    ]

    # Ask for input variables
    print("Would you like to set up the parameters manually or go with the default values?")
    print("Population size: 100")
    print("Number of generations: 250")
    print("Number of runs: 20")
    setup = input("\nGo with default values? [y/n] ")
    pop_size = 100 if setup == "y" else int(input("Set a population size: "))
    num_generations = 250 if setup == "y" else int(input("Set a number of generations: "))
    num_runs = 20 if setup == "y" else int(input("Set the number of runs: "))

    results = {}
    for de_class in [HybridDE]: 
        results[de_class.__name__] = {}

        print(f"\n\n\n\n{de_class.__name__}\n\n\n\n")

        for problem in problems:
            bestOverall, fitness_history, individual_history = run_experiments_hybrid_de(
                de_class, 
                [problem], 
                pop_size=pop_size, 
                num_generations=num_generations, 
                num_runs=num_runs
            )
            
            fitnesses = np.array([fitness for fitness, _ in bestOverall], dtype=float)
            individuals = np.array([individual for _, individual in bestOverall], dtype=float)
            results[de_class.__name__][problem[2]] = {
                'mean': np.mean(fitnesses),
                'std': np.std(fitnesses),
                'min': np.min(fitnesses),
                'max': np.max(fitnesses),
                'fitness_history': fitness_history,
                'individual_history': individual_history,
                'results': fitnesses,
                'results (individual)': individuals
            }

    # Print the results (mean, standard deviation, min, max) for each DE and problem
    for de_name, res in results.items():
        print(f"{de_name}:")
        for prob_name, stats in res.items():
            print(f"  {prob_name}: Mean: {stats['mean']}, Std Dev: {stats['std']} ,Min: {stats['min']}, Max: {stats['max']}")
            print(f"  {prob_name}: 20 experiments: {stats['results']}")
            print(*(expResults := list(zip(stats["results"], stats["results (individual)"]))), sep="\n")

    sortedResults = sorted(expResults, key=lambda x: x[0])
    print("SORTED")
    print(*sortedResults, sep="\n")

    # Plot Landscape (3D and contour) with best solutions per generation and mark the overall best solution
    fig = plt.figure(figsize=(10, 7.5))
    for i, problem in enumerate(problems):
        fitness_func, bounds, name, dimension = problem

        x_vals = np.linspace(bounds[0], bounds[1], 100)
        y_vals = np.linspace(bounds[0], bounds[1], 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.array([[fitness_func([x, y]) for x in x_vals] for y in y_vals])

        # Plot the landscape in 3D
        ax3D = fig.add_subplot(int(f"22{2*i+1}"), projection='3d')
        ax3D.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax3D.contour(X, Y, Z, levels=20, zdir='z', offset=ax3D.get_zlim()[0], cmap='viridis', alpha=0.5)
        ax3D.set_title(f"{name} Function")
        ax3D.set_xlabel('X')
        ax3D.set_ylabel('Y')
        ax3D.set_zlabel('Fitness')

        # Plot the best solutions per generation
        best_individuals = results['HybridDE'][name]['individual_history'][0]  # Choose the first run for demonstration
        best_fitness = results['HybridDE'][name]['fitness_history'][0]

        # Plot the trajectory of the best solutions in 3D
        ax3D.scatter(
            [ind[0] for ind in best_individuals],
            [ind[1] for ind in best_individuals],
            best_fitness,
            c='r', marker='o', label="Best per Generation"
        )

        # Mark the overall best solution with an 'X' in 3D
        overall_best_ind = np.argmin(best_fitness)
        ax3D.scatter(
            best_individuals[overall_best_ind][0],
            best_individuals[overall_best_ind][1],
            best_fitness[overall_best_ind],
            c='black', marker='x', s=100, label="Overall Best"
        )
        ax3D.legend()

        # Plot the landscape in 2D
        ax2D = fig.add_subplot(int(f"22{2*i+2}"))
        ax2D.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)

        # Plot the trajectory of the best solutions in 2D
        ax2D.scatter(
            [ind[0] for ind in best_individuals],
            [ind[1] for ind in best_individuals],
            c='r', marker='o', label="Best per Generation"
        )

        # Mark the overall best solution with an 'X' in 2D
        ax2D.scatter(
            best_individuals[overall_best_ind][0],
            best_individuals[overall_best_ind][1],
            c='black', marker='x', s=100, label="Overall Best"
        )
        ax2D.set_title(f"{name} Function - Best Solutions")
        ax2D.set_xlabel('X')
        ax2D.set_ylabel('Y')
        ax2D.legend()

    plt.tight_layout()
    plt.show()
import numpy as np
import pandas as pd
import random
import math
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
        self._Pc = Pc               # Probability of crossover
        self._Pm = None             # Probability of mutation
        self._nc = nc
        self._eta_m = nm
        self._P = [1e-4 for _ in problem["Constraints"]]

    # Numerically compute the gradient
    def _gradient(self, f, x, epsilon=1e-8):
        gradient = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = np.copy(x)
            x_minus = np.copy(x)
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            gradient[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
        return gradient

    # Numerically compute the hessian matrix
    def _hessian(self, f, x, epsilon=1e-5):
        n = len(x)
        hessian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_plus_i = np.copy(x)
                x_minus_i = np.copy(x)
                x_plus_i[i] += epsilon
                x_minus_i[i] -= epsilon

                x_plus_j = np.copy(x)
                x_minus_j = np.copy(x)
                x_plus_j[j] += epsilon
                x_minus_j[j] -= epsilon

                f_ij = f(x_plus_i) - f(x_minus_i) - f(x_plus_j) + f(x_minus_j)
                hessian[i, j] = f_ij / (4 * epsilon**2)
        return hessian

    # Function to evaluate fitness including constraint handling technique transformation
    def _evaluation(self, population):
        fitnessList = []

        for individual in population:
            inequalityPenalty = 0
            equalityPenalty = 0
            for i, constraint in enumerate(self._problem["Constraints"]):
                # Compute the overall penalty for all inequality constraints G(x)
                if constraint["type"] == "inequality":
                    inequalityPenalty += self._P[i] * max(0, constraint["function"](individual))
                # Compute the overall penalty for all equality constraints H(x)
                else:
                    equalityPenalty += self._P[i] * abs(constraint["function"](individual))

                gradient = self._gradient(constraint["function"], individual)
                hessian = self._hessian(constraint["function"], individual)
                
                self._P[i] -= np.dot(gradient, np.linalg.pinv(hessian) @ gradient)

            fitness = self._problem["Equation"](individual) + (inequalityPenalty + equalityPenalty)
            fitnessList.append(fitness)

        return fitnessList

    def run(self, num_generations):
        self.initialize_population()
        self._Pm = 1/len(self._population[0])
        best_solutions = []
        best_individuals = []

        for generation in range(1, num_generations+1):
            fit_list = self._evaluation(self._getPopulation())
            selected_individuals = self._selection()
            children = self._crossover(selected_individuals)
            mutated_population = self._mutation(children, generation)
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
                #individual[gene] = np.random.uniform(self._x_min[gene], self._x_max[gene])
            elif individual[gene] < self._x_min[gene]:
                individual[gene] = self._x_min[gene]
                #individual[gene] = np.random.uniform(self._x_min[gene], self._x_max[gene])

        return individual 

    # Random parent selection
    def _selection(self):
        parents = []
        for _ in self._population:
            shuffledPop = list(self._population)
            np.random.shuffle(shuffledPop)

            randChoice = np.random.randint(0, len(shuffledPop))
            parent = shuffledPop[randChoice]
            parents.append(parent)

        return parents

    # Simulated Binary Crossover (SBX)
    def _crossover(self, parents):
        newPopulation = []

        for parent1, parent2 in list(zip(parents[::2], parents[1::2])):
            crossProb = np.random.random()

            if crossProb <= self._Pc:
                u = np.random.uniform()

                if u <= 0.5:
                    beta_m = (2*u)**(1 / (self._nc + 1))
                else:
                    beta_m = (1 / (2*(1 - u)))**(1 / (self._nc + 1))

                H1 = 0.5 * ((leftSide := (np.array(parent1) + np.array(parent2))) - (rightSide := beta_m*np.abs(np.array(parent2) - np.array(parent1))))
                H2 = 0.5 * (leftSide + rightSide)

                newPopulation.extend([self._limitIndividual(H1), self._limitIndividual(H2)])
            else:
                newPopulation.extend([parent1, parent2])

        return newPopulation


    # Parameter-based mutation
    def _mutation(self, population, genNum):
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
                eta_m = 100 + genNum
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

# Define the problem

PROBLEMS = {
    "G1" : {
        "Equation": lambda x: (
            5 * x[0] + 5 * x[1] + 5 * x[2] + 5 * x[3]
            - 5 * (x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2)
            - (x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[12])
        ),
        "Constraints": [
            {"type": "inequality", "function": lambda x: 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10},
            {"type": "inequality", "function": lambda x: 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10},
            {"type": "inequality", "function": lambda x: 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10},
            {"type": "inequality", "function": lambda x: -8 * x[0] + x[9]},
            {"type": "inequality", "function": lambda x: -8 * x[1] + x[10]},
            {"type": "inequality", "function": lambda x: -8 * x[2] + x[11]},
            {"type": "inequality", "function": lambda x: -2 * x[3] - x[4] + x[9]},
            {"type": "inequality", "function": lambda x: -2 * x[5] - x[6] + x[10]},
            {"type": "inequality", "function": lambda x: -2 * x[7] - x[8] + x[11]}
        ],
        "Optimal" : {
            "Solution" : [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1
            ],
            "Evaluation" : -15
        },
        "Bounds" : [
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 100),
            (0, 100),
            (0, 100),
            (0, 1)
        ]
    },

    "G4" : {
        "Equation": lambda x: (
            5.3578547 * x[2]**2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141
        ),
        "Constraints": [
            {"type": "inequality", "function": lambda x: 85.334407 + 0.0056858 * x[1] * x[4] + 0.00026 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92},
            {"type": "inequality", "function": lambda x: -1 * (85.334407 + 0.0056858 * x[1] * x[4] + 0.00026 * x[0] * x[3] - 0.0022053 * x[2] * x[4])},
            {"type": "inequality", "function": lambda x: 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2]**2 - 110},
            {"type": "inequality", "function": lambda x: -1 * (80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2]**2) + 90},
            {"type": "inequality", "function": lambda x: 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3] - 25},
            {"type": "inequality", "function": lambda x: -1 * (9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]) + 20}
        ],
        "Optimal" : {
            "Solution" : [
                78.0, 33.0, 29.995, 45.0, 36.776
            ],
            "Evaluation" : -30665.5
        },
        "Bounds" : [
            (78, 102),
            (33, 45),
            (27, 45),
            (27, 45),
            (27, 45)
        ]
    },

    "G5" : {
        "Equation" : lambda x: 3*x[0] + 0.000001*x[0]**3 + 2*x[1] + 0.000002/3*x[1]**3,
        "Constraints": [
            {"type": "inequality", "function": lambda x: -x[3] + x[2] - 0.55},
            {"type": "inequality", "function": lambda x: -x[2] + x[3] - 0.55},
            {"type": "equality", "function": lambda x: 1000*np.sin(-x[2] - 0.25) + 1000*np.sin(-x[3] - 0.25) + 894.8 - x[0]},
            {"type": "equality", "function": lambda x: 1000*np.sin(x[2] - 0.25) + 1000*np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1]},
            {"type": "equality", "function": lambda x: 1000*np.sin(x[3] - 0.25) + 1000*np.sin(x[3] - x[2] - 0.25) + 1294.8},
        ],
        "Optimal" : {
            "Solution" : [
                679.9453, 1026.067, 0.1188764, -0.3962336
            ],
            "Evaluation" : 5126.4981
        },
        "Bounds" : [
            (0, 1200),
            (0, 1200),
            (-0.55, 0.55),
            (-0.55, 0.55)
        ]
    },

    "G6" : {
        "Equation" : lambda x: (x[0] - 10)**3 + (x[1] - 20)**3,
        "Constraints": [
            {"type": "inequality", "function": lambda x: - (x[0] - 5)**2 - (x[1] - 5)**2 + 100},
            {"type": "inequality", "function": lambda x: -1*(-(x[0] - 6)**2 - (x[1] - 5)**2 + 82.81)},
        ],
        "Optimal" : {
            "Solution" : [
                14.095, 0.84296
            ],
            "Evaluation" : -6961.81381
        },
        "Bounds" : [
            (13, 100),
            (0, 100)
        ]
    }
}

def evaluateConstraints(individual, constraints):
    total_violation = 0.0
    violated_constraints = []

    for i, constraint in enumerate(constraints):
        if constraint["type"] == "inequality":
            violation = max(0, constraint["function"](individual))
        elif constraint["type"] == "equality":
            violation = max(0, abs(constraint["function"](individual)))

        total_violation += violation
        if violation > 0:  # Only add violated constraints
            violated_constraints.append(i)

    return total_violation, violated_constraints

import sys
print(sys.argv[1])

# Run the algorithm for each problem 30 times and store the results in separate CSV files
for problem_name, problem in PROBLEMS.items():
    results = []

    for run in range(30):
        GA = RealGA(problem, 50)
        solutions, individuals = GA.run(200)
        constraint_violation, violated_constraints = evaluateConstraints(individuals[-1], problem["Constraints"])

        results.append({
            "Problem" : problem_name,
            "Run" : run + 1,
            "Best Solution" : individuals[-1],
            "Best Value" : solutions[-1],
            "Constraint Violation": constraint_violation,
            "Violated Constraints Indexes": violated_constraints
        })

    # Convert results to DataFrame and save to CSV for the current problem
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results_ga_constrained/{sys.argv[1]}_{problem_name}_results.csv", index=False)
    

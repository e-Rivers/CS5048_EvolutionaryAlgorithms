import numpy as np
import pandas as pd

class DifferentialEvolution:
    def __init__(self, function, bounds, constraints, popSize=20, maxIter=100, F=0.1, P_r=0.8, epsilon=1e-6, penalty_factor=1e6):
        self.popSize = popSize
        self.bounds = bounds
        self.constraints = constraints
        self.maxIter = maxIter
        self.F = F
        self.P_r = P_r
        self.dims = len(bounds)
        self.epsilon = epsilon
        self.evaluateFitness = function
        self.population = self.initPopulation()
        self.objValues = [self.evaluateFitness(individual) for individual in self.population]
        self.bestVector = self.population[np.argmin(self.objValues)]
        self.bestObj = np.min(self.objValues)
        self.penalty_factor = penalty_factor

    def initPopulation(self):
        return np.array([
            [np.random.uniform(low, high) for (low, high) in self.bounds]
            for _ in range(self.popSize)
        ])

    def evaluateConstraints(self, individual):
        """Calculate total constraint violation and return violated constraints."""
        total_violation = 0.0
        violated_constraints = []

        for i, constraint in enumerate(self.constraints):
            if constraint['type'] == 'inequality':
                violation = max(0, constraint['function'](individual))**3
            elif constraint['type'] == 'equality':
                violation = max(0, abs(constraint['function'](individual)))**3

            total_violation += violation
            if violation > 0:  # Only add violated constraints
                violated_constraints.append(i)

        return total_violation, violated_constraints

    def stochasticRanking(self, population):
        """Sort the population by feasibility and fitness using stochastic ranking with probability factor F."""
        popSize = len(population)
        fit = [self.fitnessWithPenalty(ind) for ind in population]
        penalties = [self.evaluateConstraints(ind)[0] for ind in population]
        
        # Sort population based on stochastic ranking
        for i in range(popSize):
            swapped = False
            for j in range(popSize - 1):
                u = np.random.rand()
                # Check stochastic condition or equal penalties
                if (u < self.F or penalties[j] == penalties[j + 1]) and penalties[j] == 0 :
                    # Sort by fitness if condition met
                    if fit[j] > fit[j + 1]:
                        population[j], population[j + 1] = population[j + 1], population[j]
                        fit[j], fit[j + 1] = fit[j + 1], fit[j]
                        penalties[j], penalties[j + 1] = penalties[j + 1], penalties[j]
                        swapped = True
                else:
                    # Sort by penalty otherwise
                    if penalties[j] > penalties[j + 1]:
                        population[j], population[j + 1] = population[j + 1], population[j]
                        fit[j], fit[j + 1] = fit[j + 1], fit[j]
                        penalties[j], penalties[j + 1] = penalties[j + 1], penalties[j]
                        swapped = True

            if not swapped:
                break

        return population


    def mutation(self, target):
        indices = [i for i in range(self.popSize) if i != target]
        a_idx, b_idx, c_idx = np.random.choice(indices, 3, replace=False)
        a, b, c = self.population[a_idx], self.population[b_idx], self.population[c_idx]
        mutant = np.clip(a + self.F * (b - c), [low for (low, high) in self.bounds], [high for (low, high) in self.bounds])
        return mutant

    def crossover(self, target, mutant):
        trial = np.copy(self.population[target])
        for i in range(self.dims):
            if np.random.rand() < self.P_r:
                trial[i] = mutant[i]
        return trial
    
    def fitnessWithPenalty(self, individual):
        """Evaluate fitness with penalty for constraint violations."""
        fitness = self.evaluateFitness(individual)
        violation, _ = self.evaluateConstraints(individual)
        
        # Add a penalty if the individual is infeasible
        if violation > 0:
            fitness += self.penalty_factor * violation
            
        return fitness

    def optimize(self):
        for iteration in range(self.maxIter):
            newPopulation = []
            for i in range(self.popSize):
                mutant = self.mutation(i)
                trial = self.crossover(i, mutant)

                # Evaluate fitness with penalty
                fitness_trial = self.fitnessWithPenalty(trial)
                fitness_current = self.fitnessWithPenalty(self.population[i])

                if fitness_trial < fitness_current:
                    newPopulation.append(trial)
                else:
                    newPopulation.append(self.population[i])

            self.population = self.stochasticRanking(newPopulation)

            bestIndex = np.argmin([self.fitnessWithPenalty(ind) for ind in self.population])
            if self.fitnessWithPenalty(self.population[bestIndex]) < self.bestObj:
                self.bestObj = self.fitnessWithPenalty(self.population[bestIndex])
                self.bestVector = self.population[bestIndex]

        return self.bestVector, self.bestObj

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



# Run the optimizer for a specific problem

problem = PROBLEMS["G6"]
optimizer = DifferentialEvolution(
    function=problem["Equation"],
    bounds=problem["Bounds"],
    constraints=problem["Constraints"],
    popSize=50,
    maxIter=200,
    F=0.1,
    P_r=0.1
)

best_solution, best_value = optimizer.optimize()
print("Best Solution:", best_solution)
print("Best Value:", best_value)

# Check constraints for the best solution
violation, violated_constraints = optimizer.evaluateConstraints(best_solution)
if violation > 0:
    print("Total Constraint Violation:", violation)
    print("Violated Constraints Indexes:", violated_constraints)
else:
    print("All constraints satisfied.")


# Run the algorithm for each problem 30 times and store the results in separate CSV files
for problem_name, problem in PROBLEMS.items():
    results = []

    for run in range(30):
        de = DifferentialEvolution(
            function=problem["Equation"],
            bounds=problem["Bounds"],
            constraints=problem["Constraints"],
            popSize=20,
            maxIter=100,
            F=0.5,
            P_r=0.7
        )
        best_solution, best_value = de.optimize()
        constraint_violation, violated_constraints = de.evaluateConstraints(best_solution)
        results.append({
            "Problem": problem_name,
            "Run": run + 1,
            "Best Solution": best_solution,
            "Best Value": best_value,
            "Constraint Violation": constraint_violation,
            "Violated Constraints Indexes": violated_constraints
        })

    # Convert results to DataFrame and save to CSV for the current problem
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results_de_constrained/{problem_name}_results.csv", index=False)

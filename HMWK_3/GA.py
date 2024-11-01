import numpy as np

class RealGA:
    def __init__(self, lower_bound, upper_bound, var_num, objective_func, popu_size, nc=20, nm=20, Pc=0.9, Pm=0.1):
        self._x_min = lower_bound
        self._x_max = upper_bound
        self._vars = var_num
        self._objective_func = objective_func
        self._popu_size = popu_size
        self._nc = nc
        self._eta_m = nm
        self._Pc = Pc
        self._Pm = Pm
        self._population = []
        self.initialize_population()

    def initialize_population(self):
        """Randomly initialize population."""
        for _ in range(self._popu_size):
            chromosome = [np.random.uniform(self._x_min, self._x_max) for _ in range(self._vars)]
            self._population.append(chromosome)

    def _limit_individual(self, individual):
        """Ensure individuals are within the defined bounds."""
        return [max(min(gene, self._x_max), self._x_min) for gene in individual]

    def _evaluate_fitness(self):
        """Evaluate fitness of the population with constraint handling."""
        fitness = []
        for individual in self._population:
            violation = 0
            for constraint in constraints:
                constraint_value = constraint(individual)
                if constraint_value > 0:  # Constraint is violated
                    violation += constraint_value
            fitness_value = self._objective_func(individual)
            # Penalize for constraint violations
            fitness.append(fitness_value + violation)
        return np.array(fitness)

    def _selection(self, fit_list):
        """Select individuals using binary tournament selection."""
        evaluated_individuals = list(zip(self._population.copy(), fit_list))
        parents = []
        for _ in range(self._popu_size):
            shuffled_pop = list(evaluated_individuals)
            np.random.shuffle(shuffled_pop)

            rand_choice1, rand_choice2 = np.random.choice(len(shuffled_pop), 2, replace=False)
            candidate1, candidate2 = shuffled_pop[rand_choice1], shuffled_pop[rand_choice2]

            fitness_cand1, fitness_cand2 = candidate1[1], candidate2[1]
            parents.append(candidate1[0] if fitness_cand1 < fitness_cand2 else candidate2[0])

        return parents

    def _crossover(self, parents):
        """Perform simulated binary crossover (SBX)."""
        new_population = []
        for parent1, parent2 in zip(parents[::2], parents[1::2]):
            if np.random.random() <= self._Pc:
                u = np.random.uniform()
                if u <= 0.5:
                    beta_m = (2 * u) ** (1 / (self._nc + 1))
                else:
                    beta_m = (1 / (2 * (1 - u))) ** (1 / (self._nc + 1))

                H1 = 0.5 * ((np.array(parent1) + np.array(parent2)) - beta_m * np.abs(np.array(parent2) - np.array(parent1)))
                H2 = 0.5 * ((np.array(parent1) + np.array(parent2)) + beta_m * np.abs(np.array(parent2) - np.array(parent1)))
                new_population.extend([self._limit_individual(H1), self._limit_individual(H2)])
            else:
                new_population.extend([parent1, parent2])

        return new_population

    def _mutation(self, population):
        """Perform parameter-based mutation."""
        new_population = []
        for individual in population:
            if np.random.random() <= self._Pm:
                i = np.random.randint(0, len(individual))
                gene = individual[i]
                u = np.random.random()

                delta = min((gene - self._x_min), (self._x_max - gene)) / (self._x_max - self._x_min)
                if u <= 0.5:
                    delta_q = (2 * u + (1 - 2 * u) * (1 - delta) ** (self._eta_m + 1)) ** (1 / (self._eta_m + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta) ** (self._eta_m + 1)) ** (1 / (self._eta_m + 1))

                mutated_gene = gene + delta_q * (self._x_max - self._x_min)
                individual[i] = mutated_gene

            new_population.append(self._limit_individual(individual))

        return new_population

    def run(self, generations):
        """Run the Genetic Algorithm."""
        best_solutions = []
        
        for generation in range(generations):
            fit_list = self._evaluate_fitness()
            parents = self._selection(fit_list)
            offspring = self._crossover(parents)
            self._population = self._mutation(offspring)

            best_fitness_index = np.argmin(fit_list)
            best_solution = self._population[best_fitness_index]
            best_solutions.append(best_solution)

        return best_solutions


# Define the objective function and constraints for "G6"
def objectiveFunction(x):
    return (x[0] - 10) ** 3 + (x[1] - 20) ** 3

def constraint1(x):
    return (x[0] - 5) ** 2 + (x[1] - 5) ** 2 - 100  # g1(x)

def constraint2(x):
    return -(x[0] - 6) ** 2 - (x[1] - 5) ** 2 + 82.81  # g2(x)

constraints = [constraint1, constraint2]

# Setting bounds and initializing GA
lower_bounds = [13, 0]
upper_bounds = [100, 100]
var_num = 2
population_size = 100
num_generations = 100

ga = RealGA(lower_bounds, upper_bounds, var_num, objectiveFunction, population_size)

# Execute the Genetic Algorithm and collect results
num_executions = 30
results = []

for _ in range(num_executions):
    best_solutions = ga.run(num_generations)
    final_solution = best_solutions[-1]
    constraints_violated = [constraint(final_solution) for constraint in constraints]

    feasible = all(violation <= 0 for violation in constraints_violated)
    degree_of_violation = [max(0, violation) for violation in constraints_violated]
    
    results.append((final_solution, feasible, degree_of_violation))

# Reporting the results
for idx, (solution, feasible, violation) in enumerate(results):
    print(f"Execution {idx + 1}: Best Solution = {solution}, Feasible = {feasible}, Violations = {violation}")

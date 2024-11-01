import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DifferentialEvolution:
    def __init__(self, function, bounds, popSize = 20, maxIter = 100, F = 0.5, P_r = 0.75):
        self.popSize = popSize
        self.bounds = bounds
        self.maxIter = maxIter
        self.F = F
        self.P_r = P_r
        self.dims = len(bounds)
        self.population = self.initPopulation()
        self.evaluateFitness = function
        self.objValues = [self.evaluateFitness(individual) for individual in self.population]
        self.bestVector = self.population[np.argmin(self.objValues)]
        self.bestObj = np.min(self.objValues)

    def initPopulation(self):
        population = np.random.rand(self.popSize, len(self.bounds))
        constrainedPopulation = self.bounds[:, 0] + (population * (self.bounds[:, 1] - self.bounds[:, 0])) 
        return constrainedPopulation

    def mutation(self, index):
        indexes = np.random.choice(np.delete(np.arange(self.popSize), index), 3, replace = False)
        x1, x2, x3 = self.population[indexes]
        return x1 + self.F * (x2 - x3)

    def crossover(self, mutated, parent):
        p = np.random.rand(self.dims)
        offspring = []
        for i in range(self.dims):
            offspring.append(mutated[i] if p[i] < self.P_r else parent[i])
        return offspring

    def checkBounds(self, individual):
        return [np.clip(individual[i], self.bounds[i, 0], self.bounds[i, 1]) for i in range(len(self.bounds))]

    def run(self):
        generations = []
        for i in range(self.maxIter):
            for j in range(self.popSize):
                mutated = self.mutation(j)
                mutated = self.checkBounds(mutated)

                trial = self.crossover(mutated, self.population[j])

                objTarget = self.evaluateFitness(self.population[j])
                objTrial = self.evaluateFitness(trial)

                if objTrial < objTarget:
                    self.population[j] = trial
                    self.objValues[j] = objTrial

            currentBestObj = np.min(self.objValues)
            if currentBestObj < self.bestObj:
                self.bestVector = self.population[np.argmin(self.objValues)]
                self.bestObj = currentBestObj
                print(f"Generation: {i}  |  {np.around(self.bestVector, decimals=5)} = {self.bestObj:.5f}")
            generations.append((i, self.bestObj))
        return self.bestVector, self.bestObj, generations

class Report:
    def __init__(self):
        self.best_individuals = []

    def add_best_individual_at_generation(self, generation, execution, best_fitness):
        self.best_individuals.append((generation, execution, best_fitness))

    def save_to_csv(self, output_file):
        df = pd.DataFrame(self.best_individuals, columns=["Generation", "Execution", "Best Fitness"])
        df.to_csv(output_file, index=False)

    def plot_convergence(self, output_folder, func_name):
        global_minima = {
            "Layeb05 (n=2)": -6.907,
            "Layeb10 (n=2)": 0,
            "Layeb15 (n=2)": 0,
            "Layeb18 (n=2)": -6.907
        }

        if func_name in global_minima:
            plt.axhline(global_minima[func_name], color='r', linestyle='--', label='Global Minimum')

        for execution in range(len(self.best_individuals) // num_executions):
            execution_data = [record[2] for record in self.best_individuals if record[1] == execution]
            plt.plot(execution_data, label=f'Execution {execution + 1}')

        plt.xscale('log')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title(f'Convergence Plot for {func_name} DE')
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{func_name.replace(' ', '_')}_convergence.png")
        plt.clf() 



if __name__ == "__main__":
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
            term = np.tanh(2 * abs( x[i]) - x[i + 1]**2 - 1)
            if term < 0:
                term = 0
            result += 10 * np.sqrt(term) + abs(np.exp(x[i] * x[i + 1]) - 1)
        return result

    def layeb18(x):
        n = len(x)
        result = 0
        for i in range(n - 1):
            result += np.log(np.abs(np.cos(2 * x[i] * x[i+1] / np.pi)) + 0.001) / (np.abs(np.sin(x[i] + x[i+1]) * np.cos(x[i])) + 1)
        return result

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

    # Set the number of executions and the output folder for plots
    num_executions = 30
    output_folder = "experiment_results_de"
    
    # Create output folder if it doesn't exist
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    

    # Run Differential Evolution for each problem
    for func, bounds, func_name, dims in problems:
        report = Report()
        for execution in range(num_executions):
            de = DifferentialEvolution(func, np.array([bounds] * dims), popSize=50, maxIter=100)
            best_vector, best_obj, generations = de.run()
            for generation, best_fitness in generations:
                report.add_best_individual_at_generation(generation, execution, best_fitness)
        report.plot_convergence(output_folder, func_name)
        report.save_to_csv(f"{output_folder}/{func_name.replace(' ', '_')}_results.csv")

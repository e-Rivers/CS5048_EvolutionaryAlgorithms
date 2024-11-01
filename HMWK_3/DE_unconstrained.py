import numpy as np

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

        return self.bestVector, self.bestObj

# TEST EXAMPLE
function = lambda x: x[0]**2 + x[1]**2
bounds = np.array([[-5, 5], [-5.0, 5.0]])

DEoptimizer = DifferentialEvolution(function, bounds)
solution = DEoptimizer.run()
print(f"\nBest Solution Found: f{np.around(solution[0], decimals=5)} = {solution[1]:.5f}")

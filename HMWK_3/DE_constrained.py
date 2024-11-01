import numpy as np

class DifferentialEvolution:
    def __init__(self, function, bounds, constraints, popSize=20, maxIter=100, F=0.5, P_r=0.75):
        self.popSize = popSize
        self.bounds = bounds
        self.constraints = constraints
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
        indexes = np.random.choice(np.delete(np.arange(self.popSize), index), 3, replace=False)
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

    def stochasticRanking(self, population):
        popSize = len(population)
        rank = list(range(popSize))

        for i in range(popSize):
            for j in range(popSize-1):
                a, b = population[rank[j]], population[rank[j+1]]
                
                if self.evaluateConstraints(a) and not self.evaluateConstraints(b):
                    rank[j], rank[j+1] = rank[j+1], rank[j]
                elif self.evaluateConstraints(a) == self.evaluateConstraints(b):
                    if self.evaluateFitness(a) > self.evaluateFitness(b):
                        rank[j], rank[j+1] = rank[j+1], rank[j]

        return [population[r] for r in rank]

    def evaluateConstraints(self, individual):
        return all(constraint(individual) for constraint in self.constraints)

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

            # Stochastic ranking for constraint handling
            # To replace regular selection mechanism
            self.population = self.stochasticRanking(self.population)
            self.objValues = [self.evaluateFitness(individual) for individual in self.population]

            currentBestObj = np.min(self.objValues)
            if currentBestObj < self.bestObj:
                self.bestVector = self.population[np.argmin(self.objValues)]
                self.bestObj = currentBestObj
                print(f"Generation: {i}  |  {np.around(self.bestVector, decimals=5)} = {self.bestObj:.5f}")

        return self.bestVector, self.bestObj

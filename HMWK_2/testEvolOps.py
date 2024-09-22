"""
# This file is just to validate the correct functioning of the evolutionary operators:
    - Mutation
    - Selection
    - Crossover
"""

from algorithms import *
import numpy as np

# Testing Setup
realGA = RealGA(-2.0, 6.0, 2, 2)
parent1 = np.array([2.3, 4.5, -1.2, 0.8])
parent2 = np.array([1.4, -0.2, 6.7, 4.8])
individual = np.array([2.3, 4.5, -1.2, 0.8])

# Testing SBX Crossover Operator
sbxObtained = realGA._crossover(parent1, parent2, 0.42)
sbxTruthValue = (np.array([1.42,-0.067,-0.97,0.9129]), np.array([2.27,4.37,6.48,4.69]))

# Testing PM Mutation Operator
pmObtained = realGA._mutation(individual, 20, 0.72, 2)
pmTruthValue = np.array([2.3, 4.5, -1.16175, 0.8])

if __name__ == "__main__":
    print(sbxObtained)
    print(sbxTruthValue)

    print(pmObtained)
    print(pmTruthValue)

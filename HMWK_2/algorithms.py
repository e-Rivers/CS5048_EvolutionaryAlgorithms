import numpy as np
import random
from abc import ABC, abstractmethod

# Father class: GeneticAlgorithm
class GeneticAlgorithm(ABC):

    #    def __init__(self, popu_size, num_generations, ind_size, Pc=0.9, Pm=0.1):
    def __init__(self, lowerBound, upperBound, Pc=0.9, Pm=0.1):
        self._xU = upperBound
        self._xL = lowerBound
        #        self._pop_size = popu_size
        #        self._num_generations = num_generations
        #        self._ind_size = ind_size
        #        self._population = self.initialize_population()
        #        self._Pc = Pc   # Crossover Probability  
        #        self._Pm = Pm   # Mutation Probability  

    def initialize_population(self):
        raise NotImplementedError

    def run(self):
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


    


#############################################################
# Child class for Binary Encoding
class BinaryGA(GeneticAlgorithm):
    def initialize_population(self):
        
        return 

    

    def crossover(self, parent1, parent2):
        # Single-point crossover
       
        return

    def selection(self):
        
        return  
    
    def mutation(self, individual):
        # Binary mutation (flip bits)
        
        return individual

    def __str__(self):
        return "Binary Encoding"

#############################################################
# Child class for Real Encoding
class RealGA(GeneticAlgorithm):

    def __init__(self, nc = 20, nm = 20):
        super().__init__()
        self._nc = nc
        self._nm = nm

    def initialize_population(self):
        return
    
    # Binary tournament selection
    def _selection(self):
        
        return None

    # Simulated Binary Crossover (SBX)
    def _crossover(self, parent1, parent2, uTest = None):
        # Step 1. Compute a random number u between 0 and 1
        u = np.random.uniform() if uTest == None else uTest

        # Step 2. Compute beta
        if u <= 0.5:
            beta = (2*u)**(1 / (self._nc + 1))
        else:
            beta = (1 / (2*(1 - u)))**(1 / (self._nc + 1))

        # Step 3. Produce children
        H1 = 0.5 * ((leftSide := (parent1 + parent2)) - (rightSide := beta*np.abs(parent2 - parent1)))
        H2 = 0.5 * (leftSide + rightSide)

        return H1, H2

    # Parameter-based mutation
    def _mutation(self, individual, t, uTest = None, iTest = None):
        # Step 0. Randomly select the gene to be mutated
        i = np.random.uniform() if iTest == None else iTest
        gene = individual[i]

        # Step 1. Compute a random number u between 0 and 1
        u = np.random.uniform() if uTest == None else uTest

        # Step 2. Compute delta sub q
            # Step 2.1. Compute eta sub m
        eta = 100 + t
            # Step 2.2. Compute delta
        delta = min((gene - self.xL), (self.xU - gene)) / (self._xU - self._xL)
            # Step 2.3. Now compute delta sub q
        if u <= 0.5:
            delta_q = (2*u + (1-2*u)*(1-delta)**(eta+1))**(1 / (eta+1)) - 1
        else:
            delta_q = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta)**(eta+1))**(1 / (eta+1))

        # Step 3. Perform the mutation 
            # Step 3.1. Compute delta max
        deltaMax = self._xU - self._xL
            # Step 3.2. Mutate the gene
        mutatedGene = gene + delta_q*deltaMax
            # Step 3.3. Replace the old gene with the mutated one
        individual[i] = mutatedGene

        return individual 

    def __str__(self):
        return "Real Encoding"


if __name__ == "__main__":
    ##
    print("esto aún no se ha terminado")

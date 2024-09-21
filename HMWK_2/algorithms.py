import numpy as np
import random
from abc import ABC, abstractmethod

# Father class: GeneticAlgorithm
class GeneticAlgorithm(ABC):

    def __init__(self, popu_size, num_generations, ind_size, Pc=0.9, Pm=0.1):
        self._pop_size = popu_size
        self._num_generations = num_generations
        self._ind_size = ind_size
        self._population = self.initialize_population()
        self._Pc = Pc   # Crossover Probability  
        self._Pm = Pm   # Mutation Probability  

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
        
        return "Hello"

    # Simulated Binary Crossover (SBX)
    def _crossover(self, parent1, parent2):
        # Step 1. Compute a random number u between 0 and 1
        u = np.random.uniform()

        # Step 2. Compute beta
        if u <= 0.5:
            beta = (2*u)*(1/(self._nc+1))
        else:
            beta = (1/(2*(1-u)))**(1/(self._nc+1))

        # Step 3. Produce children
        


        return

    # Parameter-based mutation
    def _mutation(self, individual):
        return "Hello"

    def __str__(self):
        return "Real Encoding"


if __name__ == "__main__":
    ##
    print("esto aún no se ha terminado")

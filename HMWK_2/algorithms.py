import numpy as np
import random
from abc import ABC, abstractmethod

# Father class: GeneticAlgorithm
class GeneticAlgorithm(ABC):

    def __init__(self, popu_size, num_generations, ind_size, Pc=0.9, Pm=0.1):
        self.pop_size = popu_size
        self.num_generations = num_generations
        self.ind_size = ind_size
        self.population = self.initialize_population()
        self.Pc = Pc  
        self.Pm = Pm  

    def initialize_population(self):
        raise NotImplementedError
    
    @abstractmethod
    def crossover(self, parent1, parent2):
        raise NotImplementedError

    @abstractmethod
    def mutation(self, individual):
        raise NotImplementedError
    
    @abstractmethod
    def selection(self):
        raise NotImplementedError


    


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

# Child class for Real Encoding
class RealGA(GeneticAlgorithm):
    def initialize_population(self):
        return
    def selection(self):
        # Binary tournament selection
        
        return

    def crossover(self, parent1, parent2):
        return

    def mutation(self, individual):
        # Parameter-based mutation
        return 

    def __str__(self):
        return "Real Encoding"


if __name__ == "__main__":
    ##
    print("esto aún no se ha terminado")

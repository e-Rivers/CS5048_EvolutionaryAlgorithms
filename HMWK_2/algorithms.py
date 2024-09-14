import numpy as np
import random

# Father class: GeneticAlgorithm
class GeneticAlgorithm:
    def __init__(self, popu_size, num_generations, ind_size):
        self.pop_size = popu_size
        self.num_generations = num_generations
        self.ind_size = ind_size
        self.population = self.initialize_population()

    def initialize_population(self):
        raise NotImplementedError

    def crossover(self, parent1, parent2):
        raise NotImplementedError

    def mutation(self, individual):
        raise NotImplementedError
    
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



if __name__ == "__main__":
    ##
    print("esto aún no se ha terminado")

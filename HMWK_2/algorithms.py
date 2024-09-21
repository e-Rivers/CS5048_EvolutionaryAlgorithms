import numpy as np
import random

# Father class: GeneticAlgorithm
class GeneticAlgorithm:

    def __init__(self, popu_size, num_generations, ind_size, cross_rate, mutation_rate):
        self.pop_size = popu_size
        self.num_generations = num_generations
        self.ind_size = ind_size
        self.population = self.initialize_population()
        self.cross_rate = cross_rate  
        self.mutation_rate = mutation_rate  

        

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
    def number_bits(x_min, x_max):
        """
        Determine the number of bits needed to encode the solution

        Input:
        x_min: Minimum value that the solution can take
        x_max: Maximum value that the solution can take

        Output:
        num_bits: Represents the length of the solution
        """
        num_bits = (x_max - x_min)*10**3
        num_bits = math.log(num_bits,2)
        num_bits = math.floor(num_bits + 0.99)
        return number_bits

    def initialize_population(self, num_bits_1, num_bits_2, popu_size):
        """
        Function to initialize population by encoding variables 

        Input:
        num_bits_1: number of bits needed to encode x1
        num_bits_2: number of bits to encode x2
        popu_size: number of individuals (genomes or chromosomes)

        output:
        population: list of randomly generated solutions
        """

        genome1 = ""
        genome2 = ""
        population = []
        for i in range(popu_size):
            for j in range(num_bits_1):
                gen = random.randint(0,1)
                gen = str(gen)
                genome1 += gen
            for k in range(num_bits_2):
                gen = random.randint(0,1)
                gen = str(gen)
                genome2 += gen
        return 

    

    def crossover(self, parent1, parent2):
        # Single-point crossover
       
        return

    #Roulette wheel
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

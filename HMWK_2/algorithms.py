import numpy as np
import random
from abc import ABC, abstractmethod

# Father class: GeneticAlgorithm
class GeneticAlgorithm(ABC):

    #    def __init__(self, popu_size, num_generations, ind_size, Pc=0.9, Pm=0.1):
    def __init__(self, lowerBound, upperBound, Pc=0.9, Pm=0.1):
        self._xU = upperBound
        self._xL = lowerBound
        self._population = None
        #        self._pop_size = popu_size
        #        self._num_generations = num_generations
        #        self._ind_size = ind_size
        #        self._population = self.initialize_population()
        #        self._Pc = Pc   # Crossover Probability  
        #        self._Pm = Pm   # Mutation Probability  

    def initialize_population(self):
        raise NotImplementedError

    def run(self, verbose=False):
        pass

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
        chromosome = genome1 + genome2
        population.append(chromosome)
        genome1 = ""
        genome2 = ""
        return population

    
    def decoder(self):

        # aaaaaaa

        return

    def function_evaluation(self):
        return

    def probability_percen(fit_list):
        """
        To calculate the probability for each individual

        input:
        fit_list : list of the fitness of each individual of the population

        output:
        cum_list: list of the cumulative probability of selecting each individual
        
        """
        #Initialize the total fitness (f) and cumulative probability (q) to 0
        f=0
        q=0

        # Lists to store cumulative probabilities a probabilities of selection
        cumu_probability = []
        probability = []

        #Step 2 calculate the total fitness (f)
        for i in fit_list:
            f += (i-min(fit_list))
        
        # Step 3 calculate the probability for each element
        #In case that f is equal to zero all the individuals will have the same probability
        for i in fit_list:
            if f == 0:
                new_prob = (i-min(fit_list))/len(fit_list)
            else:
                new_prob = (i-min(fit_list))/f
            probability.append(new_prob)
        
        #Step 4 calculate the cumulative probability
        for i in probability:
            q += i  
            cumu_probability.append(q)  

        return cumu_probability

    #Roulette wheel
    def selection(self, cumu_probability, population): 
        """
        To select individuals using the roulette wheel method

        input:
        cumu_probability: list of the cummulative probabilitties of each individual
        probabilities: the individuals (or chromosomes)

        output:
        selected_individuals: list of selected chromosomes
        """
        # step 5 get a pseudo-random number between 0 and 1
        # then

        selected_individuals =[]
        for i in range(len(population)):
            r = random.random()

            # Find the first individual whose cumulative probability is greater than or equal to r
            for k in range(len(population)):
                if r <= cumu_probability[k]:
                    selected_individuals.append(population[k])
                    break   


        return selected_individuals

    def crossover(self, parents):
        """
        Function to perfomr single point crossover

        input:
        parents: list of selected chromosomes

        output:
        childrens: list of new chromosomes after the crossover
        """

        # select the position 
        children =[]
        position_cross = random.randint(1, len(parents[0])-1)

        # Make the crossover
        if len(parents) > 1:
            parent1 = parents[0]
            parent2 = parents[1]
            chromosome1 = parent1[:position_cross] + parent2[position_cross:]
            chromosome2 = parent2[:position_cross] + parent1[position_cross:]
        children.append(chromosome1)
        children.append(chromosome2)

        return children

      
    
    def mutation(self, individual):
        # Binary mutation 
        
        return individual

    def __str__(self):
        return "Binary Encoding"

#############################################################
# Child class for Real Encoding
class RealGA(GeneticAlgorithm):

    def __init__(self, lowerBound, upperBound, nc = 20, nm = 20):
        super().__init__(lowerBound, upperBound)
        self._nc = nc
        self._nm = nm

    def initialize_population(self):
        return
    
    # Binary tournament selection
    def _selection(self, population):
        # Step 1. Shuffle individuals

            
        
        return None

    # Simulated Binary Crossover (SBX)
    def _crossover(self, parent1, parent2, uTest = None):
        # Step 1. Compute a random number u between 0 and 1
        u = np.random.uniform() if uTest == None else uTest

        # Step 2. Compute beta_m
        if u <= 0.5:
            beta_m = (2*u)**(1 / (self._nc + 1))
        else:
            beta_m = (1 / (2*(1 - u)))**(1 / (self._nc + 1))

        # Step 3. Produce children
        H1 = 0.5 * ((leftSide := (parent1 + parent2)) - (rightSide := beta_m*np.abs(parent2 - parent1)))
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
        eta_m = 100 + t
        delta = min((gene - self._xL), (self._xU - gene)) / (self._xU - self._xL)
        if u <= 0.5:
            delta_q = (2*u + (1-2*u)*(1-delta)**(eta_m+1))**(1 / (eta_m+1)) - 1
        else:
            delta_q = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta)**(eta_m+1))**(1 / (eta_m+1))

        # Step 3. Perform the mutation 
        deltaMax = self._xU - self._xL
        mutatedGene = gene + delta_q*deltaMax
        individual[i] = mutatedGene

        return individual 

    def __str__(self):
        return "Real Encoding"


if __name__ == "__main__":
    ##
    print("esto aún no se ha terminado")

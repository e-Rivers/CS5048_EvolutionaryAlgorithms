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


    


#####################################################################
################## Child class for Binary Encoding ##################
#####################################################################

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
        # we calculate the length of the chromosomes needed to encode the solution using 4 numbers after the decimal point
        num_bits = (x_max - x_min)*10**4
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

        # Strings to store the binary encoding for each variable (x1 and x2)
        genome1 = ""
        genome2 = ""
        population = []
        # We will do the same procedure for each individual on the population
        for i in range(popu_size):
            for j in range(num_bits_1):
                gene = random.randint(0,1)
                gene = str(gene)
                genome1 += gene
            for k in range(num_bits_2):
                gene = random.randint(0,1)
                gene = str(gene)
                genome2 += gene
        chromosome = genome1 + genome2
        population.append(chromosome)
        genome1 = ""
        genome2 = ""
        return population

    
    def decoder(self, population, x1_min, x2_min, x1_max, x2_max, num_bits_1, num_bits_2):

        """
        Function to decode from binary numbers to real ones

        Input:
        population 
        x1_min : Minimum possible value for x1
        x2_min : Minimum possible value for x2
        x1_max : Maximum possible value for x1
        x2_max : Maximum possible value for x1
        num_bits_1 : Number of bits needed to encode x1
        num_bits_2 : Number of bits needed to encode x2

        Output:
        decode_population: list of the real values for each individual (chromosome = [x1, x2])


        """

        # Empty list to store the decoded population
        decode_population = []

        # for each chromosome
        for i in range(len(population)):
            full_chromosome = population[i]
            #extract the genome for x1 and x2
            genome_x1 = full_chromosome[:num_bits_1]
            genome_x2 = full_chromosome[num_bits_1:]

            #convert the binary string for each variable to a decimal integer
            x1_deci = int(genome_x1, 2)
            x2_deci = int(genome_x2, 2)

            #scale the decimal integer to its original range
            x1 = x1_min + x1_deci * ((x1_max - x1_min)/ (2**num_bits_1 - 1))
            x2 = x2_min + x2_deci * ((x2_max - x2_min)/ (2**num_bits_1 - 1))
            x2=round(x2,4)
            x1=round(x1,4)
            decode_population.append([x1,x2])


        return decode_population


    

    #Roulette wheel
    def selection(self, fit_list, population): 
        """
        To select individuals using the roulette wheel method

        input:
        fit_list : list of the fitness of each individual of the population
        probabilities: the individuals (or chromosomes)

        output:
        selected_individuals: list of selected chromosomes
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

      
    
    def mutation(self, population, pm):
        """
        function to mutate genes randomly

        input:
        population: list of chromosomes
        pm: probability of mutation

        output:
        population: list of individuals after mutation
        """ 

        # In each chromosome we will mutate  one gene
        for i in range(len(population)):
            chromosome = population[i]
            gene_number = random.randint(0, len(chromosome))
            gene = chromosome[gene_number]

            if gene == '0':
                gene = '1'
            else:
                gene = '0'
            chromosome_mutated = chromosome[:gene_number] + gene + chromosome[gene_number:]
            population[i] = chromosome_mutated

        
        return population

    def __str__(self):
        return "Binary Encoding"




#############################################################
############# Child class for Real Encoding #################
#############################################################

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

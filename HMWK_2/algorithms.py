import numpy as np
import random
import sympy
from sympy import symbols
import math
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import csv



#####################################################################
################## Child class for Binary Encoding ##################
#####################################################################

class BinaryGA():

    def __init__(self, lowerBound, upperBound, func, Pc= 0.9,):
        self._func = func
        

    def run(self, popu_size, num_generations, bounds):
        """
        Function to run the GA with binary encoding

        Input:
        self:
        popu_size:
        num_generations:
        bounds:

        Output:
        best_solutions: a list of the best solution found in each generation

        """
        # Initialize population
        num_bits = [self.number_bits(b[0], b[1]) for b in bounds]
        first_population = self.initialize_population(num_bits, popu_size)
        population = list(first_population)
        best_solutions = []
        print(population)


        for generation in range(num_generations):
            # Decode population
            decoded_population = self.decoder(population, bounds, num_bits)
            print(decoded_population)
        
            
            fit_list = self.function_evaluation(decoded_population, self._func)

            # Selection
            selected_individuals = self._selection(fit_list, population)

            # Crossover
            children = self._crossover(selected_individuals)

            # Mutation
            mutated_population = self._mutation(children)

            # Update population
            population = list(mutated_population)

            # print best fitness in the generation
            min_index = fit_list.index(min(fit_list))

            # Store the best fitness and its corresponding decoded point
            best_fitness = fit_list[min_index]
            #print(f"Este es el mejor fitness {best_fitness}, generación {generation}")
            best_solutions.append(best_fitness)

            
            
        return best_solutions

    def number_bits(self, x_min, x_max):
        """
        Determine the number of bits needed to encode the solution.
        
        Input:
        x_min -- Minimum value that the solution can take
        x_max -- Maximum value that the solution can take
        
        Output:
        num_bits -- Represents the length of the solution
        """
        num_bits = (x_max - x_min) * 10**4
        num_bits = math.log(num_bits, 2)
        num_bits = math.floor(num_bits + 0.99)
        return num_bits

    def initialize_population(self, num_bits, popu_size):
        """
        Function to initialize the population by encoding variables.
        
        Input:
        num_bits -- List of bits needed to encode each variable
        popu_size -- Number of individuals (genomes or chromosomes)
        
        Output:
        population -- List of randomly generated solutions
        """
        population = []
        # Initialize each individual in the population
        for _ in range(popu_size):
            chromosome = ''.join(''.join(str(random.randint(0, 1)) for _ in range(bits)) for bits in num_bits)
            population.append(chromosome)
        return population

    
    def decoder(self, population, bounds, num_bits):
        """
        Function to decode from binary numbers to real ones.
        
        Input:
        population -- List of binary encoded individuals
        bounds -- List of tuples representing min and max bounds for each variable
        num_bits -- List of bits for each variable
        
        Output:
        decode_population -- List of decoded real values for each individual
        """
        decode_population = []
        for individual in population:
            values = []
            pos = 0
            # Decode each variable
            for idx, bits in enumerate(num_bits):
                genome = individual[pos:pos+bits]
                decimal_value = int(genome, 2)
                min_bound, max_bound = bounds[idx]
                # Scale the decimal value to its original range
                real_value = min_bound + decimal_value * ((max_bound - min_bound) / (2**bits - 1))
                values.append(round(real_value, 4))
                pos += bits
            decode_population.append(values)
        return decode_population


    def function_evaluation(self, decode_population, func):
        """
        function to evaluate the population into the objective function

        input:
        decode population: list of the real values for each individual (chromosome = [x1, x2])
        fun: objective function

        ----------

        output:
        fit_list: list of the fitness of each individual
        """

        fit_list = []
        for i in decode_population:
            current_value = func(i)
            fit_list.append(current_value)
            

        return fit_list

    #Roulette wheel
    def _selection(self, fit_list, population): 
        """
        To select individuals using the roulette wheel method

        input:
        fit_list : list of the fitness of each individual of the population
        probabilities: the individuals (or chromosomes)

        ----------

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
        #revert fitness values
        for i in fit_list:
            i_new = max(fit_list) - i + 1e-6
            f += i_new
        
        # Step 3 calculate the probability for each element
        #In case that f is equal to zero all the individuals will have the same probability
        for i in fit_list:
            if f == 0:
                new_prob = 1/len(fit_list)
            else:
                new_prob = (max(fit_list) - i + 1e-6)/f
            probability.append(new_prob)
        
        #Step 4 calculate the cumulative probability
        for i in probability:
            q += i  
            cumu_probability.append(q)

        print("probabilidad",probability)
        print(fit_list)    
        print("cumul",cumu_probability)
        # step 5 get a pseudo-random number between 0 and 1


        selected_individuals =[]
        for i in range(len(population)):
            r = random.random()

            # Find the first individual whose cumulative probability is greater than or equal to r
            for k in range(len(population)):
                if r <= cumu_probability[k]:
                    selected_individuals.append(population[k])
                    break  
                 


        return selected_individuals

    def _crossover(self, parents, pc=0.9):
        """
        Function to perfomr single point crossover

        input:
        parents: list of selected chromosomes
        pc : probability of crossover (0.9 as default)

        output:
        parents: list of new  population including chromosomes after the crossover
        """

        

        #empty lists for selected individual to crossover and childrens
        individuals_cross = []
        index_cross = []
        children =[]

        #select individuals random to perform the crossover
        for i in range(0, len(parents), 2):
            r = random.random()
            if r < pc:
                individuals_cross.append(parents[i])
                individuals_cross.append(parents[i+1])
                index_cross.append(i)
                index_cross.append(i+1)


        # Make the crossover
        if len(individuals_cross) > 1:
            for i in range(0, len(individuals_cross)-1,2):
                parent1 = parents[i]
                parent2 = parents[i+1]

                # select the position for crossover
                position_cross = random.randint(1, len(parents[0])-1)

                #perform single point crossover
                chromosome1 = parent1[:position_cross] + parent2[position_cross:]
                chromosome2 = parent2[:position_cross] + parent1[position_cross:]  
            
                children.append(chromosome1)
                children.append(chromosome2)

            cont = 0
            
            for i in range(len(children)):
                parents[index_cross[i]]=children[cont]
                cont +=1


        return parents

      
    
    def _mutation(self, population):
        """
        function to mutate genes randomly

        input:
        population: list of chromosomes

        output:
        population: list of individuals after mutation
        """ 

        # In each chromosome we will mutate  one gene
        for i in range(len(population)):
            chromosome = population[i]
            gene_number = random.randint(0, len(chromosome)-1)
            gene = chromosome[gene_number]

            if gene == '0':
                gene = '1'
            else:
                gene = '0'
            chromosome_mutated = chromosome[:gene_number] + gene + chromosome[(gene_number+1):]
            population[i] = chromosome_mutated

        
        return population

    def __str__(self):
        return "Binary Encoding"




#############################################################
############# Child class for Real Encoding #################
#############################################################

class RealGA():

    def __init__(self, lowerBound, upperBound, func, nc = 20, np = 20):
        self._xL = lowerBound
        self._xU = upperBound
        self._func = func
        self._nc = nc
        self._np = np
        self._eta_m = 20

    # Binary tournament selection
    def _selection(self):
        # Step 1. Shuffle individuals
        shuffledPop = np.random.shuffle(self._population.copy())

        # Step 2. Get two random individuals
        candidate1 = np.random.choice(shuffledPop)
        candidate2 = np.random.choice(shuffledPop)

        # Step 3. Make them compete based on their fitness
        fitnessCand1 = self._getFitness(candidate1)
        fitnessCand2 = self._getFitness(candidate2)

        return candidate1 if fitnessCand1 < fitnessCand2 else candidate2

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
    def _mutation(self, individual, uTest = None, iTest = None):
        # Step 0. Randomly select the gene to be mutated
        i = np.random.uniform() if iTest == None else iTest
        gene = individual[i]

        # Step 1. Compute a random number u between 0 and 1
        u = np.random.uniform() if uTest == None else uTest

        # Step 2. Compute delta sub q
        eta_m = self._eta_m # This actually is calculated as 100 + t, where t = generation num. But for the hwmk, it was required as 20
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

    def run(self):
        # Step 1. Initialize Population
        self._population = self.initialize_population()

        # Step 2. Form the couples for creating children
        couples = []
        for _ in range(len(self._population)//2):
            couples.append((
                    self._selection(),
                    self._selection()
                ))

        # Step 3. Perform crossover only for couples with that probability (Pc)
        newPopulation = []
        for couple in couples:
            crossProb = np.random.uniform()
            if crossProb <= self._Pc:
                newPopulation.extend(self._crossover(*couple))
            else:
                newPopulation.extend(couple)

    def _getFitness(self, individual):
        return self._func(individual)

    def __str__(self):
        return "Real Encoding"
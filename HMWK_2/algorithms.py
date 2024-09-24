import numpy as np
import random
import math
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import csv


# Father class: GeneticAlgorithm
class GeneticAlgorithm(ABC):

    def __init__(self, lowerBound, upperBound, varNum, func, popu_size, Pc=0.9, Pm=0.1):
        self._x_max = upperBound
        self._x_min = lowerBound
        self._func = func
        self._population = []
        self._vars = varNum 
        self._popu_size = popu_size
        self._Pc = Pc           # Probability of crossover
        self._Pm = Pm           # Probability of mutation

    def run(self, num_generations):
        """
        Function to run the GA 

        Input:
        num_generations: Number of generations

        Output:
        best_solutions: a list of the best solution found in each generation

        """
        # Initialize population
        self.initialize_population()
        best_solutions = []

        for generation in range(num_generations):
            
            # Evaluation
            fit_list = self._evaluation(self._getPopulation())

            # Selection
            selected_individuals = self._selection(fit_list)

            # Crossover
            children = self._crossover(selected_individuals)

            # Mutation
            mutated_population = self._mutation(children)

            # Update population
            self._population = list(mutated_population)

            # print best fitness in the generation
            min_index = fit_list.index(min(fit_list))

            # Store the best fitness and its corresponding decoded point
            best_fitness = fit_list[min_index]
            #print(f"Este es el mejor fitness {best_fitness}, generación {generation}")
            best_solutions.append(best_fitness)

        return best_solutions

    def _evaluation(self, population):
        """
        function to evaluate the population into the objective function

        input:

        population: list of the real values for each individual (decoded in the case of binary)

        ----------

        output:
        fit_list: list of the fitness of each individual
        """

        fit_list = []
        for i in population:
            current_value = self._func(i)
            fit_list.append(current_value)
            
        return fit_list
    
    # ABSTRACT METHODS

    @abstractmethod
    def _getPopulation(self):
        raise NotImplementedError

    @abstractmethod
    def initialize_population(self):
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



#####################################################################
################## Child class for Binary Encoding ##################
#####################################################################

class BinaryGA(GeneticAlgorithm):

    def __init__(self, lowerBound, upperBound, varNum, func, popu_size, Pc= 0.9, Pm=0.1):
        super().__init__(lowerBound, upperBound, varNum, func, popu_size, Pc, Pm)
        self._num_bits = self.number_bits()

    def _getPopulation(self):
        return self.decoder()

    def number_bits(self):
        """
        Determine the number of bits needed to encode the solution.
        
        
        Output:
        num_bits : Represents the length of the solution
        """
        num_bits = (self._x_max - self._x_min) * 10**4
        num_bits = math.log(num_bits, 2)
        num_bits = math.floor(num_bits + 0.99)
        return num_bits

    def initialize_population(self):
        """
        Function to initialize the population by encoding variables.
        
        """
        # Initialize each individual in the population
        num_bits = self.number_bits()
        for _ in range(self._popu_size):
            chromosome = ''.join(''.join(str(random.randint(0, 1)) for _ in range(num_bits)) for _ in range(self._vars))
            self._population.append(chromosome)

    
    def decoder(self):
        """
        Function to decode from binary numbers to real ones.
        
        Output:
        decode_population -- List of decoded real values for each individual

        """
        decode_population = []
        for individual in self._population:
            values = []
            pos = 0
            # Decode each variable
            for _ in range(self._vars):
                genome = individual[pos:pos+self._num_bits]
                decimal_value = int(genome, 2)

                # Scale the decimal value to its original range
                real_value = self._x_min + decimal_value * ((self._x_max - self._x_min) / (2**self._num_bits - 1))
                values.append(round(real_value, 4))
                pos += self._num_bits
            decode_population.append(values)
        return decode_population



    #Roulette wheel
    def _selection(self, fit_list): 
        """
        To select individuals using the roulette wheel method

        input:
        fit_list : list of the fitness of each individual of the population

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

        # step 5 get a pseudo-random number between 0 and 1


        selected_individuals =[]
        for i in range(len(self._population)):
            r = random.random()

            # Find the first individual whose cumulative probability is greater than or equal to r
            for k in range(len(self._population)):
                if r <= cumu_probability[k]:
                    selected_individuals.append(self._population[k])
                    break  
                 
        return selected_individuals

    def _crossover(self, parents):
        """
        Function to perfomr single point crossover

        input:
        parents: list of selected chromosomes

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
            if r < self._Pc:
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



#############################################################
############# Child class for Real Encoding #################
#############################################################

class RealGA(GeneticAlgorithm):

    def __init__(self, lowerBound, upperBound, varNum, func, popu_size, nc=20, nm=20, Pc= 0.9, Pm=0.1):
        super().__init__(lowerBound, upperBound, varNum, func, popu_size, Pc, Pm)
        self._nc = nc
        self._eta_m = nm

    def _getPopulation(self):
        return self._population

    def initialize_population(self):
        """"
        Function to randomly initialize population
        """
        for _ in range(self._popu_size):
            chromosome = [np.random.uniform(self._x_min, self._x_max) for _ in range(self._vars)]
            self._population.append(chromosome)

    def _limitIndividual(self, individual):
        """
        Function to make sure that the individuals are inside the range

        Input:
        inividual : a real individual

        Output:
        individual : an individual inside the range
        """
        for gene in range(self._vars):
            if individual[gene] > self._x_max:
                individual[gene] = self._x_max
            elif individual[gene] < self._x_min:
                individual[gene] = self._x_min

        return individual 


    # Binary tournament selection
    def _selection(self, fit_list):
        """
        Function to select individuals using the binary tournament selection method

        Input:
        fit_list: list with the fitness for each individual

        Outpu:
        parents: list of selected individuals
        """

        evaluatedIndividuals = list(zip(self._population.copy(), fit_list))

        parents = []
        for _ in self._population:

            # Step 1. Shuffle individuals
            shuffledPop = list(evaluatedIndividuals)
            np.random.shuffle(shuffledPop)

            # Step 2. Get two random individuals
            randChoice1 = np.random.randint(0, len(shuffledPop))
            randChoice2 = np.random.randint(0, len(shuffledPop))
            candidate1 = shuffledPop[randChoice1]
            candidate2 = shuffledPop[randChoice2]

            # Step 3. Make them compete based on their fitness
            fitnessCand1 = candidate1[1]
            fitnessCand2 = candidate2[1]

            # Step 4. Select the fittest individual (the one that minimizes)
            parents.append(candidate1[0] if fitnessCand1 < fitnessCand2 else candidate2[0])

        return parents

    # Simulated Binary Crossover (SBX)
    def _crossover(self, parents):

        """
        Function to perform crossover

        Input:
        parents: list of the population

        Output:
        newPopulation : list of the new population
        """

        newPopulation = []

        for parent1, parent2 in list(zip(parents[::2], parents[1::2])):
            
            # Compute the probability of crossover for the current couple
            crossProb = np.random.random()

            if crossProb <= self._Pc:

                # Step 1. Compute a random number u between 0 and 1
                u = np.random.uniform()

                # Step 2. Compute beta_m
                if u <= 0.5:
                    beta_m = (2*u)**(1 / (self._nc + 1))
                else:
                    beta_m = (1 / (2*(1 - u)))**(1 / (self._nc + 1))

                # Step 3. Produce children
                H1 = 0.5 * ((leftSide := (np.array(parent1) + np.array(parent2))) - (rightSide := beta_m*np.abs(np.array(parent2) - np.array(parent1))))
                H2 = 0.5 * (leftSide + rightSide)

                newPopulation.extend([self._limitIndividual(H1), self._limitIndividual(H2)])
            else:
                newPopulation.extend([parent1, parent2])

        return newPopulation


    # Parameter-based mutation
    def _mutation(self, population):
        """
        Function to perform parameter-based mutation

        input:
        population: list of individuals

        output: 
        newPopulation: list of new population (including mutated individuals)
        """

        newPopulation = []
        for individual in population:

            # Compute the probability of mutation for the current individual 
            mutProb = np.random.random()

            if mutProb <= self._Pm:

                # Step 1. Randomly select the gene to be mutated
                i = np.random.randint(0, len(individual))
                gene = individual[i]

                # Step 2. Compute a random number u between 0 and 1
                u = np.random.random()

                # Step 3. Compute delta sub q
                eta_m = self._eta_m # This actually is calculated as 100 + t, where t = generation num. But for the hwmk, it was required as 20
                delta = min((gene - self._x_min), (self._x_max - gene)) / (self._x_max - self._x_min)
                if u <= 0.5:
                    delta_q = (2*u + (1-2*u)*(1-delta)**(eta_m+1))**(1 / (eta_m+1)) - 1
                else:
                    delta_q = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta)**(eta_m+1))**(1 / (eta_m+1))

                # Step 4. Perform the mutation 
                deltaMax = self._x_max - self._x_min
                mutatedGene = gene + delta_q*deltaMax
                individual[i] = mutatedGene

            newPopulation.append(self._limitIndividual(individual))

        return newPopulation 

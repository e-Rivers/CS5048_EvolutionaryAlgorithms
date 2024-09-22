import numpy as np
import random
import sympy
from sympy import symbols
import math
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


# Father class: GeneticAlgorithm
class GeneticAlgorithm(ABC):

    #    def __init__(self, popu_size, num_generations, ind_size, Pc=0.9, Pm=0.1):
    def __init__(self, lowerBound, upperBound, func, Pc=0.9, Pm=0.1):
        self._xU = upperBound
        self._xL = lowerBound
        self._func = func
        self._population = None
        #        self._pop_size = popu_size
        #        self._num_generations = num_generations
        #        self._ind_size = ind_size
        #        self._population = self.initialize_population()
        #        self._Pc = Pc   # Crossover Probability  
        #        self._Pm = Pm   # Mutation Probability  

    @abstractmethod
    def initialize_population(self):
        raise NotImplementedError

    @abstractmethod
    def run(self, verbose=False):
        pass

    def _getFitness(self, individual):
        return self._func(individual)

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

    def __init__(self, lowerBound, upperBound, func, Pm=0.7):
        super().__init__(lowerBound, upperBound, func, Pm)
        

    def run(self, popu_size, num_generations, x1_min, x1_max, x2_min, x2_max):
        # Initialize population
        num_bits_1 = self.number_bits(x1_min, x1_max)
        num_bits_2 = self.number_bits(x2_min, x2_max)
        first_population = self.initialize_population(num_bits_1, num_bits_2, popu_size)
        population = list(first_population)
        best_solutions =[]


        for generation in range(num_generations):
            # Decode population
            decoded_population = self.decoder(population, x1_min, x2_min, x1_max, x2_max, num_bits_1, num_bits_2)
            #print(decoded_population)
        
            
            fit_list = self.function_evaluation(decoded_population, self._func)

            # Selection
            selected_individuals = self._selection(fit_list, population)

            # Crossover
            children = self._crossover(selected_individuals)

            # Mutation
            mutated_population = self._mutation(children, 0.7)

            # Update population
            population = mutated_population

            # print best fitness in the generation
            min_index = fit_list.index(min(fit_list))

            # Store the best fitness and its corresponding decoded point
            #best_fitness = (fit_list[min_index], decoded_population[min_index])
            best_fitness = fit_list[min_index]
            best_solutions.append(best_fitness)

            #print(best_fitness, decoded_population )
            
            

        # Return best solution
        return best_solutions

    def number_bits(self, x_min, x_max):
        """
        Determine the number of bits needed to encode the solution

        Input:
        x_min: Minimum value that the solution can take
        x_max: Maximum value that the solution can take

        ----------

        Output:
            if
        num_bits: Represents the length of the solution
        """
        # we calculate the length of the chromosomes needed to encode the solution using 4 numbers after the decimal point
        num_bits = (x_max - x_min)*10**4
        num_bits = math.log(num_bits,2)
        num_bits = math.floor(num_bits + 0.99)
        return num_bits

    def initialize_population(self, num_bits_1, num_bits_2, popu_size):
        """
        Function to initialize population by encoding variables 

        Input:
        num_bits_1: number of bits needed to encode x1
        num_bits_2: number of bits to encode x2
        popu_size: number of individuals (genomes or chromosomes)

        ----------

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

        ----------

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
            x2 = x2_min + x2_deci * ((x2_max - x2_min) / (2**num_bits_2 - 1))
            x2=round(x2,4)
            x1=round(x1,4)
            decode_population.append([x1,x2])


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
        #x1, x2 = variables
        for i in range(len(decode_population)):
            current_value = func.subs({x1: decode_population[i][0], x2: decode_population[i][1]}).evalf()
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

    def _crossover(self, parents, pc=0.7):
        """
        Function to perfomr single point crossover

        input:
        parents: list of selected chromosomes
        pc : probability of crossover (0.7 as default)

        output:
        parents: list of new  population including chromosomes after the crossover
        """

        

        #empty lists for selected individual to crossover and childrens
        individuals_cross = []
        index_cross = []
        children =[]

        #select individuals random to perform the crossover
        for i in range(len(parents)):
            r = random.random()
            if r < pc:
                individuals_cross.append(parents[i])
                index_cross.append(i)


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

      
    
    def _mutation(self, population, pm):
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

class RealGA(GeneticAlgorithm):

    def __init__(self, lowerBound, upperBound, func, nc = 20, np = 20):
        super().__init__(lowerBound, upperBound, func)
        self._nc = nc
        self._np = np

    # Binary tournament selection
    def _selection(self, population):
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
    # Define Rastrigin function
    def rastrigin(x):
        A = 10
        return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

    def run_experiments(ga_class, problems, pop_size, num_generations, num_runs):
        results = []
        for problem in problems:
            for _ in range(num_runs):
                ga = ga_class(lowerBound=problem[1][0], upperBound=problem[1][1], func=problem[0])
                best_fitness = ga.run(popu_size=pop_size, num_generations=num_generations, 
                                                                        x1_min=problem[1][0], x1_max=problem[1][1], 
                                                                        x2_min=problem[1][0], x2_max=problem[1][1])
                best_index = np.argmin(best_fitness)
                print(best_fitness)
                fitness = list(best_fitness)
                results.append(fitness[best_index])
        return results

    # Initialize problems
    x1, x2 = symbols("x1 x2")
    problems = [
        (
            100*(x1**2 - x2)**2 + (1 - x1)**2,
            np.array([-2.048, 2.048]).astype(float),
            "Function A",
            np.array([0, 0]).astype(float) 
        ),

        (
            (4-2.1*x1**2+((x1**4)/3))*x1**2+x1*x2+(-4+4*x2**2)*x2**2,
            np.array([0.5, 1]).astype(float),
            "Function B",
            np.array([0.08984201774712433, 0.7126563947865581]).astype(float)
        ),
    ]

    num_runs = 20
    results = {}
    for ga_class in [BinaryGA]:
        results[ga_class.__name__] = {}
        for problem in problems:
            fitnesses = run_experiments(ga_class, [problem], pop_size=4, num_generations=10, num_runs=num_runs)
            results[ga_class.__name__][problem[2]] = {
                'mean': np.mean(fitnesses),
                #'std': np.std(fitnesses),
                'min': np.min(fitnesses),
                'max': np.max(fitnesses),
            }

    # Print results
    for ga_name, res in results.items():
        print(f"{ga_name}:")
        for prob_name, stats in res.items():
            print(f"  {prob_name}: Mean: {stats['mean']}, Min: {stats['min']}, Max: {stats['max']}")


    # You would call plot_convergence after running your experiments
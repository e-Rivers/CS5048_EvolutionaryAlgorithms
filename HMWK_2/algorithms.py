import numpy as np
import random
import sympy
from sympy import symbols
import math
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import csv


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
        """# Step 1. Initialize Population
        self._population = self.initialize_population()

        # Step 2. Form the couples for creating children
        couples = []
        for _ in range(len(self._population)//2):
            couples.append((
                    self._selection()
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

"""
        raise NotImplementedError

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

    def __init__(self, lowerBound, upperBound, func, Pc= 0.9,):
        super().__init__(lowerBound, upperBound, func, Pc)
        

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

class RealGA(GeneticAlgorithm):

    def __init__(self, lowerBound, upperBound, func, nc = 20, np = 20):
        super().__init__(lowerBound, upperBound, func)
        self._nc = nc
        self._np = np

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

##################################
##################################
###### MAIN PRUEBA ###############
##################################
##################################

if __name__ == "__main__":

    def run_experiments(ga_class, problems, pop_size, num_generations, num_runs):
        results = []
        fitness_history = []
        for func, bounds, description, initial_guess in problems:
            for _ in range(num_runs):
                # Initialize GA with the bounds and function
                lower_bound, upper_bound = bounds  # Extract min and max bounds
                ga = ga_class(lowerBound=lower_bound, upperBound=upper_bound, func=func)
                
                # Run GA
                best_fitness = ga.run(popu_size=pop_size, num_generations=num_generations, 
                                    bounds=[(lower_bound, upper_bound)] * len(initial_guess))
                
                # Get the best fitness value in this run
                best_index = np.argmin(best_fitness)
                #print(f"Best of generation: {best_index}")
                results.append(best_fitness[best_index])
                fitness_history.append(best_fitness)
        
        return results, fitness_history

    # Initialize problems
    problems = [
    (
        lambda x: 100*(x[0]**2 - x[1])**2 + (1 - x[0])**2,
        np.array([-2.048, 2.048]).astype(float),
        "Function A",
        np.array([0, 0]).astype(float) 
    ),
    (
        lambda x: 10*len(x) + sum([(x_i**2 - 10*np.cos(2*np.pi*x_i)) for x_i in x]),
        np.array([-5.12, 5.12]).astype(float),
        "Rastrigin (n=2)",
        np.array([0, 0]).astype(float)
    ),
    (
        lambda x: 10*len(x) + sum([(x_i**2 - 10*np.cos(2*np.pi*x_i)) for x_i in x]),
        np.array([-5.12, 5.12]).astype(float),
        "Rastrigin (n=5)",
        np.array([0, 0]).astype(float)
    )
]

    num_runs = 20
    results = {}
    for ga_class in [BinaryGA]: #aqui nomas agregamos la otra clase
        results[ga_class.__name__] = {}
        for problem in problems:
            fitnesses, fitness_history = run_experiments(ga_class, [problem], pop_size=10, num_generations=250, num_runs=num_runs)
            fitnesses = np.array(fitnesses, dtype=float)
            results[ga_class.__name__][problem[2]] = {
                'mean': np.mean(fitnesses),
                'std': np.std(fitnesses),
                'min': np.min(fitnesses),
                'max': np.max(fitnesses),
                'fitness_history' : fitness_history,
                'results' : fitnesses
            }

    # Print results
    for ga_name, res in results.items():
        print(f"{ga_name}:")
        for prob_name, stats in res.items():
            print(f"  {prob_name}: Mean: {stats['mean']}, Std Dev: {stats['std']} ,Min: {stats['min']}, Max: {stats['max']}")
            print(f"  {prob_name}: 20 experiments: {stats['results']}")

#### for the table
    rows = []
    historial = []
    auxi = []

    # Recopilar los resultados en formato de tabla
    for ga_name, res in results.items():
        for prob_name, stats in res.items():
            # Crear una fila básica con los stats
            row = [
                ga_name,
                prob_name,
                stats['mean'],
                stats['std'],
                stats['min'],
                stats['max']
            ]
            
            # Agregar los valores de "20 experiments" como columnas adicionales
            experiments = stats['results']  # Asegúrate de que esto sea una lista
            row.extend(experiments)
            
            # Añadir la fila completa a las filas
            rows.append(row)

            historial_20_experiments = stats['fitness_history']
            historial.append(historial_20_experiments)

            name=[ga_name,prob_name]
            auxi.append(name)


    with open('resultados.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        print(len(stats['results']))
        header = ['GA Name', 'Problem Name', 'Mean', 'Std Dev', 'Min', 'Max'] + [f'Experiment {i+1}' for i in range(len(stats['results']))]
        writer.writerow(header)
        
        writer.writerows(rows)

##### plot random
    #select a random experiment
    r = random.randint(0, len(historial[0])-1)
    print(auxi)
    # select and plot the history of those 
    y = np.arange(1, len(historial[0][0])+1)
    cols_plot = 3
    rows_plot = 1
    plt.figure(figsize=(10, 5 * rows_plot))

    for i in range(len(historial)) :
        plt.subplot(rows_plot, cols_plot, i + 1)  # Crear un subplot
        plt.plot(y, historial[i][r], linestyle='-', color='b')
        plt.title(f'{auxi[i][1]}')
        plt.xlabel('Number of generations')
        plt.ylabel(f'Fitness value')
        plt.grid()
        
    plt.tight_layout()  # Ajustar el layout
    plt.show()
    print(len(historial[0]))
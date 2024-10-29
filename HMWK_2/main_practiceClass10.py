from algorithms3 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_experiments(ga_class, problems, pop_size, num_generations, num_runs):
    """
    Function to run a geneic algorithm for a set of given problems

    input: 
    ga_class: The genetic algorithm class
    problems: A list of problems, each with a fitness function, bounds, name, and number of decision variables.
    pop_size: population size for the genetic algorithm
    num_generations: Number of generations
    num_runs:  Number of independent runs

    output:
    results: List of best fitness values for each run.
    fitness_history: History of fitness values across generations for each run.

    """
    results = []
    fitness_history = []
    individual_history = []
    # loop thorugh each problem
    for func, bounds, _, decision_vars in problems:
        for _ in range(num_runs):
            # Initialize GA with the bounds and function
            lower_bound, upper_bound = bounds  # Extract min and max bounds
            ga = ga_class(lowerBound=lower_bound, upperBound=upper_bound, varNum=decision_vars, func=func, popu_size=pop_size)
            
            # Run GA
            best_fitness, best_individuals = ga.run(num_generations=num_generations) 
            
            # Get the best fitness value in this run
            best_index = np.argmin(best_fitness)
            results.append((best_fitness[best_index], best_individuals[best_index]))
            fitness_history.append(best_fitness)
            individual_history.append(best_individuals)
    
    return results, fitness_history, individual_history




# Initialize problems
problems = [
    (
        lambda x: 10*len(x) + sum([(x_i**2 - 10*np.cos(2*np.pi*x_i)) for x_i in x]),
        np.array([-5.12, 5.12]).astype(float),
        "Rastrigin (n=2)",
        2
    ),
    (
        lambda x: sum([100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1)]),
        np.array([-2.5, 2.5]).astype(float),
        "Rosenbrock (n=2)",
        2
    )
]

# Ask for input variables
print("Would you like to setup the parameters manually or go with the default values?")
print("Population size: 100")
print("Number of generations: 250")
print("Number of runs: 20")
setup = input("\nGo with default values? [y/n] ")
pop_size = 100 if setup == "y" else int(input("Set a population size: "))
num_generations = 250 if setup == "y" else int(input("Set a number of generations: "))
num_runs = 20 if setup == "y" else int(input("Set the number of runs: "))

results = {}
for ga_class in [RealGA]: 
    results[ga_class.__name__] = {}

    print(f"\n\n\n\n{ga_class.__name__}\n\n\n\n")

    for problem in problems:
        bestOverall, fitness_history, individual_history = run_experiments(
            ga_class, 
            [problem], 
            pop_size=pop_size, 
            num_generations=num_generations, 
            num_runs=num_runs
        )
        
        fitnesses = np.array([fitness for fitness, _ in bestOverall], dtype=float)
        individuals = np.array([individual for _, individual in bestOverall], dtype=float)
        results[ga_class.__name__][problem[2]] = {
            'mean': np.mean(fitnesses),
            'std': np.std(fitnesses),
            'min': np.min(fitnesses),
            'max': np.max(fitnesses),
            'fitness_history': fitness_history,
            'individual_history': individual_history,
            'results': fitnesses,
            'results (individual)': individuals
        }



# Print the results (mean, standard deviation, min, max) for each GA and problem
for ga_name, res in results.items():
    print(f"{ga_name}:")
    for prob_name, stats in res.items():
        print(f"  {prob_name}: Mean: {stats['mean']}, Std Dev: {stats['std']} ,Min: {stats['min']}, Max: {stats['max']}")
        print(f"  {prob_name}: 20 experiments: {stats['results']}")
        print(*(expResults := list(zip(stats["results"], stats["results (individual)"]))), sep="\n")

sortedResults = sorted(expResults, key=lambda x: x[0])
print("SORTED")
print(*sortedResults, sep="\n")


fig = plt.figure(figsize=(10, 7.5))

for i, problem in enumerate(problems):
    fitness_func, bounds, name, dimension = problem

    x_vals = np.linspace(bounds[0], bounds[1], 100)
    y_vals = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([[fitness_func([x, y]) for x in x_vals] for y in y_vals])

    ax3D = fig.add_subplot(int(f"22{2*i+1}"), projection='3d')
    ax3D.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax3D.contour(X, Y, Z, levels=20, zdir='z', offset=ax3D.get_zlim()[0], cmap='viridis', alpha=0.5)
    ax3D.set_title(f"{name} Function")
    ax3D.set_xlabel('X')
    ax3D.set_ylabel('Y')
    ax3D.set_zlabel('Fitness')

    ax2D = fig.add_subplot(int(f"22{2*i+2}"))
    ax2D.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)

    # Plot the best solution in each generation
    for run in results['RealGA'][name]['individual_history']:
        ax2D.plot([ind[0] for ind in run], [ind[1] for ind in run], 'o', color='r', label = "Best per generation")

    # Plot the best overall solution with a distinct 'X'
    best_solution = sortedResults[0][1]  # Get the overall best solution (sorted by fitness)
    ax2D.plot(best_solution[0], best_solution[1], marker ='x', color='black', markersize=10, label="Best Overall Solution")

    ax2D.set_title(f"{name} Function (Best Solutions)")
    ax2D.set_xlabel('X')
    ax2D.set_ylabel('Y')

plt.tight_layout()
plt.show()
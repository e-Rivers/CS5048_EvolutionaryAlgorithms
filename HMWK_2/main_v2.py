from algorithms import *
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
    
    return results, fitness_history, best_individuals



c1 = 3250   # Penalty for the equality constraint
c2 = 2000   # Penalty for the inequality constraint
ß = 2       # Beta: Penalty exponent for inequality constraint
γ = 2       # Gamma: Penalty exponent for equality constraint

def constrained_objective(x):
    # Objective function (x² + y²)
    objective = (x[0]**2 + x[1]**2)
    
    # Inequality constraint (penalty if x > y)
    inequality_constraint = max(0, x[0] - x[1])
    inequality_penalty = c1 * inequality_constraint**ß
    
    # Equality constraint (penalty for x^2 + y^2 not being 1/2)
    equality_constraint = (x[0]**2 + x[1]**2) - 0.5
    equality_penalty = c2 * abs(equality_constraint)**γ
    
    # Total fitness 
    total_fitness = (objective + (inequality_penalty + equality_penalty))
    
    return total_fitness

# Initialize problems
problems = [
    (
        constrained_objective,
        np.array([-1, 1]).astype(float),
        "Function Phi Φ",
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
for ga_class in [BinaryGA, RealGA]: 
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

def plot_landscape(bounds, fitness_func):
    x_vals = np.linspace(bounds[0], bounds[1], 100)
    y_vals = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([[fitness_func([x, y]) for x in x_vals] for y in y_vals])
    Z1 = np.array([[(lambda x, y: (x**2 + y**2))(x,y) for x in x_vals] for y in y_vals])

    # Plot the 3D landscape
    fig = plt.figure(figsize=(10, 7.5))
    #esta linea en caso de que queramos añadir el contour plot enseguida
    ax = fig.add_subplot(221, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.contour(X, Y, Z, levels=20, zdir='z', offset=ax.get_zlim()[0], cmap='viridis', alpha=0.5)
    ax.plot(sortedResults[-1][1][0], sortedResults[-1][1][1], sortedResults[-1][0], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    ax.set_title('Fitness Landscape\n(With Penalties)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')

    ax = fig.add_subplot(223)
    ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
    for sortedResult in sortedResults:
        ax.plot(sortedResult[1][0], sortedResult[1][1], marker="o", markersize=10, markeredgecolor="black", markerfacecolor="black")
    ax.plot(sortedResults[-1][1][0], sortedResults[-1][1][1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")

    ax = fig.add_subplot(222, projection='3d')
    ax.plot_surface(X, Y, Z1, cmap='plasma', edgecolor='none')
    ax.contour(X, Y, Z1, levels=20, zdir='z', offset=ax.get_zlim()[0], cmap='plasma', alpha=0.5)
    ax.plot(sortedResults[-1][1][0], sortedResults[-1][1][1], sortedResults[-1][0], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    ax.set_title('Fitness Landscape\n(Original Function)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')

    ax = fig.add_subplot(224)
    ax.contour(X, Y, Z1, levels=20, cmap='plasma', alpha=0.5)
    for sortedResult in sortedResults:
        ax.plot(sortedResult[1][0], sortedResult[1][1], marker="o", markersize=10, markeredgecolor="black", markerfacecolor="black")
    ax.plot(sortedResults[-1][1][0], sortedResults[-1][1][1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")


    plt.tight_layout()
    plt.show()

bounds = problems[0][1]
fitness_func = problems[0][0]
plot_landscape(bounds, fitness_func)




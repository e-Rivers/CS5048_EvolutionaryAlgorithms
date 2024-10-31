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
    # loop thorugh each problem
    for func, bounds, _, decision_vars in problems:
        for _ in range(num_runs):
            # Initialize GA with the bounds and function
            lower_bound, upper_bound = bounds  # Extract min and max bounds
            ga = ga_class(lowerBound=lower_bound, upperBound=upper_bound, varNum=decision_vars, func=func, popu_size=pop_size)
            
            # Run GA
            best_fitness = ga.run(num_generations=num_generations) 
            
            # Get the best fitness value in this run
            best_index = np.argmin(best_fitness)
            results.append(best_fitness[best_index])
            fitness_history.append(best_fitness)
    
    return results, fitness_history



c1 = 1.5  # Penalty for the equality constraint
c2 = 2.3  # Penalty for the inequality constraint
B = 1      # Penalty exponent for inequality constraint
A = 1

def constrained_objective(x):
    # Objective function (maximize x^2 + y^2)
    objective = (x[0]**2 + x[1]**2)
    
    # Equality constraint (penalty for x^2 + y^2 1/2)
    equality_constraint = (x[0]**2 + x[1]**2) - 0.5
    equality_penalty = c1 * abs(equality_constraint)**A
    
    # Inequality constraint 
    inequality_constraint = max(0, x[0] - x[1])  # Apply penalty if x > y
    inequality_penalty = c2 * inequality_constraint**B
    
    # Total fitness 
    total_fitness = (objective + equality_penalty + inequality_penalty)
    
    return total_fitness

# Initialize problems
problems = [
    (
        constrained_objective,
        np.array([-1, 1]).astype(float),
        "Function phi",
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
        fitnesses, fitness_history = run_experiments(
            ga_class, 
            [problem], 
            pop_size=pop_size, 
            num_generations=num_generations, 
            num_runs=num_runs
        )
        
        fitnesses = np.array(fitnesses, dtype=float)
        results[ga_class.__name__][problem[2]] = {
            'mean': np.mean(fitnesses),
            'std': np.std(fitnesses),
            'min': np.min(fitnesses),
            'max': np.max(fitnesses),
            'fitness_history': fitness_history,
            'results': fitnesses
        }


# Print the results (mean, standard deviation, min, max) for each GA and problem
for ga_name, res in results.items():
    print(f"{ga_name}:")
    for prob_name, stats in res.items():
        print(f"  {prob_name}: Mean: {stats['mean']}, Std Dev: {stats['std']} ,Min: {stats['min']}, Max: {stats['max']}")
        print(f"  {prob_name}: 20 experiments: {stats['results']}")

rows = []
historial = []
auxi = []

# Collect results for CSV writing
for ga_name, res in results.items():
    for prob_name, stats in res.items():
        # Row with the stats
        row = [
            ga_name,
            prob_name,
            stats['mean'],
            stats['std'],
            stats['min'],
            stats['max']
        ]
        
        # experiment results as additional columns
        experiments = stats['results']  
        row.extend(experiments)
        rows.append(row)
        historial_20_experiments = stats['fitness_history']
        historial.append(historial_20_experiments)

        # Append GA and problem names to the auxiliary list
        name=[ga_name,prob_name]
        auxi.append(name)

# Write results to CSV
with open('resultados.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    print(len(stats['results']))
    header = ['GA Name', 'Problem Name', 'Mean', 'Std Dev', 'Min', 'Max'] + [f'Experiment {i+1}' for i in range(len(stats['results']))]
    writer.writerow(header)
    writer.writerows(rows)

# Plot a random experiment's fitness history
r = random.randint(0, len(historial[0])-1)




"""cols_plot = 1
rows_plot = 3
plt.figure(figsize=(10, 2.5 * rows_plot))

# Plot fitness history for the selected experiment
for i in range(len(historial)//2) :
    plt.subplot(rows_plot, cols_plot, i + 1)  # Crear un subplot
    y1 = np.arange(1, len(historial[i][r])+1)
    y2 = np.arange(1, len(historial[i+3][r])+1)
    plt.plot(y1, historial[i][r], linestyle='-', color='b', label="Binary")
    plt.plot(y2, historial[i+3][r], linestyle='-', color='r', label="Real")
    plt.title(f'{auxi[i][1]}')
    plt.xlabel('Number of generations')
    plt.ylabel('Best Fitness value')
    plt.legend()
    plt.grid()
    
plt.tight_layout()  
plt.show()"""


def plot_landscape(bounds, fitness_func):
    x_vals = np.linspace(bounds[0], bounds[1], 100)
    y_vals = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([[fitness_func([x, y]) for x in x_vals] for y in y_vals])

    # Plot the 3D landscape
    fig = plt.figure()
    #esta linea en caso de que queramos añadir el contour plot enseguida
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('Fitness Landscape (with Penalties)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')

    contour_g = ax.contour(X, Y, Z, levels=20, zdir='z', offset=ax.get_zlim()[0], cmap='viridis', alpha=0.5)

    plt.tight_layout()

    plt.show()

bounds = problems[0][1]
fitness_func = problems[0][0]
plot_landscape(bounds, fitness_func)



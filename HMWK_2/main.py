from algorithms import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_experiments(ga_class, problems, pop_size, num_generations, num_runs):
    results = []
    fitness_history = []
    for func, bounds, _, decision_vars in problems:
        for _ in range(num_runs):
            # Initialize GA with the bounds and function
            lower_bound, upper_bound = bounds  # Extract min and max bounds
            ga = ga_class(lowerBound=lower_bound, upperBound=upper_bound, varNum=decision_vars, func=func, popu_size=pop_size)
            
            # Run GA
            best_fitness = ga.run(num_generations=num_generations) 
            
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
        2
    ),
    (
        lambda x: 10*len(x) + sum([(x_i**2 - 10*np.cos(2*np.pi*x_i)) for x_i in x]),
        np.array([-5.12, 5.12]).astype(float),
        "Rastrigin (n=2)",
        2
    ),
    (
        lambda x: 10*len(x) + sum([(x_i**2 - 10*np.cos(2*np.pi*x_i)) for x_i in x]),
        np.array([-5.12, 5.12]).astype(float),
        "Rastrigin (n=5)",
        5
    )
]


num_runs = 20
results = {}
for ga_class in [BinaryGA, RealGA]: #aqui nomas agregamos la otra clase
    results[ga_class.__name__] = {}

    print("\n\n\n\n",ga_class.__name__,"\n\n\n\n")

    for problem in problems:
        fitnesses, fitness_history = run_experiments(ga_class, [problem], pop_size=50, num_generations=250, num_runs=num_runs)
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

# plot random
#select a random experiment
r = random.randint(0, len(historial[0])-1)
print(auxi)
# select and plot the history of those 
y = np.arange(1, len(historial[0][0])+1)

cols_plot = 1
rows_plot = 3
plt.figure(figsize=(10, 2.5 * rows_plot))

print(len(historial))
for i in range(len(historial)//2) :
    plt.subplot(rows_plot, cols_plot, i + 1)  # Crear un subplot
    plt.plot(y, historial[i][r], linestyle='-', color='b', label="Binary")
    plt.plot(y, historial[i+3][r], linestyle='-', color='r', label="Real")
    plt.title(f'{auxi[i][1]}')
    plt.xlabel('Number of generations')
    plt.ylabel('Fitness value')
    plt.legend()
    plt.grid()
    
plt.tight_layout()  # Ajustar el layout
plt.show()
print(len(historial[0]))

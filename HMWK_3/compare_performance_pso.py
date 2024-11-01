import pandas as pd
from scipy.stats import ranksums

# List of function names
functions = ["Layeb05_(n=2)", "Layeb10_(n=2)", "Layeb15_(n=2)", "Layeb18_(n=2)"]
results = []

for func_name in functions:
    # Load CSV files for PSO, GA, and DE methods for the current function
    pso_df = pd.read_csv(f'/Users/cdr_c/Documents/CS5048_EvolutionaryAlgorithms/HMWK_3/experiment_results_pso/{func_name}_results.csv')
    ga_df = pd.read_csv(f'/Users/cdr_c/Documents/CS5048_EvolutionaryAlgorithms/HMWK_3/experiment_results_ga/{func_name}_results.csv')
    de_df = pd.read_csv(f'/Users/cdr_c/Documents/CS5048_EvolutionaryAlgorithms/HMWK_3/experiment_results_de/{func_name}_results.csv')

    # Extract the best fitness for each execution
    pso_best = pso_df.groupby('Execution')['Best Fitness'].min()
    ga_best = ga_df.groupby('Execution')['Best Fitness'].min()
    de_best = de_df.groupby('Execution')['Best Fitness'].min()

    # Perform Wilcoxon rank-sum test between each pair of methods
    pso_ga_stat, pso_ga_p = ranksums(pso_best, ga_best)
    pso_de_stat, pso_de_p = ranksums(pso_best, de_best)
    ga_de_stat, ga_de_p = ranksums(ga_best, de_best)

    # Store results for the current function
    results.append({
        'Function': func_name,
        'Comparison': 'PSO vs GA',
        'Statistic': pso_ga_stat,
        'P-Value': pso_ga_p
    })
    results.append({
        'Function': func_name,
        'Comparison': 'PSO vs DE',
        'Statistic': pso_de_stat,
        'P-Value': pso_de_p
    })
    results.append({
        'Function': func_name,
        'Comparison': 'GA vs DE',
        'Statistic': ga_de_stat,
        'P-Value': ga_de_p
    })

# Convert the results to a DataFrame and save as CSV
results_df = pd.DataFrame(results)
output_path = 'wilcoxon_test_results_all_functions.csv'
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

import pandas as pd
from scipy.stats import ranksums

# List of function names
functions = ["G1", "G4", "G5", "G6"]
results = []

for func_name in functions:
    # Load CSV files for GA, and DE methods for the current function
    ga_df = pd.read_csv(f'/Users/cdr_c/Documents/CS5048_EvolutionaryAlgorithms/HMWK_3/results_ga_constrained/{func_name}_results.csv')
    de_df = pd.read_csv(f'/Users/cdr_c/Documents/CS5048_EvolutionaryAlgorithms/HMWK_3/results_de_constrained/{func_name}_results.csv')

    # Extract the best fitness for each execution (aqui sustituimos por lo que vayamos a usar fitness o penalty)
    ga_best = ga_df.groupby('Execution')['Fitness'].min()
    de_best = de_df.groupby('Execution')['Best Fitness'].min()

    # Perform Wilcoxon rank-sum test between each pair of methods
    ga_de_stat, ga_de_p = ranksums(ga_best, de_best)

    # Store results for the current function
    
    results.append({
        'Function': func_name,
        'Comparison': 'GA vs DE',
        'Statistic': ga_de_stat,
        'P-Value': ga_de_p
    })

# Convert the results to a DataFrame and save as CSV
results_df = pd.DataFrame(results)
output_path = 'wilcoxon_test_results_all_functions_constrained.csv'
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

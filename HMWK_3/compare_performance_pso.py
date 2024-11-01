import pandas as pd
from scipy.stats import ranksums

# Load CSV files for PSO, GA, and DE methods
# ESTO LO MODIFICAMOS YA DESPUES QUE TENGAMOS LOS ARCHIVOS
pso_df = pd.read_csv('/Users/cdr_c/Documents/CS5048_EvolutionaryAlgorithms/HMWK_3/experiment_results_pso/problem.csv')
ga_df = pd.read_csv('ga_results.csv')
de_df = pd.read_csv('de_results.csv')


pso_best = pso_df.groupby('Execution')['Best Fitness'].min()
ga_best = ga_df.groupby('Execution')['Best Fitness'].min()
de_best = de_df.groupby('Execution')['Best Fitness'].min()

# Perform Wilcoxon rank-sum test between each pair of methods
pso_ga_stat, pso_ga_p = ranksums(pso_best, ga_best)
pso_de_stat, pso_de_p = ranksums(pso_best, de_best)
ga_de_stat, ga_de_p = ranksums(ga_best, de_best)

print("Wilcoxon Rank-Sum Test Results:")
print(f"PSO vs GA: statistic = {pso_ga_stat}, p-value = {pso_ga_p}")
print(f"PSO vs DE: statistic = {pso_de_stat}, p-value = {pso_de_p}")
print(f"GA vs DE: statistic = {ga_de_stat}, p-value = {ga_de_p}")

results_df = pd.DataFrame({
    'Comparison': ['PSO vs GA', 'PSO vs DE', 'GA vs DE'],
    'Statistic': [pso_ga_stat, pso_de_stat, ga_de_stat],
    'P-Value': [pso_ga_p, pso_de_p, ga_de_p]
})

# Save the results to a new CSV file
output_path = 'wilcoxon_test_results.csv'
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
import pandas as pd
import os

# Define the directory containing the CSV files
input_dir = '/Users/cdr_c/Documents/CS5048_EvolutionaryAlgorithms/HMWK_3/experiment_results'
output_dir = '/Users/cdr_c/Documents/CS5048_EvolutionaryAlgorithms/HMWK_3/processed_results'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each CSV file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        # Define the full path for the input file
        file_path = os.path.join(input_dir, filename)
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Calculate statistics
        min_per_execution = df.groupby('Execution')['Best Fitness'].min()
        mean_min_fitness = df.groupby('Execution')['Best Fitness'].mean()
        std_dev_min_fitness = df.groupby('Execution')['Best Fitness'].std()
        
        # Create a DataFrame with the required statistics
        stats_df = pd.DataFrame({
            'Execution': mean_min_fitness.index,
            'Min Per Execution': min_per_execution.values,
            'Mean Min Fitness': mean_min_fitness.values,
            'Std Dev Min Fitness': std_dev_min_fitness.values
        })
        
        # Define the output file path
        output_file = os.path.join(output_dir, f"stats_{filename}")
        
        # Save the statistics DataFrame to a new CSV file
        stats_df.to_csv(output_file, index_label='Execution')
        
        print(f"Processed {filename} and saved results to {output_file}")

# Homework 3. Optimization problems with PSO, GA, DE and Constraint Handling Techniques

Denisse Chacón Ramírez -  <br />
Emilio Rios Ochoa - 

## Code Execution Instructions and Explanationw

Each file already contains the necessary information to solve all the problems, so running them just requires the execution of `python3 filename.py` (`py filename.py` for Windows).<br />
In case it is desired to modify certain parameters about the execution, manual modification of the file itself is required.

### PART 1. Constrained Problems

These are the files related to solving problems **G1**, **G4**, **G5** and **G6**.

* File `GA_constrained.py` contains the implementation of a GA with a custom constrainted handling technique.
* File `DE_constrained.py` contains the implementation of DE equipped with stochastic ranking for constraint handling.

### PART 2. Unconstrained Problems

These are the files related to solving problems **Layeb05**, **Layeb10**, **Layeb15** and **Layeb18**.

* File `PSO.py` constains the implementation of a PSO algorithm.
* File `GA.py` contains the implementation of a real-encoded GA.
* File `DE.py` contains the implementation of a DE algorithm.

### Files used for analysis

* File `compare_performance_unconstrained.py` contains the implementation of the Wilcoxon rank-sum tests to compare the performance of PSO, GA and DE for the unconstrained Layeb problems.

To execute this file, it is required to:
1) First, create three folders: _experiment_results_ga_, _experiment_results_de_ and _experiment_results_pso_ to hold the CSV files of the executions of GA, DE and PSO respectively.
2) Execute all three files for the unconstrained problems, they will automatically save the results of the best individuals of each one of the 30 executions on the corresponding folders.

* File `DE_constrained.py` contains the implementation of the Wilcoxon rank-sum tests to compare the performance of GA and DE for the constrained G1, G4, G5 and G6 problems.

To execute this file, it is required to:
1) First, create two folders: _results_ga_constrained_ and _results_de_constrained_ to hold the CSV files of the executions of GA and DE respectively.
2) Execute both files for the unconstrained problems, they will automatically save the results of the best individuals of each one of the 30 executions on the corresponding folders.


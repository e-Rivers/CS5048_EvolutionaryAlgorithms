from optimizationMethods import *
import sympy
import numpy as np
import matplotlib.pyplot as plt


# Ask for input variables
print("Would you like to setup the parameters manually or go with the default values?")
print("Initial guess for step size: 0.01")
print("Tolerance with which to accept a solution: 0.001")
print("Maximum number of iterations allowed: 1000")
setup = input("\nGo with default values? [y/n] ")
stepSize = 0.01 if setup == "y" else float(input("Set an initial guess for the step size: "))
tolerance = 0.001 if setup == "y" else float(input("Set a tolerance value: "))
maxIterations = 1000 if setup == "y" else int(input("Set the maximum number of iterations: "))

# Define the variables as sympy symbols (to allow symbolic operations, e.g. derivative)
x1, x2 = sympy.symbols("x1 x2")

# Definition of the functions
# Each element in the list represents a function and contains the following elements:
# 1. The mathematical expression of the function.
# 2. The starting point for the variables x1 and x2, represented as a NumPy array.
# 3. A string with the name of the function.
# 4. The known or expected optimal solution, represented as a NumPy array.
functions = [
    (
        -(-2*x1**2 + 3*x1*x2 -1.5*x2**2-1.3),
        np.array([-4, 4]).astype(float),
        "Function A",
        np.array([0, 0]).astype(float) 
    ),
    (
        (4-2.1*x1**2+((x1**4)/3))*x1**2+x1*x2+(-4+4*x2**2)*x2**2,
        np.array([0.5, 1]).astype(float),
        "Function B",
        np.array([0.08984201774712433, 0.7126563947865581]).astype(float)
    ),
    (
        # A = 10 and n = 2 (to allow visualization)
        10 * 2 + (x1**2 - 10 * sympy.cos(2*sympy.pi*x1)) + (x2**2 - 10 * sympy.cos(2*sympy.pi*x2)),
        np.array([-2, 2]).astype(float),
        "Rastrigin Function",
        np.array([0, 0]).astype(float)
    )
]

# Definition of the solution methods
methods = [
    (
        gradientDescent,
        "Gradient Descent"
    ),
    (
        newtonMethod,
        "Newton's Method"
    ),
    (
        hill_climbing,
        "Hill Climber"
    )
]

# Create an empty array to store the results of the implementation of the methods
solutions = []

# Evaluate the functions
# For each problem, test all the solution methods
for function, startPoint, _, _ in functions:
    for method, w in methods:
        # The resulting solution (a list of points) is appended to the solutions list.
        solutions.append(method(stepSize, startPoint.copy(), function, tolerance, maxIterations, (x1, x2)))

# Lambdification of the functions
# This is so they can be called in the form f(x1, x2)
fx = [sympy.lambdify((x1, x2), func, modules="numpy") for func, _, _ , _ in functions]


# Export the results to a txt file
with open("report.txt", "w") as report:
    for i in range(len(functions)):
        report.write(f"⦿ {functions[i][2]}\n")
        report.write(f"- Real Evaluation: {fx[i](*functions[i][3])}\n")
        for j in range(len(methods)):
            report.write(f"\t◙ {methods[j][1]}\n")
            report.write(f"\t\t- Point Found: {(point := solutions[i*3 +j][-1])}\n")
            report.write(f"\t\t- Evaluation: {fx[i](*point)}\n")
            report.write(f"\t\t- Iterations: {len(solutions[i*3 +j])}\n")
            report.write(f"\t\t- Two Norm Error: {np.linalg.norm(solutions[i*3 + j][-1] - functions[i][3])}\n")
        report.write("\n\n")
        
# Creation of a grid from -6 to 6 for plotting of the contour plots
x, y = np.meshgrid((linspace := np.linspace(-6, 6, 200)), linspace)

# Create a grid of 3x3 (rows are the problems/functions & cols are the solution methods)
fig, ax = plt.subplots(3, 3, figsize=(12, 8))
fig.suptitle("Contour Plots & Solutions Reached")

for i in range(len(functions)):
    for j in range(len(methods)):
        # Plot the contour plot based on the points in the grid and the function
        contourPlot = ax[i, j].contour(x, y, fx[i](x, y), levels=20)
        # Plot the solution found by each method (all the points of all iterations)
        ax[i, j].plot(*zip(*solutions[i*3 + j]), marker=".", linestyle="-", color="red")
    ax[i, 0].set_ylabel(functions[i][2])

for i in range(len(functions)):
    ax[0, i].set_title(methods[i][1])

fig.tight_layout()
plt.show()




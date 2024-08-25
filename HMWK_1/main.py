from optimizationMethods import *
import sympy
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

x1, x2 = sympy.symbols("x1 x2")

# Definition of the functions
functions = [
#    (
#        -(-2*x1**2 + 3*x1*x2 -1.5*x2**2-1.3),
#        np.array([-4, 4]).astype(float),
#        "Function A"
#    ),
    (
        (4-2.1*x1**2+((x1**4)/3))*x1**2+x1*x2+(-4+4*x2**2)*x2**2,
        np.array([0.5, 1]).astype(float),
        "Function B"
    ),
#    (
#        # A = 10 and n = 2 (to allow visualization)
#        10 * 2 + (x1**2 - 10 * sympy.cos(2*sympy.pi*x1)) + (x2**2 - 10 * sympy.cos(2*sympy.pi*x2)),
#        np.array([-2, 2]).astype(float),
#        "Rastrigin Function"
#    )
]

# Definition of the solution methods
methods = [
    (
        gradientDescent,
        "Gradient Descent"
    ),
#    (
#        newtonMethod,
#        "Newton's Method"
#    ),
#    (
#        newtonMethod,
#        "Hill Climber"
#    )
]

# Evaluate the functions
solutions = []
for function, startPoint, _ in functions:
    for method, w in methods:
        solutions.append(method(1, startPoint.copy(), function, 0.001, x1, x2))

# Lambdification of the functions
fx = [sympy.lambdify((x1, x2), func, modules="numpy") for func, _, _ in functions]


# Save a report with the results
with open("report.txt", "w") as report:
    for i in range(len(functions)):
        report.write(f"⦿ {functions[i][2]}\n")
        for j in range(len(methods)):
            report.write(f"\t◙ {methods[j][1]}\n")
            report.write(f"\t\t- Point Found: {(point := solutions[i*3 +j][-1])}\n")
            report.write(f"\t\t- Evaluation: {fx[i](*point)}\n")
            report.write(f"\t\t- Iterations: {len(solutions[i*3 +j])}\n")
            report.write(f"\t\t- Two Norm Error: MISSING...\n")
            print(f"{solutions[i*3 + j]}\n\n")
        report.write("\n\n")
        
# Plotting of the contour plots and the points found by the algorithms
x, y = np.meshgrid((linspace := np.linspace(-6, 6, 200)), linspace)

fig, ax = plt.subplots(3, 3, figsize=(12, 8))
fig.suptitle("Contour Plots & Solutions Reached")

for i in range(len(functions)):
    for j in range(len(methods)):
        contourPlot = ax[i, j].contour(x, y, fx[i](x, y), levels=20)
        ax[i, j].plot(*zip(*solutions[i*3 + j]), marker=".", linestyle="-", color="red")
    ax[i, 0].set_ylabel(functions[i][2])

for i in range(len(functions)):
    ax[0, i].set_title(methods[i][1])

fig.tight_layout()
plt.show()




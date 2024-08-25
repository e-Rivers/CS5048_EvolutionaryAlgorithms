import numpy as np
import sympy

def getStepSize_WolfeConditions(t, x, p, f, df):
    c1 = 1e-4
    c2 = 0.9 
    flag = False

    while not flag:
        # Wolfe Condition 1 (Sufficient Decrease)
        if f(*(x + t*p)) > f(*x) + c1*t*np.matmul(df(*x).T, p):
            t /= 2
        # Wolfe Condition 2 (Curvature)
        elif np.matmul(df(*(x+t*p)).T, p) < c2*np.matmul(df(*x).T, p):
            t *= 2
        # Both conditions are met
        else:
            flag = True

    return t

def gradientDescent(t, x, func, tol, *args):

    variables = args
    # Definition of the 1st Derivative (Jacobian Matrix)
    jacobianMat = sympy.Matrix([[sympy.diff(func, var)] for var in variables])

    # Lambdification of functions to allow substitution f(x)
    f = sympy.lambdify(variables, func, modules="numpy")
    df = sympy.lambdify(variables, jacobianMat, modules="numpy")

    # To store the history of solutions
    solHistory = [x.copy()]

    while True:
        # Compute the search direction -df(x)
        p = -np.array(df(*x)).astype(float).flatten()

        # Compute the step size
        t = getStepSize_WolfeConditions(t, x, p, f, df)

        # Update the current solution
        x += t * p

        solHistory.append(x.copy())

        # Exit condition (when tolerance is exceeded)
        if np.linalg.norm(np.array(df(*x)).astype(float).flatten()) < tol:
            break

    return solHistory



##############################################################
##################### NEWTON'S METHOD ########################
##############################################################
def newtonMethod(t, x, func, tol, *args):

    variables = args 
    # Definition of the 1st Derivative (Jacobian Matrix)
    jacobianMat = sympy.Matrix([[sympy.diff(func, var)] for var in variables])
    # Definition of the 2nd Derivative (Hessian Matrix)
    hessianMat = sympy.hessian(func, variables)

    # Lambdification of functions to allow substitution f(x)
    f = sympy.lambdify(variables, func, modules="numpy")
    df = sympy.lambdify(variables, jacobianMat, modules="numpy")
    ddf = sympy.lambdify(variables, hessianMat, modules="numpy")

    # To store the history of solutions
    solHistory = [x.copy()]

    while True:
        # Update the current solution
        x = x - np.matmul(np.linalg.inv(np.array(ddf(*x))), (grad := np.array(df(*x)).flatten()))

        solHistory.append(x.copy())

        # Exit condition (when tolerance is exceeded)
        if np.linalg.norm(grad) < tol:
            break

    return solHistory


##############################################################
####################### HILL CLIMBING ########################
##############################################################

def hill_climbing(t, x, func, tol, *args):

    # t param : 
    # x param :
    # func param :
    # tol param :
    # args param : 

    variables = args
    
    # Start point
    #agragar lo de que regrese la lista de soluciones
    current_solution = [0.5,1]
    current_value = objective_functionb(current_solution[0], current_solution[1])
    
    step_size = 0.01
    max_iterations = 1000

    #counter to see in how many iterations the solution is reached
    c = 0

    for _ in range(max_iterations):
        x, y = current_solution
        
        # Generate neighbors by adjusting each variable separately + -
        neighbors = [
            [x + step_size, y],
            [x - step_size, y],
            [x, y + step_size],
            [x, y - step_size]
        ]
        
        # Evaluate neighbors and select the one that will give the maximum function
        next_solution = min(neighbors, key=lambda sol: objective_functionb(sol[0], sol[1]))
        next_value = objective_functionb(next_solution[0], next_solution[1])
        
        # Check if the neighbor is better
        if next_value < current_value:
            current_solution = next_solution
            current_value = next_value
            c += 1
        else:
            # If no better neighbors, return current solution
            break

    return current_solution, current_value, c

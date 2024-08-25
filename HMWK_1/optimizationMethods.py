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

    print(func)

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

    # t param : Step size
    # x param : Start point
    # func param : Function
    # tol param : Tolerance
    # args param : variables (x1, x2)

    variables = args
    current_solution = x
    x1, x2 = sympy.symbols('x1 x2')
    current_value = func.subs({x1: x[0], x2: x[1]}).evalf()


    
    # Start point
    #agragar lo de que regrese la lista de soluciones
    #current_solution = [0.5,1]
    #current_value = objective_functionb(current_solution[0], current_solution[1])
    
    step_size = t
    max_iterations = 1000

    # To store the history of solutions
    solHistory = [x.copy()]

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

        next_solution = min(neighbors, key=lambda sol: func.subs({x1: sol[0], x2: sol[1]}).evalf())
        next_value = func.subs({x1: next_solution[0], x2: next_solution[1]}).evalf()
    
        
        # Check if the neighbor is better
        if next_value < current_value:
            current_solution = next_solution
            current_value = next_value
            solHistory.append(current_solution.copy())
        else:
            # If no better neighbors, return current solution
            break

    return solHistory

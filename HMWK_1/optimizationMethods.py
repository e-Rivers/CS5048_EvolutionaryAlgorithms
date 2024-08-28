import numpy as np
import sympy
import math

##############################################################
#################### GRADIENT DESCENT ########################
##############################################################

def getStepSize_WolfeConditions(t, x, p, f, df):

    """
    This function is an extention for the Gradient Descent (and could
    even be used for the Newton Method) to compute the step size based
    on the Wolfe Conditions

    -----------
    Parameters:
    
    t : float
        current step size

    x : array
        current point coordinates

    p : array
        direction in which to move

    f : lambidied sympy expression
        objective function to minimize

    df : lambidied sympy expression
        first derivative of the objective function

    ----------------
    Output: float
        Returns the new step size for this iteration.
    """

    # Default values for the constants and parameters
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
        # Both conditions are met (exit the loop)
        else:
            flag = True

    return t

def gradientDescent(t, x, func, tol, *args):

    """
    This function performs the Gradient Descent optimization method.

    -----------
    Parameters:
    
    t : float
        initial guess of step size

    x : array
        starting point coordinates

    func : sympy expression
        objective function to minimize

    tol : float
    tolerance

    args : 
        Additional arguments (e.g., variables x1, x2)

    ----------------
    Output: list 
        It returns the history of the solutions explored, 
        being the last one the point that according to this 
        method minimize the objective function.
    """

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

    """
    This function performs the Newton's optimization method.

    -----------
    Parameters:
    
    t : float
        initial guess of step size

    x : array
        starting point coordinates

    func : sympy expression
        objective function to minimize

    tol : float
    tolerance

    args : 
        Additional arguments (e.g., variables x1, x2)

    ----------------
    Output: list 
        It returns the history of the solutions explored, 
        being the last one the point that according to this 
        method minimize the objective function.
    """

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

def generate_neighbors(current_solution, step_size, radius):

    """
    This function generates the neighborhood.

    -----------
    Parameters:

    current_solution : array
        The current solution coordinates (x, y).

    step_size : float 
        The step size to move in the neighborhood.

    radius : int
        The radius within which to generate neighbors.

    -----------
    Output: list    
        A list of neighboring solutions.

    """
    x, y = current_solution
    neighbors = []
    
    # Generate neighbors within the specified radius
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            distance = math.sqrt(dx**2 + dy**2)
            if distance <= radius:
                neighbors.append([x + dx * step_size, y + dy * step_size])
    
    return neighbors


def hill_climbing(t, x, func, tol, *args):

    """
    This function performs the Hill Climber optimization method.

    -----------
    Parameters:
    
    t : float
        initial guess of step size

    x : array
        starting point coordinates

    func : sympy expression
        objective function to minimize

    tol : float
    tolerance

    args : 
        Additional arguments (e.g., variables x1, x2)

    ----------------
    Output: list
        It returns the history of the solutions explored, 
        being the last one the point that according to this 
        method minimize the objective function.
    """

    # Rename variables for readability
    step_size = t
    max_iterations = 1000
    current_solution = x

    # Function evaluation for the starting point
    x1, x2 = sympy.symbols('x1 x2')
    current_value = func.subs({x1: x[0], x2: x[1]}).evalf()

    # To store the history of solutions
    solHistory = [current_solution.copy()]


    # Iterate until the solution is found
    for _ in range(max_iterations):
        x, y = current_solution
        
        #Generate neighbors within a specified radius
        radius = 2
        neighbors = generate_neighbors(current_solution, step_size, radius)
        
        # Evaluate neighbors and select the one that minimizes the objective function
        next_solution = min(neighbors, key=lambda sol: func.subs({x1: sol[0], x2: sol[1]}).evalf())
        next_value = func.subs({x1: next_solution[0], x2: next_solution[1]}).evalf()
    
        
        # If the best neighbor improves the current solution, move to it
        if next_value < current_value:
            current_solution = next_solution
            current_value = next_value
            solHistory.append(current_solution.copy())
        else:
            # If no improvement, stop the search
            break

    return solHistory

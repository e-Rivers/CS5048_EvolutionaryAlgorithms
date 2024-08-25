import numpy as np
import sympy

def getStepSize_WolfeConditions(t, x, p, f, df):
    c1 = 1e-4
    c2 = 0.9 
    flag = False

    while not flag:
        print(t)
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
# Newton's Method
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


# Bologna,  28/11/2024
# Flexible Robotic Arm Equilibrium Points
# This Python file implements the Newton Method for root finding in order to find zeros for f(x) dynamic function

from numpy import *
from dynamics import dynamicFRA as f
from parameters import *

def getEquilibriumPoints(uu):
    """
    Equilibrium points of the Flexible Robotic Arm for input uu
    Arguments:
    - uu: 1x1 input value to use to compute Flexible Robotic Arm equilibria
    Returns:
    - xeq: 4x1 column vector equilibrium states (recall that x=[x0, x1, x2, x3]'=[ϑ1, ϑ2, dϑ1dt, dϑ2dt]')
    """
    # Notice that the definition of the discretized function f=[f0, f1, f2, f3]' is the following:
    # f[0] = xx[0] + dt*xx[2]
    # f[1] = xx[1] + dt*xx[3]
    # f[2:4] = xx[2:4] + dt*invM@(U-G-F@array([xx[2], xx[3]])-C)

    uu = uu.squeeze() # remove eventually present singleton dimensions

    maximumIteration = int(5e3)
    stepsize = 1e-2
    tol = 1e-3

    ϑ1_initial = pi/2;
    ϑ2_initial = pi/4;
    xx = zeros((ns, maximumIteration))
    xx[:,0] = array([ϑ1_initial, ϑ2_initial, 0, 0]);

    epIteration = 1;

    for i in range(maximumIteration-1):
        print("Iteration number: ", i)
        print(xx[:,i])
        xxp, dfdx, dfdu = f(xx[:,i], uu)
        direction = -linalg.inv(dfdx)@xxp;
        xx[:,i+1] = xx[:,i] + (stepsize*direction).reshape(1, ns);
        print(xx[:,i+1])
        if (abs(xx[:,i+1]-xx[:,i]) < tol).all:
            eqIteration = epIteration+1
            break
    if eqIteration < 2:
        raise TimeoutError("No equilibrium point found in the maximum number of iterations")
    
    return xx[:,i+1];

print(getEquilibriumPoints(array([0])));
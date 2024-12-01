# Bologna,  28/11/2024
# Flexible Robotic Arm Equilibrium Points

from numpy import *
from dynamics import discretizedDynamicFRA as f
from parameters import *

def getEquilibriumPoints(uu, xx0):
    """
    Equilibrium points of the Flexible Robotic Arm for input uu.
    These points are computed using the Newton Method for zero root finding applied on the function
    r(x)=f(x)-x where f(x) is the discretized dynamic function of the Flexible Robotic Arm.
    Arguments:
    - uu: 1x1 input value to use to compute Flexible Robotic Arm equilibria
    - xx0: 4x1 column vector state initial guess for the equilibrium point (recall that x=[x0, x1, x2, x3]'=[ϑ1, ϑ2, dϑ1dt, dϑ2dt]')
    Returns:
    - xeq: 4x1 column vector equilibrium states
    """
    # Notice that the definition of the discretized function f=[f0, f1, f2, f3]' is the following:
    # f[0] = xx[0] + dt*xx[2]
    # f[1] = xx[1] + dt*xx[3]
    # f[2:4] = xx[2:4] + dt*invM@(U-G-F@array([xx[2], xx[3]])-C)

    uu = uu.squeeze() # remove eventually present singleton dimensions

    maximumIteration = int(5e3)
    stepsize = 1e-2
    tolerance = 1e-10
    xx = zeros((ns, maximumIteration))
    xx[:,0] = xx0;
    solved = False;

    for i in range(maximumIteration-1):
        
        xxp, dfdx, dfdu = f(xx[:,i], uu)
        xxp = xxp - xx[:,i].reshape(ns, 1)
        dfdx = dfdx - eye(ns)
        direction = -linalg.inv(dfdx)@xxp;
        xx[:,i+1] = xx[:,i] + (stepsize*direction).reshape(1, ns);
        # print("Iteration number: ", i)
        # print("Jacobian\n:", dfdx)
        # print("New value:", xx[:,i+1])
        if (abs(xx[:,i+1]-xx[:,i]) < tolerance).all(): # if linalg.norm(direction) < tolerance:
            solved = True
            break
    if not solved:
        raise TimeoutError("No equilibrium point found in " + str(maximumIteration) + " maximum number of iterations")
    
    return xx[:,i+1];


# Equilibrium points search (notice that different initial guesses lead to different equilibrium points):

print("\nEquilibrium points FRA:")

ϑ1_initial = pi - pi/180*20
ϑ2_initial = 0
print(getEquilibriumPoints(
    array([0]),
    array([ϑ1_initial, ϑ2_initial, 0, 0])
));

ϑ1_initial = pi - pi/180*20
ϑ2_initial = -pi/180*90
print(getEquilibriumPoints(
    array([0]),
    array([ϑ1_initial, ϑ2_initial, 0, 0])
));

ϑ1_initial = pi/180*20
ϑ2_initial = pi - pi/180*20
print(getEquilibriumPoints(
    array([0]),
    array([ϑ1_initial, ϑ2_initial, 0, 0])
));

ϑ1_initial = pi/180*20
ϑ2_initial = pi/180*20
print(getEquilibriumPoints(
    array([0]),
    array([ϑ1_initial, ϑ2_initial, 0, 0])
));
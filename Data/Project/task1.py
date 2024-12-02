# Bologna,  30/11/2024
# Flexible Robotic Arm Task1: from a desired step trajectory to an optimal one
# thanks to the regularized Newton's like Method (in its closed-loop version)

from parameters import *
from numpy import *

dt = dtCollection.task0_discretizationStep;
T = TCollection.task1_trajectoryDuration;

TT = int(T/dt) # number of time steps (each one of duration dt, enough for evolve from t=0 to t=T)

# Definition of the desired trajectory (step from eq. point [0, 0, 0, 0]' to eq. point [pi, 0, 0, 0]')
xx_des = zeros((ns, TT))
uu_des = zeros((ni, TT))
xx_des[0, int(TT/2):] = ones((1, int(TT/2)))*pi

# Notice: first of all, is needed the definition of a cost function l (with its jacobian and hessian wrt x and u)

# Now implements (in another file) N.M. with Armijo's rule, regularized, closed-loop version, in order to get a feasible opt trj.
# At each iteration, evaluate dynamic, solve lambda co-state equation, compute descendent direction solving a LQOCP (considering
# the regularized version), unpate the input (in closed-loop version) and then update the state running the dynamic!
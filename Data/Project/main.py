# Bologna,  30/11/2024
# Flexible Robotic Arm Task1: from a desired step trajectory (that evolves from one equilibrium to
# another) to an optimal one thanks to the regularized Newton's like Method (in its closed-loop version)

from parameters import *
from numpy import *
from equilibria import getAnEquilibriumPoint
from matplotlib import pyplot
from trajectories import *
from methods import *
from dynamics import discretizedDynamicFRA as discretizedDynamicFunction
from costs import stageCostTrkTrj, termCostTrkTrj

dt = dtCollection.task0_discretizationStep;
T = TCollection.task1_trajectoryDuration;

task = [1, 1, 0, 1, 0]

QQ = 0.1*diag([100, 100, 1, 1])
RR = 0.01*eye(ni)
def stageCostFunctionFRA(xx, uu, xx_des, uu_des):
    return stageCostTrkTrj(xx, uu, xx_des, uu_des, QQ, RR)
def terminalCostFunctionFRA(xT, xT_des):
    return termCostTrkTrj(xT, xT_des, QQ)

############################
########## TASK 1 ##########
############################

if(task[1]):

    TT = int(T/dt) # number of time steps (each one of duration dt, enough for evolve from t=0 to t=T)

    uu_equilibrium = 31.21523
    xx_equilibrium1 = getAnEquilibriumPoint(array([uu_equilibrium]), array([0, 0, 0, 0]))
    xx_equilibrium2 = getAnEquilibriumPoint(array([-uu_equilibrium]), array([0, 0, 0, 0]))

    uu_des, tu = sigmoidTrajectory(T, T/2, array([uu_equilibrium]), array([-uu_equilibrium]), dt)
    xx_des, tx = sigmoidTrajectory(T, T/2, xx_equilibrium1, xx_equilibrium2, dt)

    newtonMethodMaxIterations = 10
    xx_opt, uu_opt = runNewtonMethod(
    xx_des, uu_des, xx_equilibrium1, TT, newtonMethodMaxIterations,
    discretizedDynamicFunction, stageCostFunctionFRA, terminalCostFunctionFRA,
    1e-3
    )

############################
########## TASK 2 ##########
############################

if(task[2]):
    xx_des, tx = PascalSnail(TT, TT/2, dt, a = 0.5, b = 1)


############################
########## TASK 3 ##########
############################

if(task[3]):
    # Computation of local linearization around optimal trajectory and consequently LQR gain KK
    KK = ComputeLocalLin(xx_opt, uu_opt, QQ, RR, QQ, TT, discretizedDynamicFunction, solveLinearLQP)[0]

    # Design of the LQR with noise
    noise = GenerateNoise(xx_opt, noise_std_percentage=0.2)
    xx_track, uu_track = SolveLQPwithNoise(xx_opt, uu_opt, KK, noise, TT, discretizedDynamicFunction)


############################
########## TASK 4 ##########
############################

if(task[4]): 
    # Computing again local linearization aroun opt. traj. obtained through Newton's method in task 2
    AA_opt, BB_opt = ComputeLocalLin(xx_opt, uu_opt, QQ, RR, QQ, TT, discretizedDynamicFunction, solveLinearLQP)[1:]

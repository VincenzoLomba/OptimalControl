# Bologna,  30/11/2024
# Flexible Robotic Arm Task1: from a desired step trajectory (that evolves from one equilibrium to
# another) to an optimal one thanks to the regularized Newton's like Method (in its closed-loop version)

import parameters as params
from numpy import *
from equilibria import getAFRAEquilibriumPoint
from matplotlib import pyplot
from trajectories import *
from methods import runNewtonMethodTrkTrj
from dynamics import discretizedDynamicFRA as discretizedDynamicFuntion

def task1():

    dt = params.discretizationStep;
    T = params.TCollection.task1_trajectoryDuration;
    ns = params.ns
    ni = params.ni

    TT = int(T/dt) # number of time steps (each one of duration dt, enough for evolve from t=0 to t=T)

    # Searching for two symmetric equilibrium points
    uu_equlibrium = 31.21523
    xx_equilibrium1 = getAFRAEquilibriumPoint(array([uu_equlibrium]), array([0, 0, 0, 0]))
    xx_equilibrium2 = getAFRAEquilibriumPoint(array([-uu_equlibrium]), array([0, 0, 0, 0]))

    # Defining as desired input-state a sigmoid-junction between the two equilibrium points
    deltsSigmoid = T/4*3
    uu_des, tu = sigmoidTrajectory(T, deltsSigmoid, array([uu_equlibrium]), array([-uu_equlibrium]), dt)
    xx_des, tx = sigmoidTrajectory(T, deltsSigmoid, xx_equilibrium1, xx_equilibrium2, dt)

    # Defining cost matrices (for a trajectory tracking optimization problem) and applying the Newton Method
    QQ = diag([10, 10, 1, 1])
    RR = zeros((ni, ni, TT))
    RRzeroSlice = int(TT/16)
    RR[:,:,0:RRzeroSlice] = 100*eye(ni)
    RR[:,:,RRzeroSlice:TT-RRzeroSlice] = 0.01*eye(ni)
    RR[:,:,TT-RRzeroSlice:TT] = 100*eye(ni)
    QQT = 10**6*eye(ns)
    newtonMethodMaxIterations = 20
    xx_opt, uu_opt = runNewtonMethodTrkTrj(
        xx_des, uu_des, xx_equilibrium1, TT, newtonMethodMaxIterations,
        discretizedDynamicFuntion, 1e-4,
        QQ, RR, QQT, None
    )

    # Plotting results
    pyplot.close('all')
    pyplot.pause(1)
    pyplot.figure()
    for i in range(0, ns): pyplot.plot(tx, array(xx_des[i,:]), label='ϑ'+str(i)+' desired')
    pyplot.legend(); pyplot.show(block=False); pyplot.pause(0.5)
    pyplot.plot(tu, uu_des, label='u desired')
    pyplot.legend(); pyplot.show(block=False); pyplot.pause(0.5)
    _, ax = pyplot.subplots()
    for i in range(ns):
        line_des, = ax.plot(tx, xx_des[i, :], label='ϑ' + str(i) + ' desired')
        color = line_des.get_color()
        ax.plot(tx, xx_opt[i, :], '--', color=color, label='ϑ' + str(i) + ' optimal')
    ax.legend(loc='upper left')
    pyplot.show(block=False); pyplot.pause(0.5)
    _, ax = pyplot.subplots()
    line_u_des, = ax.plot(tu, uu_des, label='u desired')
    color_u = line_u_des.get_color()
    ax.plot(tu, uu_opt[0, :], '--', color=color_u, label='u optimal')
    ax.legend()
    pyplot.show();

    return xx_opt, uu_opt

if __name__ == "__main__":
    task1()
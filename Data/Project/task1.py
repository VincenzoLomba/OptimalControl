# Bologna,  30/11/2024
# Flexible Robotic Arm Task1: from a desired step trajectory (that evolves from one equilibrium to
# another) to an optimal one thanks to the regularized Newton's like Method (in its closed-loop version)

from parameters import *
from numpy import *
from equilibria import getAnEquilibriumPoint
from matplotlib import pyplot
from trajectories import *
from methods import runNewtonMethod
from dynamics import discretizedDynamicFRA as discretizedDynamicFuntion
from costs import stageCostTrkTrj, termCostTrkTrj

dt = dtCollection.task0_discretizationStep;
T = TCollection.task1_trajectoryDuration;

TT = int(T/dt) # number of time steps (each one of duration dt, enough for evolve from t=0 to t=T)

uu_equlibrium = 31.21523
xx_equilibrium1 = getAnEquilibriumPoint(array([uu_equlibrium]), array([0, 0, 0, 0]))
xx_equilibrium2 = getAnEquilibriumPoint(array([-uu_equlibrium]), array([0, 0, 0, 0]))

uu_des, tu = sigmoidTrajectory(T, T/2, array([uu_equlibrium]), array([-uu_equlibrium]), dt)
xx_des, tx = sigmoidTrajectory(T, T/2, xx_equilibrium1, xx_equilibrium2, dt)


QQ = 0.1*diag([100, 100, 1, 1])
RR = 0.01*eye(ni)
def stageCostFunctionFRA(xx, uu, xx_des, uu_des):
    return stageCostTrkTrj(xx, uu, xx_des, uu_des, QQ, RR)
def terminalCostFunctionFRA(xT, xT_des):
    return termCostTrkTrj(xT, xT_des, QQ)
newtonMethodMaxIterations = 10
xx_opt, uu_opt = runNewtonMethod(
   xx_des, uu_des, xx_equilibrium1, TT, newtonMethodMaxIterations,
   discretizedDynamicFuntion, stageCostFunctionFRA, terminalCostFunctionFRA,
   1e-3
)

for i in range(0, ns):
    pyplot.plot(tx, array(xx_des[i,:]), label='ϑ'+str(i)+' desired')
pyplot.legend(); pyplot.show()
pyplot.plot(tu, uu_des, label='u desired')
pyplot.legend(); pyplot.show()

fig, ax = pyplot.subplots()

for i in range(ns-2):
    line_des, = ax.plot(tx, xx_des[i, :], label='ϑ' + str(i) + ' desired')
    color = line_des.get_color()
    ax.plot(tx, xx_opt[i, :], '--', color=color, label='ϑ' + str(i) + ' optimal')

ax.legend()
pyplot.show()

fig, ax = pyplot.subplots()
line_u_des, = ax.plot(tu, uu_des, label='u desired')
color_u = line_u_des.get_color()
ax.plot(tu, uu_opt[0, :], '--', color=color_u, label='u optimal')

ax.legend()
pyplot.show()

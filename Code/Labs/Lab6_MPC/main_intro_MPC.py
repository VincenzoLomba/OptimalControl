# Model Predictive Control for Linear Systems
# One round prediction - for introduction
# OPTCON 2024
# Lorenzo Sforni, Marco Falotico
#

import numpy as np

import matplotlib.pyplot as plt

from dynamics import nominal_dynamics

from solver import unconstrained_lqr,solver_linear_mpc


ns = 2
ni = 1

xx = np.zeros((ns,1))
uu = np.zeros((ni,1))

###########################
#MPC Parameters Initialization
###########################

xx0 = np.array([[3],[0]])
umax = 1
umin = -umax
x1max = 3                   # State constraints
x1min = -x1max
x2max = 2
x2min = -x2max
T_pred = 5

########################
# Linear Dynamics - get nominal A,B matrices
########################

fx, fu = nominal_dynamics(xx,uu)[1:]

AAnom = fx.T  # nominal A
BBnom = fu.T  # nominal B

########################
# Cost
########################

# state cost

QQ = np.eye(ns) 
QQf = 10*QQ

# input cost

r = 5e-1
RR = r*np.eye(ni) 


#####################
# Simple prediction - solve only one MPC problem
####################

xx_mpc, uu_mpc = solver_linear_mpc(AAnom, BBnom, QQ, RR, QQf, xx0, x1_max=x1max, x1_min=x1min, x2_max=x2max, x2_min=x2min, umax=umax, umin=umin, T_pred=T_pred)[1:]


#######################################
# Plots
#######################################


T_pred = np.shape(xx_mpc)[1]

time = np.arange(T_pred)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(time, xx_mpc[0,:], linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_1$')

axs[0].plot(time, np.ones(T_pred)*x1max, '--g', linewidth=1.5)
axs[0].plot(time, np.ones(T_pred)*x1min, '--g', linewidth=1.5)


axs[1].plot(time, xx_mpc[1,:], linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$x_2$')

axs[1].plot(time, np.ones(T_pred)*x2max, '--g', linewidth=1.5)
axs[1].plot(time, np.ones(T_pred)*x2min, '--g', linewidth=1.5)

axs[2].plot(time, uu_mpc[0,:],'r', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$u$')
axs[2].set_xlabel('time')


axs[2].plot(time, np.ones(T_pred)*umax, '--g', linewidth=1.5)
axs[2].plot(time, np.ones(T_pred)*umin, '--g', linewidth=1.5)

fig.align_ylabels(axs)

plt.show()


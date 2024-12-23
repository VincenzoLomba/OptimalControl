# Model Predictive Control for Linear Systems
# OPTCON 2024
# Lorenzo Sforni, Marco Falotico
#

import numpy as np

import matplotlib.pyplot as plt

from dynamics import nominal_dynamics, real_dynamics

import control as ctrl

from solver import unconstrained_lqr,solver_linear_mpc

ns = 2
ni = 1

xx = np.zeros((ns,1))
uu = np.zeros((ni,1))

###############################
#MPC Parameters Initialization
###############################

xx0 = np.array([[15],[0]])  # initial condition
Tsim = 50                   # simulation horizon

T_pred = 5                  # MPC Prediction horizon
umax = 10                   # Upper input constraint
umin = -umax                # Lower input constraint

x1max = 20                  # State constraints
x1min = -x1max
x2max = 5
x2min = -x2max

########################
# Cost
########################

# state cost

QQ = np.eye(ns) 
QQf = 10*QQ

# input cost

r = 5e-1
RR = r*np.eye(ni) 


########################
# Linear Dynamics - get nominal A,B matrices
########################

fx, fu = nominal_dynamics(xx,uu)[1:]

AAnom = fx.T  # nominal A
BBnom = fu.T  # nominal B

#############################
# Unconstrained LQR solution
#############################


xx_opt, uu_opt = unconstrained_lqr(AAnom, BBnom, QQ, RR, QQf, xx0, Tsim)  # optimal trajectory

xx_real_opt = np.zeros((ns,Tsim))
uu_real_opt = np.zeros((ni,Tsim))

xx_real_opt[:,0] = xx0.squeeze()

for tt in range(Tsim-1):

    if tt%5 == 0: # print every 5 time instants
      print('LQR:\t t = {}'.format(tt))

    # System evolution - real with optimal control input (open-loop)
    uu_real_opt[:,tt] = uu_opt[:,tt]

    xx_real_opt[:,tt+1] = real_dynamics(xx_real_opt[:,tt], uu_real_opt[:,tt])[0]


#############################
# Model Predictive Control
#############################

xx_real_mpc = np.zeros((ns,Tsim))
uu_real_mpc = np.zeros((ni,Tsim))

xx_mpc = np.zeros((ns, T_pred, Tsim))

xx_real_mpc[:,0] = xx0.squeeze()

for tt in range(Tsim-1):
    # System evolution - real with MPC

    xx_t_mpc = xx_real_mpc[:,tt] # get initial condition

    # Solve MPC problem - apply first input

    if tt%5 == 0: # print every 5 time instants
      print('MPC:\t t = {}'.format(tt))

    uu_real_mpc[:,tt], xx_mpc[:,:,tt] = solver_linear_mpc(AAnom, BBnom, QQ,RR, QQf, xx_t_mpc, umax=umax, umin=umin, x1_min=x1min, x1_max=x1max, x2_min = x2min, x2_max = x2max, T_pred = T_pred)[:2]
    
    xx_real_mpc[:,tt+1] = real_dynamics(xx_real_mpc[:,tt], uu_real_mpc[:,tt])[0]

#######################################
# Plots
#######################################

time = np.arange(Tsim)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(time, xx_real_mpc[0,:], linewidth=2)
axs[0].plot(time, xx_real_opt[0,:],'--r', linewidth=2)

axs[0].plot(time, np.ones(Tsim)*x1max, '--g', linewidth=1.5)
axs[0].plot(time, np.ones(Tsim)*x1min, '--g', linewidth=1.5)


axs[0].grid()
axs[0].set_ylabel('$x_1$')

axs[0].set_xlim([-1,Tsim])
axs[0].legend(['MPC', 'LQR'])

axs[1].plot(time, xx_real_mpc[1,:], linewidth=2)
axs[1].plot(time, xx_real_opt[1,:], '--r', linewidth=2)

axs[1].plot(time, np.ones(Tsim)*x2max, '--g', linewidth=1.5)
axs[1].plot(time, np.ones(Tsim)*x2min, '--g', linewidth=1.5)

axs[1].grid()
axs[1].set_ylabel('$x_2$')

axs[1].set_xlim([-1,Tsim])
axs[1].legend(['MPC', 'LQR'])


axs[2].plot(time, uu_real_mpc[0,:],'g', linewidth=2)
axs[2].plot(time, uu_real_opt[0,:],'--r', linewidth=2)

axs[2].plot(time, np.ones(Tsim)*umax, '--g', linewidth=1.5)
axs[2].plot(time, np.ones(Tsim)*umin, '--g', linewidth=1.5)

axs[2].grid()
axs[2].set_ylabel('$u$')
axs[2].set_xlabel('time')

axs[2].set_xlim([-1,Tsim])
axs[2].legend(['MPC', 'LQR'])


fig.align_ylabels(axs)

plt.show()


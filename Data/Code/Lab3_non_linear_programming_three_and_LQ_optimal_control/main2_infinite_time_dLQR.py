#
# Infinite-time LQR for regulation
# Lorenzo Sforni, Marco Falotico
# Bologna, 8/11/2024
#

import numpy as np
import matplotlib.pyplot as plt

import control as ctrl  #control package python


# Import mass-spring-damper cart dynamics
import dynamics as dyn

# Import LTI LQR solver
from solver_lti_LQR import lti_LQR
# from homework_solver_lti_LQR_gaps import lti_LQR
# from homework_solver_ltv_LQR_gaps import ltv_LQR

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})



#######################################
# Parameters
#######################################

tf = 1 # final time in seconds
# tf = 10 # final time in seconds

dt = dyn.dt   # get discretization step from dynamics

TT = int(tf/dt) # discrete-time samples


#######################################
# Dynamics
#######################################

ns = 2
ni = 1

x0 = np.array([2, 1])

xdummy = np.array([0, 0])
udummy = np.array([0])

dxf,duf = dyn.dynamics(xdummy,udummy)[1:]

AA = dxf.T
BB = duf.T


#######################################
# Cost
#######################################

QQ = np.array([[1e2, 0], [0, 1]])
RR = 1e-2*np.eye(ni)
# RR = 1e-1*np.eye(ni)
# RR = 1e-0*np.eye(ni)
# RR = 1e1*np.eye(ni)

# different possibilities
QQf = np.array([[1e2, 0], [0, 100]])


# Infinite-horizon gain
P_inf,_,GG = ctrl.dare(AA,BB,QQ,RR)

KK_inf = -GG


#######################################
# Main
#######################################

xx_inf = np.zeros((ns,TT))
uu_inf = np.zeros((ni,TT))

KK_fin, PP_fin = lti_LQR(AA,BB,QQ,RR,QQf,TT) # for comparison
xx_fin = np.zeros((ns,TT))
uu_fin = np.zeros((ni,TT))

xx_inf[:,0] = x0
xx_fin[:,0] = x0

for tt in range(TT-1):
  
  # infinite hor
  uu_inf[:,tt] = KK_inf@xx_inf[:,tt]
  xx_inf[:,tt+1] = AA@xx_inf[:,tt] + BB@uu_inf[:,tt]

  # finite hor
  uu_fin[:,tt] = KK_fin[:,:,tt]@xx_fin[:,tt]
  xx_fin[:,tt+1] = AA@xx_fin[:,tt] + BB@uu_fin[:,tt]


#######################################
# Plots
#######################################

tt_hor = np.linspace(0,tf,TT)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')


axs[0].plot(tt_hor, xx_fin[0,:],'b--' ,linewidth=2, label = 'fin')
axs[0].plot(tt_hor, xx_inf[0,:],'b',linewidth=2, label = 'inf')
axs[0].grid()
axs[0].set_ylabel('$x_1$')

axs[1].plot(tt_hor, xx_fin[1,:],'b--', linewidth=2, label = 'fin')
axs[1].plot(tt_hor, xx_inf[1,:],'b',linewidth=2, label = 'inf')

axs[1].grid()
axs[1].set_ylabel('$x_2$')

axs[2].plot(tt_hor, uu_fin[0,:],'r--', linewidth=2, label = 'fin')
axs[2].plot(tt_hor, uu_inf[0,:],'r', linewidth=2, label = 'inf')
axs[2].grid()
axs[2].set_ylabel('$u$')
axs[2].set_xlabel('time')


fig.align_ylabels(axs)



#######################################
# Gain Comparison Plot Preparation
#######################################

# Prepare arrays for plotting the gains
K_inf_values = np.tile(KK_inf.flatten(), (TT, 1)).T  # Constant over time
K_fin_values = np.array([KK_fin[:, :, tt].flatten() for tt in range(TT)]).T  # Varies over time

# Time vector for plotting
tt_hor = np.linspace(0, tf, TT)

# Create a new figure for the gain plots
fig_K, axs_K = plt.subplots(KK_inf.size, 1, sharex='all')
fig_K.suptitle("Gain Matrix K Comparison (Infinite vs. Finite Time LQR)")

# print(KK_inf.shape);exit()

# Plot each element of the K matrix
for i in range(KK_inf.shape[1]):
    axs_K[i].plot(tt_hor, (K_fin_values[i, :]), 'g--', linewidth=2, label='Finite Time')
    axs_K[i].plot(tt_hor, (K_inf_values[i, :]), 'g', linewidth=2, label='Infinite Time')
    axs_K[i].grid()
    axs_K[i].set_ylabel(f'$K_{{{i//KK_inf.shape[1]+1},{i%KK_inf.shape[1]+1}}}$')
    if i == 0:
        axs_K[i].legend()

axs_K[-1].set_xlabel('Time')

fig_K.align_ylabels(axs_K)

######################################
# P Matrix Comparison Plot Preparation
######################################

# Prepare arrays for plotting the P values
P_inf_values = np.tile(P_inf.flatten(), (TT, 1)).T  # Constant over time
P_fin_values = PP_fin.reshape((PP_fin.shape[0]**2,TT))

# Create a new figure for the P matrix plots
fig_P, axs_P = plt.subplots(P_inf.size, 1, sharex='all')
fig_P.suptitle("P Matrix Comparison (Infinite vs. Finite Time LQR)")

# Plot each element of the P matrix
for i in range(P_inf.size):
    axs_P[i].plot(tt_hor, (P_fin_values[i, :]), 'm--', linewidth=2, label='Finite Time', color = "b")
    axs_P[i].plot(tt_hor, (P_inf_values[i, :]), 'm', linewidth=2, label='Infinite Time', color = "b")
    axs_P[i].grid()
    axs_P[i].set_ylabel(f'$P_{{{i//P_inf.shape[1]+1},{i%P_inf.shape[1]+1}}}$')
    if i == 0:
        axs_P[i].legend()

axs_P[-1].set_xlabel('Time')

fig_P.align_ylabels(axs_P)



plt.show()









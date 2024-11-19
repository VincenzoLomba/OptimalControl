# Simulation of a Discrete-time Linear System
# Lorenzo Sforni
# Bologna, 21/09/2023

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
nx = 3 # number of states
nu = 1 # number of inputs
ny = 2 # number of outputs

# Diagonal A
dA = 2*(np.random.rand(nx)-0.5) # dA = np.random.rand(nx)
A = np.diag(dA)
print("The eigenvalues of A are = ", np.linalg.eigvals(A))

B = np.random.rand(nx,nu)
C = np.random.rand(ny,nx)
D = np.zeros((ny,nu))

T = 20
horizon = np.arange(0,T)

###############################################################################
# uu=0*horizon
# uu=np.cos(horizon)
uu = np.random.rand(T,nu)
# print("size = ", np.size(uu))
xx = np.zeros((T,nx))
yy = np.zeros((T,ny))
x0 = np.random.rand(nx)
xx[0] = x0 # xx[0,:] = x0

for t in range(T-1):
  xx[t+1] = A@xx[t] + B@uu[t]
  yy[t] = C@xx[t] + D@uu[t]

###############################################################################
fig, axs = plt.subplots(3)
fig.tight_layout(pad=3.0) # to add spacing between subplots
fig.suptitle('Evolution of the state and the input')

axs[0].plot(horizon, xx, '--', linewidth=3)
axs[0].grid()
axs[0].set_xlim(0,T)
axs[0].set_ylim(np.min(xx),np.max(xx))
axs[0].set_ylabel(r"$x(t)$")

axs[1].plot(horizon, yy, '--', linewidth=3)
axs[1].grid()
axs[1].set_xlim(0,T)
axs[1].set_ylim(np.min(yy),np.max(yy))
axs[1].set_ylabel(r"$y(t)$")

axs[2].plot(horizon, uu, '--', linewidth=3)
axs[2].grid()
axs[2].set_xlim(0,T)
axs[2].set_ylim(np.min(uu),np.max(uu))
axs[2].set_ylabel(r"$u(t)$")

axs[2].set_xlabel(r"Time $t$")

plt.show()

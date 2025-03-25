#
# Gradient method for Optimal Control
# Main
# Lorenzo Sforni, Marco Falotico
# Bologna, 8/11/2024
#


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# import pendulum dynamics
import dynamics as dyn

# import cost functions
import cost as cst

# import armijo stepsize selector
import armijo

# import reference trajectory generator
import reference_trajectory as ref_gen



# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

#######################################
# Algorithm parameters
#######################################

max_iters = int(1e1)
# max_iters = int(3e2)
fixed_stepsize = 0.1

# ARMIJO PARAMETERS
stepsize_0 = 1
Armijo = True
cc = 0.5
beta = 0.7
armijo_maxiters = 20 # number of Armijo iterations

term_cond = 1e-6

visu_descent_plot = True

#######################################
# Trajectory parameters
#######################################

tf = 10 # final time in seconds

dt = dyn.dt   # get discretization step from dynamics
ns = dyn.ns
ni = dyn.ni

TT = int(tf/dt) # discrete-time samples

######################################
# Reference curve
######################################

step_reference = True

xx_ref, uu_ref = ref_gen.gen(step_reference=step_reference,tf=tf,dt=dt,ns=ns,ni=ni)

x0 = xx_ref[:,0]

######################################
# Initial guess
######################################

xx_init = np.zeros((ns, TT))
uu_init = np.zeros((ni, TT))

######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.

lmbd = np.zeros((ns, TT, max_iters)) # lambdas - costate seq.

deltau = np.zeros((ni,TT, max_iters)) # Du - descent direction
dJ = np.zeros((ni,TT, max_iters))     # DJ - gradient of J wrt u

JJ = np.zeros(max_iters)      # collect cost
descent = np.zeros(max_iters) # collect descent direction
descent_arm = np.zeros(max_iters) # collect descent direction

######################################
# Main
######################################

print('-*-*-*-*-*-')

kk = 0

xx[:,:,0] = xx_init
uu[:,:,0] = uu_init

for kk in range(max_iters-1):

  JJ[kk] = 0
  
  # calculate cost
  for tt in range(TT-1):
    temp_cost = ... #TODO
    JJ[kk] += temp_cost
  
  temp_cost = ... #TODO
  JJ[kk] += temp_cost


  ##################################
  # Descent direction calculation
  ##################################

  lmbd_temp = ... #TODO
  lmbd[:,TT-1,kk] = lmbd_temp.squeeze()

  for tt in reversed(range(TT-1)):  # integration backward in time

    qt, rt = ... #TODO
    dfx, dfu = ... #TODO

    At = ... #TODO
    Bt = ... #TODO

    lmbd_temp = ... #TODO    # costate equation
    dJ_temp = ... #TODO       # gradient of J wrt u
    deltau_temp = - dJ_temp

    lmbd[:,tt,kk] = lmbd_temp.squeeze()
    dJ[:,tt,kk] = dJ_temp.squeeze()
    deltau[:,tt,kk] = deltau_temp.squeeze()

    descent[kk] += deltau[:,tt,kk].T@deltau[:,tt,kk]
    descent_arm[kk] += dJ[:,tt,kk].T@deltau[:,tt,kk]

  ##################################
  # Stepsize selection - ARMIJO
  ##################################

  if Armijo:
    stepsize = armijo.select_stepsize(stepsize_0, armijo_maxiters, cc, beta,
                                deltau[:,:,kk], xx_ref, uu_ref, x0, 
                                uu[:,:,kk], JJ[kk], descent_arm[kk], visu_descent_plot)
  else:
     stepsize = fixed_stepsize

    
  ############################
  # Update the current solution
  ############################

  xx_temp = np.zeros((ns,TT))
  uu_temp = np.zeros((ni,TT))

  xx_temp[:,0] = x0

  for tt in range(TT-1):
    uu_temp[:,tt] = ... #TODO
    xx_temp[:,tt+1] = ... #TODO

  xx[:,:,kk+1] = xx_temp
  uu[:,:,kk+1] = uu_temp

  ############################
  # Termination condition
  ############################

  print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk,descent[kk], JJ[kk]))

  if descent[kk] <= term_cond:

    max_iters = kk

    break

xx_star = xx[:,:,max_iters-1]
uu_star = uu[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2] # for plotting purposes

############################
# Plots
############################

# cost and descent

plt.figure('descent direction')
plt.plot(np.arange(max_iters), descent[:max_iters])
plt.xlabel('$k$')
plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
plt.yscale('log')
plt.grid()
plt.show(block=False)


plt.figure('cost')
plt.plot(np.arange(max_iters), JJ[:max_iters])
plt.xlabel('$k$')
plt.ylabel('$J(\\mathbf{u}^k)$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

# optimal trajectory

tt_hor = np.linspace(0,tf,TT)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(tt_hor, xx_star[0,:], linewidth=2)
axs[0].plot(tt_hor, xx_ref[0,:], 'g--', linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_1$')

axs[1].plot(tt_hor, xx_star[1,:], linewidth=2)
axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$x_2$')

axs[2].plot(tt_hor, uu_star[0,:],'r', linewidth=2)
axs[2].plot(tt_hor, uu_ref[0,:], 'r--', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$u$')
axs[2].set_xlabel('time')
  

plt.show()
